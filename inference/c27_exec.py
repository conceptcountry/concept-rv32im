"""Run a C program free-running on the chip layers of the merged 27B checkpoint, via the
Triton executor rather than the chip layers' stock HF modules.

    QWP_RESIDENT=11 python c27_exec.py MERGED --elf c-gpt4 --out c-gpt4.json

Why this exists: c27.py drives the HF module path, which measured 196 instructions/s on
the MNIST program. The GPT demonstrations are 23.6M and 60.1M instructions, which is 33
and 85 hours there. The executor runs the IR's arithmetic directly (one CUDA graph per
token) and the paper measures it at 3,769 instructions/s at batch 1, so the same programs
land in under two and under five hours.

The executor exposes a different interface from the HF view, which is why this is a
separate script rather than a flag on c27.py: there is no DynamicCache to inject into, and
a chip token's next token comes from the kept output-code dims (`code_token`) instead of
from a full residual (`circuit_token`). RvChat._exec_start seeds it with the text prompt;
every step after that is _exec_step.

Grading is identical to c27.py: `exact_chip_stream` is the metric. In a merged checkpoint
each printed byte is followed by its Qwen text-token echo, so the echoes are checked to be
exactly the printed bytes and then excluded from the comparison.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/qwenputer"))
import stream  # noqa: E402

stream.patch_from_pretrained()
from chipformer.rv import core, elf, firmware, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402
from chipformer.rv.chat import RvChat  # noqa: E402
from chipformer.rv.measure import DATA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--elf", default="c-gpt4")
ap.add_argument("--out")
ap.add_argument("--progress", type=int, default=2000,
                help="report every N chip tokens; these runs are hours long")
a = ap.parse_args()

chat = RvChat(a.model, executor=True)
prog = elf.program(DATA / f"{a.elf}.elf")
want = ref.Machine.from_program(prog).run(10**10)
print(f"[ref] {a.elf}: {len(want):,} reference tokens, "
      f"{ref.trace_stats(want)['instructions']:,} instructions", flush=True)

with torch.no_grad():
    # Order matters and mirrors RvChat.run: the program is snapshotted into the circuit
    # first, then _exec_start seeds the executor from the text prompt (one newline).
    core.snapshot(chat.c, ref.Machine.from_program(prog))
    chat._exec_start([chat.T.NL])
    nxt = chat.v0 + S.tok("GO")
    got = []
    n_code = len(chat.code_dims)
    torch.cuda.synchronize()
    t0 = time.time()
    while len(got) < 2 * len(want) + 100:
        keep = chat._exec_step(nxt)
        if nxt >= chat.v0:
            t = chat.code_token(keep[:n_code])
        else:
            t = chat.circuit_token(chat._exec_hc(nxt, keep))
        if t is None:
            break
        got.append(t - chat.v0 if t >= chat.v0 else ("text", t))
        nxt = t
        if len(got) % a.progress == 0:
            el = time.time() - t0
            print(f"  {len(got):,}/{len(want) - 1:,} tokens  {el / 60:.1f} min  "
                  f"{len(got) / el:.0f} tok/s", flush=True)
        if t >= chat.v0 and S.family(t - chat.v0)[0] == "HALT":
            break
    torch.cuda.synchronize()
    dt = time.time() - t0

exact = got == want[1:]
echo_ok, chip_only, prev = True, [], None
for t in got:
    if isinstance(t, tuple):
        echo_ok &= prev is not None and S.family(prev)[0] == "BYTE" and \
            chat.T.byte_token.get(S.family(prev)[1]) == t[1]
    else:
        chip_only.append(t)
    prev = t if not isinstance(t, tuple) else prev
exact_chip = chip_only == want[1:]
bad = next((i for i in range(min(len(got), len(want) - 1)) if got[i] != want[i + 1]), None)

res = {"model": a.model, "program": a.elf, "tokens": len(got),
       "reference_tokens": len(want) - 1,
       "context": None if bad is None else {
           "got": [S.name(t) if isinstance(t, int) else str(t)
                   for t in got[max(0, bad - 6):bad + 4]],
           "want": [S.name(t) for t in want[1:][max(0, bad - 6):bad + 4]]},
       "instructions": ref.trace_stats(want)["instructions"], "exact": exact,
       "exact_chip_stream": exact_chip,
       "echoes": sum(isinstance(t, tuple) for t in got),
       "echoes_equal_printed_bytes": bool(echo_ok),
       "halted": bool(chip_only) and S.family(chip_only[-1])[0] == "HALT",
       "first_mismatch": bad,
       "output": firmware.output_text([t for t in got if isinstance(t, int)]),
       "seconds": dt, "tok_per_s": len(got) / dt,
       "path": "Triton executor (Executor2) over the merged checkpoint's chip layers, "
               "fp32 IR arithmetic, one CUDA graph per token, batch 1"}
print(json.dumps(res), flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
