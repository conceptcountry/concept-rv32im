"""Can the processor be bolted onto quantized text layers?

quant_probe.py answered a different question: it quantized the CHIP layers and found that
every format breaks the machine. But the artifact worth shipping is the other arrangement
-- the 11 chip layers kept in bf16, Qwen's 64 text layers quantized -- and there is a
strong prior that this works, because during a program trace Qwen's layers are never
executed. c27.py says so directly: "Qwen's layers are never run (a program trace is chip
tokens only)."

A prior is not a result. Two things have to hold for a mixed-precision artifact to be
honest, and they are different claims:

  1. the machine stays exact. Quantizing weights that are off the trace path should change
     nothing, and if it does, the "never executed" assumption is wrong and we want to know.
  2. the text bridge still works. The calc tool call DOES run Qwen's layers: the host emits
     the tool call, the machine computes, and the host reads the answer back. That path is
     quantization-sensitive in the ordinary way, and it is what a user would actually hit.

This probe quantizes only the text layers, in the same grids as quant_probe.py, and checks
both. Same caveat as before: simulating the weight grid is not simulating a kernel, so a
pass here is necessary and not sufficient.

usage: quant_probe_text.py MERGED_CKPT [--formats ...] [--progs ...]
"""
import argparse
import json
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/qwenputer"))
import stream  # noqa: E402

stream.patch_from_pretrained()
from chipformer.rv import core, elf, hf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402
from chipformer.rv.chat import RvChat  # noqa: E402
from chipformer.rv.measure import DATA  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_grids import FORMATS  # noqa: E402  the same numeric grids, one definition

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--formats", default="bf16,fp8,nvfp4a16,q8_0,q4_k_m")
ap.add_argument("--progs", default="rv32ui-add,rv32um-mul,c-crc32")
ap.add_argument("--out", default=None)
a = ap.parse_args()

chat = RvChat(a.model, executor=False)

# Qwen's own layers: everything after the chip block. These are the ones a shipped
# quantization would touch.
EXCLUDE = ("norm", "conv1d", "in_proj_a", "in_proj_b", ".gate", "A_log", "dt_bias")
originals, skipped = {}, 0
text_layers = range(chat.n_chip, len(chat.lm.layers))
meta = 0
for li in text_layers:
    for name, p in chat.lm.layers[li].named_parameters():
        if p.is_meta:
            # A meta tensor has shape and dtype but no storage: under QWP_RESIDENT the text
            # layers are never materialised during a program trace, which is itself the
            # result. Record it and move on rather than crashing.
            meta += 1
            continue
        if any(t in name for t in EXCLUDE) or p.numel() == 0:
            skipped += 1
            continue
        originals[(li, name)] = p.data.clone()
if meta:
    print(f"  {meta} text-layer tensors are META (no storage): under this residency the "
          f"text layers are never materialised during a trace", flush=True)
print(f"  {len(originals)} text-layer tensors quantizable across layers "
      f"{chat.n_chip}..{len(chat.lm.layers) - 1}, {skipped} excluded", flush=True)
print(f"  chip layers 0..{chat.n_chip - 1} stay bf16 and are never touched", flush=True)


def run_program(name):
    prog = elf.program(DATA / f"{name}.elf")
    want = ref.Machine.from_program(prog).run(10**9)
    with torch.no_grad():
        ccache = chat._chip_cache()
        x = torch.tensor([[chat.T.NL]], device=chat.dev)
        h = chat.cm.embed_tokens(x)
        for L in range(chat.n_chip):
            h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                  position_ids=None, past_key_values=ccache)
        core.snapshot(chat.c, ref.Machine.from_program(prog))
        hf.inject(chat.c, ccache)
        if chat.graph is None:
            chat._capture()
        g, nxt, got = 1, chat.v0 + S.tok("GO"), []
        while len(got) < 2 * len(want) + 100:
            t = chat.circuit_token(chat._chip_step(nxt, g, ccache))
            g += 1
            if t is None:
                break
            got.append(t - chat.v0 if t >= chat.v0 else ("text", t))
            nxt = t
            if t >= chat.v0 and S.family(t - chat.v0)[0] == "HALT":
                break
    chip_only = [t for t in got if not isinstance(t, tuple)]
    return {"exact_chip_stream": chip_only == want[1:],
            "instructions": ref.trace_stats(want)["instructions"]}


results = {}
for fmt in a.formats.split(","):
    fn = FORMATS[fmt]
    t0 = time.time()
    with torch.no_grad():
        max_rel = 0.0
        for (li, name), orig in originals.items():
            p = dict(chat.lm.layers[li].named_parameters())[name]
            w32 = orig.to(torch.float32).cpu()
            q = fn(w32)
            denom = w32.abs().max().item() or 1.0
            max_rel = max(max_rel, (q - w32).abs().max().item() / denom)
            p.data.copy_(q.to(orig.dtype).to(orig.device))
    chat.graph = None
    per = {}
    for name in a.progs.split(","):
        try:
            per[name] = run_program(name)
        except Exception as exc:
            per[name] = {"exact_chip_stream": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"  {fmt:>9} {name:<14} machine exact={per[name]['exact_chip_stream']}",
              flush=True)
    results[fmt] = {"max_relative_weight_error": max_rel, "programs": per,
                    "n_exact": sum(1 for v in per.values() if v["exact_chip_stream"]),
                    "n_programs": len(per), "seconds": time.time() - t0}
    with torch.no_grad():
        for (li, name), orig in originals.items():
            dict(chat.lm.layers[li].named_parameters())[name].data.copy_(orig)
    chat.graph = None

out = {"model": a.model, "text_layer_meta_tensors": meta,
       "note": "text layers quantized, chip layers left bf16. Weight-grid simulation only: "
               "activations are not quantized, so a pass is necessary but not sufficient.",
       "chip_layers_bf16": chat.n_chip,
       "text_layers_quantized": len(list(text_layers)),
       "n_quantized_tensors": len(originals), "n_excluded_tensors": skipped,
       "programs": a.progs.split(","), "formats": results}
print(json.dumps(out, indent=1))
if a.out:
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
