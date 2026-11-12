"""Does the processor in the weights survive quantization?

The host is a text model and quantizing a text model degrades it gracefully: a point of
GPQA, invisible in conversation. The chip layers are not a text model. They are a CPU, and
their arithmetic is exact by construction, so a perturbation does not degrade the processor
gracefully -- it breaks instruction decode and exact_chip_stream goes false.

That makes "which quantizations keep the computer a computer" an empirical question rather
than a matter of taste, and this answers it the cheap way before 80 GB of conversions get
built. For each candidate format we round-trip only the CHIP layers' weights through that
format's numeric grid, leave everything else alone, and run the ISA corpus on the result.

This is a simulation of the weight grid, not of a kernel: real FP8/NVFP4 inference also
quantizes activations and accumulates differently. A format that fails here cannot work in
practice; a format that passes here still has to be confirmed on the real artifact. The
asymmetry is the point -- this is a cheap filter, not a substitute for the sweep.

usage: quant_probe.py MERGED_CKPT [--formats bf16,fp8,nvfp4a16,q8_0,q4_k_m] [--progs ...]
"""
import argparse
import json
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/qwenputer"))
import stream  # noqa: E402

stream.patch_from_pretrained()
from chipformer.rv import core, elf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402
from chipformer.rv.chat import RvChat  # noqa: E402
from chipformer.rv.measure import DATA  # noqa: E402


from quant_grids import FORMATS  # the shipped formats' numeric grids

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--formats", default="bf16,fp8,nvfp4a16,q8_0,q4_k_m")
# Cheap but discriminating: an ALU test, a shifter, a multiplier, and one real C program.
# Program names match the ELF basenames in chipformer.rv.measure.DATA.
ap.add_argument("--progs", default="rv32ui-add,rv32ui-sll,rv32um-mul,c-crc32")
ap.add_argument("--out", default=None)
a = ap.parse_args()

chat = RvChat(a.model, executor=False)
progs = [p for p in a.progs.split(",") if p]

# The production recipes (wsp_quant.py) never quantize norms, conv1d, the GatedDeltaNet
# in_proj_a/in_proj_b, or MLP gates -- they stay bf16, matching Qwen's own FP8 layout.
# Quantizing them here would make this probe harsher than any artifact we would ship, so
# the same exclusions apply: what is tested is what would actually be quantized.
EXCLUDE = ("norm", "conv1d", "in_proj_a", "in_proj_b", ".gate", "embed_tokens", "lm_head",
           "A_log", "dt_bias")


def excluded(name):
    return any(tok in name for tok in EXCLUDE)


# Keep a pristine copy of every chip-layer weight so each format starts from bf16 rather
# than from the previous format's rounding.
originals, skipped = {}, 0
for li in range(chat.n_chip):
    for name, p in chat.cm.layers[li].named_parameters():
        if excluded(name) or p.numel() == 0:
            skipped += 1
            continue
        originals[(li, name)] = p.data.clone()
print(f"  {len(originals)} chip-layer tensors quantizable, {skipped} excluded "
      f"(norms, conv1d, in_proj_a/b, gates: bf16 in the production recipe)", flush=True)


def run_program(name):
    """Free-run one program and report whether the machine's stream is exact."""
    prog = elf.program(DATA / f"{name}.elf")
    want = ref.Machine.from_program(prog).run(10**9)
    nl = chat.T.NL
    with torch.no_grad():
        ccache = chat._chip_cache()
        x = torch.tensor([[nl]], device=chat.dev)
        h = chat.cm.embed_tokens(x)
        for L in range(chat.n_chip):
            h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                  position_ids=None, past_key_values=ccache)
        core.snapshot(chat.c, ref.Machine.from_program(prog))
        from chipformer.rv import hf
        hf.inject(chat.c, ccache)
        if chat.graph is None:
            chat._capture()
        g, nxt, got = 1, chat.v0 + S.tok("GO"), []
        while len(got) < 2 * len(want) + 100:
            hc = chat._chip_step(nxt, g, ccache)
            g += 1
            t = chat.circuit_token(hc)
            if t is None:
                break
            got.append(t - chat.v0 if t >= chat.v0 else ("text", t))
            nxt = t
            if t >= chat.v0 and S.family(t - chat.v0)[0] == "HALT":
                break
    chip_only = [t for t in got if not isinstance(t, tuple)]
    first_bad = next((i for i in range(min(len(chip_only), len(want) - 1))
                      if chip_only[i] != want[i + 1]), None)
    return {"exact_chip_stream": chip_only == want[1:],
            "instructions": ref.trace_stats(want)["instructions"],
            "first_mismatch": first_bad,
            "halted": bool(chip_only) and S.family(chip_only[-1])[0] == "HALT"}


results = {}
for fmt in a.formats.split(","):
    fn = FORMATS[fmt]
    t0 = time.time()
    with torch.no_grad():
        max_rel = 0.0
        for (li, name), orig in originals.items():
            if orig.numel() == 0:      # some chip tensors are zero-size placeholders
                continue
            p = dict(chat.cm.layers[li].named_parameters())[name]
            w32 = orig.to(torch.float32).cpu()
            q = fn(w32)
            denom = w32.abs().max().item() or 1.0
            max_rel = max(max_rel, (q - w32).abs().max().item() / denom)
            p.data.copy_(q.to(orig.dtype).to(orig.device))
    # the CUDA graph captured pointers to the old weights; force a recapture
    chat.graph = None
    per = {}
    for name in progs:
        try:
            per[name] = run_program(name)
        except Exception as exc:                      # a broken decode can raise, not just diverge
            per[name] = {"exact_chip_stream": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"  {fmt:>9} {name:<16} exact={per[name]['exact_chip_stream']}", flush=True)
    results[fmt] = {"max_relative_weight_error": max_rel,
                    "programs": per,
                    "n_exact": sum(1 for v in per.values() if v["exact_chip_stream"]),
                    "n_programs": len(per),
                    "seconds": time.time() - t0}
    # restore before the next format
    with torch.no_grad():
        for (li, name), orig in originals.items():
            if orig.numel() == 0:
                continue
            dict(chat.cm.layers[li].named_parameters())[name].data.copy_(orig)
    chat.graph = None

out = {"model": a.model, "note": "chip-layer weight-grid simulation; activations are not "
       "quantized, so a pass here is necessary but not sufficient",
       "excluded_from_quantization": list(EXCLUDE),
       "n_quantized_tensors": len(originals), "n_excluded_tensors": skipped,
       "programs": progs, "formats": results}
print(json.dumps(out, indent=1))
if a.out:
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
