#!/usr/bin/env python3
"""Is the greedy divergence caused by the bf16 cast of the rescaled final norm?

The merge rescales Qwen's final RMSNorm gain to compensate for the wider hidden
state:

    norm[:d0] = (1.0 + base_norm.float()) * scale - 1.0      # float32
    ...
    DST + "norm.weight": norm.to(torch.bfloat16)             # lossy cast

scale = sqrt(5120/6656) ~= 0.877. Every gain is multiplied by an irrational
factor and then forced back into bf16's 8 mantissa bits. The hypothesis is that
this rounding, not anything structural, is what moves 4 of 20 greedy
continuations.

This does not run the model. It asks a narrower question first, because if the
numbers here are tiny the hypothesis is dead and running a 27B merge would be a
waste of an hour:

  1. How much relative error does the cast introduce per gain?
  2. Is the error big enough to flip a logit ordering, given the observed
     teacher-forced KL?

If the per-gain relative error is on the order of 2^-9 (bf16 epsilon) and the
observed max KL is ~3e-2, the story is consistent: most positions are unaffected
and a handful of near-ties flip.
"""
import json
import os
import pathlib

import torch
from safetensors import safe_open

SRC = pathlib.Path(os.environ.get("CONCEPT_SRC", "./models/concept-27b"))
DST = pathlib.Path(os.environ.get("CONCEPT_SRC", "./models/concept-27b"))

cfg_src = json.loads((SRC / "config.json").read_text())
d0 = cfg_src.get("hidden_size") or cfg_src["text_config"]["hidden_size"]

cfg_dst = json.loads((DST / "config.json").read_text())
hidden = cfg_dst["hidden_size"]

scale = (d0 / hidden) ** 0.5
print(f"  d0={d0}  hidden={hidden}  scale={scale!r}")
print(f"  scale is exactly representable in bf16? {float(torch.tensor(scale, dtype=torch.bfloat16)) == scale}")


def find(root: pathlib.Path, needle: str):
    for f in sorted(root.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                if k.endswith(needle) and "layers" not in k:
                    return fh.get_tensor(k)
    return None


base_norm = find(SRC, "norm.weight")
merged_norm = find(DST, "norm.weight")
if base_norm is None or merged_norm is None:
    raise SystemExit(f"  could not locate final norm (base={base_norm is None}, merged={merged_norm is None})")

base = base_norm.float()
# Reproduce the merge arithmetic exactly.
exact = (1.0 + base) * scale - 1.0
cast = exact.to(torch.bfloat16).float()

err = (cast - exact).abs()
denom = exact.abs().clamp_min(1e-12)
rel = err / denom

print(f"\n  final-norm gains compared: {base.numel()}")
print(f"  absolute error   max={err.max():.3e}  mean={err.mean():.3e}")
print(f"  relative error   max={rel.max():.3e}  mean={rel.mean():.3e}")
print(f"  bf16 epsilon     {2 ** -8:.3e}  (8 mantissa bits)")
print(f"  gains changed by the cast: {(err > 0).sum().item()} / {base.numel()}")

# What the merged file actually holds, vs what an exact merge would hold.
actual = merged_norm.float()[:d0]
drift = (actual - exact[:d0]).abs()
print(f"\n  shipped vs exact-arithmetic over the Qwen rows ({d0}):")
print(f"    max={drift.max():.3e}  mean={drift.mean():.3e}  nonzero={(drift > 0).sum().item()}")

# A logit shift of ~1e-3 relative is enough to reorder a near-tie. The paper
# reports max KL 3.086e-2 and 5/960 top-1 disagreements, so we want to see
# whether this error lands in a plausible range rather than orders of magnitude off.
print(f"\n  per-gain relative error at the 99th percentile: {rel.quantile(0.99):.3e}")
print(f"  (paper observes max KL 3.086e-2 and 5/960 top-1 flips)")
