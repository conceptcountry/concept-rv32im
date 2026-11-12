#!/usr/bin/env python3
"""Can the RMSNorm rescale be made exact in bf16?

merge27.py widens the hidden state from d0=5120 to hidden=6656 by zero-padding.
RMSNorm divides by sqrt(mean(x^2)) over the full width, so padding with zeros
shrinks the RMS by sqrt(d0/hidden) and every norm gain is multiplied by

    scale = sqrt(5120/6656) = 0.8770580193070292...

to compensate. The factor is irrational, so (1+g)*scale-1 lands between bf16
values and the cast rounds. Measured: all 1152 final-norm gains perturbed, and
the same happens to every per-layer input_layernorm and post_attention_layernorm.

Three candidate fixes, cheapest first. This script decides between them WITHOUT
running a merge, because a merge is an hour.

  A. float32 norms. Exact, but mixes dtypes in one checkpoint.

  B. Choose a hidden size where d0/hidden is a perfect square of a dyadic
     rational, making scale a power of two and therefore exact in bf16 —
     multiplying by a power of two only changes the exponent, never the
     mantissa. Requires hidden = d0 * 4^k, which pick_hidden cannot currently
     produce.

  C. Leave the geometry alone and absorb the scale elsewhere: if the chip dims
     are excluded from the variance, no compensation is needed at all and
     Concept's gains carry through bit-identical.

For each, the question is the same: after the transform, is every gain exactly
representable in bf16? That is the codebase's own criterion — check_bf16()
raises on any value that is not.
"""
import json
import os
import pathlib

import torch
from safetensors import safe_open

SRC = pathlib.Path(os.environ.get("CONCEPT_SRC", "./models/concept-27b"))

cfg = json.loads((SRC / "config.json").read_text())
tc0 = cfg.get("text_config", cfg)
d0 = tc0["hidden_size"]
HIDDEN = 6656


def exact_in_bf16(t: torch.Tensor) -> tuple[int, int]:
    """(n_not_representable, n_total) — the check_bf16 criterion, counted not raised."""
    b = t.to(torch.bfloat16).to(t.dtype)
    bad = (b != t) | ~torch.isfinite(b)
    return int(bad.sum()), t.numel()


def collect_norms() -> dict[str, torch.Tensor]:
    """Every tensor merge27 rescales: WIDE_NORMS = input/post_attention layernorms + final norm."""
    out = {}
    for f in sorted(SRC.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                if not k.startswith("model.language_model."):
                    continue
                if k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")) or \
                   k == "model.language_model.norm.weight":
                    out[k] = fh.get_tensor(k)
    return out


norms = collect_norms()
total = sum(t.numel() for t in norms.values())
print(f"  rescaled tensors: {len(norms)}  gains: {total}")

print("\n  --- current behaviour (scale = sqrt(d0/hidden), cast to bf16) ---")
scale = (d0 / HIDDEN) ** 0.5
bad = tot = 0
worst = 0.0
for k, w in norms.items():
    g = (1.0 + w.float()) * scale - 1.0
    b, n = exact_in_bf16(g)
    bad += b
    tot += n
    err = (g.to(torch.bfloat16).float() - g).abs()
    worst = max(worst, float((err / g.abs().clamp_min(1e-12)).max()))
print(f"    scale            {scale!r}")
print(f"    gains NOT exactly representable: {bad} / {tot}  ({100*bad/tot:.1f}%)")
print(f"    worst relative error: {worst:.3e}   bf16 eps {2**-8:.3e}")

print("\n  --- option A: keep float32 for the rescaled norms ---")
print("    exact by construction (no cast), but the checkpoint then mixes")
print("    bf16 and float32 norm tensors.")

print("\n  --- option B: hidden where d0/hidden is a power of 4 ---")
print("    scale is then a power of two, which only shifts the exponent,")
print("    so every gain stays bit-identical. Candidate geometries:")
found = []
for k in range(1, 4):
    h = d0 * (4 ** k)
    found.append((h, 2.0 ** -k))
for h, s in found:
    ok = h >= HIDDEN
    print(f"      hidden={h:7d}  scale={s}  {'usable' if ok else 'too small'}"
          f"  {'' if ok else ''}  params x{h/HIDDEN:.2f} vs current")
print("    The smallest usable one is 20480 — 3.1x the current width.")
print("    Rejected: the checkpoint would roughly triple in size.")

print("\n  --- option C: no rescale at all ---")
bad = tot = 0
for k, w in norms.items():
    g = w.float()  # unchanged: Concept's own gains
    b, n = exact_in_bf16(g)
    bad += b
    tot += n
print(f"    gains NOT exactly representable: {bad} / {tot}")
print("    Zero, necessarily — these are the source's own bf16 values.")
print("    Requires the norm to divide by sqrt(mean over d0 dims) rather than")
print("    over all `hidden` dims, i.e. the chip dims must not enter the variance.")
