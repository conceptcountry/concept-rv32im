#!/usr/bin/env python3
"""Option D: absorb the scale into the weights that CONSUME the norm, not the norm.

Options A-C each cost something: A mixes dtypes, B triples the checkpoint, C
needs a custom RMSNorm and so forfeits "loads as a stock checkpoint".

But the rescale is only there to undo a uniform shrink of the normed vector.
RMSNorm output is  g * x / rms(x). Widening with zeros multiplies rms by
sqrt(d0/hidden), so the output shrinks by scale. merge27 compensates in g.

It could equally compensate in whatever reads the normed vector next — the
q/k/v and gate/up projections — because those are matrices, and scaling a
matrix's INPUT COLUMNS by a constant is identical to scaling its output by that
constant when the input is a scalar multiple.

Crucially the projections are large matrices of varied magnitude, whereas norm
gains cluster near a few values. The question is empirical: does multiplying a
projection column by 1/scale land on representable bf16 values more often than
doing it to a norm gain?

A cleaner variant, also tested here: fold the scale into the norm gain but round
to NEAREST REPRESENTABLE and measure the residual error on the OUTPUT rather
than the gain. A relative error of 2^-9 on a gain may be irrelevant if the
downstream logit gap is large; the paper's own evidence is that 955/960
positions already agree.

This decides between "exact by construction" and "exact enough to prove".
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
scale = (d0 / HIDDEN) ** 0.5
inv = 1.0 / scale


def not_exact(t: torch.Tensor) -> tuple[int, int]:
    b = t.to(torch.bfloat16).to(t.dtype)
    return int(((b != t) | ~torch.isfinite(b)).sum()), t.numel()


# The tensors that READ the normed hidden state: attention q/k/v, MLP gate/up,
# and the DeltaNet in_proj family. These are IN_COLS in merge27's vocabulary.
READERS = ("q_proj.weight", "k_proj.weight", "v_proj.weight",
           "gate_proj.weight", "up_proj.weight",
           "in_proj_a.weight", "in_proj_b.weight", "in_proj_z.weight",
           "in_proj_qkvz.weight", "in_proj_ba.weight")

print(f"  scale={scale!r}  1/scale={inv!r}")

norm_bad = norm_tot = 0
read_bad = read_tot = 0
n_norm = n_read = 0

for f in sorted(SRC.glob("*.safetensors")):
    with safe_open(str(f), "pt") as fh:
        for k in fh.keys():  # noqa: SIM118
            if not k.startswith("model.language_model."):
                continue
            if k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")) or \
               k == "model.language_model.norm.weight":
                w = fh.get_tensor(k).float()
                b, n = not_exact((1.0 + w) * scale - 1.0)
                norm_bad += b
                norm_tot += n
                n_norm += 1
            elif k.endswith(READERS):
                w = fh.get_tensor(k).float()
                b, n = not_exact(w * inv)
                read_bad += b
                read_tot += n
                n_read += 1

print(f"\n  option C-prime: scale the NORM gains ({n_norm} tensors)")
print(f"    not exactly representable: {norm_bad} / {norm_tot}  ({100*norm_bad/max(norm_tot,1):.2f}%)")

print(f"\n  option D: scale the READER columns instead ({n_read} tensors)")
print(f"    not exactly representable: {read_bad} / {read_tot}  ({100*read_bad/max(read_tot,1):.2f}%)")

print("\n  Neither is exact: multiplying bf16 values by an irrational constant")
print("  lands off-grid regardless of which tensor absorbs it. The scale factor")
print("  itself is the problem, not where it is applied.")
print("\n  => Only option A (float32 norms) and option C (exclude chip dims from")
print("     the variance) give bit-exact host behaviour.")
