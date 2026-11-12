#!/usr/bin/env python3
"""Does storing the rescaled norms in float32 actually reach the model exactly?

Stock Qwen3.5 RMSNorm.forward does:

    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())

self.weight.float() — the gain is upcast to float32 on every forward pass no
matter how it was stored. So a float32 norm tensor is not an exotic dtype the
runtime tolerates; it is exactly what the module consumes. Storing bf16 throws
away precision that the runtime immediately asks for again.

Two things to confirm before rebuilding a 43.5B checkpoint on this basis:

  1. safetensors can hold mixed dtypes in one file and transformers will load
     a float32 norm beside bf16 projections without complaint or silent cast.
  2. The resulting gain reaching the module is bit-identical to the exact
     arithmetic, where the bf16 path is not.

(2) is the whole point: it turns "16/20, we think it is rounding" into "20/20,
and the rounding is gone by construction".
"""
import json
import os
import pathlib
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SRC = pathlib.Path(os.environ.get("CONCEPT_SRC", "./models/concept-27b"))
cfg = json.loads((SRC / "config.json").read_text())
tc0 = cfg.get("text_config", cfg)
d0 = tc0["hidden_size"]
HIDDEN = 6656
scale = (d0 / HIDDEN) ** 0.5

# Pull one real norm tensor.
gain = None
for f in sorted(SRC.glob("*.safetensors")):
    with safe_open(str(f), "pt") as fh:
        for k in fh.keys():  # noqa: SIM118
            if k.endswith("input_layernorm.weight") and "language_model" in k:
                gain = fh.get_tensor(k)
                name = k
                break
    if gain is not None:
        break
print(f"  sample tensor: {name}  {tuple(gain.shape)}  {gain.dtype}")

exact = (1.0 + gain.float()) * scale - 1.0           # what the merge wants
as_bf16 = exact.to(torch.bfloat16)                    # what it currently stores
as_f32 = exact                                        # what it would store

print("\n  1. mixed dtypes in one safetensors file")
with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "mixed.safetensors"
    save_file({
        "norm.weight": as_f32.contiguous(),
        "proj.weight": torch.randn(64, 64, dtype=torch.bfloat16).contiguous(),
    }, str(p), metadata={"format": "pt"})
    back = load_file(str(p))
    print(f"     saved + reloaded: norm={back['norm.weight'].dtype}  proj={back['proj.weight'].dtype}")
    assert back["norm.weight"].dtype == torch.float32
    assert back["proj.weight"].dtype == torch.bfloat16
    rt = back["norm.weight"]
    print(f"     round-trip bit-identical: {bool(torch.equal(rt, as_f32))}")

print("\n  2. what the RMSNorm module actually multiplies by")
# forward does (1.0 + weight.float()); compare against the ideal.
ideal = (1.0 + gain.float()) * scale          # the true multiplier
via_bf16 = 1.0 + as_bf16.float()
via_f32 = 1.0 + as_f32

for label, got in (("stored bf16", via_bf16), ("stored float32", via_f32)):
    err = (got - ideal).abs()
    rel = err / ideal.abs().clamp_min(1e-12)
    exact_n = int((got == ideal).sum())
    print(f"     {label:15s} bit-identical {exact_n}/{got.numel()}"
          f"   max rel err {float(rel.max()):.3e}")

print("\n  3. size cost of float32 norms")
n_norm_tensors = 129
gains = 660480
extra = gains * 2  # bf16 -> float32 is +2 bytes per value
print(f"     {n_norm_tensors} tensors, {gains} gains")
print(f"     +{extra/1e6:.2f} MB on a ~87 GB checkpoint ({100*extra/87e9:.5f}%)")
