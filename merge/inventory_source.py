#!/usr/bin/env python3
"""Inventory the source checkpoint: what the merge keeps, and what it drops.

Two questions:
  1. Which tensor is the TEXT final norm (the one merge27.py rescales)?
  2. How many vision / MTP tensors exist, and what are their shapes?

(2) matters because the merge skips them with a `continue` on anything not
matching the text prefix. They are not missing from the source; they are
excluded by the merge. Restoring them means widening them like the text path.
"""
import collections
import os
import pathlib

from safetensors import safe_open

SRC = pathlib.Path(os.environ.get("CONCEPT_SRC", "./models/concept-27b"))

keys = []
shapes = {}
for f in sorted(SRC.glob("*.safetensors")):
    with safe_open(str(f), "pt") as fh:
        for k in fh.keys():  # noqa: SIM118
            keys.append(k)
            shapes[k] = tuple(fh.get_slice(k).get_shape())

print(f"  total tensors: {len(keys)}")

prefixes = collections.Counter(k.split(".")[0] + "." + k.split(".")[1] for k in keys if "." in k)
print("\n  top-level groups:")
for p, n in prefixes.most_common(12):
    print(f"    {p:28s} {n}")

print("\n  non-layer norm tensors (candidates for the final norm):")
for k in sorted(k for k in keys if "norm" in k and "layers" not in k):
    print(f"    {k:44s} {shapes[k]}")

print("\n  tensors the merge DROPS (not under the text prefix):")
SRC_TEXT_GUESSES = ("model.language_model.", "model.text_model.", "model.")
dropped = collections.Counter()
for k in keys:
    if k == "lm_head.weight":
        continue
    if k.startswith("model.visual") or ".mtp" in k or k.startswith("mtp"):
        dropped[k.split(".")[1] if "." in k else k] += 1
for g, n in dropped.most_common():
    print(f"    {g:28s} {n}")
print(f"    TOTAL DROPPED: {sum(dropped.values())}")
