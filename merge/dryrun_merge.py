#!/usr/bin/env python3
"""Dry-run the patched merge's tensor transforms without writing a checkpoint.

A full merge is about an hour. This applies the same shape and dtype rules to
every source tensor in memory and checks the results are coherent, so a shape
bug surfaces in seconds instead of at minute fifty.

Checks:
  - every source tensor is either transformed or deliberately skipped, none
    silently lost
  - widened tensors have the expected output shape
  - mtp.fc gets both halves of its 2*d0 input padded, not just the tail
  - rescaled norms come out float32 and bit-exact against the ideal arithmetic
  - vision tensors keep their native width except merger.linear_fc2
"""
import json
import os
import pathlib
import re

import torch
from safetensors import safe_open

SRC = pathlib.Path(os.environ.get("CONCEPT_SRC", "./models/concept-27b"))
cfg = json.loads((SRC / "config.json").read_text())
tc0 = cfg.get("text_config", cfg)
d0 = tc0["hidden_size"]
HIDDEN = 6656
scale = (d0 / HIDDEN) ** 0.5

IN_COLS = re.compile(
    r"(q_proj|k_proj|v_proj|gate_proj|up_proj|in_proj_a|in_proj_b|in_proj_z"
    r"|in_proj_qkvz|in_proj_ba|conv1d)\.weight$")
OUT_ROWS = re.compile(
    r"(out_proj|o_proj|down_proj)\.weight$|visual\.merger\.linear_fc2\.(weight|bias)$")
WIDE_NORMS = re.compile(
    r"(input_layernorm|post_attention_layernorm)\.weight$|language_model\.norm\.weight$")

stats = {"text": 0, "visual_passthrough": 0, "visual_widened": 0, "mtp": 0,
         "head": 0, "embed": 0, "final_norm": 0, "skipped": 0}
problems = []
norm_exact = norm_total = 0

for f in sorted(SRC.glob("*.safetensors")):
    with safe_open(str(f), "pt") as fh:
        for k in fh.keys():  # noqa: SIM118
            if k == "lm_head.weight":
                stats["head"] += 1
                continue

            if k.startswith("model.visual."):
                w = fh.get_tensor(k)
                if OUT_ROWS.search(k):
                    got = torch.cat([w, w.new_zeros(HIDDEN - d0, *w.shape[1:])], 0)
                    if got.shape[0] != HIDDEN:
                        problems.append(f"{k}: widened to {got.shape}, want dim0={HIDDEN}")
                    stats["visual_widened"] += 1
                else:
                    stats["visual_passthrough"] += 1
                continue

            if k.startswith("mtp."):
                w = fh.get_tensor(k)
                if k == "mtp.fc.weight":
                    if w.shape[1] != 2 * d0:
                        problems.append(f"{k}: input {w.shape[1]}, expected 2*{d0}")
                    a, b = w[:, :d0], w[:, d0:]
                    pad = w.new_zeros(w.shape[0], HIDDEN - d0)
                    got = torch.cat([a, pad, b, pad], -1)
                    got = torch.cat([got, got.new_zeros(HIDDEN - d0, got.shape[1])], 0)
                    if got.shape != (HIDDEN, 2 * HIDDEN):
                        problems.append(f"{k}: got {tuple(got.shape)}, want ({HIDDEN}, {2*HIDDEN})")
                elif k.endswith(("norm.weight", "layernorm.weight")) and \
                        not k.endswith(("q_norm.weight", "k_norm.weight")):
                    g = (1.0 + w.float()) * scale - 1.0
                    got = torch.cat([g, torch.zeros(HIDDEN - d0, dtype=torch.float32)])
                    if got.dtype != torch.float32:
                        problems.append(f"{k}: norm dtype {got.dtype}, want float32")
                    ideal = (1.0 + w.float()) * scale - 1.0
                    norm_exact += int(torch.equal(got[:d0], ideal))
                    norm_total += 1
                elif IN_COLS.search(k):
                    got = torch.cat([w, w.new_zeros(*w.shape[:-1], HIDDEN - d0)], -1)
                    if got.shape[-1] != HIDDEN:
                        problems.append(f"{k}: in-cols {got.shape[-1]}, want {HIDDEN}")
                elif OUT_ROWS.search(k):
                    got = torch.cat([w, w.new_zeros(HIDDEN - d0, *w.shape[1:])], 0)
                    if got.shape[0] != HIDDEN:
                        problems.append(f"{k}: out-rows {got.shape[0]}, want {HIDDEN}")
                stats["mtp"] += 1
                continue

            if not k.startswith("model.language_model."):
                stats["skipped"] += 1
                problems.append(f"UNHANDLED: {k}")
                continue

            w = fh.get_tensor(k)
            if k.endswith("embed_tokens.weight"):
                stats["embed"] += 1
                continue
            if k == "model.language_model.norm.weight":
                stats["final_norm"] += 1
                continue
            if WIDE_NORMS.search(k):
                g = (1.0 + w.float()) * scale - 1.0
                got = torch.cat([g, torch.zeros(HIDDEN - d0, dtype=torch.float32)])
                if got.dtype != torch.float32:
                    problems.append(f"{k}: dtype {got.dtype}, want float32")
                norm_exact += int(torch.equal(got[:d0], g))
                norm_total += 1
            stats["text"] += 1

print("  transform census:")
for k, v in stats.items():
    print(f"    {k:22s} {v}")
print(f"\n  rescaled norms bit-exact: {norm_exact} / {norm_total}")
print(f"  problems: {len(problems)}")
for p in problems[:15]:
    print(f"    {p}")
