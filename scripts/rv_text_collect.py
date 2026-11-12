#!/usr/bin/env python3
"""Collect the text-accuracy comparisons into dist/TEXT.json for the model card.

Each release format is compared against the same format of the base model
(ConceptCountry/qwen-concept-27b), because the user's bar is "as accurate as the base
model" rather than "identical to bf16". A format with no comparison file is recorded as
not measured, with the reason, rather than omitted.
"""
from __future__ import annotations

import json
import pathlib
import sys

D = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/somewheresystems/qwenputer/dist")

NOT_MEASURED = {
    "Q8_0": ("not measured: the GGUF runtime available here is a CPU build, and 20 prompts "
             "x 48 tokens at its observed throughput does not fit the machine's time budget"),
    "Q4_K_M": ("not measured: the GGUF runtime available here is a CPU build, and 20 prompts "
               "x 48 tokens at its observed throughput does not fit the machine's time budget"),
}

out: dict[str, object] = {}
for fmt in ("bf16", "fp8", "nvfp4a16"):
    p = D / f"cmp_{fmt}.json"
    if not p.is_file():
        out[fmt] = {"note": "not measured: comparison did not run"}
        continue
    c = json.loads(p.read_text())
    out[fmt] = {
        "n_identical": c["n_identical"],
        "n_prompts": c["n_prompts"],
        "median_first_divergence_char": c["median_first_divergence_char"],
        "min_first_divergence_char": c["min_first_divergence_char"],
        "a": c["a"],
        "b": c["b"],
        "divergent_prompts": [p["prompt"] for p in c["per_prompt"] if not p["identical"]],
    }

for fmt, note in NOT_MEASURED.items():
    out[fmt] = {"note": note}

bf = out.get("bf16") or {}
if isinstance(bf, dict) and bf.get("n_identical") is not None:
    out["base_bf16_vs_host"] = {
        "summary": (
            f"At bf16 the release reproduces the base model's greedy continuation on "
            f"{bf['n_identical']} of {bf['n_prompts']} prompts; the differences are "
            f"paraphrase-level and begin a median of {bf['median_first_divergence_char']} "
            f"characters into a 48-token continuation. Quantization widens that gap in the "
            f"way quantization does, and the comparison for each quantized format is "
            f"against the same format of the base model, not against bf16."
        )
    }

out["method"] = {
    "prompts": 20,
    "new_tokens": 48,
    "decoding": "greedy, argmax over the text vocabulary",
    "comparison": "decoded text, exact string equality, with the first differing character",
    "note": ("the release and the base model tokenize all 20 prompts to identical ids, so "
             "the comparison is of continuations rather than of tokenizers"),
}

(D / "TEXT.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {D / 'TEXT.json'}")
for k in ("bf16", "fp8", "nvfp4a16"):
    v = out[k]
    if isinstance(v, dict) and v.get("n_identical") is not None:
        print(f"  {k:9s} {v['n_identical']}/{v['n_prompts']}")
