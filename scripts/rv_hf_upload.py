#!/usr/bin/env python3
"""Upload Concept RV32IM 27B to HuggingFace.

Same shape as ~/wsp/hf_upload.py, which published Qwen Concept 27B: one repo, one folder
per format, each carrying its own SHA256SUMS, uploads resumable so a rerun after an
interruption costs a listing rather than 280 GB.

The card is built from dist/VERIFY.json and the text-accuracy comparisons rather than
written by hand, so a number in the table is a number that was measured. Cells that were
not measured say so.

    rv_hf_upload.py [--dry-run] [--card-only]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from huggingface_hub import HfApi

ORG = "ConceptCountry"
REPO = f"{ORG}/concept-rv32im-27b"
DIST = pathlib.Path(os.environ.get("DIST", "/home/somewheresystems/qwenputer/dist"))
FOLDERS = ("bf16", "fp8", "nvfp4a16", "gguf")

PAPER = "https://concept.country/research/concept-rv32im"
CODE = "https://github.com/conceptcountry/concept-rv32im"
BASE = "https://huggingface.co/ConceptCountry/qwen-concept-27b"

FORMAT_ROWS = [
    ("bf16", "bf16/", "bfloat16",
     "Full precision. 1,353 tensors, including the vision tower and the MTP head."),
    ("fp8", "fp8/", "FP8_BLOCK",
     "FP8 e4m3 weights in 128x128 blocks, matching Qwen's own FP8 layout."),
    ("nvfp4a16", "nvfp4a16/", "NVFP4 W4A16",
     "NVFP4 weights (group 16, fp8 group scales, fp32 global), bf16 activations."),
    ("Q8_0", "gguf/", "GGUF Q8_0", "llama.cpp."),
    ("Q4_K_M", "gguf/", "GGUF Q4_K_M", "llama.cpp, quantized from the bf16 GGUF, not from the Q8_0."),
]


def gb(n: int) -> str:
    return f"{n / 1e9:.0f} GB"


def folder_size(p: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()
               and not f.name.startswith(".tmp"))


def fmt_size(name: str) -> str:
    """On-disk size of one format, from the files that will actually be uploaded."""
    if name in ("bf16", "fp8", "nvfp4a16"):
        d = DIST / name
        return gb(folder_size(d)) if d.is_dir() else "not built"
    f = DIST / "gguf" / f"concept-rv32im-27b-{name}.gguf"
    return gb(f.stat().st_size) if f.is_file() else "not built"


def build_card() -> str:
    V = json.loads((DIST / "VERIFY.json").read_text())
    text = {}
    tp = DIST / "TEXT.json"
    if tp.is_file():
        text = json.loads(tp.read_text())

    rows = []
    for key, folder, scheme, note in FORMAT_ROWS:
        v = V["formats"].get(key, {})
        if key == "bf16":
            machine = "reference"
        elif not v:
            machine = "not measured"
        elif v.get("kind") == "gguf":
            # GGUF: the claim is that nothing in the machine's region was quantized.
            n = v.get("n_machine_tensors")
            machine = (f"bf16/f32 ({n}/{n})" if v.get("machine_unchanged")
                       else f"**QUANTIZED** ({len(v.get('quantized_tensors_in_machine', []))})")
        else:
            n_machine = v.get("n_machine_tensors")
            ident = v.get("identical", n_machine if v.get("machine_unchanged") else None)
            machine = (f"identical ({ident}/{n_machine})" if v.get("machine_unchanged")
                       else f"**CHANGED** ({ident}/{n_machine})")
        loads = v.get("loads")
        load_s = {True: "yes", False: "no", None: "not measured"}[loads]
        t = text.get(key) or {}
        if t.get("n_identical") is not None:
            acc = f"{t['n_identical']}/{t['n_prompts']}"
        else:
            acc = t.get("note") or "not measured"
        rows.append(f"| `{folder}` | {scheme} | {fmt_size(key)} | {machine} | {load_s} | {acc} |")

    base_bf16 = (text.get("base_bf16_vs_host") or {}).get("summary")
    bf16_line = ""
    if base_bf16:
        bf16_line = f"\n{base_bf16}\n"

    return f"""---
license: apache-2.0
base_model: ConceptCountry/qwen-concept-27b
tags:
  - weight-space-computation
  - risc-v
  - rv32im
  - model-merging
  - concept-country
---

# Concept RV32IM 27B

Qwen Concept 27B with an exact RV32IM processor compiled into eleven additional
transformer layers, placed in front of the host model. The machine's memory is the
Gated DeltaNet recurrent state: 256 KB of code ROM, 128 KB of rodata and 32 KB of RAM
live in a linear-attention cache. The host model calls it through its own native
tool-call syntax, and no tool runs outside the weights.

A port of [Qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal
(code MIT, weights Apache-2.0) from Qwen3.5-4B onto a 27B hybrid-attention model.

Paper, artifacts and the code that generates every number: <{PAPER}>
Reproduction repository: <{CODE}>

## What was measured on which weights

The paper's results were measured on the text-only merge. This release is the
vision+MTP build: it keeps the vision tower (333 tensors) and the MTP head (15), and its
154 machine-layer tensors are byte-identical to the measured text-only build. The only
difference between the two is that 129 RMSNorm gains are stored float32 here rather than
bf16. The release layout loads as the stock `Qwen3_5ForConditionalGeneration` class with
no missing tensors and no shape mismatches.

## Formats

In every quantized format the eleven machine layers (language-model layers 0-10), the
token embedding and the output head stay bf16; only the host's layers are quantized. That
is not a precaution: a weight-grid probe found that FP8, Q8_0 and Q4_K_M each break the
machine when applied to its layers, taking it from 4 of 4 programs exact to 0 of 4, while
the bf16 control stays exact. The processor is not a text model and does not degrade
gracefully.

| folder | scheme | size | machine layers vs bf16 | loads | text identity vs base |
|---|---|---|---|---|---|
{chr(10).join(rows)}

"machine layers vs bf16" is the sha256 of every tensor in language-model layers 0-10,
`embed_tokens` and `lm_head`, compared against this release's bf16 weights. The GGUF
conversion renames, fuses and transposes relative to the HuggingFace layout, so for those
builds the check is that no tensor in `blk.0`-`blk.10`, `token_embd` or `output` carries a
quantized type: they are all BF16 or F32, while the host's layers are quantized. "text
identity vs base" is greedy continuation of 20 prompts for 48 new tokens against the
*same format* of [Qwen Concept 27B]({BASE}), counting identical continuations.
{bf16_line}
Each folder carries a `SHA256SUMS`. The GGUF folder also carries
`mmproj-concept-rv32im-27b-bf16.gguf` for the vision tower. The GGUF builds exclude the
MTP head, as the base model's own GGUF does; it is present in the safetensors formats.

## Also available

Mirrored on the Concept Country registry, which speaks the HF Hub API:

```sh
HF_ENDPOINT=https://models.concept.country hf download ConceptCountry/concept-rv32im-27b
```

## Status

Experimental, under evaluation.

## License

Apache-2.0. The Qwenputer port's code is MIT; its weights are Apache-2.0.
"""


def gate(V: dict) -> tuple[set[str], list[str]]:
    """Which formats passed the upload gate, and why the others did not.

    The gate is checks (a) and (b): the machine's tensors are unchanged, and the format
    loads. Text accuracy is reported in the card either way. This is enforced here rather
    than left to the operator, so a format that fails cannot be published by rerunning the
    script.
    """
    ok: set[str] = set()
    refused: list[str] = []
    for key, folder, _scheme, _note in FORMAT_ROWS:
        v = V["formats"].get(key)
        if not v:
            refused.append(f"{key}: not verified")
            continue
        if key != "bf16" and not v.get("machine_unchanged"):
            refused.append(f"{key}: machine tensors differ from bf16")
            continue
        if v.get("loads") is not True:
            refused.append(f"{key}: did not load")
            continue
        ok.add(key)
    return ok, refused


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--card-only", action="store_true")
    a = ap.parse_args()

    V = json.loads((DIST / "VERIFY.json").read_text())
    passed, refused = gate(V)
    for r in refused:
        print(f"  GATE REFUSED {r}", flush=True)
    print(f"  gate passed: {sorted(passed)}", flush=True)

    # A folder is uploaded only if every format it carries passed the gate.
    folder_formats = {"bf16": ["bf16"], "fp8": ["fp8"], "nvfp4a16": ["nvfp4a16"],
                      "gguf": ["Q8_0", "Q4_K_M"]}
    uploadable = [f for f in FOLDERS
                  if all(k in passed for k in folder_formats[f])]
    if not uploadable:
        print("  nothing passed the gate; refusing to upload", flush=True)
        return 1

    card_text = build_card()
    if a.dry_run:
        print(card_text)
        for f in uploadable:
            src = DIST / f
            if src.is_dir():
                n = sum(1 for x in src.rglob("*") if x.is_file() and not x.name.startswith(".tmp"))
                print(f"  would upload {f}: {n} files, {gb(folder_size(src))}")
        for f in FOLDERS:
            if f not in uploadable:
                print(f"  would NOT upload {f}: a format in it failed the gate")
        return 0

    api = HfApi()
    print(f"  authenticated as {api.whoami()['name']}", flush=True)
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    print(f"  repo ready: https://huggingface.co/{REPO}", flush=True)

    card = DIST / "README.md"
    card.write_text(card_text)
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=REPO, repo_type="model")
    print("  card uploaded", flush=True)
    # VERIFY.json travels with the weights: the table above is derived from it.
    api.upload_file(path_or_fileobj=str(DIST / "VERIFY.json"), path_in_repo="VERIFY.json",
                    repo_id=REPO, repo_type="model")
    print("  VERIFY.json uploaded", flush=True)
    if (DIST / "TEXT.json").is_file():
        api.upload_file(path_or_fileobj=str(DIST / "TEXT.json"), path_in_repo="TEXT.json",
                        repo_id=REPO, repo_type="model")
        print("  TEXT.json uploaded", flush=True)
    if a.card_only:
        return 0

    for folder in uploadable:
        src = DIST / folder
        if not src.is_dir():
            print(f"  SKIP {folder}: not built", flush=True)
            continue
        n = sum(1 for x in src.rglob("*") if x.is_file())
        print(f"  uploading {folder}: {n} files, {gb(folder_size(src))}", flush=True)
        api.upload_folder(
            folder_path=str(src), path_in_repo=folder, repo_id=REPO, repo_type="model",
            # .tmp-bf16.gguf is the 88 GB intermediate Q4_K_M was quantized from; it is
            # build scratch, not a release artifact.
            ignore_patterns=["**/.cache/**", "**/.git/**", ".tmp*", "**/.tmp*"],
        )
        print(f"  {folder} done", flush=True)

    print(f"\n  https://huggingface.co/{REPO}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
