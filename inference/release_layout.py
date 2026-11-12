"""Rewrite the vision+MTP merge into the layout the original Qwen Concept 27B ships in.

merge27_exact.py keeps the vision tower and MTP head but writes the language model under
`model.layers.*` with a text-only config, so a stock loader builds Qwen3_5ForCausalLM and
never attaches the 333 vision tensors. This renames the language-model keys back to
`model.language_model.*` and wraps the config as Qwen3_5ForConditionalGeneration, the same
shape as the source checkpoint. No tensor value changes: every tensor is hashed on the way
in and again after the shard is written, and the run fails on the first mismatch.

    python release_layout.py MERGED_EXACT SOURCE_BF16 DST
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def digest(t):
    return hashlib.sha256(t.contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def rename(k):
    if k.startswith(("model.visual.", "mtp.")) or k == "lm_head.weight":
        return k
    assert k.startswith("model."), k
    return "model.language_model." + k[len("model."):]


def main(merged, source, dst):
    merged, source, dst = Path(merged), Path(source), Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    files = sorted(merged.glob("*.safetensors"))
    wm, total, n = {}, 0, 0
    for f in files:
        out, want = {}, {}
        with safe_open(f, "pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                t = fh.get_tensor(k)
                nk = rename(k)
                out[nk], want[nk] = t, (str(t.dtype), tuple(t.shape), digest(t))
        save_file(out, str(dst / f.name), metadata={"format": "pt"})
        del out
        with safe_open(dst / f.name, "pt") as fh:
            assert sorted(fh.keys()) == sorted(want), f.name
            for k, w in want.items():
                t = fh.get_tensor(k)
                assert (str(t.dtype), tuple(t.shape), digest(t)) == w, k
                wm[k] = f.name
                total += t.numel() * t.element_size()
                n += 1
        print(f.name, len(want), "verified", flush=True)

    top = json.loads((source / "config.json").read_text())
    text = json.loads((merged / "config.json").read_text())
    text.pop("architectures", None)
    text["model_type"] = top["text_config"]["model_type"]
    top["text_config"] = text
    top["vision_config"] = dict(top["vision_config"], out_hidden_size=text["hidden_size"])
    (dst / "config.json").write_text(json.dumps(top, indent=2) + "\n")
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": wm}, indent=1))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json", "chipformer.json", "LICENSE"):
        shutil.copy(merged / f, dst / f)
    for f in ("preprocessor_config.json", "video_preprocessor_config.json"):
        shutil.copy(source / f, dst / f)
    print(json.dumps({"tensors": n, "bytes": total, "shards": len(files),
                      "language_model": sum(k.startswith("model.language_model.") for k in wm),
                      "visual": sum(k.startswith("model.visual.") for k in wm),
                      "mtp": sum(k.startswith("mtp.") for k in wm)}), flush=True)
    print("LAYOUT_DONE", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
