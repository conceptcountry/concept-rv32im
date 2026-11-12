"""Quantize Concept RV32IM 27B with the machine's layers left in bf16.

Same schemes and exclusions as the Qwen Concept 27B release (FP8_BLOCK matching Qwen's
official FP8 layout, and NVFP4 weights with bf16 activations), both data-free. One addition:
the first 11 language-model layers are the processor, and every lossy weight grid we tested
breaks it, so they are excluded and stay bf16.

    rv_quant.py {fp8|nvfp4a16} SRC DST
"""
import glob
import json
import os
import shutil
import sys

IGNORE = ["lm_head", "re:.*embed_tokens.*", "re:.*visual.*", "re:.*linear_attn\\.in_proj_a$",
          "re:.*linear_attn\\.in_proj_b$", "re:.*mlp\\.gate$", "re:.*shared_expert_gate$",
          "re:.*conv1d$", "re:.*norm$", "re:.*mtp.*",
          "re:.*language_model\\.layers\\.([0-9]|10)\\..*"]
SCHEME = {"fp8": "FP8_BLOCK", "nvfp4a16": "NVFP4A16"}


def main(mode, src, dst):
    from llmcompressor import model_free_ptq
    model_free_ptq(src, dst, scheme=SCHEME[mode], ignore=IGNORE, max_workers=1, device="cpu")
    for f in glob.glob(os.path.join(src, "*")):
        b = os.path.basename(f)
        if os.path.isfile(f) and not b.endswith(".safetensors") \
                and b not in ("model.safetensors.index.json", "config.json") \
                and not os.path.exists(os.path.join(dst, b)):
            shutil.copy(f, dst)
    json.dump({"mode": mode, "scheme": SCHEME[mode], "ignore": IGNORE},
              open(os.path.join(dst, "quant_recipe.json"), "w"), indent=1)
    print("QUANT_DONE", dst)


if __name__ == "__main__":
    main(*sys.argv[1:4])
