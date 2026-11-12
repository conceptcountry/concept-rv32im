"""Greedy text accuracy of a quantized release against the same format of the base model.

The question the release card has to answer is the user's: is the quantized RV32IM
checkpoint as accurate as the quantized base model it was built from. So each format is
compared against *its own* counterpart in ~/wsp/dist/qwen-concept-27b, not against bf16:
fp8 against fp8, nvfp4a16 against nvfp4a16. The comparison is greedy continuation of the
same 20 prompts for 48 new tokens, counting identical continuations and the position of
the first divergence.

The two checkpoints have different vocabularies and different layer counts (the RV32IM
build carries 11 extra machine layers in front of the host and a widened hidden size), so
"identical" means the decoded text matches, not that token ids match. Text is compared
after decoding with each model's own tokenizer.

Memory: layers are streamed from disk one at a time by stream_dq.stream_model, which
dequantizes compressed shards to bf16 as it reads them, so peak residency is the resident
tensors plus a couple of layers rather than the whole checkpoint.

    rv_text_eval.py --model DIR --prompts P.json --n-new 48 --out gen.json
    rv_text_eval.py --compare a.json b.json --out cmp.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def generate(model_dir: str, prompts_path: str, n_new: int, out_path: str,
             text_vocab: int | None) -> None:
    import torch

    sys.path.insert(0, os.path.expanduser("~/qwenputer"))
    sys.path.insert(0, HERE)
    from stream_dq import stream_model  # noqa: E402
    from transformers import AutoTokenizer, DynamicCache  # noqa: E402

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    P = json.load(open(prompts_path))
    model = stream_model(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    lm = model.model
    L = len(lm.layers)
    # The merged build's vocabulary holds machine tokens above the text vocabulary; argmax
    # is taken over the text range so a machine token cannot win a text decode. For the
    # base model the full vocabulary *is* the text vocabulary.
    tv = text_vocab if text_vocab else int(model.lm_head.weight.shape[0])

    rows = []
    t0 = time.time()
    with torch.no_grad():
        model.stream_stats["hold"] = True
        caches = [DynamicCache(config=model.config) for _ in P]
        gen: list[list[int]] = [[] for _ in P]
        inp = [list(p["ids"]) for p in P]
        pos = [0] * len(P)
        for step in range(n_new):
            hs, pes, pids = [], [], []
            for j in range(len(P)):
                x = torch.tensor([inp[j]], device="cuda")
                h = lm.embed_tokens(x)
                p = torch.arange(pos[j], pos[j] + len(inp[j]), device="cuda")
                p4 = p.view(1, 1, -1).expand(4, 1, -1)
                pes.append(lm.rotary_emb(h, p4[1:]))
                pids.append(p4[0])
                hs.append(h)
            for i in range(L):
                model.stream_load(i)
                for j in range(len(P)):
                    hs[j] = lm.layers[i](hs[j], position_embeddings=pes[j],
                                         attention_mask=None, position_ids=pids[j],
                                         past_key_values=caches[j])
                model.stream_drop(i)
            for j in range(len(P)):
                logits = model.lm_head(lm.norm(hs[j][:, -1:]))[0, -1]
                t = int(torch.argmax(logits[:tv]))
                gen[j].append(t)
                pos[j] += len(inp[j])
                inp[j] = [t]
            print(f"  step {step + 1}/{n_new} {time.time() - t0:.0f}s", flush=True)
        model.stream_stats["hold"] = False
    for j, p in enumerate(P):
        rows.append({"prompt": p["prompt"], "ids": gen[j],
                     "text": tok.decode(gen[j], skip_special_tokens=True)})
    json.dump({"model": model_dir, "n_new": n_new, "text_vocab": tv,
               "seconds": time.time() - t0, "rows": rows},
              open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {out_path} in {time.time() - t0:.0f}s")


def compare(a_path: str, b_path: str, out_path: str) -> None:
    A = json.load(open(a_path))
    B = json.load(open(b_path))
    ra, rb = A["rows"], B["rows"]
    assert len(ra) == len(rb), "prompt count differs"
    per = []
    for x, y in zip(ra, rb):
        ta, tb = x["text"], y["text"]
        if ta == tb:
            per.append({"prompt": x["prompt"], "identical": True, "first_divergence": None})
            continue
        k = 0
        for ca, cb in zip(ta, tb):
            if ca != cb:
                break
            k += 1
        per.append({"prompt": x["prompt"], "identical": False, "first_divergence_char": k,
                    "a": ta[:200], "b": tb[:200]})
    ident = sum(1 for p in per if p["identical"])
    div = [p["first_divergence_char"] for p in per if not p["identical"]]
    out = {"a": A["model"], "b": B["model"], "n_prompts": len(per),
           "n_identical": ident,
           "median_first_divergence_char": (sorted(div)[len(div) // 2] if div else None),
           "min_first_divergence_char": (min(div) if div else None),
           "per_prompt": per}
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"{ident}/{len(per)} identical continuations -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--prompts")
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--text-vocab", type=int, default=0)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.compare:
        compare(a.compare[0], a.compare[1], a.out)
    else:
        generate(a.model, a.prompts, a.n_new, a.out, a.text_vocab or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
