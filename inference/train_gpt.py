"""Train a tiny character-level GPT and emit int8 weights for the in-weights RV32IM machine.

The machine is RV32IM_Zba: integer multiply and divide, no FPU. Measured on the machine
itself, a soft-float multiply-accumulate costs 189 instructions against 4.22 for an int8
MAC, and `expf` costs 2,185 -- 518 int8 MACs. A float softmax would therefore cost more
than every matmul in this model combined. So the inference is integer end to end:

  * int8 symmetric per-tensor weights, int32 accumulators
  * requantization by a calibrated fixed-point multiplier (M0, shift), the gemmlowp form:
    out = (acc * M0) >> shift, with M0 near 2^30. RV32M has MULH, so a 32x32 -> 64 bit
    product is two instructions; a soft-float multiply is 189.
  * RMSNorm with an exact integer floor-sqrt. Normalization is scale-invariant, which is
    why RMSNorm quantizes cleanly where LayerNorm would need a mean in the same units.
  * softmax by a 256-entry exponential lookup table indexed by (max - score), with the
    table built from the *calibrated* score scale rather than a guessed decay constant.

Every scale here is measured on real activations, not assumed. `int_forward` is the
specification; gpt.c mirrors it operation for operation.

Writes:
  gpt_weights.h  int8 weights, multipliers, exp LUT, prompt -- all const (DROM)
  gpt_meta.json  float vs integer loss, the calibrated constants, expected generation
  gpt_state.pt   checkpoint, for the diagnostic script
"""
import json
import math
import pathlib
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(1337)
np.random.seed(1337)

D, L, H, FF, T = 64, 2, 2, 256, 64
HD = D // H
NGEN = 24
PROMPT = "ROMEO:"

# ------------------------------------------------------------------ corpus
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
src = OUT / "tinyshakespeare.txt"
if not src.exists():
    urllib.request.urlretrieve(URL, src)
text = src.read_text(encoding="utf-8")
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in text], dtype=np.int64)
train, val = data[: int(0.9 * len(data))], data[int(0.9 * len(data)) :]
print(f"corpus {len(text):,} chars, vocab {V}", flush=True)


# ------------------------------------------------------------------ float model
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gn1 = nn.Parameter(torch.ones(D))
        self.wq, self.wk = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.wv, self.wo = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.gn2 = nn.Parameter(torch.ones(D))
        self.w1, self.w2 = nn.Linear(D, FF, bias=False), nn.Linear(FF, D, bias=False)

    @staticmethod
    def rms(x, g):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g

    def forward(self, x, mask):
        B, Sq, _ = x.shape
        n = self.rms(x, self.gn1)
        q = self.wq(n).view(B, Sq, H, HD).transpose(1, 2)
        k = self.wk(n).view(B, Sq, H, HD).transpose(1, 2)
        v = self.wv(n).view(B, Sq, H, HD).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(HD)
        a = a.masked_fill(mask[:, :, :Sq, :Sq] == 0, float("-inf")).softmax(-1)
        y = (a @ v).transpose(1, 2).reshape(B, Sq, D)
        x = x + self.wo(y)
        return x + self.w2(F.relu(self.w1(self.rms(x, self.gn2))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.ModuleList(Block() for _ in range(L))
        self.gnf = nn.Parameter(torch.ones(D))
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, T, T)))

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x, self.mask)
        return Block.rms(x, self.gnf) @ self.tok.weight.T


dev = "cuda" if torch.cuda.is_available() else "cpu"
m = GPT().to(dev)
print(f"params {sum(p.numel() for p in m.parameters()):,}  device {dev}", flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
BS, STEPS = 64, 3000


def batch(split):
    d = train if split == "train" else val
    ix = np.random.randint(0, len(d) - T - 1, BS)
    x = np.stack([d[i : i + T] for i in ix])
    y = np.stack([d[i + 1 : i + T + 1] for i in ix])
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


vl = float("nan")
for step in range(STEPS):
    x, y = batch("train")
    loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0 or step == STEPS - 1:
        with torch.no_grad():
            vx, vy = batch("val")
            vl = F.cross_entropy(m(vx).reshape(-1, V), vy.reshape(-1)).item()
        print(f"step {step:>4}  train {loss.item():.3f}  val {vl:.3f}", flush=True)
float_val_loss = vl
torch.save({"model": {k: v.cpu() for k, v in m.state_dict().items()},
            "stoi": stoi, "itos": itos}, OUT / "gpt_state.pt")

P = {k: v.detach().float().cpu().numpy() for k, v in m.state_dict().items()}


# ------------------------------------------------------------------ quantization machinery
def qsym(w):
    s = float(np.abs(w).max()) / 127.0
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), s


def multiplier(ratio):
    """Express a positive real as (M0, shift) with out = (acc * M0) >> shift, M0 < 2^31."""
    assert ratio > 0, ratio
    shift = 0
    r = ratio
    while r < (1 << 29):
        r *= 2.0
        shift += 1
        if shift > 62:
            break
    while r >= (1 << 31):
        r /= 2.0
        shift -= 1
    return int(round(r)), shift


Wq, S = {}, {}
for k in P:
    Wq[k], S[k] = qsym(P[k])


# ------------------------------------------------------------------ calibration
# Observe real activations on real text, so every scale is measured.
def float_trace(toks):
    tr = {}
    x = P["tok.weight"][toks] + P["pos.weight"][: len(toks)]
    tr["res.-1"] = x.copy()
    for li in range(L):
        b = f"blocks.{li}."
        n = x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6) * P[b + "gn1"]
        tr[f"n1.{li}"] = n.copy()
        q = n @ P[b + "wq.weight"].T
        k = n @ P[b + "wk.weight"].T
        v = n @ P[b + "wv.weight"].T
        tr[f"qkv.{li}"] = np.concatenate([q, k, v]).copy()
        att = np.zeros_like(q)
        scores = []
        for h in range(H):
            sl = slice(h * HD, (h + 1) * HD)
            sc = q[:, sl] @ k[:, sl].T / math.sqrt(HD)
            msk = np.tril(np.ones_like(sc))
            scores.append(sc[msk > 0])
            sc = np.where(msk > 0, sc, -1e9)
            p = np.exp(sc - sc.max(-1, keepdims=True))
            att[:, sl] = (p / p.sum(-1, keepdims=True)) @ v[:, sl]
        tr[f"score.{li}"] = np.concatenate(scores)
        tr[f"att.{li}"] = att.copy()
        x = x + att @ P[b + "wo.weight"].T
        tr[f"res1.{li}"] = x.copy()
        n2 = x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6) * P[b + "gn2"]
        tr[f"n2.{li}"] = n2.copy()
        hh = np.maximum(n2 @ P[b + "w1.weight"].T, 0)
        tr[f"h.{li}"] = hh.copy()
        x = x + hh @ P[b + "w2.weight"].T
        tr[f"res2.{li}"] = x.copy()
    tr["nf"] = (x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6) * P["gnf"]).copy()
    return tr


obs = {}
for c in range(24):
    i = c * 1511
    tr = float_trace(list(val[i : i + T]))
    for k, a in tr.items():
        obs.setdefault(k, []).append(float(np.percentile(np.abs(a), 99.9)))
CAL = {k: max(float(np.mean(v)), 1e-6) for k, v in obs.items()}

# One residual scale for the whole stream, from the largest residual observed.
res_absmax = max(CAL[k] for k in CAL if k.startswith("res"))
S_RES = res_absmax / 8.0e6          # int32 residual, generous headroom
S_ACT = {k: CAL[k] / 127.0 for k in CAL}

print("calibrated activation absmax (99.9pct):", flush=True)
for k in ["res.-1", "n1.0", "qkv.0", "score.0", "att.0", "res1.0", "h.0", "nf"]:
    print(f"  {k:<9} {CAL[k]:>8.3f}", flush=True)

# RMSNorm fixed-point: normalized = xs * (1<<RN_K) / rms, then * g.
RN_PRE, RN_K = 4, 20
# exp LUT over the calibrated score scale.
S_SCORE = CAL["score.0"] * 2.0 / 127.0      # score int unit, in nats
LUT_MAX = 16384
EXP_LUT = np.array([int(round(LUT_MAX * math.exp(-i * S_SCORE))) for i in range(256)],
                   dtype=np.int32)

MUL = {}
# Position embeddings carry their own quantization scale; this Q10 numerator converts
# them into token-embedding units with a shift, which is what the C can do cheaply.
POS_NUM = int(round(S["pos.weight"] / S["tok.weight"] * 1024))


def set_mul(name, ratio):
    M0, sh = multiplier(ratio)
    MUL[name] = (M0, sh)
    return M0, sh


# embedding -> residual
set_mul("emb", S["tok.weight"] / S_RES)
for li in range(L):
    b = f"blocks.{li}."
    # RMSNorm out: (xs * inv * g) >> ?, see int_forward for the algebra.
    set_mul(f"n1.{li}", S[b + "gn1"] / (S_ACT[f"n1.{li}"] * (1 << RN_K)))
    set_mul(f"n2.{li}", S[b + "gn2"] / (S_ACT[f"n2.{li}"] * (1 << RN_K)))
    set_mul(f"q.{li}", S_ACT[f"n1.{li}"] * S[b + "wq.weight"] / S_ACT[f"qkv.{li}"])
    set_mul(f"k.{li}", S_ACT[f"n1.{li}"] * S[b + "wk.weight"] / S_ACT[f"qkv.{li}"])
    set_mul(f"v.{li}", S_ACT[f"n1.{li}"] * S[b + "wv.weight"] / S_ACT[f"qkv.{li}"])
    # score: (q.k) * s_qkv^2 / sqrt(HD), expressed in S_SCORE units
    set_mul(f"sc.{li}", S_ACT[f"qkv.{li}"] ** 2 / math.sqrt(HD) / S_SCORE)
    set_mul(f"o.{li}", S_ACT[f"qkv.{li}"] * S[b + "wo.weight"] / S_RES)
    set_mul(f"h.{li}", S_ACT[f"n2.{li}"] * S[b + "w1.weight"] / S_ACT[f"h.{li}"])
    set_mul(f"m.{li}", S_ACT[f"h.{li}"] * S[b + "w2.weight"] / S_RES)
set_mul("nf", S["gnf"] / (S_ACT["nf"] * (1 << RN_K)))


def rq(acc, name):
    """out = (acc * M0) >> shift, in int64 then clamped to int8."""
    M0, sh = MUL[name]
    a = np.asarray(acc, dtype=np.int64) * M0
    return np.clip(a >> sh, -127, 127).astype(np.int8)


def isqrt(n):
    return math.isqrt(int(n)) if n > 0 else 0


def rmsnorm_q(x32, g, name):
    """int32 residual -> int8 normalized. Scale-invariant, so no residual scale enters."""
    xs = np.asarray(x32, np.int64) >> RN_PRE
    ss = int((xs * xs).sum())
    r = max(isqrt(ss // D), 1)
    inv = (1 << RN_K) // r
    return rq((xs * inv) * g.astype(np.int64), name)


def int_forward(toks):
    kc = np.zeros((L, T, D), np.int8)
    vc = np.zeros((L, T, D), np.int8)
    x = np.zeros(D, np.int64)
    for pos, tok in enumerate(toks):
        # Position embeddings are quantized against their own scale, so they must be
        # rescaled into token-embedding units. The C does this as (v * POS_NUM) >> 10,
        # an arithmetic shift; mirror that exactly rather than rounding in float, or the
        # two paths disagree in the last bit and greedy decode amplifies it.
        pos_rescaled = (Wq["pos.weight"][pos].astype(np.int64) * POS_NUM) >> 10
        emb = Wq["tok.weight"][tok].astype(np.int64) + pos_rescaled
        M0, sh = MUL["emb"]
        x = (emb * M0) >> sh
        for li in range(L):
            b = f"blocks.{li}."
            n = rmsnorm_q(x, Wq[b + "gn1"], f"n1.{li}")
            qv = rq(n.astype(np.int32) @ Wq[b + "wq.weight"].astype(np.int32).T, f"q.{li}")
            kv = rq(n.astype(np.int32) @ Wq[b + "wk.weight"].astype(np.int32).T, f"k.{li}")
            vv = rq(n.astype(np.int32) @ Wq[b + "wv.weight"].astype(np.int32).T, f"v.{li}")
            kc[li, pos], vc[li, pos] = kv, vv
            att = np.zeros(D, np.int8)
            for h in range(H):
                sl = slice(h * HD, (h + 1) * HD)
                raw = kc[li, : pos + 1, sl].astype(np.int64) @ qv[sl].astype(np.int64)
                M0, sh = MUL[f"sc.{li}"]
                s_i = (raw * M0) >> sh
                d = np.clip(int(s_i.max()) - s_i, 0, 255)
                p = EXP_LUT[d].astype(np.int64)
                psum = max(int(p.sum()), 1)
                acc = (p[:, None] * vc[li, : pos + 1, sl].astype(np.int64)).sum(0)
                att[sl] = np.clip(acc // psum, -127, 127).astype(np.int8)
            x = x + rq_i32(att.astype(np.int32) @ Wq[b + "wo.weight"].astype(np.int32).T,
                           f"o.{li}")
            n2 = rmsnorm_q(x, Wq[b + "gn2"], f"n2.{li}")
            hh = np.maximum(n2.astype(np.int32) @ Wq[b + "w1.weight"].astype(np.int32).T, 0)
            hq = rq(hh, f"h.{li}")
            x = x + rq_i32(hq.astype(np.int32) @ Wq[b + "w2.weight"].astype(np.int32).T,
                           f"m.{li}")
    nf = rmsnorm_q(x, Wq["gnf"], "nf")
    return nf.astype(np.int32) @ Wq["tok.weight"].astype(np.int32).T


def rq_i32(acc, name):
    """Requantize into residual units (int32, no int8 clamp)."""
    M0, sh = MUL[name]
    return (np.asarray(acc, np.int64) * M0) >> sh


ptoks = [stoi[c] for c in PROMPT]
gen = list(ptoks)
for _ in range(NGEN):
    gen.append(int(np.argmax(int_forward(gen[-T:]))))
int_text = "".join(itos[t] for t in gen)

with torch.no_grad():
    fg = list(ptoks)
    for _ in range(NGEN):
        fg.append(int(m(torch.tensor([fg[-T:]], device=dev))[0, -1].argmax()))
float_text = "".join(itos[t] for t in fg)

print(f"float   : {float_text!r}", flush=True)
print(f"integer : {int_text!r}", flush=True)

nval, tot = 48, 0.0
# Integer logits are in units of (nf activation scale x token-embedding weight scale).
# Converting to nats is what makes the loss comparable with the float model; argmax is
# scale-invariant, so generation never depended on this.
LOGIT_UNIT = S_ACT["nf"] * S["tok.weight"]
for i in range(nval):
    lg = int_forward(list(val[i * 97 : i * 97 + T])).astype(np.float64) * LOGIT_UNIT
    lg -= lg.max()
    pr = np.exp(lg) / np.exp(lg).sum()
    tot += -math.log(max(pr[val[i * 97 + T]], 1e-12))
int_val_loss = tot / nval
print(f"float val loss {float_val_loss:.3f}   integer val loss {int_val_loss:.3f}", flush=True)

# ------------------------------------------------------------------ emit C
def carr(name, arr, ctype):
    f = np.asarray(arr).reshape(-1)
    return f"static const {ctype} {name}[{f.size}] = {{" + ",".join(str(int(v)) for v in f) + "};\n"


mul_names = ["emb", "nf"] + [f"{p}.{li}" for li in range(L)
                             for p in ["n1", "n2", "q", "k", "v", "sc", "o", "h", "m"]]
mul_idx = {n: i for i, n in enumerate(mul_names)}
M0s = np.array([MUL[n][0] for n in mul_names], np.int64)
SHs = np.array([MUL[n][1] for n in mul_names], np.int32)

h = ["/* GENERATED by train_gpt.py -- do not edit.\n",
     f"   Character-level GPT: vocab {V}, d_model {D}, {L} layers, {H} heads, ff {FF}, ctx {T}.\n",
     f"   float val loss {float_val_loss:.3f}, integer val loss {int_val_loss:.3f}.\n",
     "   int8 weights in DROM, calibrated fixed-point requantization, LUT softmax,\n",
     "   integer floor-sqrt RMSNorm. No floating point anywhere. */\n",
     "#pragma once\n#include <stdint.h>\n\n",
     f"#define GV {V}\n#define GD {D}\n#define GL {L}\n#define GH {H}\n#define GHD {HD}\n",
     f"#define GFF {FF}\n#define GT {T}\n#define GNGEN {NGEN}\n#define GNPROMPT {len(ptoks)}\n",
     f"#define RN_PRE {RN_PRE}\n#define RN_K {RN_K}\n#define POS_NUM {POS_NUM}\n\n"]
for n, i in mul_idx.items():
    h.append(f"#define MUL_{n.replace('.', '_').upper()} {i}\n")
h.append("\n")
h.append(carr("g_m0", M0s, "int32_t"))
h.append(carr("g_sh", SHs, "int32_t"))
h.append(carr("g_tok", Wq["tok.weight"], "int8_t"))
h.append(carr("g_pos", Wq["pos.weight"], "int8_t"))
for p, nm in [("wq", "g_wq"), ("wk", "g_wk"), ("wv", "g_wv"), ("wo", "g_wo"),
              ("w1", "g_w1"), ("w2", "g_w2")]:
    h.append(carr(nm, np.stack([Wq[f"blocks.{i}.{p}.weight"] for i in range(L)]), "int8_t"))
for p, nm in [("gn1", "g_gn1"), ("gn2", "g_gn2")]:
    h.append(carr(nm, np.stack([Wq[f"blocks.{i}.{p}"] for i in range(L)]), "int8_t"))
h.append(carr("g_gnf", Wq["gnf"], "int8_t"))
h.append(carr("g_exp", EXP_LUT, "int32_t"))
h.append(carr("g_prompt", np.array(ptoks), "uint8_t"))
h.append(carr("g_chars", np.array([ord(c) for c in chars]), "uint8_t"))
# g_expect is NOT emitted here: it is derived from this header by gen_expect.py, so the
# specification the C is checked against uses the shipped int8 arithmetic rather than
# anything this process holds in float.
(OUT / "gpt_weights.h").write_text("".join(h))

nb = int(sum(np.asarray(Wq[k]).size for k in Wq))
meta = {
    "arch": f"char GPT V={V} d={D} L={L} H={H} ff={FF} ctx={T}",
    "params_int8": nb,
    "rodata_bytes": nb + EXP_LUT.size * 4 + M0s.size * 8,
    "float_val_loss": round(float(float_val_loss), 4),
    "int_val_loss": round(float(int_val_loss), 4),
    "prompt": PROMPT,
    "float_generation": float_text,
    "int_generation": int_text,
    "generations_agree": int_text == float_text,
    "expected_tokens": [int(t) for t in gen],
    "calibration": {k: round(v, 5) for k, v in sorted(CAL.items())},
    "score_scale_nats": round(S_SCORE, 6),
    "multipliers": {n: list(MUL[n]) for n in mul_names},
}
(OUT / "gpt_meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps({k: meta[k] for k in ("params_int8", "rodata_bytes", "float_val_loss",
                                       "int_val_loss", "generations_agree")}, indent=1))
print("wrote gpt_weights.h and gpt_meta.json")
