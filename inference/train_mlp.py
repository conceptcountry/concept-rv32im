"""Train a tiny MNIST MLP and emit int8 weights as a C header for the RV32IM machine.

784 -> 32 -> 10, ReLU, no biases on the output layer beyond int32 accumulators.
Everything is integer at inference: int8 weights, int8 activations, int32 accumulators,
with a per-layer power-of-two requantization shift so the RV32 code needs no division
and no floating point (the machine is RV32IM_Zba: integer only, no FPU).

Writes:
  mlp_weights.h   const int8_t arrays in rodata, plus the shifts
  mlp_meta.json   accuracy and quantization metadata for the record
"""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------- data
# MNIST straight from the idx files: torchvision is not needed for four arrays, and
# keeping the dependency surface small matters more here than convenience.
import gzip  # noqa: E402
import urllib.request  # noqa: E402

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def _idx(path):
    with open(path, "rb") as raw_fh:
        raw = gzip.decompress(raw_fh.read())
    magic = int.from_bytes(raw[:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw[4 + 4 * ndim :], dtype=np.uint8).reshape(dims)


cache = OUT / "mnist"
cache.mkdir(parents=True, exist_ok=True)
data = {}
for key, name in FILES.items():
    dest = cache / name
    if not dest.exists():
        urllib.request.urlretrieve(MIRROR + name, dest)
    data[key] = _idx(dest)

Xtr = torch.from_numpy(data["train_x"].reshape(-1, 784).copy()).float() / 255.0
Ytr = torch.from_numpy(data["train_y"].copy()).long()
Xte = torch.from_numpy(data["test_x"].reshape(-1, 784).copy()).float() / 255.0
Yte = torch.from_numpy(data["test_y"].copy()).long()
print(f"MNIST: train {tuple(Xtr.shape)}  test {tuple(Xte.shape)}", flush=True)

H = 32


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, H, bias=True)
        self.fc2 = nn.Linear(H, 10, bias=True)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


m = MLP()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for epoch in range(8):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 256):
        idx = perm[i : i + 256]
        opt.zero_grad()
        loss = F.cross_entropy(m(Xtr[idx]), Ytr[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (m(Xte).argmax(1) == Yte).float().mean().item()
    print(f"epoch {epoch}  test acc {acc:.4f}", flush=True)

float_acc = acc

# ---------------------------------------------------------------- quantize
# Symmetric per-tensor int8. Activations enter as uint8 pixels (0..255) scaled to int8
# range by >>1 so the first matmul's accumulator cannot overflow int32:
#   784 * 127 * 127 = 12.6M, well inside int32.
W1 = m.fc1.weight.detach().numpy()  # (H, 784)
B1 = m.fc1.bias.detach().numpy()
W2 = m.fc2.weight.detach().numpy()  # (10, H)
B2 = m.fc2.bias.detach().numpy()


def qsym(w):
    s = np.abs(w).max() / 127.0
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), s


W1q, s1 = qsym(W1)
W2q, s2 = qsym(W2)

# Input scale: x in [0,1] mapped to 0..127 (int8-safe).
sx = 1.0 / 127.0
# Bias in accumulator units of layer 1: acc1 ~ (x/sx) . (W1/s1) so bias/(sx*s1).
B1q = np.round(B1 / (sx * s1)).astype(np.int32)

# Requantize layer-1 accumulator to int8 activations with a power-of-two shift.
with torch.no_grad():
    h_float = F.relu(m.fc1(Xte[:2000])).numpy()
amax = np.percentile(h_float, 99.9)
sh = amax / 127.0 if amax > 0 else 1.0
# acc1 * (sx*s1) = h_float  ->  h_int8 = acc1 * (sx*s1/sh); choose shift ~ that ratio.
ratio = (sx * s1) / sh
SHIFT1 = int(np.clip(np.round(-np.log2(ratio)), 0, 30))
B2q = np.round(B2 / (sh * s2)).astype(np.int32)

print(f"s1={s1:.6g} s2={s2:.6g} sx={sx:.6g} sh={sh:.6g} SHIFT1={SHIFT1}", flush=True)


def infer_int(x_u8):
    """Exactly what the C will do, in numpy, for cross-checking."""
    a0 = (x_u8.astype(np.int32) >> 1)  # 0..127
    acc1 = a0 @ W1q.astype(np.int32).T + B1q
    h = np.clip(acc1 >> SHIFT1, 0, 127)
    acc2 = h @ W2q.astype(np.int32).T + B2q
    return acc2


Xte_u8 = (Xte.numpy() * 255.0).round().astype(np.uint8)
logits = infer_int(Xte_u8)
int_acc = float((logits.argmax(1) == Yte.numpy()).mean())
print(f"float acc {float_acc:.4f}   int8 acc {int_acc:.4f}", flush=True)

# ---------------------------------------------------------------- pick test digits
# Ten test images, one per class, that the integer model classifies correctly.
chosen = []
pred = logits.argmax(1)
for d in range(10):
    cand = np.where((Yte.numpy() == d) & (pred == d))[0]
    chosen.append(int(cand[0]))
imgs = Xte_u8[chosen]
labels = Yte.numpy()[chosen]

# ---------------------------------------------------------------- emit C
def carr(name, arr, ctype="int8_t"):
    flat = arr.reshape(-1)
    body = ",".join(str(int(v)) for v in flat)
    return f"static const {ctype} {name}[{flat.size}] = {{{body}}};\n"


h = ['/* GENERATED by train_mlp.py -- do not edit.\n']
h.append(f"   MNIST MLP 784->{H}->10, symmetric int8, power-of-two requantization.\n")
h.append(f"   float test acc {float_acc:.4f}, integer test acc {int_acc:.4f} over 10000 images.\n")
h.append("   Weights live in rodata (DROM); only activations touch RAM. */\n")
h.append("#pragma once\n#include <stdint.h>\n\n")
h.append(f"#define MLP_IN   784\n#define MLP_HID  {H}\n#define MLP_OUT  10\n")
h.append(f"#define MLP_SHIFT1 {SHIFT1}\n#define MLP_NTEST {len(chosen)}\n\n")
h.append(carr("mlp_w1", W1q))           # (H,784) row-major: w1[j*784+i]
h.append(carr("mlp_b1", B1q, "int32_t"))
h.append(carr("mlp_w2", W2q))           # (10,H)
h.append(carr("mlp_b2", B2q, "int32_t"))
h.append(carr("mlp_test_img", imgs, "uint8_t"))
h.append(carr("mlp_test_lab", labels, "uint8_t"))
(OUT / "mlp_weights.h").write_text("".join(h))

meta = {
    "arch": f"784-{H}-10 MLP, ReLU",
    "float_test_acc": round(float_acc, 4),
    "int8_test_acc": round(int_acc, 4),
    "shift1": SHIFT1,
    "params_int8": int(W1q.size + W2q.size),
    "rodata_bytes": int(W1q.size + W2q.size + B1q.size * 4 + B2q.size * 4 + imgs.size + 10),
    "test_indices": chosen,
    "test_labels": [int(v) for v in labels],
    "expected_logits": [[int(v) for v in row] for row in logits[chosen]],
    "expected_pred": [int(v) for v in pred[chosen]],
}
(OUT / "mlp_meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps({k: meta[k] for k in ("float_test_acc", "int8_test_acc", "params_int8", "rodata_bytes")}, indent=1))
print("wrote mlp_weights.h and mlp_meta.json")
