"""Are the machine's tensors the same bytes in the text-only build and the vision+MTP build?"""
import hashlib, json, re, sys
from pathlib import Path
from safetensors import safe_open
A, B, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
wa = json.loads((A / "model.safetensors.index.json").read_text())["weight_map"]
wb = json.loads((B / "model.safetensors.index.json").read_text())["weight_map"]
def get(root, wm, k):
    with safe_open(root / wm[k], "pt") as fh:
        t = fh.get_tensor(k)
    return str(t.dtype), tuple(t.shape), hashlib.sha256(t.contiguous().view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes()).hexdigest()
L = re.compile(r"^model\.layers\.(\d+)\.")
res = {"a": str(A), "b": str(B), "tensors_a": len(wa), "tensors_b": len(wb),
       "only_in_b": {}, "only_in_a": 0, "groups": {}}
for k in wb:
    if k not in wa:
        p = k.split(".")[1] if k.startswith("model.") else k.split(".")[0]
        res["only_in_b"][p] = res["only_in_b"].get(p, 0) + 1
res["only_in_a"] = sum(k not in wb for k in wa)
diffs = []
for k in sorted(wa):
    if k not in wb:
        continue
    m = L.match(k)
    grp = ("chip_layers" if int(m.group(1)) < 11 else "host_layers") if m else "embed_head_norm"
    da, db = get(A, wa, k), get(B, wb, k)
    g = res["groups"].setdefault(grp, {"n": 0, "identical": 0, "dtype_differs": 0})
    g["n"] += 1
    if da == db: g["identical"] += 1
    else:
        if da[0] != db[0]: g["dtype_differs"] += 1
        diffs.append([k, da[0], db[0], list(da[1]), list(db[1])])
    print(grp, k, da == db, flush=True)
res["differing"] = diffs
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "differing"}))
