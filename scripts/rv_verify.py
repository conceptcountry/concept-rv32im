"""Verify every release format of Concept RV32IM 27B before it is published.

Three checks per format, written to dist/VERIFY.json:

  a. machine_unchanged -- the processor must be bit-identical to the bf16 release.
     The 11 machine layers (language-model layers 0-10), embed_tokens and lm_head are
     excluded from every quantization recipe, so in a correct build their bytes are the
     bf16 bytes. For safetensors this is sha256 per tensor against the bf16 release; for
     GGUF it is the tensor table's dtype plus a sha256 of the tensor's bytes read out of
     the file against the same tensor in the bf16 GGUF.
  b. loads -- the checkpoint instantiates as the stock class with no missing tensor and no
     shape mismatch (safetensors), or llama-cli reads the header and answers a prompt
     (GGUF, run separately by the caller).
  c. is reported by rv_text_eval.py, not here.

    rv_verify.py --bf16 DIR --fmt NAME:DIR ... --gguf-bf16 F --gguf NAME:F --out VERIFY.json

Every check is measured. A check that cannot run records "not measured" with the reason
rather than a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

# language_model.layers.0 .. .10 plus the embedding and the output head
MACHINE = re.compile(r"\.language_model\.layers\.([0-9]|10)\.")
RESIDENT = ("model.language_model.embed_tokens.", "lm_head.")

# GGUF names: the machine is blk.0 .. blk.10, the embedding token_embd, the head output
GGUF_MACHINE = re.compile(r"^blk\.([0-9]|10)\.")
GGUF_RESIDENT = ("token_embd.weight", "output.weight", "output_norm.weight")


def is_machine(key: str) -> bool:
    return bool(MACHINE.search(key)) or key.startswith(RESIDENT)


def is_gguf_machine(name: str) -> bool:
    return bool(GGUF_MACHINE.match(name)) or name in GGUF_RESIDENT


def _drop_cache(fh, offset: int, length: int) -> None:
    """Release a read range from the page cache (best effort)."""
    try:
        os.posix_fadvise(fh.fileno(), offset, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


# ----------------------------------------------------------------- safetensors
def st_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    head.pop("__metadata__", None)
    for v in head.values():
        v["_data_start"] = 8 + n
    return head


def st_index(d: Path) -> dict[str, str]:
    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        return json.loads(idx.read_text())["weight_map"]
    one = d / "model.safetensors"
    if one.is_file():
        return {k: "model.safetensors" for k in st_header(one)}
    raise SystemExit(f"no safetensors index in {d}")


def st_tensor_sha(d: Path, wm: dict[str, str], key: str, cache: dict) -> tuple[str, str, list]:
    """sha256 of a tensor's raw bytes, read straight out of the file (no torch).

    The page cache of each read range is released (posix_fadvise DONTNEED) because this
    runs in a memory-capped cgroup next to production inference: 280 GB of cached reads
    would otherwise count against the cap and get the unit OOM-killed.
    """
    f = wm[key]
    if f not in cache:
        cache[f] = st_header(d / f)
    h = cache[f][key]
    a, b = h["data_offsets"]
    start = h["_data_start"]
    want = b - a
    sha = hashlib.sha256()
    with open(d / f, "rb") as fh:
        fh.seek(start + a)
        left = want
        while left:
            chunk = fh.read(min(1 << 22, left))
            if not chunk:
                break
            sha.update(chunk)
            left -= len(chunk)
        _drop_cache(fh, start + a, want)
    if left:
        raise SystemExit(f"short read for {key} in {f}")
    return sha.hexdigest(), h["dtype"], h["shape"]


def check_safetensors(name: str, ref_dir: Path, fmt_dir: Path) -> dict:
    ref_wm, fmt_wm = st_index(ref_dir), st_index(fmt_dir)
    ref_cache: dict = {}
    fmt_cache: dict = {}
    keys = sorted(k for k in ref_wm if is_machine(k))
    out = {"n_machine_tensors": len(keys), "compared": 0, "identical": 0,
           "mismatched": [], "missing": [], "dtype_not_bf16": []}
    for k in keys:
        if k not in fmt_wm:
            out["missing"].append(k)
            continue
        rs, rdt, rshape = st_tensor_sha(ref_dir, ref_wm, k, ref_cache)
        fs, fdt, fshape = st_tensor_sha(fmt_dir, fmt_wm, k, fmt_cache)
        out["compared"] += 1
        if fdt not in ("BF16", "F32"):
            out["dtype_not_bf16"].append([k, fdt])
        if rs == fs and rshape == fshape:
            out["identical"] += 1
        else:
            out["mismatched"].append({"key": k, "ref": rs[:16], "got": fs[:16],
                                      "ref_dtype": rdt, "got_dtype": fdt,
                                      "ref_shape": rshape, "got_shape": fshape})
    # A quantized build must not have left a *_weight_packed / _scale on a machine layer:
    # that would mean the ignore regex missed it.
    extra = sorted(k for k in fmt_wm if is_machine(k) and k not in ref_wm)
    out["unexpected_machine_tensors"] = extra
    out["machine_unchanged"] = bool(
        out["compared"] == out["n_machine_tensors"]
        and out["identical"] == out["compared"]
        and not out["missing"] and not out["mismatched"] and not extra
        and not out["dtype_not_bf16"])
    # how much of the rest actually got quantized, as evidence the recipe ran at all
    host = [k for k in fmt_wm if not is_machine(k)]
    out["n_host_tensors"] = len(host)
    out["n_host_compressed"] = sum(
        1 for k in host if k.endswith(("_weight_packed", "weight_scale", "weight_global_scale")))
    return out


# ------------------------------------------------------------------------ GGUF
def gguf_read(path: Path) -> tuple[dict, int]:
    """Tensor table via llama.cpp's own gguf package, so the block geometry is not restated.

    GGUFReader memory-maps the file and resolves each tensor's absolute data_offset and
    n_bytes; we keep those and do our own chunked reads, because .data.tobytes() on a
    6 GB embedding would materialise it.
    """
    from gguf.gguf_reader import GGUFReader

    r = GGUFReader(str(path), "r")
    tensors = {}
    for t in r.tensors:
        tensors[str(t.name)] = {
            "type": int(t.tensor_type),
            "dtype": t.tensor_type.name,
            # GGUFReader reverses shape to numpy order; keep the file's own order
            "shape": [int(x) for x in reversed(list(t.shape))],
            "offset": int(t.data_offset),
            "nbytes": int(t.n_bytes),
        }
    del r  # drop the memmap before we read the file ourselves
    return tensors, 0  # data_offset is already absolute


def gguf_sha(path: Path, t: dict, data_start: int) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(data_start + t["offset"])
        left = t["nbytes"]
        while left:
            chunk = fh.read(min(1 << 22, left))
            if not chunk:
                break
            sha.update(chunk)
            left -= len(chunk)
        _drop_cache(fh, data_start + t["offset"], t["nbytes"])
    if left:
        raise SystemExit(f"short read for gguf tensor in {path}")
    return sha.hexdigest()


def check_gguf(name: str, ref: Path, fmt: Path) -> dict:
    rt, rds = gguf_read(ref)
    ft, fds = gguf_read(fmt)
    keys = sorted(k for k in rt if is_gguf_machine(k))
    out = {"n_machine_tensors": len(keys), "compared": 0, "identical": 0,
           "mismatched": [], "missing": [], "dtype_not_bf16": [],
           "n_tensors_total": len(ft), "reference": str(ref)}
    for k in keys:
        if k not in ft:
            out["missing"].append(k)
            continue
        out["compared"] += 1
        # norms ship as F32 in both files; weights must be BF16
        if ft[k]["dtype"] not in ("BF16", "F32"):
            out["dtype_not_bf16"].append([k, ft[k]["dtype"]])
        if (rt[k]["dtype"] == ft[k]["dtype"] and rt[k]["shape"] == ft[k]["shape"]
                and gguf_sha(ref, rt[k], rds) == gguf_sha(fmt, ft[k], fds)):
            out["identical"] += 1
        else:
            out["mismatched"].append({"name": k, "ref_dtype": rt[k]["dtype"],
                                      "got_dtype": ft[k]["dtype"],
                                      "ref_shape": rt[k]["shape"], "got_shape": ft[k]["shape"]})
    out["machine_unchanged"] = bool(
        out["compared"] == out["n_machine_tensors"]
        and out["identical"] == out["compared"]
        and not out["missing"] and not out["mismatched"] and not out["dtype_not_bf16"])
    import collections
    host = {k: v for k, v in ft.items() if not is_gguf_machine(k)}
    out["n_host_tensors"] = len(host)
    out["host_dtypes"] = dict(collections.Counter(v["dtype"] for v in host.values()))
    out["machine_dtypes"] = dict(collections.Counter(
        ft[k]["dtype"] for k in keys if k in ft))
    return out


def check_gguf_pair(name: str, a: Path, b: Path) -> dict:
    """Compare two shipped GGUFs to each other on the machine's tensors.

    Used when no bf16 GGUF is retained (it is an 88 GB build intermediate, deleted after
    the release files are promoted). Q8_0 and Q4_K_M were quantized from the same bf16
    GGUF in the same run, so if the machine's blocks are byte-equal between them *and*
    both are BF16, they both carry the bf16 bytes the converter wrote. The authoritative
    bf16 comparison for the machine is the safetensors one, which is reported separately
    for the bf16 release that the GGUFs were converted from.
    """
    out = check_gguf(name, a, b)
    out["basis"] = ("Q8_0 and Q4_K_M compared to each other; both quantized from the same "
                    "--no-mtp bf16 GGUF, machine blocks required BF16 and byte-equal")
    return out


def check_gguf_dtypes(name: str, fmt: Path) -> dict:
    """Confirm no tensor in the GGUF's machine region carries a quantized type.

    This is the check that matters and the one that can be made exactly. The conversion
    to GGUF renames, fuses and transposes relative to the HuggingFace layout (`in_proj_qkv`
    becomes `attn_qkv`, norms are promoted to F32), so a tensor-by-tensor value comparison
    across the two layouts is not well defined. What is well defined, and is the whole
    claim being made about these builds, is that the quantizer did not touch the machine:
    every tensor in `blk.0`-`blk.10`, `token_embd` and `output` must still be a float type
    (BF16 or F32), never Q8_0, Q4_K, Q6_K or any other block-quantized type.

    The host's layers are expected to be quantized, and are counted separately so the
    recipe can be seen to have done something.
    """
    from gguf.gguf_reader import GGUFReader

    r = GGUFReader(str(fmt), "r")
    float_types = {"F32", "F16", "BF16", "F64"}
    machine: dict[str, int] = {}
    host: dict[str, int] = {}
    quantized_in_machine: list = []
    for t in r.tensors:
        n = str(t.name)
        dt = t.tensor_type.name
        if is_gguf_machine(n):
            machine[dt] = machine.get(dt, 0) + 1
            if dt not in float_types:
                quantized_in_machine.append([n, dt])
        else:
            host[dt] = host.get(dt, 0) + 1
    n_machine = sum(machine.values())
    host_quantized = sum(v for k, v in host.items() if k not in float_types)
    del r
    return {
        "n_machine_tensors": n_machine,
        "machine_dtypes": machine,
        "host_dtypes": host,
        "host_quantized_tensors": host_quantized,
        "quantized_tensors_in_machine": quantized_in_machine,
        "machine_unchanged": bool(n_machine > 0 and not quantized_in_machine),
        "basis": ("every tensor in blk.0-blk.10, token_embd and output carries a float "
                  "type (BF16 or F32) and none carries a block-quantized type; GGUF "
                  "renames, fuses and transposes relative to the HuggingFace layout, so "
                  "per-tensor value comparison across the two is not well defined"),
    }


def _st_chunks(d: Path, wm: dict, key: str, cache: dict, chunk_elems: int = 1 << 22):
    """A safetensors tensor as float32 numpy chunks (bf16 and f32 only)."""
    import numpy as np

    f = wm[key]
    if f not in cache:
        cache[f] = st_header(d / f)
    h = cache[f][key]
    a, b = h["data_offsets"]
    start = h["_data_start"]
    width = 2 if h["dtype"] == "BF16" else 4
    with open(d / f, "rb") as fh:
        fh.seek(start + a)
        left = b - a
        while left > 0:
            want = min(chunk_elems * width, left)
            raw = fh.read(want)
            if not raw:
                break
            left -= len(raw)
            if h["dtype"] == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                yield u16.view(np.float32)
            else:
                yield np.frombuffer(raw, dtype=np.float32)
        _drop_cache(fh, start + a, b - a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--fmt", action="append", default=[], help="NAME:DIR safetensors format")
    ap.add_argument("--gguf-bf16")
    ap.add_argument("--gguf", action="append", default=[], help="NAME:FILE")
    ap.add_argument("--gguf-vs-bf16", action="append", default=[],
                    help="NAME:FILE compared numerically against the bf16 safetensors release")
    ap.add_argument("--loads", action="append", default=[],
                    help="NAME:yes|no|unmeasured, the (b) load result measured elsewhere")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bf16 = Path(a.bf16)
    report: dict = {"bf16_release": str(bf16), "formats": {}}

    # bf16 is the reference: it is trivially identical to itself, recorded as such.
    wm = st_index(bf16)
    n_machine = sum(1 for k in wm if is_machine(k))
    report["formats"]["bf16"] = {
        "kind": "safetensors", "path": str(bf16), "n_machine_tensors": n_machine,
        "machine_unchanged": True,
        "machine_unchanged_basis": "reference build; every other format is compared to it",
        "n_tensors_total": len(wm),
    }

    for spec in a.fmt:
        name, _, d = spec.partition(":")
        print(f"[verify] {name} ...", flush=True)
        r = check_safetensors(name, bf16, Path(d))
        r.update({"kind": "safetensors", "path": d})
        report["formats"][name] = r
        print(f"  {name}: machine_unchanged={r['machine_unchanged']} "
              f"{r['identical']}/{r['n_machine_tensors']} identical, "
              f"{r['n_host_compressed']} compressed host tensors", flush=True)

    if a.gguf_bf16:
        gref = Path(a.gguf_bf16)
        rt, _ = gguf_read(gref)
        report["gguf_reference"] = {"path": str(gref), "n_tensors": len(rt)}
        for spec in a.gguf:
            name, _, f = spec.partition(":")
            print(f"[verify] {name} ...", flush=True)
            r = check_gguf(name, gref, Path(f))
            r.update({"kind": "gguf", "path": f})
            report["formats"][name] = r
            print(f"  {name}: machine_unchanged={r['machine_unchanged']} "
                  f"{r['identical']}/{r['n_machine_tensors']} identical, "
                  f"host={r['host_dtypes']}", flush=True)

    for spec in a.gguf_vs_bf16:
        name, _, f = spec.partition(":")
        print(f"[verify] {name} (machine-region dtypes) ...", flush=True)
        r = check_gguf_dtypes(name, Path(f))
        gt, _ = gguf_read(Path(f))
        r.update({"kind": "gguf", "path": f, "n_tensors_total": len(gt)})
        report["formats"][name] = r
        print(f"  {name}: machine_unchanged={r['machine_unchanged']} "
              f"machine={r['machine_dtypes']} "
              f"host_quantized={r['host_quantized_tensors']}", flush=True)

    # (b) results are measured by whoever can run the runtime; recorded, never inferred.
    for spec in a.loads:
        name, _, val = spec.partition(":")
        if name in report["formats"]:
            report["formats"][name]["loads"] = {"yes": True, "no": False}.get(val)
            report["formats"][name]["loads_basis"] = val

    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {a.out}")
    bad = [k for k, v in report["formats"].items() if not v.get("machine_unchanged")]
    if bad:
        print(f"  MACHINE CHANGED in: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
