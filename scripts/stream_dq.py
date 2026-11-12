"""stream.py for compressed checkpoints: dequantize each shard's tensors as they are read.

~/qwenputer/stream.py reads safetensors straight into bf16 modules, which works for the
bf16 release but not for the FP8_BLOCK and NVFP4A16 builds: those store weight (or
weight_packed) plus weight_scale, and stream.py's dtype map has no FP8 or packed-FP4
entry. Rather than loading a compressed checkpoint resident (the thing that must not
happen on this machine), this wraps stream.py: the same per-layer streaming, with a
dequantization step between the file and the module, so peak residency stays at the
resident tensors plus a couple of layers.

Dequantization follows compressed-tensors' own definitions:

  FP8_BLOCK   w = w_q.float() * scale, scale broadcast over 128x128 weight blocks
  NVFP4A16    w = unpack_fp4(packed) * scale.float() * global_scale, group size 16

Both reconstruct the values the quantized kernel would see for the weights; activations
stay bf16, which is what NVFP4A16 does by construction and what dequantizing FP8_BLOCK
to bf16 approximates for FP8 (its activation quantization is dynamic per token group, so
this is the weight-faithful, not the kernel-faithful, path). That is the right comparison
for "does quantizing the host damage the text", and the card says which it is.

    from stream_dq import stream_model
    model = stream_model(path)     # bf16, fp8 or nvfp4a16 release directory
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.expanduser("~/qwenputer"))
import stream as _s  # noqa: E402

# e2m1 magnitudes, indexed by the low 3 bits; the sign is bit 3. This mirrors
# compressed_tensors.compressors.nvfp4.helpers.unpack_fp4_from_uint8 exactly (kE2M1ToFloat
# with a separate sign), rather than restating a 16-entry signed table.
KE2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _dq_fp8_block(wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8_BLOCK: one scale per 128x128 block of the weight."""
    w = wq.to(torch.float32)
    r, c = w.shape
    br = -(-r // 128)
    bc = -(-c // 128)
    s = scale.to(torch.float32)
    assert s.shape == (br, bc), f"fp8 scale {tuple(s.shape)} != {(br, bc)}"
    s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:r, :c]
    return (w * s).to(torch.bfloat16)


def _dq_nvfp4(packed: torch.Tensor, scale: torch.Tensor,
              gscale: torch.Tensor) -> torch.Tensor:
    """NVFP4: two 4-bit codes per byte, one fp8 scale per group of 16, one fp32 global.

    Nibble order, sign handling and the scale composition follow compressed-tensors:
    the low nibble is the first element of each pair, and the effective scale is
    scale / global_scale (forward_helpers._quantize_dequantize).
    """
    r, half = packed.shape
    flat = packed.reshape(-1)
    low = flat & 0x0F
    high = (flat >> 4) & 0x0F
    combined = torch.stack((low, high), dim=1).reshape(-1)
    signs = (combined & 0x08).to(torch.bool)
    mags = (combined & 0x07).to(torch.long)
    vals = KE2M1.to(packed.device)[mags] * torch.where(signs, -1.0, 1.0)
    vals = vals.reshape(r, half * 2)
    eff = scale.to(torch.float32) / gscale.to(torch.float32).reshape(())
    eff = eff.repeat_interleave(16, dim=1)[:, : vals.shape[1]]
    return (vals * eff).to(torch.bfloat16)


def _patch_files(files: "_s._Files", quant: str) -> None:
    """Make files.get(key) return a dequantized bf16 tensor for compressed weights."""
    raw_get = files.get
    hdr = files.hdr
    # FP8 shards store F8_E4M3, which torch can view; add it to stream.py's dtype map.
    if "F8_E4M3" not in files.DT:
        files.DT["F8_E4M3"] = torch.float8_e4m3fn

    def get(key, device):
        if quant == "fp8":
            sk = key + "_scale"
            if key.endswith(".weight") and sk in hdr:
                wq = raw_get(key, device)
                sc = raw_get(sk, device)
                return _dq_fp8_block(wq, sc)
            return raw_get(key, device)
        if quant == "nvfp4":
            if key.endswith(".weight"):
                pk = key + "_packed"
                if pk in hdr:
                    return _dq_nvfp4(raw_get(pk, device), raw_get(key + "_scale", device),
                                     raw_get(key + "_global_scale", device))
            return raw_get(key, device)
        return raw_get(key, device)

    files.get = get


def _quant_kind(path: Path) -> str:
    cfg = json.loads((path / "config.json").read_text())
    qc = cfg.get("quantization_config")
    if not qc:
        return "bf16"
    groups = qc.get("config_groups", {})
    names = " ".join(groups.keys()).upper()
    fmt = str(qc.get("format", "")).lower()
    if "NVFP4" in names or "nvfp4" in fmt:
        return "nvfp4"
    if "FP8" in names or "float-quantized" in fmt:
        return "fp8"
    raise SystemExit(f"unrecognised quantization_config in {path}")


def stream_model(path, device: str = "cuda", **kw):
    """stream.py's stream_model, with compressed shards dequantized on read.

    The release layout wraps the text config as `text_config` under
    `Qwen3_5ForConditionalGeneration`, while stream.py builds the text-only
    `Qwen3_5ForCausalLM` and reads `cfg.vocab_size` at the top level. For a release
    directory the text config is unwrapped into a temporary config directory, and the
    language-model key prefix is stripped, so stream.py sees the layout it expects.
    """
    path = Path(path)
    kind = _quant_kind(path)
    print(f"[stream_dq] {path.name}: {kind}", flush=True)

    cfg = json.loads((path / "config.json").read_text())
    wrapped = "text_config" in cfg

    orig_Files = _s._Files
    orig_cfg_from = None

    if wrapped:
        # stream.py calls AutoConfig.from_pretrained(path); hand it the text config.
        from transformers import AutoConfig
        orig_cfg_from = AutoConfig.from_pretrained

        def _from(p, *a, **k):
            if str(p) == str(path):
                import tempfile
                tc = dict(cfg["text_config"])
                tc.setdefault("architectures", ["Qwen3_5ForCausalLM"])
                for key in ("tie_word_embeddings", "torch_dtype", "dtype"):
                    if key in cfg and key not in tc:
                        tc[key] = cfg[key]
                d = tempfile.mkdtemp(prefix="stream_dq_cfg_")
                (Path(d) / "config.json").write_text(json.dumps(tc))
                return orig_cfg_from(d, *a, **k)
            return orig_cfg_from(p, *a, **k)

        AutoConfig.from_pretrained = _from

    class _DQFiles(orig_Files):
        """stream.py's reader with two translations layered on.

        1. Key names: the release layout prefixes the text model with
           `model.language_model.`, while the text-only class stream.py builds expects
           `model.`. The weight map is rewritten to the logical names and every read is
           translated back to the name the file actually holds.
        2. Compressed weights: `X.weight` is served by dequantizing `X.weight_packed`
           (NVFP4) or `X.weight` + `X.weight_scale` (FP8) on read.
        """

        def __init__(self, d: Path):
            super().__init__(d)
            if kind != "bf16":
                _patch_files(self, kind)
            raw_get = self.get
            logical = {}
            to_orig = {}
            for k, f in self.wm.items():
                if kind != "bf16":
                    if k.endswith("_packed"):
                        k_logical = k[: -len("_packed")]
                    elif k.endswith(("_scale", "_global_scale")):
                        continue
                    else:
                        k_logical = k
                else:
                    k_logical = k
                orig = k_logical  # the name _patch_files/get expects for this tensor
                if k_logical.startswith("model.language_model."):
                    k_logical = "model." + k_logical[len("model.language_model."):]
                elif k_logical.startswith("model.visual.") or k_logical.startswith("mtp."):
                    continue  # not part of the text graph stream.py builds
                logical[k_logical] = f
                to_orig[k_logical] = orig
            self.wm = logical

            def get(key, dev):
                return raw_get(to_orig.get(key, key), dev)

            self.get = get

    _s._Files = _DQFiles
    try:
        model = _s.stream_model(path, device=device, **kw)
    finally:
        _s._Files = orig_Files
        if wrapped and orig_cfg_from is not None:
            from transformers import AutoConfig
            AutoConfig.from_pretrained = orig_cfg_from
    return model
