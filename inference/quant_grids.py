"""The numeric grids of each shipped weight format, as pure functions.

Separated from the probes so both quant_probe.py (chip layers) and
quant_probe_text.py (text layers) round-trip through exactly the same arithmetic.
Each takes an fp32 tensor and returns it rounded onto that format's grid.
"""
import torch


def q_bf16(w):
    """The checkpoint's own format: the control. Must come back exact."""
    return w.to(torch.bfloat16).to(torch.float32)


def q_fp8_e4m3(w):
    """FP8_BLOCK, as in Qwen's official layout: e4m3 weights, 128x128 block scales."""
    out = w.clone().reshape(-1)
    n = out.numel()
    blk = 128 * 128
    for i in range(0, n, blk):
        chunk = out[i:i + blk]
        amax = chunk.abs().max()
        if amax == 0:
            continue
        # e4m3 max is 448; scale the block into range, round on the grid, scale back
        scale = amax / 448.0
        q = (chunk / scale).to(torch.float8_e4m3fn).to(torch.float32)
        out[i:i + blk] = q * scale
    return out.reshape(w.shape)


def q_nvfp4a16(w):
    """NVFP4 weights: e2m1 on groups of 16, fp8 group scales, fp32 global scale."""
    flat = w.reshape(-1)
    pad = (-flat.numel()) % 16
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    g = flat.reshape(-1, 16)
    amax = g.abs().amax(dim=1, keepdim=True)
    # e2m1 representable magnitudes
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    scale = (amax / 6.0).clamp(min=1e-12)
    # Group scales are stored as fp8 relative to one fp32 global scale per tensor, chosen so
    # the largest group scale lands on the top of the e4m3 range. Casting the raw scales to
    # fp8 without it underflows every small group to zero.
    glob = (scale.max() / 448.0).clamp(min=1e-30)
    scale = ((scale / glob).to(torch.float8_e4m3fn).to(torch.float32) * glob).clamp(min=1e-12)
    norm = (g / scale).abs()
    idx = (norm.unsqueeze(-1) - grid).abs().argmin(dim=-1)
    q = grid[idx] * torch.sign(g) * scale
    return q.reshape(-1)[: w.numel()].reshape(w.shape)


def q_q8_0(w):
    """llama.cpp Q8_0: int8 with one fp16 scale per 32 weights."""
    flat = w.reshape(-1)
    pad = (-flat.numel()) % 32
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    g = flat.reshape(-1, 32)
    amax = g.abs().amax(dim=1, keepdim=True)
    scale = (amax / 127.0).to(torch.float16).to(torch.float32).clamp(min=1e-12)
    q = torch.round(g / scale).clamp(-127, 127) * scale
    return q.reshape(-1)[: w.numel()].reshape(w.shape)


def q_q4_k_m(w):
    """A stand-in for Q4_K_M's 4-bit grid: 4 bits over 32-weight blocks.

    Real Q4_K uses a two-level superblock scale and leaves some tensors at higher
    precision. This is deliberately the optimistic version: if even this breaks the
    processor, the real thing certainly does.
    """
    flat = w.reshape(-1)
    pad = (-flat.numel()) % 32
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    g = flat.reshape(-1, 32)
    amax = g.abs().amax(dim=1, keepdim=True)
    scale = (amax / 7.0).to(torch.float16).to(torch.float32).clamp(min=1e-12)
    q = torch.round(g / scale).clamp(-8, 7) * scale
    return q.reshape(-1)[: w.numel()].reshape(w.shape)


FORMATS = {"bf16": q_bf16, "fp8": q_fp8_e4m3, "nvfp4a16": q_nvfp4a16,
           "q8_0": q_q8_0, "q4_k_m": q_q4_k_m}
