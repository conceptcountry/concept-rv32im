#!/usr/bin/env python3
"""Patch merge27.py for an exact merge that keeps vision and MTP.

Three defects in the shipped merge, all verified by probe before this was written:

1. NORM PRECISION. Widening 5120 -> 6656 with zeros shrinks the RMS by
   sqrt(d0/hidden), so every norm gain is rescaled by that irrational factor and
   then cast back to bf16. Measured: 660,479 of 660,480 gains across 129 tensors
   land off the bf16 grid. Exactly one survives.

   Stock Qwen3.5 RMSNorm.forward does `self.weight.float()` — it upcasts the
   gain to float32 on every forward pass regardless of stored dtype. So storing
   the rescaled norm in float32 is not a workaround; it is the dtype the module
   consumes, and bf16 storage was discarding precision the runtime immediately
   asks for. Verified: float32 storage gives 5120/5120 bit-identical gains at
   zero error, for +1.32 MB on an ~87 GB checkpoint.

   Scaling the consumer projections instead (so the norms stay bf16) was tested
   and rejected: 14,116,454,400 of 14,116,454,400 values land off-grid. The
   irrational factor is the problem, not where it is applied.

   A geometry where d0/hidden is a power of four would make the scale a power
   of two and therefore exact, but the smallest usable width is 20480 — 3.1x
   the checkpoint. Rejected.

2. VISION DROPPED. `if not k.startswith(SRC_TEXT): continue` skips 333 visual
   tensors. The vision tower runs in its own 1152-dim space and only touches the
   text hidden state at merger.linear_fc2, whose output is 5120. Upstream's
   OUT_ROWS pattern already matches that tensor, so the widening rule exists and
   was simply never reached.

3. MTP DROPPED. Same `continue` skips 15 mtp.* tensors, and the config sets
   mtp_num_hidden_layers=0. MTP operates at 5120 throughout and needs the same
   IN_COLS/OUT_ROWS/WIDE_NORMS treatment as the text path.

Usage:  python patch_merge27.py /path/to/merge27.py
Writes merge27.py in place after asserting every anchor matched exactly once.
"""
import pathlib
import sys

PATCHES = [
    # ---- 1. exact norms: per-layer -------------------------------------------------
    (
        """                elif WIDE_NORMS.search(k):
                    g = (1.0 + w.float()) * scale - 1.0
                    w = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d0)])""",
        """                elif WIDE_NORMS.search(k):
                    # Exact rescale. The factor sqrt(d0/hidden) is irrational, so
                    # (1+g)*scale-1 lands off the bf16 grid for essentially every
                    # gain (measured: 660479/660480). Stock RMSNorm.forward calls
                    # self.weight.float(), so float32 is the dtype the module
                    # actually consumes; storing bf16 here discards precision the
                    # runtime asks for one line later. Costs ~1.3 MB in total.
                    g = (1.0 + w.float()) * scale - 1.0
                    w = torch.cat([g, torch.zeros(hidden - d0, dtype=torch.float32)])""",
    ),
    # ---- 1b. exact norms: final ----------------------------------------------------
    (
        """    norm = torch.full((hidden,), -1.0)
    norm[:d0] = (1.0 + base_norm.float()) * scale - 1.0""",
        """    # float32 throughout: see the WIDE_NORMS note above. The chip gains below
    # (code_gain, TAU_GAIN) are exact in bf16 already; the Qwen rows are not.
    norm = torch.full((hidden,), -1.0, dtype=torch.float32)
    norm[:d0] = (1.0 + base_norm.float()) * scale - 1.0""",
    ),
    (
        """    check_bf16("final norm", norm[d0:])
    W.write({"lm_head.weight": head, DST + "norm.weight": norm.to(torch.bfloat16)})""",
        """    check_bf16("final norm", norm[d0:])
    # The chip gains above are exact in bf16 by construction; check_bf16 proves
    # it. The Qwen rows are not, which is why the tensor stays float32 — assert
    # that what we store is bit-identical to the exact arithmetic, so the merge
    # proves its own claim rather than asserting it in prose.
    _want = (1.0 + base_norm.float()) * scale - 1.0
    assert torch.equal(norm[:d0], _want), "final norm host rows are not exact"
    W.write({"lm_head.weight": head, DST + "norm.weight": norm})""",
    ),
    # ---- 2 + 3. keep vision and MTP -------------------------------------------------
    (
        """                if not k.startswith(SRC_TEXT):
                    continue  # vision tower, mtp""",
        """                if k.startswith("model.visual."):
                    # The vision tower lives in its own 1152-dim space and only
                    # meets the text hidden state at merger.linear_fc2, whose
                    # output is d0. OUT_ROWS already matches that tensor.
                    if OUT_ROWS.search(k):
                        w = fh.get_tensor(k)
                        w = torch.cat([w, w.new_zeros(hidden - d0, *w.shape[1:])], 0)
                        out[k] = w
                    else:
                        out[k] = fh.get_tensor(k)
                    continue
                if k.startswith("mtp."):
                    # MTP runs at d0 end to end, so it takes the same widening as
                    # the text path. Its layernorms get the same exact float32
                    # rescale; mtp.fc reads a 2*d0 concatenation of embedding and
                    # hidden, so both halves must be padded, not just the tail.
                    w = fh.get_tensor(k)
                    if k == "mtp.fc.weight":
                        a, b = w[:, :d0], w[:, d0:]
                        pad = w.new_zeros(w.shape[0], hidden - d0)
                        w = torch.cat([a, pad, b, pad], -1)
                        w = torch.cat([w, w.new_zeros(hidden - d0, w.shape[1])], 0)
                    elif k.endswith(("norm.weight", "layernorm.weight")) and \\
                            not k.endswith(("q_norm.weight", "k_norm.weight")):
                        g = (1.0 + w.float()) * scale - 1.0
                        w = torch.cat([g, torch.zeros(hidden - d0, dtype=torch.float32)])
                    elif IN_COLS.search(k):
                        w = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d0)], -1)
                    elif OUT_ROWS.search(k):
                        w = torch.cat([w, w.new_zeros(hidden - d0, *w.shape[1:])], 0)
                    out[k] = w
                    continue
                if not k.startswith(SRC_TEXT):
                    continue""",
    ),
    # ---- 3b. advertise MTP in the config -------------------------------------------
    (
        """    tc.update(hidden_size=hidden, num_hidden_layers=n_chip + n0, mtp_num_hidden_layers=0,""",
        """    tc.update(hidden_size=hidden, num_hidden_layers=n_chip + n0,
              mtp_num_hidden_layers=tc0.get("mtp_num_hidden_layers", 0),""",
    ),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = pathlib.Path(sys.argv[1])
    src = path.read_text()

    for i, (old, new) in enumerate(PATCHES, 1):
        n = src.count(old)
        if n != 1:
            print(f"  PATCH {i} FAILED: anchor appears {n} times, expected 1")
            print(f"  anchor begins: {old.splitlines()[0][:72]}")
            return 1
        src = src.replace(old, new)
        print(f"  patch {i}/{len(PATCHES)} applied")

    path.write_text(src)
    print(f"\n  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
