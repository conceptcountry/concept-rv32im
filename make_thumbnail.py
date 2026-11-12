#!/usr/bin/env python3
"""Thumbnail for the Concept RV32IM paper. Reads results/results.json only.

The paper has no figures of its own: its claims are counts and identities rather
than distributions, and a chart of "722 of 722" would be decoration. But the
research index and the Library both show a cover image, so this draws the one
comparison that is genuinely a measurement rather than a count — teacher-forced
top-1 agreement against the harness self-agreement floor, which is the evidence
that the failed criterion sits at the noise floor rather than above it.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = json.loads((HERE / "results" / "results.json").read_text())
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

INK = "#1b3a5c"
MUTED = "#b8b8b8"

tf = RESULTS["teacher_forced"]
positions = tf["positions"]
agree = tf["top1_agree"]
floor = tf["harness_floor"]

text = RESULTS["text"]
identical = text["n_identical"]
prompts = text["n_prompts"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})

fig, ax = plt.subplots(figsize=(6.9, 2.9))

# Disagreements, not agreements: the interesting quantity is the gap, and at this
# scale three bars near 100% would look identical.
rows = [
    ("merged vs host", positions - agree, INK),
    ("host vs itself\n(harness floor)", positions - floor, MUTED),
]

ys = range(len(rows))
for y, (label, value, colour) in zip(ys, rows):
    ax.barh(y, value, height=0.45, color=colour, zorder=3)
    ax.text(value + 0.06, y, f"{value} of {positions}", va="center", ha="left",
            fontsize=10, color="#202020")

ax.set_yticks(list(ys))
ax.set_yticklabels([label for label, _, _ in rows], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("teacher-forced top-1 disagreements (lower is closer)")
ax.set_xlim(0, max(positions - floor, positions - agree) * 1.45)
ax.grid(axis="x", alpha=0.25, zorder=0)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)

ax.set_title(
    f"The merged model differs from its host by less than the harness differs from itself\n"
    f"Strict greedy text identity still holds on only {identical} of {prompts} prompts",
    fontsize=10.5, pad=11, loc="left",
)

fig.tight_layout()
path = OUT / "noise-floor-2x.png"
fig.savefig(path, bbox_inches="tight", facecolor="white")
plt.close(fig)

from PIL import Image
with Image.open(path) as im:
    print(f"wrote {path.relative_to(HERE)}  {im.width}x{im.height}")
