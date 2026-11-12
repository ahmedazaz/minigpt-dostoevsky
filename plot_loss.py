"""Plot validation bits-per-character for each training run in logs/.

Usage:
    python3 plot_loss.py  ->  assets/loss_curve.png
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = [
    ("logs/chinchilla_971steps.txt", "Chinchilla budget (971 steps)", "#eb6834"),
    ("logs/steps_5000.txt", "5000 steps", "#2a78d6"),
]
LINE = re.compile(r"Step (\d+)/\d+ .*val bpc ([\d.]+)")

TEXT = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"


def read_run(path):
    steps, bpc = [], []
    for line in Path(path).read_text().splitlines():
        m = LINE.search(line)
        if m:
            steps.append(int(m.group(1)))
            bpc.append(float(m.group(2)))
    return steps, bpc


fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for path, label, color in RUNS:
    steps, bpc = read_run(path)
    ax.plot(steps, bpc, color=color, linewidth=2, label=label)
    ax.scatter(steps[-1], bpc[-1], color=color, s=40, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"{bpc[-1]:.3f}", (steps[-1], bpc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=TEXT, fontsize=10)

ax.axvline(971, color=MUTED, linewidth=1, linestyle=":")
ax.text(990, ax.get_ylim()[1] * 0.98, "Chinchilla budget\n(20 tokens / param)",
        color=MUTED, fontsize=9, va="top")

ax.set_title("Validation bits-per-character (lower is better)", color=TEXT, loc="left", fontsize=12)
ax.set_xlabel("Training step", color=MUTED)
ax.set_ylabel("Val bpc", color=MUTED)
ax.tick_params(colors=MUTED)
ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#c9c8c2")
ax.legend(frameon=False, labelcolor=TEXT)
ax.set_xlim(0, 5400)

out = Path("assets/loss_curve.png")
out.parent.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out, facecolor=SURFACE)
print(f"saved {out}")
