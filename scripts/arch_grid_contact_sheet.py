"""Stitch the per-frame overlay PNGs from compare_arch_grid.py into one ranked
contact sheet (worst -> best, per outputs/arch_grid_ranking.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent.parent
ranking = json.loads((ROOT / "outputs" / "arch_grid_ranking.json").read_text())

n = len(ranking)
cols = 5
rows = -(-n // cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = axes.flatten()
for i, (ax, r) in enumerate(zip(axes, ranking), 1):
    img = mpimg.imread(r["image"])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"#{i}  {r['dist_px']:.0f}px / {r['pct_of_chord']:.1f}%", fontsize=9)
for ax in axes[len(ranking):]:
    ax.axis("off")
fig.suptitle("Nick vs Veerle arch tips, ranked worst (top-left) to best (bottom-right)", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = ROOT / "outputs" / "arch_grid_contact_sheet.png"
fig.savefig(out, dpi=100, facecolor="white")
print(f"saved {out}")
