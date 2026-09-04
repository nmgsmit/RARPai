"""
2D (arch-scaled) vs metric-depth SUL: agreement plot + the frames that agree least/most.

Reads the csv sul_measure_depth.py writes, ranks the videos by |depth - 2D|, and puts
out two things:

  sul_2d_vs_depth.png       scatter (2D vs depth, y=x line) + signed-difference bar chart,
                            with the 4 best / 4 worst videos labelled.
  sul_2d_vs_depth_grid.png  those 8 urethra frames, mask outline + mirror axis drawn,
                            captioned with both lengths and the difference.

python scripts/visualize_sul_2d_vs_depth.py \
    --root ../transfer_atlas_mod/workspace/SUL_img3x --out outputs/sul_2d_vs_depth
"""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sul_measure as S

N_SHOW = 4


def load_rows(path):
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sul_2d_mm"] and r["sul_depth_mm"]]
    for r in rows:
        r["d2"] = float(r["sul_2d_mm"])
        r["d3"] = float(r["sul_depth_mm"])
        r["diff"] = r["d3"] - r["d2"]
        r["adiff"] = abs(r["diff"])
    return sorted(rows, key=lambda r: r["adiff"])


def overlay(root, stem, caption):
    """Urethra frame with the mask outline and the mirror axis that was measured."""
    img = cv2.cvtColor(np.array(Image.open(root / "images" / f"{stem}.png").convert("RGB")),
                       cv2.COLOR_RGB2BGR)
    mask = np.array(Image.open(root / "masks" / f"{stem}.png")) == S.URETHRA
    ax = S.mirror_axis(mask)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, S.URETHRA_BGR, 2)
    q0, q1 = (np.round(p).astype(int) for p in (ax["p0"], ax["p1"]))
    unit = (q1 - q0) / max(np.linalg.norm(q1 - q0), 1e-6)
    perp = np.array([-unit[1], unit[0]]) * 25
    cv2.line(img, tuple(q0), tuple(q1), S.AXIS_BGR, 3)
    for q in (q0, q1):
        cv2.line(img, tuple(np.round(q - perp).astype(int)),
                 tuple(np.round(q + perp).astype(int)), S.AXIS_BGR, 3)
    for text, y, colour in caption:
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def stem_for(root, video):
    """The urethra frame of that video (groups are keyed by clip, not video id)."""
    for clip, frames in S.load_groups(root).items():
        if S.video_id(clip) == video and S.URETHRA in frames:
            return frames[S.URETHRA]
    return None


def agreement_figure(rows, best, worst, out):
    d2 = np.array([r["d2"] for r in rows])
    d3 = np.array([r["d3"] for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    lim = [0, max(d2.max(), d3.max()) * 1.08]
    ax1.plot(lim, lim, "k--", lw=1, label="depth = 2D")
    ax1.scatter(d2, d3, s=45, c="#7f8c9a", zorder=3, label="video")
    ax1.scatter([r["d2"] for r in best], [r["d3"] for r in best], s=90, c="#2a9d5c",
                zorder=4, label=f"{N_SHOW} closest")
    ax1.scatter([r["d2"] for r in worst], [r["d3"] for r in worst], s=90, c="#c0392b",
                zorder=4, label=f"{N_SHOW} furthest")
    ax1.set(xlim=lim, ylim=lim, xlabel="2D SUL (mm, arch scale)",
            ylabel="depth SUL (mm, metric model)",
            title=f"n={len(rows)}   r={np.corrcoef(d2, d3)[0, 1]:.2f}   "
                  f"median ratio {np.median(d3 / d2):.2f}")
    ax1.legend(frameon=False)
    ax1.grid(alpha=.25)

    order = sorted(rows, key=lambda r: r["diff"])
    colours = ["#c0392b" if r in worst else "#2a9d5c" if r in best else "#b8c0c8"
               for r in order]
    ax2.bar(range(len(order)), [r["diff"] for r in order], color=colours)
    ax2.axhline(0, color="k", lw=1)
    ax2.set(xlabel="video (sorted by signed difference)", ylabel="depth - 2D (mm)",
            title=f"mean |difference| {np.mean([r['adiff'] for r in rows]):.1f} mm")
    ax2.grid(alpha=.25, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def grid_figure(root, best, worst, out):
    fig, axes = plt.subplots(2, N_SHOW, figsize=(4.6 * N_SHOW, 8.4))
    for r_i, (label, group, colour) in enumerate(
            [(f"closest agreement", best, "#2a9d5c"),
             (f"largest disagreement", worst, "#c0392b")]):
        for c_i, row in enumerate(group):
            ax = axes[r_i, c_i]
            ax.axis("off")
            stem = stem_for(root, row["video"])
            if stem is None:
                ax.set_title("frame not found")
                continue
            cap = [(f"2D    {row['d2']:.1f} mm", 42, S.AXIS_BGR),
                   (f"depth {row['d3']:.1f} mm", 84, (255, 200, 60)),
                   (f"diff  {row['diff']:+.1f} mm", 126, (60, 60, 255))]
            ax.imshow(overlay(root, stem, cap))
            ax.set_title(f"{label} #{c_i + 1}   |diff| {row['adiff']:.1f} mm\n"
                         f"{row['video'][:34]}", fontsize=9, color=colour)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../transfer_atlas_mod/workspace/SUL_img3x")
    ap.add_argument("--csv", default=None, help="default: <root>/sul_depth_compare.csv")
    ap.add_argument("--out", default="outputs/sul_2d_vs_depth",
                    help="prefix: writes <out>.png and <out>_grid.png")
    args = ap.parse_args()

    root = Path(args.root)
    rows = load_rows(Path(args.csv) if args.csv else root / "sul_depth_compare.csv")
    if not rows:
        raise SystemExit("no rows with both a 2D and a depth length")
    best, worst = rows[:N_SHOW], rows[-N_SHOW:][::-1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    agreement_figure(rows, best, worst, out.with_suffix(".png"))
    grid_figure(root, best, worst, out.parent / f"{out.name}_grid.png")

    print(f"[rank] {len(rows)} videos by |depth - 2D|")
    for tag, group in (("closest", best), ("furthest", worst)):
        for r in group:
            print(f"  {tag:8s} {r['adiff']:6.1f} mm  "
                  f"2D {r['d2']:6.1f}  depth {r['d3']:6.1f}  {r['video']}")
    print(f"[done] {out.with_suffix('.png')} and {out.parent / (out.name + '_grid.png')}")


if __name__ == "__main__":
    main()
