"""Plot the metric (mm) per-video disagreement distributions computed by
compute_metric_distribution_per_video.py -- same shared-x-axis layout as the
pixel version (compare_arch_deciles_per_video.py's distribution figure), but
in millimeters, and using each video's mm-based 5 sample bins (independent of
the earlier pixel-based bins, since the ranking can shift -- see arch_px_vs_mm.png).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_BINS = 5


def pick_bins(values, n_bins):
    """values: list of {frame, mean_pairwise_mm}. Returns n_bins picks (bin center-nearest)."""
    arr = np.array([v["mean_pairwise_mm"] for v in values])
    edges = np.linspace(arr.min(), arr.max(), n_bins + 1)
    picks, used = [], set()
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [v for v in values if lo <= v["mean_pairwise_mm"] <= hi and v["frame"] not in used]
        center = (lo + hi) / 2
        pool = in_bin or [v for v in values if v["frame"] not in used]
        pick = min(pool, key=lambda v: abs(v["mean_pairwise_mm"] - center))
        used.add(pick["frame"])
        picks.append(pick)
    return picks, edges


def main():
    data = json.load(open(ROOT / "outputs" / "arch_metric_distribution_per_video.json"))
    videos = sorted(data)

    per_video = {}
    for video in videos:
        values = data[video]["values"]
        picks, edges = pick_bins(values, N_BINS)
        per_video[video] = dict(vals=np.array([v["mean_pairwise_mm"] for v in values]),
                                 edges=edges, picks=picks)

    global_min = min(d["vals"].min() for d in per_video.values())
    global_max = max(d["vals"].max() for d in per_video.values())
    shared_bins = np.linspace(global_min, global_max, 25)

    cols = 3
    rows = -(-len(videos) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.6))
    axes = np.atleast_2d(axes)
    for i, video in enumerate(videos):
        ax = axes[i // cols, i % cols]
        d = per_video[video]
        ax.hist(d["vals"], bins=shared_bins, color="#4c72b0", edgecolor="white", alpha=0.85)
        for e in d["edges"]:
            ax.axvline(e, color="gray", lw=0.7, ls="--", alpha=0.6)
        ax.set_xlim(global_min, global_max)
        ymax = ax.get_ylim()[1]
        for j, p in enumerate(d["picks"]):
            ax.axvline(p["mean_pairwise_mm"], color="crimson", lw=1.2)
            ax.annotate(f"{j+1}", (p["mean_pairwise_mm"], ymax * 0.92), color="crimson",
                        fontsize=8, ha="center", fontweight="bold")
        n_full = data[video]["n_full"]
        n_samp = data[video]["n_sampled"]
        ax.set_title(f"{video[:40]}\n(n={n_samp} of {n_full} frames, subsampled)", fontsize=8)
        ax.set_xlabel("mean pairwise tip dist (mm)", fontsize=7)
        ax.tick_params(labelsize=7)
    for i in range(len(videos), rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle(f"Per-video distribution of 3-way arch-tip disagreement, METRIC (mm) "
                 f"(shared x-axis: {global_min:.1f}-{global_max:.1f}mm)\n"
                 "gray dashed = this video's own 5 mm sample-bin edges | red = the 5 sampled frames\n"
                 "each video subsampled to <=60 evenly-spaced frames (full depth inference "
                 "on all frames would take hours on CPU)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = ROOT / "outputs" / "arch_deciles_per_video_distributions_mm.png"
    fig.savefig(out, dpi=110, facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
