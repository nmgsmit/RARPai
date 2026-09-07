"""Per-video version of visualize_metric_ranked_5.py: for EACH of the 7
videos, rank that video's own mm-subsample (outputs/arch_metric_distribution_per_video.json,
the 60-frame-per-video depth subsample) and pick the 0th/25th/50th/75th/100th
percentile frame (i.e. min, Q1, median, Q3, max of that video's own
disagreement) -- 5 images x 7 videos = 35 panels total.

Also redraws the per-video mm distribution histograms (shared x-axis across
videos, like arch_deciles_per_video_distributions_mm.png) with these 5
percentile picks marked instead of the earlier equal-width-bin picks, so the
image grid and the histogram show the exact same frames.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_arch_multi import ANNOTATORS, REFERENCE, find_arches, load_frames
from visualize_multi_annotators import COLORS, arch_points, get_frame, offset_for

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data" / "InterAnnotator_Arch_Comparison"
PCT_LABELS = ["0%\n(best)", "25%", "50%\n(median)", "75%", "100%\n(worst)"]


def percentile_picks(values):
    """values: list of {frame, mean_pairwise_mm}, ascending-sorted. 5 picks at
    evenly spaced RANK positions (0/25/50/75/100th percentile of this video's
    own subsample), each tagged with its percentile label."""
    values = sorted(values, key=lambda v: v["mean_pairwise_mm"])
    idx = np.round(np.linspace(0, len(values) - 1, 5)).astype(int)
    return [dict(pct=p, **values[i]) for p, i in zip([0, 25, 50, 75, 100], idx)], values


def main():
    dist = json.load(open(ROOT / "outputs" / "arch_metric_distribution_per_video.json"))
    videos = sorted(dist)
    names = list(ANNOTATORS)

    per_video_picks, per_video_vals = {}, {}
    for video in videos:
        picks, sorted_vals = percentile_picks(dist[video]["values"])
        per_video_picks[video] = picks
        per_video_vals[video] = np.array([v["mean_pairwise_mm"] for v in sorted_vals])
        print(f"{video[:50]:<52}", "  ".join(f"{p}%={v['mean_pairwise_mm']:.2f}mm"
              for p, v in zip([0, 25, 50, 75, 100], picks)))

    all_frames, offsets = {}, {}
    for video in videos:
        for name in names:
            path = find_arches(ANNOTATORS[name], video)
            frames, _ = load_frames(path)
            all_frames[(name, video)] = {int(k): v for k, v in frames.items()}
        for name in names:
            if name != REFERENCE:
                offsets[(name, video)] = offset_for(video, name, all_frames)

    # --- 7x5 image grid -----------------------------------------------------
    fig, axes = plt.subplots(len(videos), 5, figsize=(5 * 3.4, len(videos) * 3.6))
    for r, video in enumerate(videos):
        crop = json.load(open(DATA / "JSONfileNick" / video / "source_crop.json"))["crop"]
        for c, p in enumerate(per_video_picks[video]):
            ax = axes[r, c]
            frame_idx = p["frame"]
            img = get_frame(DATA / video, frame_idx, crop)
            if img is None:
                ax.axis("off")
                continue
            h, w = img.shape[:2]
            ax.imshow(img, extent=(0, w, h, 0))
            apex_by_name = {}
            for name in names:
                entry = all_frames[(name, video)][frame_idx]
                pts, apex = arch_points(entry)
                if name != REFERENCE:
                    pts = pts - offsets[(name, video)]
                    apex = apex - offsets[(name, video)]
                color = np.array(COLORS[name]) / 255.0
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2)
                ax.scatter(*apex, color=color, s=90, edgecolor="black", zorder=5, marker="*")
                apex_by_name[name] = apex
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    ax.plot([apex_by_name[a][0], apex_by_name[b][0]],
                            [apex_by_name[a][1], apex_by_name[b][1]], "w--", lw=0.8, alpha=0.7)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.axis("off")
            ax.set_title(f"{PCT_LABELS[c]}  {p['mean_pairwise_mm']:.2f}mm", fontsize=8)
            if c == 0:
                ax.text(-0.08, 0.5, video[:24], transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)
    fig.suptitle("Per-video: 0/25/50/75/100th percentile of that video's own METRIC (mm) "
                 "disagreement\ncyan=Nick  magenta=Veerle  green=Aron", fontsize=13)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    grid_path = ROOT / "outputs" / "arch_mm_percentiles_per_video_grid.png"
    fig.savefig(grid_path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"\nsaved {grid_path}")

    # --- distribution histograms, shared x-axis, percentile picks marked ---
    global_min = min(v.min() for v in per_video_vals.values())
    global_max = max(v.max() for v in per_video_vals.values())
    shared_bins = np.linspace(global_min, global_max, 25)

    cols = 3
    rows = -(-len(videos) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.6))
    axes = np.atleast_2d(axes)
    for i, video in enumerate(videos):
        ax = axes[i // cols, i % cols]
        vals = per_video_vals[video]
        ax.hist(vals, bins=shared_bins, color="#4c72b0", edgecolor="white", alpha=0.85)
        ax.set_xlim(global_min, global_max)
        ymax = ax.get_ylim()[1]
        for j, p in enumerate(per_video_picks[video]):
            ax.axvline(p["mean_pairwise_mm"], color="crimson", lw=1.2)
            ax.annotate(f"{p['pct']}%", (p["mean_pairwise_mm"], ymax * 0.92), color="crimson",
                        fontsize=8, ha="center", fontweight="bold")
        n_full = dist[video]["n_full"]
        n_samp = dist[video]["n_sampled"]
        ax.set_title(f"{video[:40]}\n(n={n_samp} of {n_full} frames, subsampled)", fontsize=8)
        ax.set_xlabel("mean pairwise tip dist (mm)", fontsize=7)
        ax.tick_params(labelsize=7)
    for i in range(len(videos), rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle(f"Per-video METRIC (mm) distribution (shared x-axis: {global_min:.1f}-{global_max:.1f}mm)\n"
                 "red = the 0/25/50/75/100th percentile frames shown in the grid above", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    hist_path = ROOT / "outputs" / "arch_mm_percentiles_per_video_distributions.png"
    fig.savefig(hist_path, dpi=110, facecolor="white")
    print(f"saved {hist_path}")

    out = {v: [dict(p) for p in per_video_picks[v]] for v in videos}
    (ROOT / "outputs" / "arch_mm_percentiles_per_video.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
