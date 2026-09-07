"""Re-render the 7-video x 5-sample contact sheet with METRIC (mm) tip
disagreement labeled alongside the pixel figure, using
outputs/arch_deciles_per_video_metric.json (apply_metric_depth_to_samples.py),
and plot px vs mm across the 35 samples to show where perspective/depth
changes the picture pixel distance alone would give.
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


def main():
    metric = json.load(open(ROOT / "outputs" / "arch_deciles_per_video_metric.json"))
    by_key = {(r["video"], r["frame"]): r for r in metric}
    samples = json.load(open(ROOT / "outputs" / "arch_deciles_per_video.json"))
    videos = list(samples)
    names = list(ANNOTATORS)

    all_frames, offsets = {}, {}
    for video in videos:
        for name in names:
            path = find_arches(ANNOTATORS[name], video)
            frames, _ = load_frames(path)
            all_frames[(name, video)] = {int(k): v for k, v in frames.items()}
        for name in names:
            if name != REFERENCE:
                offsets[(name, video)] = offset_for(video, name, all_frames)

    fig, axes = plt.subplots(len(videos), 5, figsize=(5 * 3.4, len(videos) * 3.6))
    for r, video in enumerate(videos):
        crop = json.load(open(DATA / "JSONfileNick" / video / "source_crop.json"))["crop"]
        for c, p in enumerate(samples[video]["picks"]):
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
            m = by_key.get((video, frame_idx))
            title = f"{p['mean_pairwise_px']:.0f}px" + (f"  |  {m['mean_pairwise_mm']:.2f}mm" if m else "")
            ax.set_title(title, fontsize=9)
            if c == 0:
                ax.text(-0.08, 0.5, video[:24], transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)
    fig.suptitle("Per-video samples: pixel gap vs METRIC gap (mono depth model, GUI masked)\n"
                 "cyan=Nick  magenta=Veerle  green=Aron", fontsize=13)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    out = ROOT / "outputs" / "arch_deciles_per_video_metric_contact_sheet.png"
    fig.savefig(out, dpi=100, facecolor="white")
    print(f"saved {out}")

    # px vs mm scatter, colored by video
    fig2, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    for i, video in enumerate(videos):
        rows = [r for r in metric if r["video"] == video]
        px = [r["mean_pairwise_px"] for r in rows]
        mm = [r["mean_pairwise_mm"] for r in rows]
        ax.scatter(px, mm, color=cmap(i % 10), label=video[:24], s=60, edgecolor="black")
    ax.set_xlabel("pixel-space mean pairwise tip distance (px)")
    ax.set_ylabel("metric (depth-backprojected) mean pairwise tip distance (mm)")
    px_all = np.array([r["mean_pairwise_px"] for r in metric])
    mm_all = np.array([r["mean_pairwise_mm"] for r in metric])
    corr = np.corrcoef(px_all, mm_all)[0, 1]
    ax.set_title(f"Pixel vs metric tip disagreement, 35 samples (corr={corr:.2f})\n"
                 "same px gap can mean a different mm gap depending on depth/scale", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")
    fig2.tight_layout()
    out2 = ROOT / "outputs" / "arch_px_vs_mm.png"
    fig2.savefig(out2, dpi=130, facecolor="white")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
