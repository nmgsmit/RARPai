"""Recreate the best/worst contact sheet, but ranked on METRIC (mm) tip
disagreement instead of pixels, with 5 panels: best, three evenly-spaced
in-betweens, and worst.

Pool is outputs/arch_metric_distribution_per_video.json (the 448-frame,
7-video subsample compute_metric_distribution_per_video.py already ran depth
on) -- the only frames with an actual mm number. Sorted ascending by
mean_pairwise_mm, 5 picks are taken at evenly spaced positions across that
ranked list (index 0 = best, last = worst, 3 more at ~25/50/75%).

Arch curves are redrawn from each annotator's raw arches.json (full
left/right/height/power), offset-corrected the same way as everywhere else,
on the actual video frame -- not just the tip point.
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
N_PICKS = 5
LABELS = ["BEST", "IN-BETWEEN 1", "IN-BETWEEN 2", "IN-BETWEEN 3", "WORST"]


def main():
    dist = json.load(open(ROOT / "outputs" / "arch_metric_distribution_per_video.json"))
    pool = [dict(video=video, frame=v["frame"], mean_pairwise_mm=v["mean_pairwise_mm"])
            for video, d in dist.items() for v in d["values"]]
    pool.sort(key=lambda r: r["mean_pairwise_mm"])

    idx = np.round(np.linspace(0, len(pool) - 1, N_PICKS)).astype(int)
    picks = [pool[i] for i in idx]
    for label, p in zip(LABELS, picks):
        print(f"{label:<14} {p['video']:<50} frame {p['frame']:<6} {p['mean_pairwise_mm']:.2f}mm")

    names = list(ANNOTATORS)
    videos_needed = sorted({p["video"] for p in picks})
    all_frames, offsets = {}, {}
    for video in videos_needed:
        for name in names:
            path = find_arches(ANNOTATORS[name], video)
            frames, _ = load_frames(path)
            all_frames[(name, video)] = {int(k): v for k, v in frames.items()}
        for name in names:
            if name != REFERENCE:
                offsets[(name, video)] = offset_for(video, name, all_frames)

    fig, axes = plt.subplots(1, N_PICKS, figsize=(N_PICKS * 4.2, 4.6))
    for ax, label, p in zip(axes, LABELS, picks):
        video, frame_idx = p["video"], p["frame"]
        crop = json.load(open(DATA / "JSONfileNick" / video / "source_crop.json"))["crop"]
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
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.2, label=f"{name}'s arch")
            ax.scatter(*apex, color=color, s=130, edgecolor="black", zorder=5, marker="*")
            apex_by_name[name] = apex
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ax.plot([apex_by_name[a][0], apex_by_name[b][0]],
                        [apex_by_name[a][1], apex_by_name[b][1]], "w--", lw=1, alpha=0.7)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"{label}\n{video[:28]}\nframe {frame_idx}  |  {p['mean_pairwise_mm']:.2f}mm",
                     fontsize=9)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

    fig.suptitle(f"3-way arch-tip disagreement ranked by METRIC (mm) distance, "
                 f"n={len(pool)} pooled frames\ncyan=Nick  magenta=Veerle  green=Aron", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = ROOT / "outputs" / "arch_mm_ranked_5.png"
    fig.savefig(out, dpi=120, facecolor="white")
    print(f"\nsaved {out}")

    (ROOT / "outputs" / "arch_mm_ranked_5.json").write_text(
        json.dumps([dict(label=l, **p) for l, p in zip(LABELS, picks)], indent=2))


if __name__ == "__main__":
    main()
