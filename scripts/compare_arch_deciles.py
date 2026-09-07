"""Sample one frame from each tenth of the 3-way disagreement RANGE (not
percentile -- equal-width bins from min to max mean-pairwise-tip-distance),
draw all three annotators' arches on each, and plot the full distribution
with the decile edges and picked samples marked.

Reads outputs/arch_tip_comparison_multi_triple.json (compare_arch_multi.py).
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_arch_multi import ANNOTATORS, REFERENCE, find_arches, load_frames
from visualize_multi_annotators import COLORS, arch_points, get_frame, offset_for

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data" / "InterAnnotator_Arch_Comparison"
OUT_DIR = ROOT / "outputs" / "arch_deciles"
N_BINS = 10


def main():
    triple = json.load(open(ROOT / "outputs" / "arch_tip_comparison_multi_triple.json"))
    vals = np.array([r["mean_pairwise_px"] for r in triple])
    edges = np.linspace(vals.min(), vals.max(), N_BINS + 1)

    picks = []
    used = set()
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [r for r in triple if lo <= r["mean_pairwise_px"] <= hi
                  and (r["video"], r["frame"]) not in used]
        center = (lo + hi) / 2
        if in_bin:
            pick = min(in_bin, key=lambda r: abs(r["mean_pairwise_px"] - center))
            fallback = False
        else:
            candidates = [r for r in triple if (r["video"], r["frame"]) not in used]
            pick = min(candidates, key=lambda r: abs(r["mean_pairwise_px"] - center))
            fallback = True
        used.add((pick["video"], pick["frame"]))
        picks.append(dict(bin=i, lo=lo, hi=hi, fallback=fallback, **pick))

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for p in picks:
        video, frame_idx = p["video"], p["frame"]
        crop = json.load(open(DATA / "JSONfileNick" / video / "source_crop.json"))["crop"]
        img = get_frame(DATA / video, frame_idx, crop)
        if img is None:
            print(f"[warn] could not read {video} frame {frame_idx}")
            continue

        fig, ax = plt.subplots(figsize=(8, 7))
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
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.5, label=f"{name}'s arch")
            ax.scatter(*apex, color=color, s=160, edgecolor="black", zorder=5, marker="*")
            apex_by_name[name] = apex
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ax.plot([apex_by_name[a][0], apex_by_name[b][0]],
                        [apex_by_name[a][1], apex_by_name[b][1]], "w--", lw=1, alpha=0.7)
        ax.legend(loc="upper right", fontsize=8)
        fallback_note = "  (no sample fell in this bin -- nearest shown)" if p["fallback"] else ""
        ax.set_title(f"decile {p['bin']+1}/10  [{p['lo']:.0f}-{p['hi']:.0f}px]{fallback_note}\n"
                     f"{video}  frame {frame_idx}  |  {p['mean_pairwise_px']:.0f}px "
                     f"({p['mean_pairwise_pct_chord']:.1f}% chord)", fontsize=8)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_facecolor("black")
        fig.tight_layout()
        out_path = OUT_DIR / f"decile{p['bin']+1:02d}__{video}__f{frame_idx}.png"
        fig.savefig(out_path, dpi=105, facecolor="white")
        plt.close(fig)
        saved.append(dict(p, image=str(out_path)))
        print(f"decile {p['bin']+1}/10 [{p['lo']:.0f}-{p['hi']:.0f}px]: {video} frame {frame_idx} "
              f"= {p['mean_pairwise_px']:.1f}px{' (fallback)' if p['fallback'] else ''}")

    # contact sheet, low -> high
    cols, rows = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.3))
    for i, s in enumerate(saved):
        ax = axes[i // cols, i % cols]
        ax.imshow(plt.imread(s["image"]))
        ax.axis("off")
        ax.set_title(f"#{i+1}  {s['mean_pairwise_px']:.0f}px", fontsize=10)
    fig.suptitle("One sample per tenth of the disagreement range (low -> high)\n"
                 "cyan=Nick  magenta=Veerle  green=Aron", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    sheet_path = ROOT / "outputs" / "arch_deciles_contact_sheet.png"
    fig.savefig(sheet_path, dpi=105, facecolor="white")
    print(f"\nsaved {sheet_path}")

    # distribution plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(vals, bins=40, color="#4c72b0", edgecolor="white", alpha=0.85)
    for e in edges:
        ax.axvline(e, color="gray", lw=0.8, ls="--", alpha=0.6)
    ymax = ax.get_ylim()[1]
    for i, s in enumerate(saved):
        ax.axvline(s["mean_pairwise_px"], color="crimson", lw=1.5)
        ax.annotate(f"#{i+1}", (s["mean_pairwise_px"], ymax * (0.95 - 0.05 * (i % 3))),
                    color="crimson", fontsize=9, ha="center", fontweight="bold")
    ax.set_xlabel("3-way mean pairwise tip distance (px)")
    ax.set_ylabel("frame count")
    ax.set_title(f"Distribution of 3-way arch-tip disagreement across all {len(vals)} shared frames\n"
                 f"dashed lines = decile edges (equal-width bins of the range) | "
                 f"red lines = the 10 sampled frames above", fontsize=11)
    fig.tight_layout()
    dist_path = ROOT / "outputs" / "arch_deciles_distribution.png"
    fig.savefig(dist_path, dpi=130, facecolor="white")
    print(f"saved {dist_path}")

    (ROOT / "outputs" / "arch_deciles_ranking.json").write_text(json.dumps(saved, indent=2))


if __name__ == "__main__":
    main()
