"""Same idea as compare_arch_deciles.py (one sample per Nth of the disagreement
RANGE + a distribution plot), but computed separately PER VIDEO instead of
pooling all videos together, with N_BINS=5 samples per video.

Each video gets its own min-max range for the 3-way mean-pairwise-tip-distance
metric, split into 5 equal-width bins; one representative frame is drawn from
each. Output: one combined contact sheet (rows=videos, cols=5 samples) and one
combined distribution figure (one histogram per video, decile edges + picks
marked), plus per-video JSON.

Videos without all 3 annotators (currently RARP_075: Veerle's export for it
is corrupt) are skipped -- noted in the printed summary.
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
OUT_DIR = ROOT / "outputs" / "arch_deciles_per_video"
N_BINS = 5


def pick_bins(rows, n_bins):
    vals = np.array([r["mean_pairwise_px"] for r in rows])
    edges = np.linspace(vals.min(), vals.max(), n_bins + 1)
    picks, used = [], set()
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [r for r in rows if lo <= r["mean_pairwise_px"] <= hi
                  and (r["video"], r["frame"]) not in used]
        center = (lo + hi) / 2
        pool = in_bin or [r for r in rows if (r["video"], r["frame"]) not in used]
        pick = min(pool, key=lambda r: abs(r["mean_pairwise_px"] - center))
        used.add((pick["video"], pick["frame"]))
        picks.append(dict(bin=i, lo=lo, hi=hi, fallback=not in_bin, **pick))
    return picks, edges, vals


def render_sample(ax, p, all_frames, offsets, names):
    video, frame_idx = p["video"], p["frame"]
    crop = json.load(open(DATA / "JSONfileNick" / video / "source_crop.json"))["crop"]
    img = get_frame(DATA / video, frame_idx, crop)
    if img is None:
        ax.axis("off")
        ax.set_title("frame read failed", fontsize=8)
        return
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
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, label=f"{name}")
        ax.scatter(*apex, color=color, s=90, edgecolor="black", zorder=5, marker="*")
        apex_by_name[name] = apex
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ax.plot([apex_by_name[a][0], apex_by_name[b][0]],
                    [apex_by_name[a][1], apex_by_name[b][1]], "w--", lw=0.8, alpha=0.7)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fb = " (fallback)" if p["fallback"] else ""
    ax.set_title(f"{p['mean_pairwise_px']:.0f}px{fb}", fontsize=9)


def main():
    triple = json.load(open(ROOT / "outputs" / "arch_tip_comparison_multi_triple.json"))
    names = list(ANNOTATORS)
    videos = sorted({r["video"] for r in triple})
    print(f"videos with all {len(names)} annotators: {len(videos)}")
    skipped = sorted(set(p.name for p in ANNOTATORS[REFERENCE].iterdir() if p.is_dir()) - set(videos))
    if skipped:
        print(f"skipped (missing an annotator's data): {skipped}")

    all_frames, offsets = {}, {}
    per_video_picks = {}
    for video in videos:
        rows = [r for r in triple if r["video"] == video]
        picks, edges, vals = pick_bins(rows, N_BINS)
        per_video_picks[video] = dict(picks=picks, edges=edges, vals=vals)

        for name in names:
            path = find_arches(ANNOTATORS[name], video)
            frames, _ = load_frames(path)
            all_frames[(name, video)] = {int(k): v for k, v in frames.items()}
        for name in names:
            if name != REFERENCE:
                offsets[(name, video)] = offset_for(video, name, all_frames)

        print(f"\n{video}  (n={len(rows)}, range {vals.min():.1f}-{vals.max():.1f}px)")
        for p in picks:
            print(f"  bin {p['bin']+1}/{N_BINS} [{p['lo']:.0f}-{p['hi']:.0f}px]: "
                  f"frame {p['frame']} = {p['mean_pairwise_px']:.1f}px{' (fallback)' if p['fallback'] else ''}")

    # --- combined contact sheet: rows=videos, cols=N_BINS -------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(videos), N_BINS, figsize=(N_BINS * 3.2, len(videos) * 3.4))
    for r, video in enumerate(videos):
        for c, p in enumerate(per_video_picks[video]["picks"]):
            ax = axes[r, c]
            render_sample(ax, p, all_frames, offsets, names)
            if c == 0:
                ax.text(-0.08, 0.5, video[:24], transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)
    fig.suptitle(f"Per-video: one sample per 1/{N_BINS} of that video's own disagreement range "
                 f"(low -> high, left to right)\ncyan=Nick  magenta=Veerle  green=Aron", fontsize=13)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    sheet_path = ROOT / "outputs" / "arch_deciles_per_video_contact_sheet.png"
    fig.savefig(sheet_path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"\nsaved {sheet_path}")

    # --- combined distribution figure: one histogram panel per video -------
    # shared x-axis (same min/max, same bin edges) across all panels so the
    # videos are visually comparable; the 5 per-video sample bin edges (gray
    # dashed, computed on that video's own range) still differ panel to panel.
    global_min = min(d["vals"].min() for d in per_video_picks.values())
    global_max = max(d["vals"].max() for d in per_video_picks.values())
    shared_bins = np.linspace(global_min, global_max, 31)

    cols = 3
    rows = -(-len(videos) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.6))
    axes = np.atleast_2d(axes)
    for i, video in enumerate(videos):
        ax = axes[i // cols, i % cols]
        d = per_video_picks[video]
        ax.hist(d["vals"], bins=shared_bins, color="#4c72b0", edgecolor="white", alpha=0.85)
        for e in d["edges"]:
            ax.axvline(e, color="gray", lw=0.7, ls="--", alpha=0.6)
        ax.set_xlim(global_min, global_max)
        ymax = ax.get_ylim()[1]
        for j, p in enumerate(d["picks"]):
            ax.axvline(p["mean_pairwise_px"], color="crimson", lw=1.2)
            ax.annotate(f"{j+1}", (p["mean_pairwise_px"], ymax * 0.92), color="crimson",
                        fontsize=8, ha="center", fontweight="bold")
        ax.set_title(video[:40], fontsize=9)
        ax.set_xlabel("mean pairwise tip dist (px)", fontsize=7)
        ax.tick_params(labelsize=7)
    for i in range(len(videos), rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle(f"Per-video distribution of 3-way arch-tip disagreement (shared x-axis: "
                 f"{global_min:.0f}-{global_max:.0f}px)\n"
                 "gray dashed = this video's own 5 sample-bin edges | red = the 5 sampled frames", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    dist_path = ROOT / "outputs" / "arch_deciles_per_video_distributions.png"
    fig.savefig(dist_path, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"saved {dist_path}")

    out_json = {v: dict(picks=d["picks"], range=[float(d["vals"].min()), float(d["vals"].max())])
                for v, d in per_video_picks.items()}
    (ROOT / "outputs" / "arch_deciles_per_video.json").write_text(json.dumps(out_json, indent=2))
    print(f"saved outputs/arch_deciles_per_video.json")


if __name__ == "__main__":
    main()
