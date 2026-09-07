"""Visualize worst / average / best 3-way arch-tip disagreement samples
(Nick, Veerle, Aron), on the actual video frame, drawing each annotator's
full arch curve (not just the tip).

Reads outputs/arch_tip_comparison_multi_triple.json (from compare_arch_multi.py)
to pick N_EACH samples each from the worst, closest-to-median, and best end
of the mean-pairwise-distance ranking; re-reads each annotator's raw
arches.json for the full curve (left/right/height/power), applies the same
per-video offset correction as compare_arch_multi.py, draws all three curves
on the source frame, and stitches everything into one ranked contact sheet.
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_arch_multi import ANNOTATORS, REFERENCE, apex_of, find_arches, load_frames

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data" / "InterAnnotator_Arch_Comparison"
OUT_DIR = ROOT / "outputs" / "arch_multi_samples"
N_EACH = 2
COLORS = {"Nick": (0, 255, 255), "Veerle": (255, 0, 255), "Aron": (0, 200, 0)}  # RGB, 0-255


def arch_points(frame, n=200):
    left = np.array(frame["left"], dtype=np.float64)
    right = np.array(frame["right"], dtype=np.float64)
    mid = (left + right) / 2
    chord = right - left
    d = float(np.linalg.norm(chord)) / 2
    t = chord / (2 * d) if d > 1e-9 else np.array([1.0, 0.0])
    n_vec = np.array([t[1], -t[0]])
    power = frame.get("power", 2.0)
    us = np.linspace(-1.0, 1.0, n)
    shape = 1 - np.abs(us) ** power
    pts = mid[None, :] + (us[:, None] * d) * t[None, :] + (shape[:, None] * frame["height"]) * n_vec[None, :]
    apex = mid + frame["height"] * n_vec
    return pts, apex


def get_frame(video_path, frame_idx, crop):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
    return img[y:y + h, x:x + w]


def offset_for(video, other_name, all_frames):
    ref = all_frames[(REFERENCE, video)]
    oth = all_frames[(other_name, video)]
    shared = sorted(set(ref) & set(oth))
    ref_apex = np.array([apex_of(ref[fi])[0] for fi in shared])
    oth_apex = np.array([apex_of(oth[fi])[0] for fi in shared])
    return (oth_apex - ref_apex).mean(axis=0)


def main():
    triple = json.loads((ROOT / "outputs" / "arch_tip_comparison_multi_triple.json").read_text())
    med = np.median([r["mean_pairwise_px"] for r in triple])

    def top_n_distinct_videos(rows, key, n):
        """n picks by `key`, at most one per video, to avoid near-duplicate frames."""
        out, used = [], set()
        for r in sorted(rows, key=key):
            if r["video"] in used:
                continue
            out.append(r)
            used.add(r["video"])
            if len(out) == n:
                break
        return out

    worst = top_n_distinct_videos(triple, lambda r: -r["mean_pairwise_px"], N_EACH)
    best = top_n_distinct_videos(triple, lambda r: r["mean_pairwise_px"], N_EACH)
    average = top_n_distinct_videos(triple, lambda r: abs(r["mean_pairwise_px"] - med), N_EACH)
    picks = [("WORST", r) for r in worst] + [("AVERAGE", r) for r in average] + [("BEST", r) for r in best]

    names = list(ANNOTATORS)
    videos_needed = sorted({r["video"] for _, r in picks})

    # load raw frames + per-video offsets only for the videos we need
    all_frames = {}
    offsets = {}
    for video in videos_needed:
        for name in names:
            path = find_arches(ANNOTATORS[name], video)
            frames, err = load_frames(path)
            all_frames[(name, video)] = {int(k): v for k, v in frames.items()}
        for name in names:
            if name != REFERENCE:
                offsets[(name, video)] = offset_for(video, name, all_frames)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for tag, r in picks:
        video, frame_idx = r["video"], r["frame"]
        crop = json.loads((DATA / "JSONfileNick" / video / "source_crop.json").read_text())["crop"]
        img = get_frame(DATA / video, frame_idx, crop)
        if img is None:
            print(f"[warn] could not read {video} frame {frame_idx}")
            continue

        fig, ax = plt.subplots(figsize=(9, 8))
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
            ax.scatter(*apex, color=color, s=170, edgecolor="black", zorder=5, marker="*")
            apex_by_name[name] = apex
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ax.plot([apex_by_name[a][0], apex_by_name[b][0]],
                        [apex_by_name[a][1], apex_by_name[b][1]], "w--", lw=1, alpha=0.7)

        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"[{tag}] {video}\nframe {frame_idx}  |  mean pairwise gap: "
                     f"{r['mean_pairwise_px']:.0f}px ({r['mean_pairwise_pct_chord']:.1f}% chord)\n"
                     + "  ".join(f"{k}={v:.0f}px" for k, v in r["dists"].items()), fontsize=8)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_facecolor("black")
        fig.tight_layout()
        out_path = OUT_DIR / f"{tag}__{video}__f{frame_idx}.png"
        fig.savefig(out_path, dpi=110, facecolor="white")
        plt.close(fig)
        saved.append(dict(tag=tag, video=video, frame=frame_idx,
                           mean_pairwise_px=r["mean_pairwise_px"],
                           mean_pairwise_pct_chord=r["mean_pairwise_pct_chord"], image=str(out_path)))
        print(f"[{tag}] {video} frame {frame_idx}: {r['mean_pairwise_px']:.1f}px -> {out_path.name}")

    cols = N_EACH
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    for i, s in enumerate(saved):
        ax = axes[i // cols, i % cols]
        ax.imshow(plt.imread(s["image"]))
        ax.axis("off")
        ax.set_title(f"{s['tag']}: {s['mean_pairwise_px']:.0f}px ({s['mean_pairwise_pct_chord']:.1f}%)", fontsize=10)
    fig.suptitle("3-way arch disagreement: worst (top) / average (mid) / best (bottom)\n"
                 "cyan=Nick  magenta=Veerle  green=Aron", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    sheet_path = ROOT / "outputs" / "arch_multi_contact_sheet.png"
    fig.savefig(sheet_path, dpi=110, facecolor="white")
    print(f"\nsaved contact sheet: {sheet_path}")

    (ROOT / "outputs" / "arch_multi_samples_ranking.json").write_text(json.dumps(saved, indent=2))


if __name__ == "__main__":
    main()
