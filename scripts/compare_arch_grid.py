"""Superimpose Nick's + Veerle's arch on 3 sample frames per video (30 total),
using the actual videos in data/InterAnnotator_Arch_Comparison/, and rank all
30 by tip disagreement (worst to best).

Per video, 3 frames are picked evenly spaced across the range where BOTH
annotators have arch data (min/mid/max shared frame index) -- covering the
clip without depending on either annotator's manual keyframes, which almost
never coincide (see compare_arch_annotators.py).

Frames are read directly from the .mp4 with OpenCV. Arch coordinates are in
Nick's cropped-frame space (source_crop.json), so frames are cropped to that
box before drawing; Veerle's arch is shifted by that video's estimated
constant coordinate offset (see compare_arch_annotators.py) before drawing,
so what's shown is the real, offset-corrected disagreement.
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data" / "InterAnnotator_Arch_Comparison"
NICK_DIR = DATA / "JSONfileNick"
VEERLE_DIR = DATA / "JSONfilesVeerle"
VEERLE_PREFIX = "InterAnnotator_Arch_Comparison__"
VEERLE_SUFFIX = "_arches.json"
OUT_DIR = ROOT / "outputs" / "arch_grid"
N_FRAMES_PER_VIDEO = 3


def load_json(path):
    try:
        return json.loads(path.read_text()), None
    except FileNotFoundError:
        return None, "missing"
    except json.JSONDecodeError as e:
        return None, f"corrupt/truncated JSON ({e})"


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(p.name for p in NICK_DIR.iterdir() if p.is_dir())
    results = []

    for video in videos:
        nick_data, n_err = load_json(NICK_DIR / video / "arches.json")
        veerle_data, v_err = load_json(VEERLE_DIR / f"{VEERLE_PREFIX}{video}{VEERLE_SUFFIX}")
        crop_data, c_err = load_json(NICK_DIR / video / "source_crop.json")
        video_path = DATA / video
        if n_err or v_err or c_err or not video_path.exists():
            print(f"[skip] {video}: nick={n_err or 'ok'} veerle={v_err or 'ok'} "
                  f"crop={c_err or 'ok'} video={'ok' if video_path.exists() else 'MISSING'}")
            continue

        nick_frames = {int(k): v for k, v in nick_data["frames"].items()}
        veerle_frames = {int(k): v for k, v in veerle_data["frames"].items()}
        shared = sorted(set(nick_frames) & set(veerle_frames))
        if not shared:
            print(f"[skip] {video}: no overlapping frame indices")
            continue

        # per-video constant coordinate offset (see compare_arch_annotators.py)
        all_apex = [(arch_points(nick_frames[fi])[1], arch_points(veerle_frames[fi])[1])
                    for fi in shared]
        dx = np.mean([v_apex[0] - n_apex[0] for n_apex, v_apex in all_apex])
        dy = np.mean([v_apex[1] - n_apex[1] for n_apex, v_apex in all_apex])
        offset = np.array([dx, dy])

        # 3 frames spread across the shared range: min, mid, max
        picks = sorted({shared[0], shared[len(shared) // 2], shared[-1]})
        while len(picks) < N_FRAMES_PER_VIDEO and len(picks) < len(shared):
            picks.append(shared[len(picks)])
        picks = picks[:N_FRAMES_PER_VIDEO]

        for frame_idx in picks:
            nf, vf = nick_frames[frame_idx], veerle_frames[frame_idx]
            n_pts, n_apex = arch_points(nf)
            v_pts, v_apex = arch_points(vf)
            v_pts_corr, v_apex_corr = v_pts - offset, v_apex - offset
            dist = float(np.linalg.norm(np.array(n_apex) - v_apex_corr))
            avg_chord = (np.linalg.norm(np.subtract(nf["right"], nf["left"])) +
                         np.linalg.norm(np.subtract(vf["right"], vf["left"]))) / 2
            pct = 100.0 * dist / avg_chord if avg_chord > 1e-9 else float("nan")

            img = get_frame(video_path, frame_idx, crop_data["crop"])
            if img is None:
                print(f"[warn] {video} frame {frame_idx}: could not read video frame")
                continue

            out_path = OUT_DIR / f"{video}__f{frame_idx}.png"
            fig, ax = plt.subplots(figsize=(9, 8))
            h, w = img.shape[:2]
            ax.imshow(img, extent=(0, w, h, 0))
            ax.plot(n_pts[:, 0], n_pts[:, 1], color="cyan", lw=3, label="Nick's arch")
            ax.plot(v_pts_corr[:, 0], v_pts_corr[:, 1], color="magenta", lw=3, label="Veerle's arch")
            ax.scatter(*n_apex, color="cyan", s=140, edgecolor="black", zorder=5, marker="*")
            ax.scatter(*v_apex_corr, color="magenta", s=140, edgecolor="black", zorder=5, marker="*")
            ax.plot([n_apex[0], v_apex_corr[0]], [n_apex[1], v_apex_corr[1]], "w--", lw=1.5)
            mid = (np.array(n_apex) + v_apex_corr) / 2
            ax.annotate(f"{dist:.0f}px / {pct:.1f}%", mid, color="yellow", fontsize=12,
                        fontweight="bold", ha="center", va="bottom")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"{video}\nframe {frame_idx}  |  tip gap: {dist:.0f}px "
                         f"({pct:.1f}% of chord)", fontsize=9)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_facecolor("black")
            fig.tight_layout()
            fig.savefig(out_path, dpi=110, facecolor="white")
            plt.close(fig)

            results.append(dict(video=video, frame=frame_idx, dist_px=dist, pct_of_chord=pct,
                                 image=str(out_path)))
            print(f"[ok] {video} frame {frame_idx}: {dist:.1f}px ({pct:.1f}%) -> {out_path.name}")

    results.sort(key=lambda r: -r["dist_px"])
    rank_path = ROOT / "outputs" / "arch_grid_ranking.json"
    rank_path.write_text(json.dumps(results, indent=2))

    print(f"\n=== ranking: worst to best ({len(results)} images) ===")
    print(f"{'rank':>4} {'dist_px':>8} {'%chord':>7}  video  (frame)")
    for i, r in enumerate(results, 1):
        print(f"{i:>4} {r['dist_px']:>8.1f} {r['pct_of_chord']:>6.1f}%  {r['video']} (f{r['frame']})")
    print(f"\nranking written to {rank_path}")
    print(f"images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
