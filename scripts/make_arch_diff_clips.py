"""Make short comparison video clips (both arches overlaid) for the worst-N and
best-N samples from outputs/arch_grid_ranking.json (written by
compare_arch_grid.py), then concatenate them into one combined video.

For each picked (video, frame) sample, a window of frames around it (only
frames present in BOTH annotators' arches.json, i.e. actually comparable) is
read from the source .mp4, cropped to Nick's crop box, and drawn on with
Nick's arch (cyan) and Veerle's arch (magenta, shifted by that video's
estimated constant coordinate offset -- the real, offset-corrected position).
Playback is slowed to READOUT_FPS so the drift/agreement is easy to watch.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data" / "InterAnnotator_Arch_Comparison"
NICK_DIR = DATA / "JSONfileNick"
VEERLE_DIR = DATA / "JSONfilesVeerle"
VEERLE_PREFIX = "InterAnnotator_Arch_Comparison__"
VEERLE_SUFFIX = "_arches.json"
OUT_DIR = ROOT / "outputs" / "arch_diff_clips"

WINDOW_FRAMES = 120     # +/- frames (source fps) around the picked sample
READOUT_FPS = 15        # output playback speed (slower than source ~60fps)
LABEL_SECONDS = 1.2     # how long the rank/gap title card shows at each clip's start


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


def draw_arch(img, pts, color, thickness=3):
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i], False, color, thickness, cv2.LINE_AA)


def make_clip(video, center_frame, rank, gap_px, gap_pct, writer, crop, offset,
              nick_frames, veerle_frames, source_fps):
    cap = cv2.VideoCapture(str(DATA / video))
    x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]

    shared = sorted(set(nick_frames) & set(veerle_frames))
    lo, hi = center_frame - WINDOW_FRAMES, center_frame + WINDOW_FRAMES
    window = [fi for fi in shared if lo <= fi <= hi]
    if not window:
        window = [center_frame]
    step = max(1, round(source_fps / READOUT_FPS))
    window = window[::step]

    label_frames = int(LABEL_SECONDS * READOUT_FPS)
    for i, fi in enumerate(window):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        crop_img = frame[y:y + h, x:x + w].copy()  # BGR, matches cv2 drawing colors below

        nf, vf = nick_frames.get(fi), veerle_frames.get(fi)
        if nf is not None:
            n_pts, n_apex = arch_points(nf)
            draw_arch(crop_img, n_pts, (255, 255, 0))       # cyan in BGR
            cv2.drawMarker(crop_img, tuple(np.round(n_apex).astype(int)), (255, 255, 0),
                            cv2.MARKER_STAR, 26, 3)
        if vf is not None:
            v_pts, v_apex = arch_points(vf)
            v_pts, v_apex = v_pts - offset, v_apex - offset
            draw_arch(crop_img, v_pts, (255, 0, 255))       # magenta in BGR
            cv2.drawMarker(crop_img, tuple(np.round(v_apex).astype(int)), (255, 0, 255),
                            cv2.MARKER_STAR, 26, 3)

        cv2.putText(crop_img, f"#{rank}  {video}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(crop_img, f"frame {fi}  |  tip gap ~{gap_px:.0f}px ({gap_pct:.1f}% chord)",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(crop_img, "cyan = Nick   magenta = Veerle", (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if i < label_frames:
            cv2.putText(crop_img, "WORST" if rank <= 2 else "BEST", (w - 260, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255) if rank <= 2 else (0, 200, 0), 3, cv2.LINE_AA)

        writer.write(crop_img)
    cap.release()


def main():
    ranking = json.loads((ROOT / "outputs" / "arch_grid_ranking.json").read_text())
    picks = [dict(r, rank=i + 1) for i, r in enumerate(ranking)]
    selected = picks[:2] + picks[-2:]  # worst 2 + best 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "worst2_vs_best2.mp4"

    # figure output frame size from the first sample's crop box
    crops = {}
    offsets = {}
    frames_cache = {}
    for s in selected:
        v = s["video"]
        if v not in crops:
            crops[v] = json.loads((NICK_DIR / v / "source_crop.json").read_text())["crop"]
            nick_frames = {int(k): val for k, val in
                            json.loads((NICK_DIR / v / "arches.json").read_text())["frames"].items()}
            veerle_frames = {int(k): val for k, val in
                              json.loads((VEERLE_DIR / f"{VEERLE_PREFIX}{v}{VEERLE_SUFFIX}").read_text())["frames"].items()}
            frames_cache[v] = (nick_frames, veerle_frames)
            shared = sorted(set(nick_frames) & set(veerle_frames))
            n_apex_all = [arch_points(nick_frames[fi])[1] for fi in shared]
            v_apex_all = [arch_points(veerle_frames[fi])[1] for fi in shared]
            dx = np.mean([va[0] - na[0] for na, va in zip(n_apex_all, v_apex_all)])
            dy = np.mean([va[1] - na[1] for na, va in zip(n_apex_all, v_apex_all)])
            offsets[v] = np.array([dx, dy])

    w, h = crops[selected[0]["video"]]["w"], crops[selected[0]["video"]]["h"]
    # all crops share the same w/h in this dataset (source_crop.json identical across videos)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, READOUT_FPS, (w, h))

    for s in selected:
        v = s["video"]
        cap = cv2.VideoCapture(str(DATA / v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        nick_frames, veerle_frames = frames_cache[v]
        print(f"rendering #{s['rank']} {v} frame {s['frame']} "
              f"(gap {s['dist_px']:.0f}px / {s['pct_of_chord']:.1f}%)...")
        make_clip(v, s["frame"], s["rank"], s["dist_px"], s["pct_of_chord"], writer,
                  crops[v], offsets[v], nick_frames, veerle_frames, fps)

    writer.release()
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
