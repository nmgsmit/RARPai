"""Metric (mm) version of the per-video disagreement distribution.

The pixel-space histograms (compare_arch_deciles_per_video.py) used every
shared frame per video (183-2027 frames/video). The depth model is ~2.9s/frame
on this CPU, so running it on all 5914 frames would take ~5 hours. Instead,
each video is subsampled to at most CAP_PER_VIDEO evenly-spaced frames (from
its full shared-frame set) -- enough to estimate the shape of the
distribution without an all-night run.

For each sampled frame: read it (GUI-masked), run the depth model, sample
depth at each annotator's already offset-corrected tip (from
outputs/arch_tip_comparison_multi_triple.json), back-project to 3D mm, take
the mean pairwise 3D distance -- the metric analogue of mean_pairwise_px.

Every depth map computed is saved, not thrown away, in TWO forms under
outputs/depth_maps_arch_samples/<video>/frame_<idx>:
  - <...>.jpg   colorized (magma) visualization, for looking at
  - <...>.npz   raw float32 depth in mm under key 'depth' (+ json 'meta'),
                for anything downstream that needs the actual numbers --
                a jpg alone would quantize mm depth to 8 bits and lose it

Saves outputs/arch_metric_distribution_per_video.json; the plotting itself is
in visualize_metric_distribution_per_video.py (kept separate so replotting
doesn't require re-running the model).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gui_mask import gui_mask
from gui_depth_measure import (DepthBackend, DEFAULT_CKPT, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH,
                               DEFAULT_MAX_DEPTH, DEFAULT_K_NORM, backproject, colorize, sample_depth)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data" / "InterAnnotator_Arch_Comparison"
DEPTH_OUT = ROOT / "outputs" / "depth_maps_arch_samples"
CAP_PER_VIDEO = 60


def get_frame_masked(video_path, frame_idx, crop):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
    img[gui_mask(img, scale=False)] = 0
    return cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)


def main():
    triple = json.load(open(ROOT / "outputs" / "arch_tip_comparison_multi_triple.json"))
    by_video = {}
    for r in triple:
        by_video.setdefault(r["video"], []).append(r)

    print("loading depth model...")
    backend = DepthBackend(DEFAULT_CKPT, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH, DEFAULT_MAX_DEPTH).load()
    print("loaded.\n")

    out = {}
    for video, rows in sorted(by_video.items()):
        rows = sorted(rows, key=lambda r: r["frame"])
        idx = np.unique(np.round(np.linspace(0, len(rows) - 1, min(CAP_PER_VIDEO, len(rows)))).astype(int))
        picked = [rows[i] for i in idx]
        crop = json.load(open(DATA / "JSONfileNick" / video / "source_crop.json"))["crop"]

        video_out_dir = DEPTH_OUT / video
        video_out_dir.mkdir(parents=True, exist_ok=True)

        mm_vals = []
        for r in picked:
            img_rgb = get_frame_masked(DATA / video, r["frame"], crop)
            if img_rgb is None:
                continue
            h, w = img_rgb.shape[:2]
            depth = backend.predict(Image.fromarray(img_rgb))

            stem = video_out_dir / f"frame_{r['frame']:06d}"
            Image.fromarray(colorize(depth)).save(stem.with_suffix(".jpg"), quality=92)
            np.savez(stem.with_suffix(".npz"), depth=depth.astype(np.float32),
                     meta=json.dumps({"video": video, "frame": r["frame"],
                                      "crop": crop, "checkpoint": str(DEFAULT_CKPT),
                                      "min_depth": DEFAULT_MIN_DEPTH, "max_depth": DEFAULT_MAX_DEPTH,
                                      "note": "depth in mm, GUI-masked+cropped frame, "
                                              "gui-masked BEFORE depth prediction"}))

            tips_mm = {}
            for name, (x, y) in r["tips"].items():
                u_n, v_n = x / w, y / h
                z = sample_depth(depth, u_n, v_n, radius=3)
                tips_mm[name] = backproject(u_n, v_n, z, DEFAULT_K_NORM)
            names = list(r["tips"])
            dists = [float(np.linalg.norm(tips_mm[a] - tips_mm[b]))
                     for i, a in enumerate(names) for b in names[i + 1:]]
            mm_vals.append(dict(frame=r["frame"], mean_pairwise_mm=float(np.mean(dists))))

        vals = np.array([v["mean_pairwise_mm"] for v in mm_vals])
        out[video] = dict(n_full=len(rows), n_sampled=len(mm_vals), values=mm_vals)
        print(f"{video[:50]:<52} sampled {len(mm_vals)}/{len(rows)}  "
              f"mm: mean={vals.mean():.2f} median={np.median(vals):.2f} "
              f"min={vals.min():.2f} max={vals.max():.2f}")

    out_path = ROOT / "outputs" / "arch_metric_distribution_per_video.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
