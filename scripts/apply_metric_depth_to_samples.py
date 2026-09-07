"""Convert the pixel-space tip disagreement (outputs/arch_deciles_per_video.json,
35 samples: 7 videos x 5 range-bins) into METRIC (mm) disagreement, using the
trained monocular depth model (outputs/depth_ruler_range_sw05/best.pth).

For each sample frame:
  1. read the frame, cropped to Nick's crop box (same crop the arch pixel
     coords are already in -- see source_crop.json)
  2. mask the da Vinci GUI overlay (bottom info bar + tab) with gui_mask's
     fixed geometry, so the depth model isn't fed console UI pixels
  3. run the depth model -> a per-pixel depth map in mm
  4. sample depth at each annotator's (already offset-corrected) tip pixel,
     back-project to 3D with the model's camera intrinsics, and take the
     pairwise 3D (mm) distance between annotators -- the metric analogue of
     the pixel-space mean_pairwise_px used earlier

This is a MONOCULAR estimate, not a calibrated caliper measurement -- the
depth model's own doc says holdout scale error is ~2.6% on average; treat the
mm numbers as "same order of magnitude, real units" rather than exact.

Every depth map computed is saved, not thrown away, under
outputs/depth_maps_arch_samples/<video>/frame_<idx>: a colorized .jpg (to
look at) and a raw float32-mm .npz under key 'depth' (a jpg alone would
quantize the mm values to 8 bits and lose them).
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


def get_frame_bgr(video_path, frame_idx, crop):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
    # GUI mask uses the ORIGINAL (uncropped) frame geometry (fixed_gui_mask assumes 1920x1080),
    # then gets cropped the same way as the image so it lines up with the arch pixel coords.
    mask_full = gui_mask(img, scale=False)
    img[mask_full] = 0
    crop_img = img[y:y + h, x:x + w]
    return cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)


def main():
    samples = json.load(open(ROOT / "outputs" / "arch_deciles_per_video.json"))
    print("loading depth model...")
    backend = DepthBackend(DEFAULT_CKPT, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH, DEFAULT_MAX_DEPTH).load()
    print("loaded.\n")

    results = []
    for video, d in samples.items():
        crop = json.load(open(DATA / "JSONfileNick" / video / "source_crop.json"))["crop"]
        for p in d["picks"]:
            frame_idx = p["frame"]
            img_rgb = get_frame_bgr(DATA / video, frame_idx, crop)
            if img_rgb is None:
                print(f"[warn] {video} frame {frame_idx}: could not read frame")
                continue
            h, w = img_rgb.shape[:2]
            depth = backend.predict(Image.fromarray(img_rgb))

            video_out_dir = DEPTH_OUT / video
            video_out_dir.mkdir(parents=True, exist_ok=True)
            stem = video_out_dir / f"frame_{frame_idx:06d}"
            Image.fromarray(colorize(depth)).save(stem.with_suffix(".jpg"), quality=92)
            np.savez(stem.with_suffix(".npz"), depth=depth.astype(np.float32),
                     meta=json.dumps({"video": video, "frame": frame_idx, "crop": crop,
                                      "checkpoint": str(DEFAULT_CKPT),
                                      "min_depth": DEFAULT_MIN_DEPTH, "max_depth": DEFAULT_MAX_DEPTH,
                                      "note": "depth in mm, GUI-masked+cropped frame, "
                                              "gui-masked BEFORE depth prediction"}))

            tips_mm = {}
            for name, (x, y) in p["tips"].items():
                u_n, v_n = x / w, y / h
                z = sample_depth(depth, u_n, v_n, radius=3)
                tips_mm[name] = backproject(u_n, v_n, z, DEFAULT_K_NORM)

            names = list(p["tips"])
            pair_mm = {f"{a}-{b}": float(np.linalg.norm(tips_mm[a] - tips_mm[b]))
                       for i, a in enumerate(names) for b in names[i + 1:]}
            mean_mm = float(np.mean(list(pair_mm.values())))

            results.append(dict(video=video, frame=frame_idx, bin=p["bin"],
                                 mean_pairwise_px=p["mean_pairwise_px"],
                                 mean_pairwise_pct_chord=p["mean_pairwise_pct_chord"],
                                 mean_pairwise_mm=mean_mm, pair_mm=pair_mm,
                                 depth_at_tips_mm={n: round(float(sample_depth(
                                     depth, p["tips"][n][0] / w, p["tips"][n][1] / h, 3)), 1)
                                     for n in names}))
            print(f"{video[:40]:<42} f{frame_idx:<6} px={p['mean_pairwise_px']:6.1f}  "
                  f"mm={mean_mm:6.2f}  depths={results[-1]['depth_at_tips_mm']}")

    out_path = ROOT / "outputs" / "arch_deciles_per_video_metric.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path} ({len(results)} samples)")

    px = np.array([r["mean_pairwise_px"] for r in results])
    mm = np.array([r["mean_pairwise_mm"] for r in results])
    print(f"\npx  : mean={px.mean():.1f} median={np.median(px):.1f} min={px.min():.1f} max={px.max():.1f}")
    print(f"mm  : mean={mm.mean():.2f} median={np.median(mm):.2f} min={mm.min():.2f} max={mm.max():.2f}")
    corr = np.corrcoef(px, mm)[0, 1]
    print(f"corr(px, mm) across the 35 samples: {corr:.3f}")


if __name__ == "__main__":
    main()
