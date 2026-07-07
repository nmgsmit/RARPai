"""
Build metric-depth ground truth for SCARED test keyframes from the STEREO pairs.

The SCARED challenge *test* release (dataset_8/9) ships only the keyframe stereo pair
(Left_Image.png / Right_Image.png), the sequence video, and endoscope_calibration.yaml
-- the structured-light GT was withheld. But the calibration gives the full stereo rig
(M1,M2 intrinsics + R,T with a ~4.35 mm baseline), so we recover METRIC depth (mm) from
the pair by rectify -> SGBM disparity -> reproject. That's a legitimate calibrated-stereo
reference (what Hamlyn-style benchmarks use); the model never sees the right image, so the
comparison is fair. Noisier than structured light and only the 10 keyframes, but real and
metric, needing no further download.

Output (consumed as-is by scripts/eval_scared.py):
  <out>/frames/<dataset>_<keyframe>.png   rectified LEFT image (what the model runs on)
  <out>/gt_depths.npz                     key "data", (N,H,W) float32 depth in mm, 0 = invalid

Run (Snellius):
    python scripts/export_scared_stereo_gt.py \
        --scared-root ../data/SCARED --out ../data/SCARED/stereo_gt

ponytail: OpenCV SGBM is the stdlib-equivalent here (no deep stereo net for a 10-frame
reference). Tune --num-disp / --block if the depth stats look wrong for a keyframe.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np

# SCARED test working distances: reject disparities implying depth outside this band (mm).
MIN_MM, MAX_MM = 10.0, 300.0


def read_calib(yaml_path):
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    g = lambda k: fs.getNode(k).mat()
    calib = dict(M1=g("M1"), D1=g("D1"), M2=g("M2"), D2=g("D2"), R=g("R"), T=g("T"))
    fs.release()
    calib["T"] = calib["T"].reshape(3, 1)                 # yaml stores it 1x3
    return calib


def stereo_depth(left, right, c, num_disp, block):
    """Rectify the calibrated pair, SGBM disparity, reproject -> (rect_left_bgr, depth_mm)."""
    h, w = left.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        c["M1"], c["D1"], c["M2"], c["D2"], (w, h), c["R"], c["T"],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1x, m1y = cv2.initUndistortRectifyMap(c["M1"], c["D1"], R1, P1, (w, h), cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(c["M2"], c["D2"], R2, P2, (w, h), cv2.CV_32FC1)
    lr = cv2.remap(left, m1x, m1y, cv2.INTER_LINEAR)
    rr = cv2.remap(right, m2x, m2y, cv2.INTER_LINEAR)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(lr, rr).astype(np.float32) / 16.0  # SGBM returns fixed-point *16

    xyz = cv2.reprojectImageTo3D(disp, Q)
    depth = xyz[:, :, 2]                                   # Z in mm (calibration units)
    valid = (disp > 0) & np.isfinite(depth) & (depth > MIN_MM) & (depth < MAX_MM)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    return lr, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scared-root", default="../data/SCARED",
                    help="dir holding test_dataset_*/keyframe_*/{Left,Right}_Image.png")
    ap.add_argument("--out", default="../data/SCARED/stereo_gt")
    ap.add_argument("--num-disp", type=int, default=192, help="SGBM disparity range (÷16)")
    ap.add_argument("--block", type=int, default=5, help="SGBM matched block size (odd)")
    args = ap.parse_args()

    kfs = sorted(Path(args.scared_root).glob("test_dataset_*/keyframe_*"))
    assert kfs, f"no test_dataset_*/keyframe_* under {args.scared_root}"
    frames_dir = Path(args.out) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    depths, names = [], []
    for kf in kfs:
        left = cv2.imread(str(kf / "Left_Image.png"))
        right = cv2.imread(str(kf / "Right_Image.png"))
        if left is None or right is None:
            print(f"[skip] {kf} missing Left/Right_Image.png"); continue
        calib = read_calib(kf / "endoscope_calibration.yaml")
        rect_left, depth = stereo_depth(left, right, calib, args.num_disp, args.block)

        name = f"{kf.parent.name}_{kf.name}"               # test_dataset_8_keyframe_0
        cv2.imwrite(str(frames_dir / f"{name}.png"), rect_left)
        depths.append(depth); names.append(name)
        v = depth[depth > 0]
        print(f"[ok] {name}: valid {100*(depth>0).mean():4.1f}%  "
              f"depth med={np.median(v):6.1f}  p5={np.percentile(v,5):6.1f}  "
              f"p95={np.percentile(v,95):6.1f} mm", flush=True)

    assert depths, "no keyframes exported"
    np.savez_compressed(Path(args.out) / "gt_depths.npz", data=np.stack(depths))
    print(f"\n[done] {len(depths)} keyframes -> {args.out}/gt_depths.npz + frames/  "
          f"(order: {names[0]} .. {names[-1]})")


if __name__ == "__main__":
    main()
