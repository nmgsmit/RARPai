#!/usr/bin/env python3
"""Rank videos by sharpness / blur using no-reference focus measures.

Per clip we sample N frames, resize to a fixed height (so resolution does not
bias the score), and on the non-black (unmasked) region compute:

  sharp   = variance of the Laplacian            -- classic focus measure
  nsharp  = var(Laplacian) / var(gray)           -- contrast-normalised, so a
                                                    dark/low-contrast clip is
                                                    not punished twice
  blur%   = fraction of 64x64 tiles whose Laplacian variance is below
            --tile-thresh  -- local haze, water/smoke splats, out-of-focus
            corners show up here even when the frame average looks fine

Reported per clip: median over frames (robust to a few motion-blurred ones).

    python scripts/rank_sharpness.py <dir> --csv out.csv
    python scripts/rank_sharpness.py --selftest
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np

TILE = 64


def frame_metrics(bgr, height, tile_thresh):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if height and g.shape[0] != height:
        s = height / g.shape[0]
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    valid = g > 8  # ponytail: fixed cutoff for the GUI/pillarbox black mask
    if valid.mean() < 0.05:
        return None
    lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
    sharp = lap[valid].var()
    gv = g[valid].astype(np.float32).var()
    nsharp = sharp / gv if gv > 1e-6 else 0.0

    h, w = (np.array(g.shape) // TILE) * TILE
    if h and w:
        t_lap = lap[:h, :w].reshape(h // TILE, TILE, w // TILE, TILE)
        t_val = valid[:h, :w].reshape(h // TILE, TILE, w // TILE, TILE)
        keep = t_val.mean(axis=(1, 3)) > 0.9          # tiles fully inside the image
        tv = t_lap.var(axis=(1, 3))[keep]
        blur_frac = float((tv < tile_thresh).mean()) if tv.size else float("nan")
    else:
        blur_frac = float("nan")
    return sharp, nsharp, blur_frac


def score_video(path, n_frames, height, tile_thresh):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idx = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    rows, want = [], set(idx.tolist())
    for i in range(total):                    # sequential read beats seeking
        ok = cap.grab()
        if not ok:
            break
        if i in want:
            ok, frame = cap.retrieve()
            if ok:
                m = frame_metrics(frame, height, tile_thresh)
                if m:
                    rows.append(m)
    cap.release()
    if not rows:
        return None
    a = np.array(rows)
    return dict(zip(("sharp", "nsharp", "blur_frac"), np.median(a, axis=0)))


def selftest():
    rng = np.random.default_rng(0)
    img = rng.integers(40, 215, (512, 512, 3), dtype=np.uint8)
    soft = cv2.GaussianBlur(img, (0, 0), 3)
    s_sharp = frame_metrics(img, 512, 50)
    s_soft = frame_metrics(soft, 512, 50)
    assert s_sharp[0] > s_soft[0] * 5, (s_sharp, s_soft)
    assert s_sharp[2] < s_soft[2], (s_sharp, s_soft)
    half = img.copy()
    half[:, 256:] = cv2.GaussianBlur(half[:, 256:], (0, 0), 3)
    assert 0.3 < frame_metrics(half, 512, 50)[2] < 0.7   # ~half the tiles blurry
    print("selftest ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", help="directory searched recursively for videos")
    p.add_argument("--ext", default="mp4")
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--height", type=int, default=512, help="0 = keep native size")
    p.add_argument("--tile-thresh", type=float, default=50.0,
                   help="tile Laplacian variance below this counts as blurry")
    p.add_argument("--csv")
    p.add_argument("--sort", default="sharp", choices=["sharp", "nsharp", "blur_frac"])
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.root:
        p.error("root is required")

    vids = sorted(Path(a.root).rglob(f"*.{a.ext}"))
    out = []
    for i, v in enumerate(vids, 1):
        r = score_video(v, a.frames, a.height, a.tile_thresh)
        print(f"[{i}/{len(vids)}] {v.name} {'skipped' if not r else ''}",
              file=sys.stderr, flush=True)
        if r:
            out.append({"video": str(v.relative_to(a.root)), **r})

    out.sort(key=lambda r: r[a.sort], reverse=a.sort != "blur_frac")
    print(f"\n{'sharp':>9} {'nsharp':>7} {'blur%':>6}  video")
    for r in out:
        print(f"{r['sharp']:9.1f} {r['nsharp']:7.3f} {100*r['blur_frac']:6.1f}  {r['video']}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, ["video", "sharp", "nsharp", "blur_frac"])
            w.writeheader(); w.writerows(out)
        print(f"\nwrote {a.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
