"""
Calibrate the assumed intrinsics against the catheter, across every clip in a SUL workspace.

Every clip has one manually drawn catheter arch whose chord is a known 16/3 mm. Back-projecting
that chord with the depth map and DEFAULT_K_NORM over-reads it (~1.36x), so one multiplier on
(fx, fy) puts it right:  in-plane length scales as 1/f, so  f_new = f_old * (measured / 5.333).

What this can and cannot separate: the arch lies flat on (dz ~ 0.1 mm), so the chord constrains
only the in-plane term (du/W) * z / fx -- a depth that is uniformly too FAR and an fx that is
too SMALL are indistinguishable from it. The fitted number therefore absorbs both; it is the
right correction for measuring lengths, and is NOT evidence the lens is what was wrong.

    python scripts/fit_fx_catheter.py --root ..\transfer_atlas_mod\workspace\SUL_img3x_5x4
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import sul_measure as S
from gui_depth_measure import (DEFAULT_CKPT, DEFAULT_K_NORM, DEFAULT_MAX_DEPTH,
                               DEFAULT_MIN_DEPTH, DEFAULT_SHAPE, DepthBackend,
                               sample_depth, segment_length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../transfer_atlas_mod/workspace/SUL_img3x_5x4")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=None, help="default: <root>/k_calibration.json")
    args = ap.parse_args()

    root = Path(args.root)
    groups, arches = S.load_groups(root), S.load_arches(root)
    backend = DepthBackend(args.ckpt, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH, DEFAULT_MAX_DEPTH).load()

    rows = []
    for clip, frames in sorted(groups.items()):
        c_stem = frames.get(S.CATHETER)
        if c_stem not in arches:
            continue
        left, right = arches[c_stem]
        chord_px = float(np.linalg.norm(left - right))
        if chord_px < 5:
            continue
        pil = Image.open(root / "images" / f"{c_stem}.png").convert("RGB")
        W, H = pil.size
        depth = backend.predict(pil)
        a, b = (left[0] / W, left[1] / H), (right[0] / W, right[1] / H)
        if not all(-0.02 <= c <= 1.02 for c in a + b):
            continue
        a = tuple(min(max(c, 0.0), 1.0) for c in a)
        b = tuple(min(max(c, 0.0), 1.0) for c in b)
        z0, z1 = sample_depth(depth, *a), sample_depth(depth, *b)
        mm = segment_length(a, b, z0, z1, DEFAULT_K_NORM)
        rows.append({"video": S.video_id(clip), "chord_px": chord_px,
                     "z_mm": 0.5 * (z0 + z1), "dz_mm": z1 - z0,
                     "measured_mm": mm, "s": mm / S.CATHETER_MM})

    if not rows:
        raise SystemExit("[abort] no usable catheter arches")

    s = np.array([r["s"] for r in rows])
    # Median, not mean: a few arches sit on an instrument edge and read wildly far.
    s_hat = float(np.median(s))
    fx, fy, cx, cy = DEFAULT_K_NORM
    k_new = (fx * s_hat, fy * s_hat, cx, cy)

    out = Path(args.out) if args.out else root / "k_calibration.json"
    out.write_text(json.dumps(
        {"n_clips": len(rows), "reference": "catheter arch chord", "reference_mm": S.CATHETER_MM,
         "k_norm_assumed": list(DEFAULT_K_NORM), "scale": s_hat, "k_norm_fitted": list(k_new),
         "scale_iqr": [float(np.percentile(s, 25)), float(np.percentile(s, 75))],
         "per_clip": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                      for r in rows]}, indent=2))

    resid = s / s_hat                      # 1.0 = this clip agrees with the global fit
    print(f"[done] {len(rows)} catheter arches -> {out}")
    print(f"  measured/true per clip: median {s_hat:.3f}  "
          f"IQR {np.percentile(s, 25):.3f}-{np.percentile(s, 75):.3f}  "
          f"range {s.min():.3f}-{s.max():.3f}")
    print(f"  fx {fx:.4f} -> {k_new[0]:.4f}   fy {fy:.4f} -> {k_new[1]:.4f}")
    print(f"  after this one global fit, per-clip error: median |resid-1| "
          f"{np.median(np.abs(resid - 1)) * 100:.1f}%, "
          f"90th pct {np.percentile(np.abs(resid - 1), 90) * 100:.1f}%")
    print(f"  (a per-clip fit would be exact by construction -- that spread is what a single "
          f"global number costs)")


if __name__ == "__main__":
    main()
