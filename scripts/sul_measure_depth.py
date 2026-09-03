"""
SUL from the urethra mask alone, measured twice, and compared.

  2D  : the mirror-axis span of the urethra mask (sul_measure.mirror_axis), scaled to mm
        with the catheter arch chord -- exactly what sul_measure.py reports.
  3D  : the same two axis endpoints back-projected with the METRIC depth model
        (outputs/depth_ruler_range_sw05/best.pth) and measured in the camera frame.
        No arch, no anchor: the depth map itself carries the millimetres.

The 3D number is >= the 2D one whenever the urethra tilts toward or away from the camera,
which is the whole point of comparing them.

python scripts/sul_measure_depth.py --root "../transfer_atlas_mod/workspace/SUL_img3x"
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import sul_measure as S
from gui_depth_measure import (DEFAULT_CKPT, DEFAULT_K_NORM, DEFAULT_MAX_DEPTH,
                               DEFAULT_MIN_DEPTH, DEFAULT_SHAPE, DepthBackend,
                               auto_bars, crop_box, sample_depth, segment_length)


def norm_point(p, box):
    """Image-coords point -> normalised coords inside the crop box (l,t,r,b)."""
    l, t, r, b = box
    return ((p[0] - l) / (r - l), (p[1] - t) / (b - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../transfer_atlas_mod/workspace/SUL_img3x")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--csv", default=None, help="default: <root>/sul_depth_compare.csv")
    ap.add_argument("--max-tilt", type=float, default=40.0)
    ap.add_argument("--no-crop", action="store_true",
                    help="skip the pillarbox auto-crop (the model expects ~5:4 framing)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.csv) if args.csv else root / "sul_depth_compare.csv"
    groups, arches = S.load_groups(root), S.load_arches(root)
    backend = DepthBackend(args.ckpt, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH, DEFAULT_MAX_DEPTH).load()

    rows = []
    for clip, frames in sorted(groups.items()):
        video = S.video_id(clip)
        row = {"video": video, "sul_2d_mm": "", "sul_depth_mm": "", "diff_mm": "",
               "ratio": "", "span_px": "", "mm_per_px": "", "z0_mm": "", "z1_mm": "",
               "dz_mm": "", "status": ""}
        if S.URETHRA not in frames:
            row["status"] = "no urethra mask"
            rows.append(row)
            continue

        stem = frames[S.URETHRA]
        u_mask = np.array(Image.open(root / "masks" / f"{stem}.png")) == S.URETHRA
        ax = S.mirror_axis(u_mask, max_tilt=args.max_tilt)
        p0, p1 = ax["p0"], ax["p1"]
        row["span_px"] = ax["span"]

        # --- 2D, as sul_measure.py does it (arch chord for the scale)
        c_stem = frames.get(S.CATHETER)
        if c_stem in arches:
            left, right = arches[c_stem]
            chord = float(np.linalg.norm(left - right))
            if chord >= 5:
                scale = S.CATHETER_MM / chord
                row["mm_per_px"] = f"{scale:.5f}"
                row["sul_2d_mm"] = f"{ax['span'] * scale:.2f}"

        # --- 3D, from the metric depth map alone
        img = Image.open(root / "images" / f"{stem}.png").convert("RGB")
        box = ((0, 0, img.size[0], img.size[1]) if args.no_crop else
               crop_box(img.size, *auto_bars(np.array(img).astype(np.float32))))
        depth = backend.predict(img.crop(box))
        a, b = norm_point(p0, box), norm_point(p1, box)
        # A mirror-axis endpoint can land a pixel outside the frame after the inverse
        # rotation; that is rounding, not a bad measurement, so clamp. Anything genuinely
        # off-frame (>2% out) means the mask is outside the picture area -- skip it.
        if not all(-0.02 <= c <= 1.02 for c in a + b):
            row["status"] = "axis endpoint outside the crop"
            rows.append(row)
            continue
        a = tuple(min(max(c, 0.0), 1.0) for c in a)
        b = tuple(min(max(c, 0.0), 1.0) for c in b)
        z0, z1 = sample_depth(depth, *a), sample_depth(depth, *b)
        mm3d = segment_length(a, b, z0, z1, DEFAULT_K_NORM)
        row.update(sul_depth_mm=f"{mm3d:.2f}", z0_mm=f"{z0:.1f}", z1_mm=f"{z1:.1f}",
                   dz_mm=f"{z1 - z0:+.1f}", status="ok")
        if row["sul_2d_mm"]:
            mm2d = float(row["sul_2d_mm"])
            row.update(diff_mm=f"{mm3d - mm2d:+.2f}", ratio=f"{mm3d / mm2d:.3f}")
        else:
            row["status"] = "ok (depth only, no arch scale)"
        rows.append(row)

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    both = [r for r in rows if r["ratio"]]
    print(f"[done] {len(rows)} videos, {len(both)} with both lengths -> {out}")
    if both:
        d2 = np.array([float(r["sul_2d_mm"]) for r in both])
        d3 = np.array([float(r["sul_depth_mm"]) for r in both])
        ratio = d3 / d2
        print(f"  2D    median {np.median(d2):6.1f} mm  range {d2.min():.1f}-{d2.max():.1f}")
        print(f"  depth median {np.median(d3):6.1f} mm  range {d3.min():.1f}-{d3.max():.1f}")
        print(f"  depth/2D median {np.median(ratio):.2f}  "
              f"mean {ratio.mean():.2f}  range {ratio.min():.2f}-{ratio.max():.2f}")
        print(f"  mean abs diff {np.abs(d3 - d2).mean():.1f} mm, "
              f"correlation {np.corrcoef(d2, d3)[0, 1]:.2f}")
    for r in rows:
        if not r["status"].startswith("ok"):
            print(f"  [skip] {r['video']}: {r['status']}")


if __name__ == "__main__":
    main()
