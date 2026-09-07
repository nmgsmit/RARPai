"""
Per-frame scale check on the catheter, whose width is KNOWN: 16 Fr = 16/3 = 5.333 mm.

For every frame that has a catheter mask, measure that width three ways and compare:

  width_px       the catheter mask's own width -- median row width with the mask rotated
                 so its long axis is vertical, so this is a true perpendicular width.
  mm_per_px      5.333 / width_px. This is the per-frame 2D scale, derived from the mask
                 alone: no hand-drawn arch, no anchor click.
  arch_mm_per_px 5.333 / arch chord, the scale sul_measure.py actually uses. Comparing the
                 two says how much the manual arch and the mask disagree, in scale terms.
  depth_width_mm the SAME two edge pixels back-projected with the metric depth model and
                 measured in 3D. It should come out 5.333 mm; how far off it lands IS the
                 depth model's scale error on that frame, which is the number we cannot get
                 any other way.

python scripts/catheter_scale_check.py --root "../transfer_atlas_mod/workspace/SUL_img3x_5x4" --no-crop
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sul_measure as S
from gui_depth_measure import (DEFAULT_CKPT, DEFAULT_K_NORM, DEFAULT_MAX_DEPTH,
                               DEFAULT_MIN_DEPTH, DEFAULT_SHAPE, DepthBackend,
                               auto_bars, crop_box, sample_depth, segment_length)


def catheter_width(mask):
    """Perpendicular width of an elongated mask, in pixels, plus the two edge points of a
    representative cross-section in image coords.

    The mask is rotated until its principal axis is vertical, so a row of the rotated mask
    IS a perpendicular cross-section. The width is the MEDIAN row width -- the catheter is
    a cylinder, so its rows agree, and the median ignores the ragged first/last rows and any
    row the segmenter clipped where an instrument crosses. The reported edge points come from
    the row whose width is closest to that median, so the pixels handed to the depth model
    are on a cross-section that is actually representative.
    """
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    side = int(np.hypot(ys.max() - ys.min(), xs.max() - xs.min())) + 8
    tilt, elong = S.principal_axis(mask)

    rot = cv2.getRotationMatrix2D((cx, cy), -tilt, 1.0)   # -tilt puts the long axis upright
    rot[0, 2] += side / 2.0 - cx
    rot[1, 2] += side / 2.0 - cy
    canvas = cv2.warpAffine(mask.astype(np.uint8), rot, (side, side), flags=cv2.INTER_NEAREST)

    rows = np.flatnonzero(canvas.any(1))
    spans = np.array([np.ptp(np.flatnonzero(canvas[r])) + 1 for r in rows])
    width = float(np.median(spans))
    r = rows[int(np.argmin(np.abs(spans - width)))]
    cols = np.flatnonzero(canvas[r])

    inv = cv2.invertAffineTransform(rot)
    ends = np.array([[cols[0], r], [cols[-1], r]], np.float64)
    p0, p1 = (inv[:, :2] @ ends.T).T + inv[:, 2]
    return width, p0, p1, elong, int(rows.size)


def norm_point(p, box):
    l, t, r, b = box
    return ((p[0] - l) / (r - l), (p[1] - t) / (b - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../transfer_atlas_mod/workspace/SUL_img3x_5x4")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--csv", default=None, help="default: <root>/catheter_scale.csv")
    ap.add_argument("--no-crop", action="store_true",
                    help="images are already in 5:4 training framing (SUL_img3x_5x4)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.csv) if args.csv else root / "catheter_scale.csv"
    groups, arches = S.load_groups(root), S.load_arches(root)
    backend = DepthBackend(args.ckpt, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH, DEFAULT_MAX_DEPTH).load()

    rows = []
    for clip, frames in sorted(groups.items()):
        row = {"video": S.video_id(clip), "catheter_frame": "", "width_px": "",
               "mm_per_px": "", "arch_chord_px": "", "arch_mm_per_px": "",
               "mask_vs_arch": "", "depth_width_mm": "", "depth_err_pct": "",
               "z0_mm": "", "z1_mm": "", "elongation": "", "rows_measured": "",
               "status": ""}
        stem = frames.get(S.CATHETER)
        if stem is None:
            row["status"] = "no catheter mask"
            rows.append(row)
            continue

        row["catheter_frame"] = stem
        mask = np.array(Image.open(root / "masks" / f"{stem}.png")) == S.CATHETER
        if mask.sum() < 50:
            row["status"] = "catheter mask too small"
            rows.append(row)
            continue

        width, p0, p1, elong, nrows = catheter_width(mask)
        row.update(width_px=f"{width:.1f}", mm_per_px=f"{S.CATHETER_MM / width:.5f}",
                   elongation=f"{elong:.2f}", rows_measured=nrows)

        if stem in arches:
            chord = float(np.linalg.norm(arches[stem][0] - arches[stem][1]))
            row["arch_chord_px"] = f"{chord:.1f}"
            if chord >= 5:
                row["arch_mm_per_px"] = f"{S.CATHETER_MM / chord:.5f}"
                row["mask_vs_arch"] = f"{width / chord:.3f}"

        img = Image.open(root / "images" / f"{stem}.png").convert("RGB")
        box = ((0, 0, img.size[0], img.size[1]) if args.no_crop else
               crop_box(img.size, *auto_bars(np.array(img).astype(np.float32))))
        depth = backend.predict(img.crop(box))
        a, b = norm_point(p0, box), norm_point(p1, box)
        if not all(-0.02 <= c <= 1.02 for c in a + b):
            row["status"] = "cross-section outside the crop"
            rows.append(row)
            continue
        a = tuple(min(max(c, 0.0), 1.0) for c in a)
        b = tuple(min(max(c, 0.0), 1.0) for c in b)
        z0, z1 = sample_depth(depth, *a), sample_depth(depth, *b)
        mm = segment_length(a, b, z0, z1, DEFAULT_K_NORM)
        row.update(depth_width_mm=f"{mm:.2f}", z0_mm=f"{z0:.1f}", z1_mm=f"{z1:.1f}",
                   depth_err_pct=f"{100 * (mm / S.CATHETER_MM - 1):+.1f}", status="ok")
        rows.append(row)

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"[done] {len(rows)} clips, {len(ok)} measured -> {out}")
    if ok:
        wpx = np.array([float(r["width_px"]) for r in ok])
        dmm = np.array([float(r["depth_width_mm"]) for r in ok])
        print(f"  mask width      median {np.median(wpx):5.1f} px  "
              f"range {wpx.min():.0f}-{wpx.max():.0f}")
        print(f"  depth width     median {np.median(dmm):5.2f} mm  "
              f"range {dmm.min():.2f}-{dmm.max():.2f}   (true {S.CATHETER_MM:.2f} mm)")
        err = dmm / S.CATHETER_MM
        print(f"  depth/true      median {np.median(err):.2f}  mean {err.mean():.2f}  "
              f"range {err.min():.2f}-{err.max():.2f}")
        ratio = np.array([float(r["mask_vs_arch"]) for r in ok if r["mask_vs_arch"]])
        if ratio.size:
            print(f"  mask/arch width median {np.median(ratio):.2f}  mean {ratio.mean():.2f}  "
                  f"range {ratio.min():.2f}-{ratio.max():.2f}  (n={ratio.size})")
    for r in rows:
        if r["status"] != "ok":
            print(f"  [skip] {r['video']}: {r['status']}")


if __name__ == "__main__":
    main()
