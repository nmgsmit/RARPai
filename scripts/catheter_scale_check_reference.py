r"""
Depth-model scale check against MANUALLY drawn catheter widths -- ground truth from a human,
not the segmentation mask (see catheter_scale_check.py for the mask version).

workspace/UMCsulsnaps/sul_reference/scale_objects.json holds a hand-placed "Catheter tip"
object (class_id 2, fixed at 16 Fr = 5.333 mm) on every frame where the catheter was clear
enough to annotate -- not all of them, some frames were skipped as unusable. For each annotated
frame, the two endpoints are back-projected with the metric depth model and measured in 3D;
comparing that to the known 5.333 mm is a scale-error reading with NO segmentation and NO
manual arch chord in the loop at all -- just the depth model against a human's own eyes.

Every depth map is written to outputs/depth_cache/<ckpt>_<fingerprint>/ (DepthCache, shared with
gui_depth_measure.py's GUI) as it's computed, so a second run -- or opening the same images in the
GUI -- reads the cache back instead of rerunning the model.

python scripts/catheter_scale_check_reference.py --root ..\transfer_atlas_mod\workspace\UMCsulsnaps\sul_reference
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gui_depth_measure import (DEFAULT_CACHE, DEFAULT_CKPT, DEFAULT_K_NORM, DEFAULT_MAX_DEPTH,
                               DEFAULT_MIN_DEPTH, DEFAULT_SHAPE, DepthBackend, DepthCache,
                               DepthProvider, sample_depth, segment_length)

CATHETER_CLASS = 2   # scale_objects.json class id for the manually drawn catheter width
NO_CROP = (0.0, 0.0, 0.0)   # sul_reference frames are already 5:4, banner already blacked


def load_reference(root):
    """image stem -> (a_px, b_px, true_mm) for every manually annotated catheter width.

    Frame keys in scale_objects.json are indices into the sorted image list -- same convention
    arches.json uses (sul_measure.load_arches)."""
    so = json.loads((root / "scale_objects.json").read_text())
    images = sorted(p.stem for p in (root / "images").glob("*.png"))
    out = {}
    for idx, objs in so["frames"].items():
        i = int(idx)
        if i >= len(images):
            continue
        for o in objs:
            if o["class_id"] == CATHETER_CLASS:
                out[images[i]] = (np.array(o["a"], float), np.array(o["b"], float), float(o["mm"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../transfer_atlas_mod/workspace/UMCsulsnaps/sul_reference")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--csv", default=None, help="default: <root>/catheter_scale_reference.csv")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.csv) if args.csv else root / "catheter_scale_reference.csv"
    ref = load_reference(root)
    if not ref:
        raise SystemExit(f"[abort] no 'Catheter tip' annotations in {root / 'scale_objects.json'}")

    cache = DepthCache(args.cache, args.ckpt, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH, DEFAULT_MAX_DEPTH)
    provider = DepthProvider(DepthBackend(args.ckpt, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH,
                                          DEFAULT_MAX_DEPTH), cache)

    rows = []
    for stem, (a_px, b_px, true_mm) in sorted(ref.items()):
        path = root / "images" / f"{stem}.png"
        img = Image.open(path).convert("RGB")
        w, h = img.size
        depth, src, _ = provider.depth_for(str(path), img, NO_CROP)
        a, b = (a_px[0] / w, a_px[1] / h), (b_px[0] / w, b_px[1] / h)
        z0, z1 = sample_depth(depth, *a), sample_depth(depth, *b)
        mm = segment_length(a, b, z0, z1, DEFAULT_K_NORM)
        rows.append({"video": stem, "true_mm": f"{true_mm:.3f}", "depth_width_mm": f"{mm:.2f}",
                     "ratio": f"{mm / true_mm:.3f}", "z0_mm": f"{z0:.1f}", "z1_mm": f"{z1:.1f}",
                     "depth_src": src})

    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ratio = np.array([float(r["ratio"]) for r in rows])
    print(f"[done] {len(rows)} manually-annotated catheter frames -> {out}")
    print(f"  cache: outputs/depth_cache/{cache.name}  ({cache.count()} maps stored)")
    print(f"  depth width     median {np.median([float(r['depth_width_mm']) for r in rows]):5.2f} mm  "
          f"(true {rows[0]['true_mm']} mm)")
    print(f"  depth/true      median {np.median(ratio):.2f}  mean {ratio.mean():.2f}  "
          f"std {ratio.std():.2f}  range {ratio.min():.2f}-{ratio.max():.2f}")


if __name__ == "__main__":
    main()
