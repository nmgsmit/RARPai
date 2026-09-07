"""Score checkpoints on the hand-annotated catheter widths -- the only metric GT with no
segmentation mask and no training anchor in it.

    python scripts/eval_catheter_ckpt.py --ref ../data/sul_reference \
        outputs/depth_ruler_range_sw05/best.pth outputs/depth_ruler_inplane/best.pth

Reuses eval_catheter_ref from finetune_depth, so the geometry is identical to the in-training
metric. err_mm is signed (+ = the model reads the catheter too WIDE); scale is measured/true.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from finetune_depth import (build_depth_model, disp_to_depth, eval_catheter_ref,
                            load_catheter_ref, round14, round32, sample_depth, segment_length)


def per_frame(model, ref, hw, device, min_depth, max_depth, k_norm):
    """Per-frame measured/true ratio, plus the object's apparent size. A model that emits ONE
    constant distance over-reads exactly on the frames where the catheter looks BIG (it is closer
    than the model thinks), so corr(ratio, apparent size) is +0.84 for the shipped checkpoint and
    should fall toward 0 for a model that tracks distance."""
    model.eval()
    ratio, apparent = [], []
    with torch.no_grad():
        for img, a, b, true_mm in ref:
            disp = F.interpolate(model(img.unsqueeze(0).to(device))[("disp", 0)], hw,
                                 mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, min_depth, max_depth)
            d = depth[0, 0].cpu().numpy()
            mm = segment_length(a, b, sample_depth(d, *a), sample_depth(d, *b), k_norm)
            ratio.append(mm / true_mm)
            apparent.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
    return np.array(ratio), np.array(apparent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--ref", default="../data/sul_reference")
    ap.add_argument("--image-shape", type=int, nargs=2, default=[392, 490])
    ap.add_argument("--intrinsics", type=float, nargs=4, default=[0.82, 1.02, 0.5, 0.5])
    ap.add_argument("--min-depth", type=float, default=20.0)
    ap.add_argument("--max-depth", type=float, default=200.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_shape = (round14(args.image_shape[0]), round14(args.image_shape[1]))
    hw = (round32(args.image_shape[0]), round32(args.image_shape[1]))
    ref = load_catheter_ref(args.ref, hw)
    if not ref:
        raise SystemExit(f"[abort] no catheter annotations under {args.ref}")
    print(f"[ref] {len(ref)} hand-annotated widths from {args.ref}", flush=True)

    model = build_depth_model(model_shape, device)
    for c in args.ckpts:
        model.load_state_dict(torch.load(c, map_location=device))
        r = eval_catheter_ref(model, ref, hw, device, args.min_depth, args.max_depth,
                              tuple(args.intrinsics))
        ratio, apparent = per_frame(model, ref, hw, device, args.min_depth, args.max_depth,
                                    tuple(args.intrinsics))
        deb = np.median(np.abs(ratio / np.median(ratio) - 1.0))
        print(f"{Path(c).parent.name:26s} err {r['err_mm']:+.3f} mm | abs {r['abs_err_mm']:.3f} "
              f"| width {r['width_mm']:.3f} (true 5.333) | scale {r['scale']:.3f} | n {int(r['n'])}",
              flush=True)
        print(f"{'':26s} after removing its own constant: median err {100 * deb:.1f}%  "
              f"| corr(ratio, apparent size) {np.corrcoef(ratio, apparent)[0, 1]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
