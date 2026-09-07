"""Score checkpoints on the hand-annotated catheter widths -- the only metric GT with no
segmentation mask and no training anchor in it.

    python scripts/eval_catheter_ckpt.py --ref ../data/sul_reference \
        outputs/depth_ruler_range_sw05/best.pth outputs/depth_ruler_inplane/best.pth

Reuses eval_catheter_ref from finetune_depth, so the geometry is identical to the in-training
metric. err_mm is signed (+ = the model reads the catheter too WIDE); scale is measured/true.
"""
import argparse
from pathlib import Path

import torch

from finetune_depth import (build_depth_model, eval_catheter_ref, load_catheter_ref, round14,
                            round32)


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
        print(f"{Path(c).parent.name:26s} err {r['err_mm']:+.3f} mm | abs {r['abs_err_mm']:.3f} "
              f"| width {r['width_mm']:.3f} (true 5.333) | scale {r['scale']:.3f} | n {int(r['n'])}",
              flush=True)


if __name__ == "__main__":
    main()
