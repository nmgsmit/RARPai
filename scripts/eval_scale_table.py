"""
How many millimetres is the depth map off?

Runs one or more checkpoints over the annotated known-size objects in scale_objects.json and
prints, per class and per annotated length, the PREDICTED metric length next to the TRUE one.
Same geometry as `scale_loss` in finetune_depth.py (sample depth at the annotated points,
back-project with K, sum the 3D polyline) -- this is a readable report of that, not a second
implementation.

`corrected` columns divide out ONE global constant (the median predicted/true ratio over all
objects of that checkpoint), i.e. what you would get from a single calibration -- the honest
comparison for a self-supervised depth map, whose scale is arbitrary until something fixes it.

    python scripts/eval_scale_table.py \
        --ckpt warmstart=../backbones/EndoDAC/depth_model.pth \
        --ckpt allcls=outputs/depth_ruler_scale/best.pth \
        --data-root ../data/processed/depthclips_ruler_NoGUI --video-split 4 5 --split test
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finetune_depth import (DEFAULT_K_NORM, RARPTriplets, build_depth_model, _filter_load,
                            disp_to_depth, round14, round32, scale_loss, split_by_video)

CLASS_NAMES = {1: "Ruler", 2: "Catheter tip", 3: "Robot arm"}


@torch.no_grad()
def collect(depth_model, loader, hw, device, min_depth, max_depth):
    """-> (pred_mm, true_mm, class_id) arrays over every annotated object in the loader."""
    depth_model.eval()
    rows = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        disp = F.interpolate(depth_model(batch[("color", 0)])[("disp", 0)], hw,
                             mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp, min_depth, max_depth)
        _, ratio, w = scale_loss(depth, batch["inv_K"], batch, hw)
        m = w > 0
        rows.append(torch.stack([ratio[m] * batch["anch_mm"][m], batch["anch_mm"][m],
                                 batch["anch_cls"][m].float()], 1).cpu())
    r = torch.cat(rows).numpy() if rows else np.zeros((0, 3), np.float32)
    return r[:, 0], r[:, 1], r[:, 2]


def summarise(pred, true, k):
    """Row of the report. k = the single global constant divided out for the corrected columns."""
    dev, cdev = pred - true, pred / k - true
    return dict(n=len(true), true_mm=true.mean(), pred_mm=pred.mean(),
                mae_mm=np.abs(dev).mean(), mae_pct=100 * np.abs(dev / true).mean(),
                med_pct=100 * np.median(np.abs(dev / true)),
                c_pred_mm=(pred / k).mean(), c_mae_mm=np.abs(cdev).mean(),
                c_mae_pct=100 * np.abs(cdev / true).mean(),
                c_med_pct=100 * np.median(np.abs(cdev / true)))


HDR = (f"{'group':<22}{'n':>5}{'true mm':>9}{'pred mm':>9}{'MAE mm':>9}{'MAE %':>9}{'med %':>8}"
       f"  | {'pred mm':>9}{'MAE mm':>9}{'MAE %':>9}{'med %':>8}")


def print_table(name, pred, true, cls, k):
    print(f"\n=== {name}   (global scale k = {k:.4f}; 'corrected' = predictions divided by k)")
    print(HDR)
    print("-" * len(HDR))

    def line(label, m):
        if m.sum() < 3:
            return
        s = summarise(pred[m], true[m], k)
        print(f"{label:<22}{s['n']:>5}{s['true_mm']:>9.2f}{s['pred_mm']:>9.2f}"
              f"{s['mae_mm']:>9.2f}{s['mae_pct']:>9.1f}{s['med_pct']:>8.1f}"
              f"  | {s['c_pred_mm']:>9.2f}{s['c_mae_mm']:>9.2f}{s['c_mae_pct']:>9.1f}"
              f"{s['c_med_pct']:>8.1f}")

    for ci in sorted(np.unique(cls)):
        m = cls == ci
        line(CLASS_NAMES.get(int(ci), f"class {int(ci)}"), m)
        for mm in sorted(np.unique(true[m]))[:6]:      # e.g. the 10 mm vs 20 mm ruler segments
            line(f"  {mm:.2f} mm", m & (true == mm))
    line("ALL", np.ones(len(true), bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable; NAME=PATH (a bare PATH is named after its parent dir)")
    ap.add_argument("--data-root", default="../data/processed/depthclips_ruler_NoGUI")
    ap.add_argument("--video-split", type=int, nargs=2, default=[4, 5])
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--image-shape", type=int, nargs=2, default=[392, 490])
    ap.add_argument("--intrinsics", type=float, nargs=4, default=list(DEFAULT_K_NORM))
    ap.add_argument("--frame-stride", type=int, default=1)
    # 20/200 mm, NOT the 0.1/150 KITTI-unit defaults: those compress the 35-120mm endoscopic
    # band into 0.2% of the sigmoid range, which makes every metric number an artifact.
    # See CLAUDE_NOTES 2026-09-02 "--min-depth/--max-depth WAS THE BUG".
    ap.add_argument("--min-depth", type=float, default=20.0)
    ap.add_argument("--max-depth", type=float, default=200.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--csv", default=None, help="also dump every object as pred_mm,true_mm,class")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_shape = (round14(args.image_shape[0]), round14(args.image_shape[1]))
    hw = (round32(args.image_shape[0]), round32(args.image_shape[1]))
    tr, va, te = split_by_video(args.data_root, *args.video_split)
    dirs = {"train": tr, "val": va, "test": te, "all": tr + va + te}[args.split]

    ds = RARPTriplets(None, hw, tuple(args.intrinsics), args.frame_stride, clip_dirs=dirs,
                      anchors_max=4, mask_overlay=False)
    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_depth_model(model_shape, device)
    dump = []
    for spec in args.ckpt:
        name, _, path = spec.rpartition("=")
        name = name or Path(path).parent.name
        _filter_load(model, path, name)
        pred, true, cls = collect(model, loader, hw, device, args.min_depth, args.max_depth)
        assert len(true), "no annotated objects in this split"
        k = float(np.median(pred / true))
        print_table(f"{name}  [{args.split}, {len(dirs)} clips]", pred, true, cls, k)
        dump += [(name, p, t, int(c)) for p, t, c in zip(pred, true, cls)]

    if args.csv:
        Path(args.csv).write_text("ckpt,pred_mm,true_mm,class_id\n" +
                                  "".join(f"{n},{p:.4f},{t:.4f},{c}\n" for n, p, t, c in dump))
        print(f"\n[csv] {len(dump)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
