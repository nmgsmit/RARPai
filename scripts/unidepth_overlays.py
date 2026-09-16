"""UniDepth V2 metric depth for a folder of mono frames, written next to the images.

Per image: `<stem>_unidepth.npz` (depth mm float32 over the auto-cropped content frame, key
'depth') and `<stem>_unidepth_overlay.png` (frame | turbo depth | blend). Same auto-crop and K
as gui_depth_measure, so `gui_depth_measure.py --ckpt unidepth` measures on exactly these maps.

    python scripts/unidepth_overlays.py --dir ../data/others_ruler
    python scripts/unidepth_overlays.py --dir ../data/others_ruler --calib outputs/metric_calib_proxy/results.json

--calib: apply the frozen ruler-set calibration of metric_calib_proxy (1/z_mm = s/z_m + b, or
z_mm = z_m/s_only with --calib-mode scale). That fit ran UniDepth WITHOUT K, so K is dropped here too.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gui_depth_measure import DEFAULT_K_NORM, IMG_EXTS, auto_bars, crop_box  # noqa: E402


def overlay_panel(crop_rgb, depth, name):
    """frame | turbo inverse depth (near = red, 2-98 pct) | blend, with a label. BGR uint8."""
    h, w = crop_rgb.shape[:2]
    inv = 1.0 / depth
    lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
    col = cv2.applyColorMap((np.clip((inv - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    col = cv2.resize(col, (w, h))
    bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    panel = np.hstack([bgr, col, cv2.addWeighted(bgr, 0.5, col, 0.5, 0)])
    cv2.putText(panel, f"{name}  median {np.median(depth):.0f} mm  range {depth.min():.0f}-{depth.max():.0f}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return panel


def frames(d, pattern="*"):
    """Source frames only: skip the npz/overlay/compare files these scripts write next to them.
    `pattern` narrows a folder holding by-products too (proxy-GT: --glob "*_left.png")."""
    return [p for p in sorted(Path(d).glob(pattern))
            if p.suffix.lower() in IMG_EXTS and "_unidepth" not in p.stem and "_endodac" not in p.stem]


def main():
    import torch
    from PIL import Image
    from unidepth.models import UniDepthV2

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default="lpiccinelli/unidepth-v2-vitl14")
    ap.add_argument("--intrinsics", type=float, nargs=4, default=list(DEFAULT_K_NORM))
    ap.add_argument("--calib", help="metric_calib_proxy results.json -> ruler-calibrated mm")
    ap.add_argument("--calib-mode", choices=["affine", "scale"], default="affine")
    ap.add_argument("--glob", default="*", help='only these files, e.g. "*_left.png" in a proxy-GT dir')
    ap.add_argument("--no-crop", action="store_true",
                    help="frames are already the content frame (rectified stereo left): no auto-crop, "
                         "so the map covers the same pixels as the stereo depth16")
    args = ap.parse_args()

    cal = json.loads(Path(args.calib).read_text())["unidepth_v2_vitl"]["calibration"] if args.calib else None
    if cal:
        print(f"ruler calibration ({args.calib_mode}): {cal}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UniDepthV2.from_pretrained(args.model).to(dev).eval()

    for p in frames(args.dir, args.glob):
        full = Image.open(p).convert("RGB")
        fracs = (0.0, 0.0, 0.0) if args.no_crop else auto_bars(np.asarray(full))
        crop = np.asarray(full.crop(crop_box(full.size, *fracs)))
        h, w = crop.shape[:2]
        fx, fy, cx, cy = args.intrinsics
        K = torch.tensor([[fx * w, 0, cx * w], [0, fy * h, cy * h], [0, 0, 1]], dtype=torch.float32)
        rgb = torch.from_numpy(crop).permute(2, 0, 1)
        if cal:                                   # same inference as the calibration fit: no K, metres
            z = model.infer(rgb.to(dev))["depth"][0, 0].float().cpu().numpy()
            depth = (z / cal["s_only"] if args.calib_mode == "scale"
                     else 1.0 / np.clip(cal["s"] / np.clip(z, 1e-6, None) + cal["b"], 1e-6, None))
        else:
            depth = model.infer(rgb.to(dev), K.to(dev))["depth"][0, 0].float().cpu().numpy() * 1000.0
        np.savez(p.with_name(p.stem + "_unidepth.npz"), depth=depth.astype(np.float32),
                 fracs=np.array(fracs), calib=args.calib_mode if cal else "none")
        cv2.imwrite(str(p.with_name(p.stem + "_unidepth_overlay.png")),
                    overlay_panel(crop, depth, f"UniDepthV2 ruler-cal {args.calib_mode}" if cal else "UniDepthV2"))
        print(f"{p.name}: median {np.median(depth):.0f} mm")


if __name__ == "__main__":
    main()
