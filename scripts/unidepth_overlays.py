"""UniDepth V2 metric depth for a folder of mono frames, written next to the images.

Per image: `<stem>_unidepth.npz` (depth mm float32 over the auto-cropped content frame, key
'depth') and `<stem>_unidepth_overlay.png` (frame | turbo depth | blend). Same auto-crop and K
as gui_depth_measure, so `gui_depth_measure.py --ckpt unidepth` measures on exactly these maps.

    python scripts/unidepth_overlays.py --dir ../data/others_ruler
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gui_depth_measure import DEFAULT_K_NORM, IMG_EXTS, auto_bars, crop_box  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--model", default="lpiccinelli/unidepth-v2-vitl14")
ap.add_argument("--intrinsics", type=float, nargs=4, default=list(DEFAULT_K_NORM))
args = ap.parse_args()

from unidepth.models import UniDepthV2  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = UniDepthV2.from_pretrained(args.model).to(dev).eval()

for p in sorted(Path(args.dir).iterdir()):
    if p.suffix.lower() not in IMG_EXTS or "_unidepth" in p.stem:
        continue
    full = Image.open(p).convert("RGB")
    fracs = auto_bars(np.asarray(full))
    crop = np.asarray(full.crop(crop_box(full.size, *fracs)))
    h, w = crop.shape[:2]
    fx, fy, cx, cy = args.intrinsics
    K = torch.tensor([[fx * w, 0, cx * w], [0, fy * h, cy * h], [0, 0, 1]], dtype=torch.float32)
    rgb = torch.from_numpy(crop).permute(2, 0, 1)
    depth = model.infer(rgb.to(dev), K.to(dev))["depth"][0, 0].float().cpu().numpy() * 1000.0
    np.savez(p.with_name(p.stem + "_unidepth.npz"), depth=depth.astype(np.float32),
             fracs=np.array(fracs))

    inv = 1.0 / depth
    lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
    col = cv2.applyColorMap((np.clip((inv - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    col = cv2.resize(col, (w, h))
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    panel = np.hstack([bgr, col, cv2.addWeighted(bgr, 0.5, col, 0.5, 0)])
    cv2.putText(panel, f"UniDepthV2  median {np.median(depth):.0f} mm  range {depth.min():.0f}-{depth.max():.0f}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.imwrite(str(p.with_name(p.stem + "_unidepth_overlay.png")), panel)
    print(f"{p.name}: median {np.median(depth):.0f} mm")
