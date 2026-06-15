"""
Run finetuned MetaFormerFPN segmentation on a video, output overlay video.

Usage:
    python scripts/run_rarp_seg.py \
        --video data/raw/RARP_voorbeeld_A.mp4 \
        --checkpoint outputs/rarp_finetune/best.pth \
        --num-classes 12
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "surgenet"))

from metaformer import MetaFormerFPN  # noqa: E402
from src.utils.viz import VideoWriter, overlay_mask_multiclass  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def preprocess(frame_rgb: np.ndarray, img_size: int) -> torch.Tensor:
    img = Image.fromarray(frame_rgb).resize((img_size, img_size), Image.BILINEAR)
    x = np.array(img).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", default="outputs/rarp_finetune/best.pth")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--out", default="outputs/rarp_seg/overlay.mp4")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MetaFormerFPN(num_classes=args.num_classes, pretrained="ImageNet", pretrained_weights=None)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    print(f"[loaded] {args.checkpoint} | device={device}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoWriter(out_path, fps=fps, size=(h, w)) as vw:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = preprocess(frame_rgb, args.img_size).to(device)
            with torch.no_grad():
                logits = model(x)  # [1, C, H, W] — already at input resolution
            class_map = logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
            # resize class map back to original frame size for overlay
            class_map = cv2.resize(class_map, (w, h), interpolation=cv2.INTER_NEAREST)
            vw.write(overlay_mask_multiclass(frame_rgb, class_map, alpha=args.alpha))
            i += 1
            if i % 50 == 0:
                print(f"  {i} frames", flush=True)

    cap.release()
    print(f"[done] {i} frames -> {out_path}")


if __name__ == "__main__":
    main()
