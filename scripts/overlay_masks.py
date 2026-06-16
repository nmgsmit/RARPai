"""
Load a fine-tuned segmentation model and overlay predicted masks on a raw video.

Usage:
    python scripts/overlay_masks.py \
        --video ../data/raw/RARP_voorbeeld_A.mp4 \
        --checkpoint outputs/rarp_higherlr/best.pth \
        --output outputs/RARP_voorbeeld_A_masked.mp4 \
        --img-size 512

Output video has catheter (class 1) in green, urethra (class 3) in blue, alpha=0.5.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "surgenet"))
from metaformer import MetaFormerFPN

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# class -> (B, G, R, alpha) for overlay
CLASS_COLORS = {
    1: (0, 255, 0, 0.5),  # catheter: green
    3: (255, 0, 0, 0.5),  # urethra: blue
}


def load_model(checkpoint_path, device):
    model = MetaFormerFPN(num_classes=4, pretrained="ImageNet", pretrained_weights=None).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[model] loaded {checkpoint_path}")
    return model


@torch.no_grad()
def segment_frame(frame_bgr, model, img_size, device):
    """
    Segment a single frame (BGR numpy array) and return mask (HxW, class ids).
    """
    h, w = frame_bgr.shape[:2]
    # RGB, resize, normalize
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((img_size, img_size), Image.BICUBIC)
    x = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = x.unsqueeze(0).to(device)
    # infer
    logits = model(x)
    pred = logits.argmax(1)[0].cpu().numpy()  # (H_model, W_model)
    # resize back to original frame size
    pred_resized = cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return pred_resized


def overlay_mask(frame_bgr, mask, class_colors=CLASS_COLORS):
    """
    Overlay class masks on frame using specified colors and alpha blending.
    """
    overlay = frame_bgr.copy()
    for class_id, (b, g, r, alpha) in class_colors.items():
        class_mask = (mask == class_id).astype(np.uint8) * 255
        # blend: overlay[mask] = (1-alpha)*frame + alpha*color
        for c, color_val in enumerate([b, g, r]):
            overlay[..., c] = np.where(
                class_mask > 0,
                (1 - alpha) * frame_bgr[..., c] + alpha * color_val,
                overlay[..., c]
            )
    return overlay.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--checkpoint", required=True, help="model checkpoint path")
    ap.add_argument("--output", required=True, help="output video path")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--num-classes", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}")

    model = load_model(args.checkpoint, device)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # fourcc for MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"[video] {num_frames} frames @ {fps:.1f} fps, {width}x{height}")

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        mask = segment_frame(frame, model, args.img_size, device)
        overlay = overlay_mask(frame, mask)
        out.write(overlay)
        if (i + 1) % 30 == 0:
            print(f"[progress] {i+1}/{num_frames}")

    cap.release()
    out.release()
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
