"""
Segmentation pipeline: runs a segmentation model over a video.
Supports both frame-by-frame models (predict) and video models (run_on_video).
"""
from __future__ import annotations
import logging
from pathlib import Path

import cv2
import numpy as np

from src.data.video_reader import VideoReader
from src.models.segmentation.registry import build_seg_model
from src.utils.viz import VideoWriter, overlay_mask

# Register all model backends
import src.models.segmentation.sam2  # noqa: F401

log = logging.getLogger(__name__)


def run_segmentation(cfg: dict) -> None:
    data_cfg = cfg["data"]
    seg_cfg = cfg["segmentation"]
    out_cfg = cfg["output"]

    # ── Data ──────────────────────────────────────────────────────────────
    resize = data_cfg.get("resize")
    if resize is not None:
        resize = tuple(resize)

    reader = VideoReader(
        video_path=data_cfg["video_path"],
        frame_skip=data_cfg.get("frame_skip", 1),
        max_frames=data_cfg.get("max_frames"),
        resize=resize,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_seg_model(seg_cfg)

    # fp16 via torch.autocast is handled inside SAM2; no wrapper needed here
    model.load()

    # ── Output setup ──────────────────────────────────────────────────────
    out_dir = Path(out_cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fps_out = out_cfg.get("fps") or reader.fps
    h, w = reader.output_size
    alpha = out_cfg.get("overlay_alpha", 0.4)

    mask_dir = out_dir / "masks"
    if out_cfg.get("save_masks", True):
        mask_dir.mkdir(exist_ok=True)

    # ── Inference ─────────────────────────────────────────────────────────
    # SAM2 needs all frames loaded first for video propagation
    if hasattr(model, "run_on_video"):
        log.info("Loading all frames for video-mode inference...")
        frames = reader.read_all()
        log.info(f"Running SAM2 on {len(frames)} frames...")
        results = model.run_on_video(frames)

        overlay_path = out_dir / "overlay.mp4"
        with VideoWriter(overlay_path, fps=fps_out, size=(h, w)) as vw:
            for i, (frame, result) in enumerate(zip(frames, results)):
                if out_cfg.get("save_masks", True):
                    cv2.imwrite(str(mask_dir / f"frame_{i:06d}.png"), result.mask)
                if out_cfg.get("save_overlay", True):
                    vw.write(overlay_mask(frame, result.mask, alpha=alpha))

    else:
        # Frame-by-frame models
        overlay_path = out_dir / "overlay.mp4"
        with VideoWriter(overlay_path, fps=fps_out, size=(h, w)) as vw:
            for i, (frame_idx, frame) in enumerate(reader):
                result = model.predict(frame)
                if out_cfg.get("save_masks", True):
                    cv2.imwrite(str(mask_dir / f"frame_{frame_idx:06d}.png"), result.mask)
                if out_cfg.get("save_overlay", True):
                    vw.write(overlay_mask(frame, result.mask, alpha=alpha))
                if (i + 1) % 50 == 0:
                    log.info(f"  {i + 1}/{len(reader)} frames")

    log.info(f"Done. Results saved to {out_dir}")
