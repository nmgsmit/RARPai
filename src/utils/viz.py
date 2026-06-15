from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


# Distinct RGB colours per class (index 0 = background, skipped)
_CLASS_COLORS = np.array([
    [  0,   0,   0],  # 0 background — transparent
    [  0, 255,   0],  # 1 green
    [255,   0,   0],  # 2 red
    [  0, 128, 255],  # 3 blue
    [255, 255,   0],  # 4 yellow
    [255,   0, 255],  # 5 magenta
    [  0, 255, 255],  # 6 cyan
    [255, 128,   0],  # 7 orange
    [128,   0, 255],  # 8 purple
    [  0, 255, 128],  # 9 mint
    [255,   0, 128],  # 10 pink
    [128, 255,   0],  # 11 lime
    [  0, 128, 128],  # 12 teal
], dtype=np.uint8)


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a binary mask (0/255) onto an RGB frame."""
    out = frame.copy()
    mask_bool = mask > 0
    out[mask_bool] = (
        (1 - alpha) * frame[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color[::-1], 2)
    return out


def overlay_mask_multiclass(
    frame: np.ndarray,
    class_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a per-pixel class map (H×W int, 0=background) onto an RGB frame."""
    out = frame.copy()
    colors = _CLASS_COLORS
    if class_map.max() >= len(colors):
        # extend palette by cycling if more classes than predefined colors
        extra = class_map.max() + 1 - len(colors)
        colors = np.vstack([colors, np.random.randint(50, 255, (extra, 3), dtype=np.uint8)])
    for cls_id in np.unique(class_map):
        if cls_id == 0:
            continue
        mask = (class_map == cls_id).astype(np.uint8)
        color = colors[cls_id].tolist()
        out[mask > 0] = (
            (1 - alpha) * frame[mask > 0] + alpha * np.array(color)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color[::-1], 2)
    return out


class VideoWriter:
    """Context manager for writing RGB frames to an mp4."""

    def __init__(self, path: str | Path, fps: float, size: Tuple[int, int]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.h, self.w = size
        self._writer: Optional[cv2.VideoWriter] = None

    def __enter__(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.w, self.h))
        return self

    def write(self, frame_rgb: np.ndarray) -> None:
        self._writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def __exit__(self, *_):
        if self._writer:
            self._writer.release()
