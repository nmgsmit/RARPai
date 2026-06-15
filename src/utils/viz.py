from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a binary mask onto an RGB frame."""
    out = frame.copy()
    mask_bool = mask > 0
    out[mask_bool] = (
        (1 - alpha) * frame[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2 uses BGR, frame is RGB — flip colour
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
