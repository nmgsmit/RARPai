from __future__ import annotations
import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoReader:
    """
    Iterates over a video file, yielding (frame_idx, rgb_frame) tuples.

    Args:
        video_path:  path to video file
        frame_skip:  yield every Nth frame (1 = all frames)
        max_frames:  stop after N yielded frames (None = no limit)
        resize:      (H, W) to resize each frame, or None to keep original
    """

    def __init__(
        self,
        video_path: str | Path,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,
    ):
        self.video_path = Path(video_path)
        self.frame_skip = max(1, frame_skip)
        self.max_frames = max_frames
        self.resize = resize  # (H, W)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        log.info(
            f"VideoReader: {self.video_path.name} | "
            f"{self.total_frames} frames @ {self.fps:.1f} fps | "
            f"{self.width}x{self.height}"
        )

    def read_all(self) -> list[np.ndarray]:
        """Read all (selected) frames into a list. Use for SAM2 video mode."""
        return [frame for _, frame in self]

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        cap = cv2.VideoCapture(str(self.video_path))
        yielded = 0
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.frame_skip == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.resize is not None:
                        h, w = self.resize
                        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                    yield frame_idx, rgb
                    yielded += 1
                    if self.max_frames is not None and yielded >= self.max_frames:
                        break
                frame_idx += 1
        finally:
            cap.release()

    def __len__(self) -> int:
        n = self.total_frames // self.frame_skip
        if self.max_frames is not None:
            n = min(n, self.max_frames)
        return n

    @property
    def output_size(self) -> Tuple[int, int]:
        """(H, W) of output frames."""
        return self.resize if self.resize else (self.height, self.width)
