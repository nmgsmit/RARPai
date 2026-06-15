from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SegmentationResult:
    """Standardised output from any segmentation model."""
    mask: np.ndarray                        # [H, W] uint8, values 0 or 255
    scores: list[float] = field(default_factory=list)
    logits: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


class BaseSegmentationModel(ABC):
    """Common interface all segmentation backends must implement."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")
        self.params = cfg.get("params", {})

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Call once before inference."""
        ...

    @abstractmethod
    def predict(self, frame: np.ndarray) -> SegmentationResult:
        """
        Run inference on a single RGB frame [H, W, 3] uint8.
        Returns SegmentationResult.
        """
        ...
