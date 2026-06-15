"""
SAM2 video segmentation backend.

Mode: prompted video propagation.
- You provide one or more point/box prompts on a chosen frame.
- SAM2 propagates the mask through the entire video.

Install:
    pip install sam2
    # or: pip install git+https://github.com/facebookresearch/sam2.git

Checkpoints (auto-downloaded from HuggingFace on first run):
    facebook/sam2-hiera-tiny    – fastest
    facebook/sam2-hiera-small
    facebook/sam2-hiera-base-plus
    facebook/sam2-hiera-large   – most accurate (default)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import numpy as np

from src.models.segmentation.base import BaseSegmentationModel, SegmentationResult
from src.models.segmentation.registry import register_seg_model

log = logging.getLogger(__name__)


@register_seg_model("sam2")
class SAM2VideoSegmentation(BaseSegmentationModel):
    """
    SAM2 prompted video segmentation.

    Config params (under segmentation.params):
        hf_model_id (str):    HuggingFace model ID (default: facebook/sam2-hiera-large)
        prompt_frame (int):   frame index to place the prompt on (default: 0)
        points (list):        [[x, y], ...] clicked points in prompt frame
        point_labels (list):  [1, 0, ...] — 1=foreground, 0=background
        box (list):           [x1, y1, x2, y2] bounding box prompt (alternative to points)
        obj_id (int):         object ID to track (default: 1)
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.hf_model_id: str = self.params.get("hf_model_id", "facebook/sam2-hiera-large")
        self.prompt_frame: int = self.params.get("prompt_frame", 0)
        self.points: Optional[list] = self.params.get("points", None)
        self.point_labels: Optional[list] = self.params.get("point_labels", None)
        self.box: Optional[list] = self.params.get("box", None)
        self.obj_id: int = self.params.get("obj_id", 1)
        self._predictor = None

    def load(self) -> None:
        try:
            import torch
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError as e:
            raise ImportError(
                "SAM2 not installed. Run: pip install sam2"
            ) from e

        log.info(f"Loading SAM2 from {self.hf_model_id} on {self.device}")
        import torch
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        self._predictor = SAM2VideoPredictor.from_pretrained(self.hf_model_id)
        self._predictor = self._predictor.to(self.device)
        log.info("SAM2 loaded.")

    def predict(self, frame: np.ndarray) -> SegmentationResult:
        """
        Single-frame predict — not the primary use case for SAM2.
        For video use run_on_video() instead.
        Runs image predictor on a single frame using stored prompts.
        """
        raise NotImplementedError(
            "SAM2 is designed for video. Use run_on_video() instead of predict()."
        )

    def run_on_video(self, frames: list[np.ndarray]) -> list[SegmentationResult]:
        """
        Run SAM2 on a list of RGB frames.

        Prompts (points or box) are applied to self.prompt_frame,
        then propagated through all frames.

        Args:
            frames: list of uint8 RGB [H, W, 3] frames

        Returns:
            list of SegmentationResult, one per frame
        """
        assert self._predictor is not None, "Call .load() first"

        import torch
        import tempfile, os
        import cv2

        # SAM2 video predictor expects frames as a directory of JPEGs
        # Write frames to a temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(tmpdir, f"{i:06d}.jpg"), bgr)

            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                state = self._predictor.init_state(video_path=tmpdir)

                # Add prompts on the chosen frame
                prompt_kwargs = dict(
                    inference_state=state,
                    frame_idx=self.prompt_frame,
                    obj_id=self.obj_id,
                )
                if self.points is not None:
                    prompt_kwargs["points"] = np.array(self.points, dtype=np.float32)
                    prompt_kwargs["labels"] = np.array(
                        self.point_labels if self.point_labels else [1] * len(self.points),
                        dtype=np.int32,
                    )
                if self.box is not None:
                    prompt_kwargs["box"] = np.array(self.box, dtype=np.float32)

                self._predictor.add_new_points_or_box(**prompt_kwargs)

                # Propagate through all frames
                results = [None] * len(frames)
                for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(state):
                    # mask_logits: [n_objs, 1, H, W]
                    mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                    results[frame_idx] = SegmentationResult(
                        mask=mask,
                        scores=[],
                        metadata={"frame_idx": frame_idx, "obj_ids": obj_ids},
                    )

        # Fill any gaps with empty masks
        h, w = frames[0].shape[:2]
        for i in range(len(results)):
            if results[i] is None:
                results[i] = SegmentationResult(
                    mask=np.zeros((h, w), dtype=np.uint8)
                )

        return results
