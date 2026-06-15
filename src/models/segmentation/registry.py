from __future__ import annotations
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.segmentation.base import BaseSegmentationModel

_REGISTRY: dict[str, Type["BaseSegmentationModel"]] = {}


def register_seg_model(name: str):
    """Class decorator: @register_seg_model('sam2')"""
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Segmentation model '{name}' already registered.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_seg_model(seg_cfg: dict) -> "BaseSegmentationModel":
    """Instantiate model from config. Call .load() before use."""
    name = seg_cfg.get("model")
    if not name:
        raise ValueError("segmentation.model must be set in config.")
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](seg_cfg)
