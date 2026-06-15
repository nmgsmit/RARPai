"""
Config loader: base.yaml → experiment yaml → CLI overrides.
CLI overrides use dot-notation: data.video_path=/path/to/video.mp4
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict:
    """
    Load config, resolve _base_ inheritance, apply CLI overrides.

    Args:
        config_path: path to experiment yaml
        overrides:   ['key.subkey=value', ...] from CLI

    Returns:
        Merged config dict
    """
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve _base_
    if "_base_" in cfg:
        base_path = (config_path.parent / cfg.pop("_base_")).resolve()
        base = load_config(base_path)
        cfg = _deep_merge(base, cfg)

    # Apply CLI overrides
    for override in (overrides or []):
        if "=" not in override:
            raise ValueError(f"Override must be 'key.path=value', got: {override!r}")
        key_path, raw_val = override.split("=", 1)
        val: Any = yaml.safe_load(raw_val)
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

    return cfg
