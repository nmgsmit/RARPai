#!/usr/bin/env python
"""
Run segmentation pipeline.

Usage:
    python scripts/run_segmentation.py --config configs/experiments/sam2_video.yaml
    python scripts/run_segmentation.py --config configs/experiments/sam2_video.yaml \\
        data.video_path=data/raw/case_001.mp4 \\
        segmentation.params.prompt_frame=10 \\
        output.dir=outputs/case_001/
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.pipelines.segmentation import run_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("overrides", nargs="*",
                        help="key.path=value overrides, e.g. data.video_path=foo.mp4")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config, overrides=args.overrides)

    if not cfg["data"].get("video_path"):
        print("ERROR: data.video_path not set. Pass it as a CLI override.")
        sys.exit(1)

    run_segmentation(cfg)


if __name__ == "__main__":
    main()
