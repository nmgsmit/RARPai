"""EndoDAC overlays next to the UniDepth ones, for a side-by-side look. Runs on CPU (laptop).

Per frame: `<stem>_endodac_overlay.png` (same layout as `_unidepth_overlay.png`) and, when the
UniDepth npz exists, `<stem>_endodac_vs_unidepth.png` = EndoDAC row stacked over UniDepth row.
Same auto-crop as the GUI; each colour bar is stretched per map, so compare SHAPE by colour and
scale by the printed mm.

    python scripts/endodac_overlays.py --dir ..\\OTHERS\\GOODrulerimg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gui_depth_measure import (DEFAULT_CKPT, DEFAULT_MAX_DEPTH, DEFAULT_MIN_DEPTH,  # noqa: E402
                               DEFAULT_SHAPE, DepthBackend, auto_bars, crop_box)
from unidepth_overlays import frames, overlay_panel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
args = ap.parse_args()

be = DepthBackend(args.ckpt, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH, DEFAULT_MAX_DEPTH).load()
for p in frames(args.dir):
    full = Image.open(p).convert("RGB")
    crop = full.crop(crop_box(full.size, *auto_bars(np.asarray(full))))
    endo = overlay_panel(np.asarray(crop), be.predict(crop), "EndoDAC")
    cv2.imwrite(str(p.with_name(p.stem + "_endodac_overlay.png")), endo)
    npz = p.with_name(p.stem + "_unidepth.npz")
    if npz.is_file():
        with np.load(npz) as z:
            uni = overlay_panel(np.asarray(crop), z["depth"], "UniDepthV2")
        cv2.imwrite(str(p.with_name(p.stem + "_endodac_vs_unidepth.png")), np.vstack([endo, uni]))
    print(p.name)
