"""Black out the da Vinci HUD and GUI cues in side-by-side (SBS) stereo stills.

    python scripts/mask_sbs_gui.py --src ../data/3D_ProxyGT/proxyGTimg \
                                   --dst ../data/3D_ProxyGT/proxyGTimg_nogui

The HUD geometry (gui_mask.py) and the popup / cue-bar templates (cut_cue_clips.py) are all
pixel-calibrated on MONO 1920x1080 console output. An SBS eye is that same picture squeezed
2.125x horizontally, so each eye is un-squeezed into the mono frame with the measured
eye->mono affine (calibrate_stereo_charuco + source_crop.json), masked there by the
unchanged mono code, and the mask is warped back.

The console composites every overlay identically into both eyes (exactly +960 px), so the
two eye masks are OR-ed and applied to both: an overlay that scores under threshold in one
eye still goes if the other eye caught it.

Writes <stem>.png (GUI black, lossless) and <stem>_mask.png (255 = GUI). KEEP THE MASK:
black is also identical in both eyes, so a stereo matcher reads blacked-out GUI as a flat
surface at the screen plane (~47 mm) that passes the LR check. Invalidate depth under it.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_stereo_charuco import EYE_DX, OUT_H, OUT_W, SRC_H, SRC_W, SRC_X, SRC_Y  # noqa: E402
from cut_cue_clips import BAR_PREFIX, GUI_TEMPLATE_DIR, content_box, full_gui_mask  # noqa: E402
from gui_mask import CONNECT_TEMPLATES, REF_H, REF_W, load_templates  # noqa: E402
from rectify_mono_clips import MONO_CROP  # noqa: E402


def eye_affine(right):
    """SBS eye pixel -> raw mono 1920x1080 console pixel."""
    sx, sy = OUT_W / SRC_W, OUT_H / SRC_H
    x0 = SRC_X + (EYE_DX if right else 0.0)
    return np.float32([[sx, 0, MONO_CROP[0] - sx * x0], [0, sy, MONO_CROP[1] - sy * SRC_Y]])


def sbs_gui_mask(sbs, panels, markers, bars, m_thr, b_thr, min_bars, dilate):
    """Bool GUI mask over the whole SBS frame, identical in both eyes."""
    h, w = sbs.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for right in (False, True):
        M = eye_affine(right)
        mono = cv2.warpAffine(sbs, M, (REF_W, REF_H), flags=cv2.INTER_CUBIC)
        box = content_box(cv2.cvtColor(mono, cv2.COLOR_BGR2GRAY))
        mm = full_gui_mask(mono, panels, markers, bars, box, m_thr, b_thr, min_bars, dilate)
        # LINEAR then > 0: any SBS pixel touched by the mono mask counts -> errs toward masking
        m |= cv2.warpAffine(mm.astype(np.uint8) * 255, M, (w, h),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    m = m > 0
    return m | np.roll(m, int(EYE_DX), axis=1)   # w == 2*EYE_DX, so the roll swaps the eyes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="directory of 1920x1080 SBS .jpg/.png stills")
    p.add_argument("--dst", required=True)
    p.add_argument("--templates", default="data/templates/Move_Que")
    # same defaults as cut_cue_clips.py (chosen there with sweep_cue_thresholds.py)
    p.add_argument("--marker-thresh", type=float, default=0.55)
    p.add_argument("--bar-thresh", type=float, default=0.90)
    p.add_argument("--min-bars", type=int, default=4)
    p.add_argument("--dilate", type=int, default=3, help="px (mono) to grow the mask")
    a = p.parse_args()

    allt = {os.path.splitext(os.path.basename(f))[0]: cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            for f in sorted(glob.glob(os.path.join(a.templates, "*.png")))}
    markers = {k: v for k, v in allt.items() if not k.startswith(BAR_PREFIX)}
    bars = {k: v for k, v in allt.items() if k.startswith(BAR_PREFIX)}
    panels = {k: v for k, v in load_templates(GUI_TEMPLATE_DIR).items()
              if k not in CONNECT_TEMPLATES}
    if not markers or not bars or not panels:
        raise SystemExit(f"missing templates: markers {len(markers)} bars {len(bars)} "
                         f"panels {len(panels)} (run from code/ or pass --templates)")

    os.makedirs(a.dst, exist_ok=True)
    paths = sorted(sum([glob.glob(os.path.join(a.src, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []))
    for f in paths:
        img = cv2.imread(f)
        if img is None or img.shape[:2] != (REF_H, 2 * int(EYE_DX)):
            print(f"  skip {os.path.basename(f)} (not a 1920x1080 SBS frame)")
            continue
        m = sbs_gui_mask(img, panels, markers, bars, a.marker_thresh, a.bar_thresh,
                         a.min_bars, a.dilate)
        img[m] = 0
        stem = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(a.dst, stem + ".png"), img)
        cv2.imwrite(os.path.join(a.dst, stem + "_mask.png"), m.astype(np.uint8) * 255)
        print(f"  {stem}  masked {100 * m.mean():.1f}%")
    print(f"{len(paths)} frames -> {a.dst}")


def _self_test():
    # the eye rectangle lands exactly on the mono 1340x1072 crop, for both eyes
    for right in (False, True):
        M = eye_affine(right)
        x0 = SRC_X + (EYE_DX if right else 0.0)
        tl = M @ [x0, SRC_Y, 1]
        br = M @ [x0 + SRC_W, SRC_Y + SRC_H, 1]
        assert np.allclose(tl, MONO_CROP, atol=1e-3), tl
        assert np.allclose(br, (MONO_CROP[0] + OUT_W, MONO_CROP[1] + OUT_H), atol=1e-3), br
    # grey frame, no templates: the fixed HUD bar must be masked in BOTH eyes, tissue not
    sbs = np.zeros((REF_H, REF_W, 3), np.uint8)
    sbs[40:1040, 170:790] = sbs[40:1040, 1130:1750] = 120
    m = sbs_gui_mask(sbs, {}, {}, {}, 0.55, 0.9, 4, 3)
    hud_y = int(SRC_Y + (REF_H - 20 - MONO_CROP[1]) * SRC_H / OUT_H)   # 20 px into the mono bar
    assert m[hud_y, 400] and m[hud_y, 1360], "HUD bar not masked in both eyes"
    assert not m[500, 400] and not m[500, 1360], "tissue masked"
    print("ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
