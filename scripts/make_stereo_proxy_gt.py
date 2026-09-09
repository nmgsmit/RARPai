"""Metric depth proxy-GT from the da Vinci 3D (side-by-side) clips.

    SBS frame -> rectified left/right (ONE resample) -> disparity -> Z_mm = f*B / d

Output lands in the SAME 1340x1072 rectified frame as the mono clips produced by
rectify_mono_clips.py, so proxy GT and training images share one camera model.

WHY ONE RESAMPLE. The naive route resamples twice -- an affine un-squeeze to 1340x1072,
then the rectification remap -- and each cubic pass softens the image. Sub-pixel disparity
is exactly what that blur costs. initUndistortRectifyMap gives, per rectified pixel, a
location in the 1340x1072 eye frame; pushing that back through the inverse eye_to_mono
affine gives a location in the original 1920x1080 SBS frame. Sample there directly and the
anamorphic squeeze comes along for free. See `rect_maps()`.

WHY RECTIFY AT ALL. Matchers search along image ROWS on the assumption that epipolar lines
are horizontal, and Z = f*B/d is only valid in rectified coordinates. Measured, not
theoretical: disparity depth off raw coordinates was 9% biased; rectified, 0.8%.

    python scripts/make_stereo_proxy_gt.py --preview out.png --video ../data/3D_ProxyGT/<clip>.mp4
    python scripts/make_stereo_proxy_gt.py --src ../data/3D_ProxyGT --dst ../data/processed/proxy_gt

Depth is stored as uint16 PNG in 1/16 mm (0 = invalid), which is ~0.06mm quantisation
against ~1mm calibration accuracy, and a fraction of the size of float arrays.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_stereo_charuco import (OUT_W, OUT_H, SRC_X, SRC_Y, SRC_W, SRC_H, EYE_DX,
                                      BANNER_ROW)

DEPTH_SCALE = 16.0          # stored value = mm * DEPTH_SCALE


def rect_maps(calib, right):
    """Composed map: rectified 1340x1072 pixel -> pixel in the raw 1920x1080 SBS frame.

    Returns (mapx, mapy, valid). `valid` is False where the rectified pixel would come from
    outside the eye's source rectangle or from the GUI banner -- the banner is composited
    identically into both eyes, so it has zero disparity by construction and would read as a
    flat wall at the convergence plane if a matcher were allowed to see it.
    """
    s = "R" if right else "L"
    K, D = np.array(calib["K" + s]), np.array(calib["D" + s])
    R = np.array(calib["R2" if right else "R1"])
    P = np.array(calib["P2" if right else "P1"])
    ex, ey = cv2.initUndistortRectifyMap(K, D, R, P, (OUT_W, OUT_H), cv2.CV_32FC1)
    valid = (ex >= 0) & (ex < OUT_W) & (ey >= 0) & (ey < BANNER_ROW)
    sx, sy = OUT_W / SRC_W, OUT_H / SRC_H
    x0 = SRC_X + (EYE_DX if right else 0.0)
    return (ex / sx + x0).astype(np.float32), (ey / sy + SRC_Y).astype(np.float32), valid


def specular_mask(img, max_blob=2000):
    """Blown-out highlights are view-dependent: they sit at different scene points in the two
    eyes, so any disparity a matcher reports there is meaningless.

    SMALL blobs only. "bright and desaturated" also describes a white da Vinci instrument
    shaft, and masking those threw away the nearest objects in frame with the sharpest depth
    discontinuities -- the most valuable supervision in the dataset. Measured over the SBS
    stills (1849 components): genuine highlights have median area 10 px and p95 128 px, while
    instrument blobs run 5k-32k, so the two are cleanly separable by size. 2000 sits above
    every plausible highlight, and anything ambiguous stays masked; the left-right check is
    the real safety net either way, since a view-dependent highlight fails it anyway.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bright = ((hsv[:, :, 2] > 240) & (hsv[:, :, 1] < 40)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < max_blob      # 0 is background
    return small[lab]


def make_sgbm(min_disp, num_disp, block=5):
    return cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=block,
        P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=150, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def sgbm_matcher(left, right, min_disp, num_disp):
    """ponytail: SGBM, no new dependency. Known ceiling: textureless blood-covered tissue,
    where a cost volume has nothing to lock onto -- measured 58% valid. `ffs` lifts that."""
    return make_sgbm(min_disp, num_disp).compute(left, right).astype(np.float32) / 16.0


def ffs_matcher(root, model_path, iters, max_disp):
    """C-Fast-FoundationStereo (NVIDIA, CVPR 2026). The checkpoint is a SERIALIZED nn.Module,
    not a state_dict, so `core` must be importable when it unpickles -- hence sys.path."""
    import torch
    sys.path.insert(0, os.path.expanduser(root))
    from core.utils.utils import InputPadder

    model = torch.load(os.path.expanduser(model_path), map_location="cpu", weights_only=False)
    # Version skew: the published checkpoint's pickled args predate keys the repo HEAD now
    # reads. Fill from the repo's OWN function defaults (build_gwc_volume_*(normalize=True))
    # rather than guessing -- a wrong value here changes the cost volume silently.
    for key, default in (("normalize", True),):
        if key not in model.args:
            print("  model.args missing %r, using repo default %r" % (key, default))
            model.args[key] = default
    model.args.valid_iters = iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    torch.autograd.set_grad_enabled(False)

    def run(left, right, min_disp, num_disp):
        h, w = left.shape[:2]
        # cv2 gives BGR; the model was trained on RGB (imageio).
        to_t = (lambda im: torch.as_tensor(np.ascontiguousarray(im[..., ::-1]))
                .cuda().float()[None].permute(0, 3, 1, 2))
        a, b = to_t(left), to_t(right)
        padder = InputPadder(a.shape, divis_by=32, force_square=False)
        a, b = padder.pad(a, b)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            d = model.forward(a, b, iters=iters, test_mode=True,
                              optimize_build_volume="pytorch1")
        return padder.unpad(d.float()).cpu().numpy().reshape(h, w)

    return run


def lr_check(dl, dr, min_disp, tol=1.5):
    """Keep only pixels where matching left->right and right->left agree. For a learned
    matcher this is what keeps the monocular prior OUT of the supervision: the network will
    happily hallucinate plausible depth on textureless tissue, and only genuine stereo
    correspondence survives agreeing in both directions."""
    h, w = dl.shape
    xs = np.arange(w)[None, :].repeat(h, 0)
    src = np.clip(xs - dl, 0, w - 1).astype(np.int32)
    agree = np.abs(dl - dr[np.arange(h)[:, None], src]) <= tol
    dl = dl.copy()
    dl[~agree | (dl <= min_disp)] = np.nan
    return dl


def match(matcher, left, right, min_disp, num_disp, scale=1.0):
    """-> disparity in FULL-resolution rectified px, NaN where the left-right check fails.
    The right pass uses the flip trick, so it needs nothing from opencv-contrib."""
    if scale != 1.0:
        sz = (int(OUT_W * scale), int(OUT_H * scale))
        left, right = cv2.resize(left, sz), cv2.resize(right, sz)
        min_disp, num_disp = int(min_disp * scale), int(round(num_disp * scale / 16)) * 16
    dl = matcher(left, right, min_disp, num_disp)
    dr = matcher(np.ascontiguousarray(right[:, ::-1]),
                 np.ascontiguousarray(left[:, ::-1]), min_disp, num_disp)[:, ::-1]
    dl = lr_check(dl, dr, min_disp)
    if scale != 1.0:
        dl = cv2.resize(dl, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST) / scale
    return dl


def frame_depth(frame, maps, cal, min_disp, num_disp, scale, matcher=sgbm_matcher):
    (mxL, myL, vL), (mxR, myR, vR) = maps
    left = cv2.remap(frame, mxL, myL, cv2.INTER_CUBIC)
    right = cv2.remap(frame, mxR, myR, cv2.INTER_CUBIC)
    disp = match(matcher, left, right, min_disp, num_disp, scale)
    disp[~(vL & vR)] = np.nan
    disp[specular_mask(left)] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        z = cal["f_times_B_rectified"] / disp
    z[~np.isfinite(z) | (z <= 0)] = 0.0
    return left, disp, z


def colorize(a, lo, hi, cmap=cv2.COLORMAP_MAGMA):
    v = np.clip((np.nan_to_num(a) - lo) / (hi - lo + 1e-9), 0, 1)
    out = cv2.applyColorMap((v * 255).astype(np.uint8), cmap)
    out[~np.isfinite(a) | (np.nan_to_num(a) == 0)] = 0
    return out


def preview(frame, maps, cal, min_disp, num_disp, scale, out, matcher=sgbm_matcher):
    left, disp, z = frame_depth(frame, maps, cal, min_disp, num_disp, scale, matcher)
    ok = z > 0
    lo, hi = np.percentile(z[ok], [5, 95]) if ok.any() else (0, 1)
    tiles = []
    for panel, label in (
            (left, "rectified LEFT eye"),
            # inverse depth so NEAR = bright, matching gui_depth_measure.colorize
            (colorize(np.where(ok, 1.0 / np.maximum(z, 1e-6), np.nan), 1 / hi, 1 / lo),
             "metric depth %.0f-%.0f mm  (bright = near)" % (lo, hi)),
            (np.dstack([(ok * 255).astype(np.uint8)] * 3),
             "valid  %.1f%% of frame" % (100 * ok.mean()))):
        t = cv2.resize(panel, (OUT_W // 2, OUT_H // 2))
        t = cv2.copyMakeBorder(t, 44, 10, 10, 10, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(t, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite(out, np.hstack(tiles))
    print("wrote %s" % out)
    if ok.any():
        print("valid %.1f%%  depth p5 %.0f  median %.0f  p95 %.0f mm"
              % (100 * ok.mean(), lo, np.median(z[ok]), hi))


def depth_jpg(left, z, path, name=""):
    """Side-by-side rectified image + metric depth with a mm scale bar. Depth alone is hard
    to read; the reference image is what makes it analysable."""
    ok = z > 0
    if not ok.any():
        return
    lo, hi = np.percentile(z[ok], [2, 98])
    # inverse depth so NEAR = bright, matching gui_depth_measure.colorize
    heat = colorize(np.where(ok, 1.0 / np.maximum(z, 1e-6), np.nan), 1 / hi, 1 / lo)
    bar_h = 34
    grad = np.linspace(1 / lo, 1 / hi, OUT_W - 200)[None, :].repeat(bar_h, 0)
    bar = colorize(grad, 1 / hi, 1 / lo)
    bar = cv2.copyMakeBorder(bar, 6, 24, 100, 100, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        mm = 1.0 / (1 / lo + frac * (1 / hi - 1 / lo))
        x = int(100 + frac * (OUT_W - 200))
        cv2.putText(bar, "%.0f" % mm, (x - 14, bar_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, "mm", (OUT_W - 88, bar_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    heat = np.vstack([heat, cv2.resize(bar, (OUT_W, bar.shape[0]))])
    left = np.vstack([left, np.full((bar.shape[0], OUT_W, 3), 20, np.uint8)])
    tiles = []
    for panel, label in ((left, "rectified LEFT   %s" % name),
                         (heat, "metric depth  %.0f-%.0f mm  (bright = near)  valid %.1f%%"
                          % (lo, hi, 100 * ok.mean()))):
        t = cv2.copyMakeBorder(panel, 44, 10, 10, 10, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(t, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite(path, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return lo, hi, float(ok.mean())


def run_images(paths, dst, maps, cal, min_disp, num_disp, scale, matcher):
    """Same pipeline as run_video, for individual SBS frames pasted in as PNGs."""
    os.makedirs(dst, exist_ok=True)
    for p in paths:
        fr = cv2.imread(p)
        if fr is None or fr.shape[1] != 1920:
            print("  skip %s (not a 1920x1080 SBS frame)" % os.path.basename(p))
            continue
        left, _, z = frame_depth(fr, maps, cal, min_disp, num_disp, scale, matcher)
        stem = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(dst, stem + "_depth16.png"),
                    np.clip(z * DEPTH_SCALE, 0, 65535).astype(np.uint16))
        r = depth_jpg(left, z, os.path.join(dst, stem + ".jpg"), stem[-12:])
        good = z > 0
        print("  %-58s valid %5.1f%%  depth p2 %.0f  med %.0f  p98 %.0f mm"
              % (stem[-58:], 100 * good.mean(),
                 *(np.percentile(z[good], [2, 50, 98]) if good.any() else (0, 0, 0))))


def run_video(video, dst, maps, cal, stride, min_disp, num_disp, scale, matcher):
    name = os.path.splitext(os.path.basename(video))[0]
    for sub in ("images", "depth"):
        os.makedirs(os.path.join(dst, name, sub), exist_ok=True)
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kept, fracs = 0, []
    for i in range(0, n, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        left, _, z = frame_depth(fr, maps, cal, min_disp, num_disp, scale, matcher)
        good = z > 0
        fracs.append(float(good.mean()))
        cv2.imwrite(os.path.join(dst, name, "images", "%07d.png" % i), left)
        cv2.imwrite(os.path.join(dst, name, "depth", "%07d.png" % i),
                    np.clip(z * DEPTH_SCALE, 0, 65535).astype(np.uint16))
        kept += 1
    cap.release()
    f = np.array(fracs) if fracs else np.zeros(1)
    with open(os.path.join(dst, name, "proxy_gt.json"), "w") as fh:
        json.dump(dict(video=os.path.basename(video), frames=kept, stride=stride,
                       matcher=getattr(matcher, "__name__", "ffs"),
                       depth_scale=DEPTH_SCALE, depth_units="mm", invalid=0,
                       f_times_B=cal["f_times_B_rectified"], scale=scale,
                       K=cal["P1"], image_size=[OUT_W, OUT_H],
                       valid_frac_mean=float(f.mean()), valid_frac_p10=float(np.percentile(f, 10))),
                  fh, indent=2)
    print("  %-46s %4d frames, valid %.1f%% (p10 %.1f%%)"
          % (name[:46], kept, 100 * f.mean(), 100 * np.percentile(f, 10)))
    return kept, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calib/stereo_calib.json")
    ap.add_argument("--src", help="directory of SBS .mp4 clips")
    ap.add_argument("--dst")
    ap.add_argument("--video", help="single clip, for --preview")
    ap.add_argument("--preview")
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--stride", type=int, default=6, help="60fps source -> ~10fps")
    ap.add_argument("--min-disp", type=int, default=16)
    ap.add_argument("--num-disp", type=int, default=192,
                    help="measured range is ~30-200 px at 1340 wide; must be /16")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="run the matcher at this fraction. Each eye is only 636px wide "
                         "natively, so 1.0 is a 2.1x upsample carrying no new information -- "
                         "measured over 9 frames, 0.5 gives 58%% valid vs 41.7%% at 1.0 with "
                         "median depths agreeing to ~1mm. 0.5*1340=670 ~= native 636.")
    ap.add_argument("--also-video", action="store_true",
                    help="by default, if --src holds still frames only those are processed")
    ap.add_argument("--matcher", default="sgbm", choices=["sgbm", "ffs"],
                    help="ffs = C-Fast-FoundationStereo (needs a GPU)")
    ap.add_argument("--ffs-root", default="~/Fast-FoundationStereo")
    ap.add_argument("--ffs-model",
                    default="~/Fast-FoundationStereo/weights/c-fast/model_best_bp2_serialize.pth")
    ap.add_argument("--ffs-iters", type=int, default=8)
    a = ap.parse_args()

    matcher = sgbm_matcher
    if a.matcher == "ffs":
        matcher = ffs_matcher(a.ffs_root, a.ffs_model, a.ffs_iters, a.num_disp)

    with open(a.calib) as fh:
        cal = json.load(fh)
    maps = (rect_maps(cal, False), rect_maps(cal, True))
    print("f*B %.1f mm*px -> Z at disparity %d..%d = %.0f..%.0f mm"
          % (cal["f_times_B_rectified"], a.min_disp, a.min_disp + a.num_disp,
             cal["f_times_B_rectified"] / (a.min_disp + a.num_disp),
             cal["f_times_B_rectified"] / a.min_disp))

    if a.preview:
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame_index)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("could not read frame %d" % a.frame_index)
        preview(fr, maps, cal, a.min_disp, a.num_disp, a.scale, a.preview, matcher)
        return

    if not (a.src and a.dst):
        raise SystemExit("need --src and --dst (or --preview with --video)")
    imgs = sorted(sum([glob.glob(os.path.join(a.src, e)) for e in ("*.png", "*.jpg", "*.jpeg")], []))
    if imgs:
        print("%d SBS still frames -> %s" % (len(imgs), a.dst))
        run_images(imgs, a.dst, maps, cal, a.min_disp, a.num_disp, a.scale, matcher)
        if not a.also_video:
            return
    vids = sorted(glob.glob(os.path.join(a.src, "*.mp4")))
    print("%d clips -> %s" % (len(vids), a.dst))
    total, allf = 0, []
    for v in vids:
        k, f = run_video(v, a.dst, maps, cal, a.stride, a.min_disp, a.num_disp, a.scale, matcher)
        total += k
        allf.append(f)
    f = np.concatenate(allf)
    print("done: %d frames, mean valid %.1f%%" % (total, 100 * f.mean()))


if __name__ == "__main__":
    main()
