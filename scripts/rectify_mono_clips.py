"""Undistort + rectify the mono 1340x1072 depth clips using the stereo calibration.

Why: the calibration (STEREO_REPORT.md §5) shows the console feed is NOT a pinhole camera at
the periphery -- 2.4px off at r=400-600 and 5.4px (max 34.7) at r=600-900. The depth training
reprojection loss assumes a pinhole, so every peripheral pixel has been reprojected wrong.

Rectifying (rather than only undistorting) puts these frames in the SAME frame as the left eye
of the 3D clips, so stereo proxy-GT depth maps land directly on top of the mono training images
with one shared K -- no second camera model to carry around.

    # see what it does before running it on anything
    python scripts/rectify_mono_clips.py --preview out.png --frame ../data/....mp4

    # batch: <root>/<video>/clip_NNN.mp4/{images,masks,scale_objects.json}
    python scripts/rectify_mono_clips.py --src ../data/processed/depthclips_ruler_NoGUI/NOgui \
                                         --dst ../data/processed/depthclips_ruler_rect/NOgui

WARNING -- unresolved: which eye the 2D console feed carries. The two eyes' principal points
differ by 97px (the 3D convergence shift), and at the periphery using the WRONG eye is worse
than doing nothing (11.6px residual vs 5.4px uncorrected). --eye defaults to left by
convention; confirm it before trusting a batch run. See STEREO_REPORT.md §8.
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np

W, H = 1340, 1072
MONO_CROP = (289, 4)          # source_crop.json: x, y of the 1340x1072 crop in 1920x1080


def load_maps(calib, eye):
    """-> (map1, map2, K, D, R, P). Maps take a rectified pixel to its source pixel."""
    s = "L" if eye == "left" else "R"
    K, D = np.array(calib["K" + s]), np.array(calib["D" + s])
    R = np.array(calib["R1" if eye == "left" else "R2"])
    P = np.array(calib["P1" if eye == "left" else "P2"])
    m1, m2 = cv2.initUndistortRectifyMap(K, D, R, P, (W, H), cv2.CV_32FC1)
    return m1, m2, K, D, R, P


def warp_points(pts, K, D, R, P):
    """Same transform as the image maps, for annotation coordinates."""
    p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K, D, R=R, P=P).reshape(-1, 2)


def mono_crop(frame):
    """1920x1080 console frame -> the 1340x1072 training crop."""
    x, y = MONO_CROP
    return frame[y:y + H, x:x + W]


def read_frame(path, index=0):
    if path.lower().endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("could not read frame %d of %s" % (index, path))
        return mono_crop(fr) if fr.shape[1] == 1920 else fr
    img = cv2.imread(path)
    if img is None:
        raise SystemExit("could not read %s" % path)
    return mono_crop(img) if img.shape[1] == 1920 else img


# ---------------------------------------------------------------- preview
def preview(img, m1, m2, K, D, out, step=90):
    """3 panels: the same grid before and after, plus the distortion field.

    The grid is drawn STRAIGHT in the rectified panel and mapped back through m1/m2 onto the
    original -- so the bend you see on the left is exactly the distortion the pinhole model
    was ignoring.
    """
    rect = cv2.remap(img, m1, m2, cv2.INTER_CUBIC)
    a, b = img.copy(), rect.copy()
    green, n = (0, 255, 90), 3
    for y in range(step, H, step):                       # horizontal lines
        xs = np.arange(0, W, n)
        src = np.stack([m1[y, xs], m2[y, xs]], 1)
        cv2.polylines(a, [src.astype(np.int32)], False, green, 2, cv2.LINE_AA)
        cv2.line(b, (0, y), (W, y), green, 2, cv2.LINE_AA)
    for x in range(step, W, step):                       # vertical lines
        ys = np.arange(0, H, n)
        src = np.stack([m1[ys, x], m2[ys, x]], 1)
        cv2.polylines(a, [src.astype(np.int32)], False, green, 2, cv2.LINE_AA)
        cv2.line(b, (x, 0), (x, H), green, 2, cv2.LINE_AA)

    # distortion-only field (excludes the rectification rotation/zoom): how far the real lens
    # puts a pixel from where a pinhole would.
    gx, gy = np.meshgrid(np.arange(0, W, 4, np.float32), np.arange(0, H, 4, np.float32))
    g = np.stack([gx, gy], -1).reshape(-1, 1, 2)
    moved = cv2.undistortPoints(g, K, D, P=K).reshape(-1, 2) - g.reshape(-1, 2)
    d = np.linalg.norm(moved, axis=1).reshape(gy.shape)
    dmax = float(d.max())
    heat = cv2.applyColorMap(np.clip(d / max(dmax, 1e-6) * 255, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)
    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)
    for lvl in (1, 2, 5, 10, 20):                        # iso-error contours, in pixels
        if lvl > dmax:
            break
        mask = (cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC) > lvl).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(heat, cs, -1, (255, 255, 255), 1)
        if cs is not None and len(cs):
            p = max(cs, key=cv2.contourArea)[:, 0].mean(0).astype(int)
            cv2.putText(heat, "%dpx" % lvl, tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (255, 255, 255), 2, cv2.LINE_AA)

    tiles = []
    for panel, label in ((a, "BEFORE  original frame, grid bends = the distortion"),
                         (b, "AFTER  undistorted + rectified, grid is straight"),
                         (heat, "distortion field, max %.1f px" % dmax)):
        t = cv2.resize(panel, (W // 2, H // 2))
        t = cv2.copyMakeBorder(t, 46, 10, 10, 10, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(t, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite(out, np.hstack(tiles))
    print("wrote %s   (max distortion %.1f px)" % (out, dmax))


# ---------------------------------------------------------------- batch
def rectify_json(path, dst, K, D, R, P):
    """Annotation points live in unrectified 1340x1072 px -- move them with the images, or the
    metric-scale loss silently samples the wrong pixels."""
    with open(path) as fh:
        js = json.load(fh)
    n = lost = 0
    for objs in js.get("frames", {}).values():
        for o in objs:
            for key in ("a", "b"):
                if key in o:
                    o[key] = warp_points([o[key]], K, D, R, P)[0].tolist()
            if "points" in o:
                p = warp_points(o["points"], K, D, R, P)
                o["points"] = p.tolist()
                # rectification with alpha=0 zooms in, so edge annotations can land off-frame
                n += len(p)
                lost += int(((p[:, 0] < 0) | (p[:, 0] >= W) |
                             (p[:, 1] < 0) | (p[:, 1] >= H)).sum())
    js["rectified"] = True
    with open(dst, "w") as fh:
        json.dump(js, fh, indent=2)
    return n, lost


def batch(src, dst, m1, m2, K, D, R, P, eye):
    clips = []
    for video in sorted(os.listdir(src)):
        vdir = os.path.join(src, video)
        if not os.path.isdir(vdir):
            continue
        clips += [c for c in (os.path.join(vdir, n) for n in sorted(os.listdir(vdir)))
                  if os.path.isdir(os.path.join(c, "images"))]
    print("%d clips" % len(clips))
    n_img = n_pt = n_lost = 0
    for i, clip in enumerate(clips):
        rel = os.path.relpath(clip, src)
        out = os.path.join(dst, rel)
        for sub, interp in (("images", cv2.INTER_CUBIC), ("masks", cv2.INTER_NEAREST)):
            d_in = os.path.join(clip, sub)
            if not os.path.isdir(d_in):
                continue
            os.makedirs(os.path.join(out, sub), exist_ok=True)
            for name in sorted(os.listdir(d_in)):
                img = cv2.imread(os.path.join(d_in, name), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                cv2.imwrite(os.path.join(out, sub, name), cv2.remap(img, m1, m2, interp))
                n_img += sub == "images"
        for name in os.listdir(clip):
            p = os.path.join(clip, name)
            if not os.path.isfile(p):
                continue
            if name == "scale_objects.json":
                a, b = rectify_json(p, os.path.join(out, name), K, D, R, P)
                n_pt, n_lost = n_pt + a, n_lost + b
            else:
                shutil.copy2(p, os.path.join(out, name))
        with open(os.path.join(out, "rectify.json"), "w") as fh:
            json.dump(dict(eye=eye, K=P[:, :3].tolist(), image_size=[W, H],
                           source=os.path.relpath(clip, src)), fh, indent=2)
        if (i + 1) % 10 == 0:
            print("  %d/%d clips, %d images" % (i + 1, len(clips), n_img))
    print("done: %d clips, %d images -> %s" % (len(clips), n_img, dst))
    if n_pt:
        print("annotation points: %d total, %d landed off-frame (%.2f%%)"
              % (n_pt, n_lost, 100 * n_lost / n_pt))
    print("rectified K: fx %.2f fy %.2f cx %.2f cy %.2f" % (P[0, 0], P[1, 1], P[0, 2], P[1, 2]))
    print("normalised: %.4f %.4f %.4f %.4f"
          % (P[0, 0] / W, P[1, 1] / H, P[0, 2] / W, P[1, 2] / H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="outputs/stereo_calib/calib.json")
    ap.add_argument("--eye", default="left", choices=["left", "right"],
                    help="which eye the 2D console feed carries -- UNCONFIRMED, see docstring")
    ap.add_argument("--preview", help="write a before/after PNG and stop")
    ap.add_argument("--frame", help="video or image for --preview")
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--src")
    ap.add_argument("--dst")
    a = ap.parse_args()

    with open(a.calib) as fh:
        calib = json.load(fh)
    m1, m2, K, D, R, P = load_maps(calib, a.eye)

    if a.preview:
        if not a.frame:
            raise SystemExit("--preview needs --frame")
        preview(read_frame(a.frame, a.frame_index), m1, m2, K, D, a.preview)
        return
    if not (a.src and a.dst):
        raise SystemExit("need --src and --dst (or --preview)")
    batch(a.src, a.dst, m1, m2, K, D, R, P, a.eye)


if __name__ == "__main__":
    main()
