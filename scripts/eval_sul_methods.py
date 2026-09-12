"""SUL methods head to head, on the frames Nick annotated, with stereo and monocular depth.

Reference: Nick's two ruler points back-projected with the STEREO depth (a 3D chord). It is the
same number in every condition, so each method is judged against a fixed target.

methods  A  mask ends          the two ends of the mask along its main axis (the idea of the repo's
                               first SUL method, sul_measure_depth.py) -> 3D chord through the depth
         B  first cylinder     the first delivered version, run from git (50834de): mask start, roof
                               end by walk-back, hidden-start check
         C  current cylinder   HEAD defaults: ROI refit, knee3 end, knee start near the mask, outward
         D  mask along tube    C's fitted axis, both ends from the mask (1st / 99th percentile)
masks    hand (Nick) | model (ureth_fn + keep-largest)
depth    stereo | mono_scaled (EndoDAC depth_ruler_range_sw05, scaled per frame so its median over the
         urethra matches the stereo: shape error only) | mono (as predicted: shape + scale)

    sbatch jobs/eval_sul_methods.sh
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import types

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urethra_cylinder as cur                                   # noqa: E402
from make_stereo_proxy_gt import DEPTH_SCALE                      # noqa: E402

RUNS = ("seg3_t30", "noocc_seg3_14s", "noocc_5e27_16s")
LABEL = {"seg3_t30": "control", "noocc_seg3_14s": "short", "noocc_5e27_16s": "long"}


def load_v1(rev):
    """The first delivered version, straight from git -- no hand-copied reimplementation."""
    src = subprocess.check_output(["git", "show", rev + ":scripts/urethra_cylinder.py"]).decode()
    mod = types.ModuleType("urethra_cylinder_" + rev)
    mod.__file__ = os.path.abspath(__file__)
    exec(compile(src, "urethra_cylinder@" + rev, "exec"), mod.__dict__)
    return mod


def backproject(q, z, K):
    fx, fy, cx, cy = K
    return np.array([(q[0] - cx) * z / fx, (q[1] - cy) * z / fy, z])


def depth_near(z, mask, q, rad=9):
    """Median depth of the MASK pixels around q: a mask end sits on the silhouette, where a plain
    patch would mix in whatever lies behind the urethra."""
    u, v = int(round(q[0])), int(round(q[1]))
    sl = (slice(max(0, v - rad), v + rad + 1), slice(max(0, u - rad), u + rad + 1))
    zz = z[sl][mask[sl] & (z[sl] > 0)]
    return float(np.median(zz)) if zz.size else float("nan")


def mask_ends(mask):
    """The two ends of the mask along its main axis: the mean position of the mask pixels in the
    outer 1% at each end. Points ON the axis line fall off a tilted or curved mask (control clip:
    every one of them missed the depth), and sul_measure.mirror_axis can't be imported here (its
    module needs openpyxl), so this is the same idea done with the pixels themselves."""
    v, u = np.nonzero(mask)
    P = np.stack([u, v], 1).astype(float)
    c = P.mean(0)
    a = np.linalg.eigh(np.cov((P - c).T))[1][:, -1]
    t = (P - c) @ a
    lo, hi = np.percentile(t, [1, 99])
    return P[t <= lo].mean(0), P[t >= hi].mean(0)


def orient2d(e0, e1, seg):
    """Which mask end is the proximal one: the one nearer the prostate/catheter, else the lower
    in the image (the same rule urethra_cylinder.orient uses, in 2D)."""
    ref = np.argwhere(np.isin(seg, (cur.PROSTATE, cur.CATHETER)))
    if len(ref) > 500:
        q = ref.mean(0)[::-1]
        return (e0, e1) if np.linalg.norm(e0 - q) < np.linalg.norm(e1 - q) else (e1, e0)
    return (e0, e1) if e0[1] > e1[1] else (e1, e0)


def top_uv(fr, t, K):
    """Pixel of the tube's top line at axial position t -- where the method puts that end."""
    if t is None or not np.isfinite(t):
        return np.array([np.nan, np.nan])
    A = fr["p"][None] + t * fr["d"][None]
    S = A + fr["r"] * cur.toward_camera(A, fr["d"])
    return cur.proj(S, K)[0] if S[0, 2] > 1 else np.array([np.nan, np.nan])


def xy(s, e):
    r2 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 1)
    return dict(s_u=r2(s[0]), s_v=r2(s[1]), e_u=r2(e[0]), e_v=r2(e[1]))


def along(fr, K, q):
    """t on fr's axis whose top-line point projects nearest to the click q (image matching)."""
    t_end = fr["t_end"] if fr.get("found") else fr["t_start"] + 30
    tt = np.arange(min(fr["t_start"], t_end) - 20, max(fr["t_start"], t_end) + 20, 0.1)
    A = fr["p"] + tt[:, None] * fr["d"]
    S = A + fr["r"] * cur.toward_camera(A, fr["d"])
    ok = S[:, 2] > 1.0
    uv = cur.proj(S[ok], K)
    return float(tt[ok][np.argmin(np.linalg.norm(uv - q, axis=1))])


def ends_err(fr, K, a, b):
    ta, tb = sorted((along(fr, K, a), along(fr, K, b)))
    return fr["t_start"] - ta, (fr["t_end"] - tb) if fr.get("found") else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/temporal_stereo")
    ap.add_argument("--out", default="outputs/sul_eval")
    ap.add_argument("--v1", default="50834de", help="git revision of the first delivered version")
    ap.add_argument("--calib", default="calib/stereo_calib.json")
    ap.add_argument("--checkpoint", default="outputs/ureth_fn/best.pth")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    from PIL import Image
    from overlay_dir import MetaFormerFPN, _keep_largest, predict
    from gui_depth_measure import (DEFAULT_CKPT, DEFAULT_MAX_DEPTH, DEFAULT_MIN_DEPTH,
                                   DEFAULT_SHAPE, DepthBackend)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    segnet = MetaFormerFPN(num_classes=sd["FPN.segmentation_head.0.bias"].shape[0],
                           pretrained="ImageNet", pretrained_weights=None)
    segnet.load_state_dict(sd)
    segnet.to(dev).eval()
    mono = DepthBackend(DEFAULT_CKPT, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH, DEFAULT_MAX_DEPTH).load()
    v1 = load_v1(a.v1)
    with open(a.calib) as fh:
        P1 = np.array(json.load(fh)["P1"])
    K = (P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2])

    rows = []
    for run in RUNS:
        R = os.path.join(a.root, run)
        with open(os.path.join(R, "hand_points.csv")) as fh:
            pts = {r["frame"]: (np.array([float(r["ax"]), float(r["ay"])]),
                                np.array([float(r["bx"]), float(r["by"])])) for r in csv.DictReader(fh)}
        for name in sorted(pts):
            img = cv2.imread(os.path.join(R, "images", name + ".png"))
            zs = cv2.imread(os.path.join(R, "depth", name + ".png"), cv2.IMREAD_UNCHANGED) / DEPTH_SCALE
            zs = zs.astype(np.float32)
            H, W = zs.shape
            zm = cv2.resize(mono.predict(Image.fromarray(img[..., ::-1])), (W, H),
                            interpolation=cv2.INTER_LINEAR).astype(np.float32)
            pa, pb = pts[name]
            ref = cur.hand_measure(zs, pa, pb, K, None)["hand_sul"]
            segs = {"hand": cur.load_mask(os.path.join(R, "hand_masks", name + ".png"), (H, W)),
                    "model": _keep_largest(predict(segnet, Image.fromarray(img[..., ::-1]), (512, 512), dev),
                                           cur.URETHRA)}
            for mname, seg in segs.items():
                um = seg == cur.URETHRA
                if um.sum() < 1500:
                    continue
                e0, e1 = mask_ends(um)
                zsu = zs[um & (zs > 0)]
                scale = float(np.median(zsu) / np.median(zm[um])) if zsu.size else float("nan")
                for dname, z in (("stereo", zs), ("mono_scaled", zm * scale), ("mono", zm)):
                    base = dict(clip=LABEL[run], frame=name, mask=mname, depth=dname, ref=round(ref, 2),
                                mono_scale=round(scale, 3))
                    # A: mask ends -> 3D chord
                    A0, A1 = (backproject(e, depth_near(z, um, e), K) for e in (e0, e1))
                    as_, ae_ = orient2d(e0, e1, seg)
                    rows.append(dict(base, method="A mask ends", sul=float(np.linalg.norm(A0 - A1)),
                                     counted=True, start_err=np.nan, end_err=np.nan, **xy(as_, ae_)))
                    # B: first delivered cylinder
                    fb = v1.analyse(z, seg, K)
                    if fb is not None:
                        se, ee = ends_err(fb, K, pa, pb)
                        rows.append(dict(base, method="B first cylinder", sul=fb["sul"],
                                         counted=bool(fb["start_ok"]) and fb["found"], start_err=se, end_err=ee,
                                         **xy(top_uv(fb, fb["t_start"], K),
                                              top_uv(fb, fb.get("t_end"), K))))
                    # C: current cylinder, and D: its axis with the mask's own ends
                    fc = cur.analyse(z, seg, K)
                    if fc is not None:
                        se, ee = ends_err(fc, K, pa, pb)
                        rows.append(dict(base, method="C current cylinder", sul=fc["sul"],
                                         counted=bool(fc["start_ok"]) and fc["found"], start_err=se, end_err=ee,
                                         **xy(top_uv(fc, fc["t_start"], K),
                                              top_uv(fc, fc.get("t_end"), K))))
                        core = cv2.erode((um & (z > 0)).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
                        v, u = np.nonzero(core)
                        if len(u) > 200:
                            X = np.stack([(u - K[2]) * z[v, u] / K[0], (v - K[3]) * z[v, u] / K[1], z[v, u]], 1)
                            t = (X - fc["p"]) @ fc["d"]
                            t0, t1 = np.percentile(t, [1, 99])
                            fd = dict(fc, t_start=float(t0), t_end=float(t1), found=True)
                            se, ee = ends_err(fd, K, pa, pb)
                            rows.append(dict(base, method="D mask along tube", sul=float(t1 - t0),
                                             counted=True, start_err=se, end_err=ee,
                                             **xy(top_uv(fd, t0, K), top_uv(fd, t1, K))))
            print("  %-8s %s done (ref %.1f mm)" % (LABEL[run], name, ref), flush=True)

    keys = list(rows[0])
    with open(os.path.join(a.out, "eval_rows.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    def stats(sel):
        err = np.array([r["sul"] - r["ref"] for r in sel], float)
        err = err[np.isfinite(err)]
        if not err.size:
            return None
        return dict(n=int(err.size), med=float(np.median(err)), mae=float(np.median(np.abs(err))),
                    p10=float(np.percentile(err, 10)), p90=float(np.percentile(err, 90)))

    summary = {}
    print("\nSUL minus Nick's 3D chord (stereo). Counted frames only. median signed | median abs | p10..p90 (n)")
    for mname in ("hand", "model"):
        for dname in ("stereo", "mono_scaled", "mono"):
            print("\n== masks %s, depth %s" % (mname, dname))
            for meth in ("A mask ends", "B first cylinder", "C current cylinder", "D mask along tube"):
                line, allsel = [], []
                for clip in ("control", "short", "long"):
                    sel = [r for r in rows if r["mask"] == mname and r["depth"] == dname and r["method"] == meth
                           and r["clip"] == clip and r["counted"]]
                    allsel += sel
                    s = stats(sel)
                    line.append("%-7s %s" % (clip, "--" if s is None else "%+5.1f |%4.1f| %+5.1f..%+5.1f (%2d)"
                                             % (s["med"], s["mae"], s["p10"], s["p90"], s["n"])))
                    summary["%s|%s|%s|%s" % (mname, dname, meth, clip)] = s
                s = stats(allsel)
                summary["%s|%s|%s|all" % (mname, dname, meth)] = s
                print("  %-19s %s   ALL |%s|" % (meth, "   ".join(line), "--" if s is None else "%.1f" % s["mae"]))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\nwrote %s/eval_rows.csv, summary.json" % a.out)


if __name__ == "__main__":
    main()
