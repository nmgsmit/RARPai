"""Urethra end point in 3D: fit a cylinder to the urethra, then follow it distally until the
tissue 'roof' comes in front of it -- that is where the urethra leaves the field.

The segmentation gives the START (prostate side) cleanly. The END is the hard one: tissue covers
it, or it runs off screen, which is why the Retzius-arch fit was tried. With stereo depth it
becomes geometry. The urethra is a tube and the tube carries on under the roof, so extend the
fitted tube and find where the observed surface stops BEING the tube and starts sitting IN FRONT
of it:

    gap(t) = Z_tube_top(t) - Z_observed(t)    ~0 while the tube is visible, grows once covered
    end    = where gap leaves ~0 for good (first sustained gap > --margin, walked back to ~0)
    SUL    = end - start, along the axis

Input is the temporal-stereo output (--save-depth): images/*.png = rectified left eye, the same
1340x1072 frame the segmentation was trained in; depth/*.png = uint16 mm*16, 0 = no depth.

    python scripts/urethra_cylinder.py --run outputs/temporal_stereo/seg3_t30
    python scripts/urethra_cylinder.py --self-test      # synthetic tube + roof, no model needed
"""
import argparse
import csv
import glob
import json
import os
import sys
import warnings

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_stereo_proxy_gt import DEPTH_SCALE, colorize   # noqa: E402

# compact ids of outputs/ureth_fn, trained with --keep-classes 1,2,4,5
URETHRA, PROSTATE, CATHETER, NONANAT = 1, 2, 3, 4


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def basis(d):
    e1 = unit(np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0]))
    return e1, np.cross(d, e1)


def toward_camera(A, d):
    """Unit vector from axis point(s) A toward the camera (at the origin), perpendicular to the
    axis. A + r*this is the tube's camera-facing top line -- the part the roof covers first."""
    w = -A
    return unit(w - (w @ d)[..., None] * d)


def proj(X, K):
    fx, fy, cx, cy = K
    return np.stack([fx * X[..., 0] / X[..., 2] + cx, fy * X[..., 1] / X[..., 2] + cy], -1)


# ------------------------------------------------------------------ fit + roof

def fit_cylinder(P):
    """Robust least squares on (distance to axis - r).

    Only the camera-facing part of the tube is ever seen, so the start point matters: axis along
    the points' principal direction, radius from their spread ACROSS it, and the axis one radius
    BEHIND the visible surface. ponytail: straight cylinder; a curved urethra biases the far end.
    """
    c = P.mean(0)
    d0 = np.linalg.svd(P - c, full_matrices=False)[2][0]
    e1, e2 = basis(d0)
    across = unit(np.cross(d0, unit(c)))
    r0 = max(0.5, np.subtract(*np.percentile((P - c) @ across, [97, 3])) / 2)
    off = r0 * unit(c)

    def model(x):
        return unit(d0 + x[0] * e1 + x[1] * e2), c + x[2] * e1 + x[3] * e2

    def res(x):
        d, p = model(x)
        return np.linalg.norm(np.cross(P - p, d), axis=1) - x[4]

    s = least_squares(res, [0, 0, off @ e1, off @ e2, r0], loss="soft_l1", f_scale=0.3)
    d, p = model(s.x)
    return p, d, abs(s.x[4]), float(np.median(np.abs(res(s.x)))), r0


def orient(p, d, P, uv, seg):
    """Point the axis DISTALLY. Proximal = the end nearer the prostate/catheter; with neither
    in view, the lower end in the image (the pubic arch is at the top in this console view)."""
    t = (P - p) @ d
    lo, hi = uv[t <= np.percentile(t, 5)].mean(0), uv[t >= np.percentile(t, 95)].mean(0)
    ref = np.argwhere(np.isin(seg, (PROSTATE, CATHETER)))
    if len(ref) > 500:
        r = ref.mean(0)[::-1]                                   # (v,u) -> (u,v)
        return (-d if np.linalg.norm(hi - r) < np.linalg.norm(lo - r) else d), "prostate/catheter"
    return (-d if hi[1] > lo[1] else d), "lower end"


def nanmedian_filter(x, k=5):
    pad = np.pad(x, k // 2, constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.lib.stride_tricks.sliding_window_view(pad, k), 1)


def gap_along(p, d, r, ts, zmap, seg, K, patch=2):
    """Tube top line minus observed surface (mm) at axial positions ts, NaN where unknown."""
    H, W = zmap.shape
    A = p + ts[:, None] * d
    S = A + r * toward_camera(A, d)
    uv = proj(S, K)
    inside = ((uv[:, 0] >= patch) & (uv[:, 0] < W - patch - 1) &
              (uv[:, 1] >= patch) & (uv[:, 1] < H - patch - 1))
    zobs = np.full(len(ts), np.nan)
    for i in np.flatnonzero(inside):
        ui, vi = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if seg is not None and seg[vi, ui] == NONANAT:
            continue                     # an instrument in front says nothing about the roof
        z = zmap[vi - patch:vi + patch + 1, ui - patch:ui + patch + 1]
        z = z[z > 0]
        if z.size:
            zobs[i] = np.median(z)
    return S[:, 2] - zobs, inside


def march(p, d, r, t0, zmap, seg, K, ext, margin, step=0.2, zero_tol=0.5, run_mm=1.0):
    """Walk the tube's top line from t0 distally and compare it with the observed surface."""
    ts = np.arange(t0, t0 + ext, step)
    raw, inside = gap_along(p, d, r, ts, zmap, seg, K)
    gap = nanmedian_filter(raw)
    k = max(1, int(round(run_mm / step)))
    covered = np.nan_to_num(gap, nan=-np.inf) > margin
    run = np.flatnonzero(np.convolve(covered, np.ones(k), "valid") == k)
    out = dict(ts=ts, gap=gap, found=False)
    if run.size:
        # the margin only makes the detection robust; the END is where the gap left ~0, so walk
        # back -- otherwise a shallow roof would push the end point distally by margin/slope
        j = run[0]
        while j > 0 and gap[j - 1] > zero_tol:
            j -= 1
        out.update(found=True, t_end=float(ts[j]), status="roof")
    elif not inside.all():
        out["status"] = "left frame at +%.1f mm" % (ts[np.argmin(inside)] - t0)
    else:
        out["status"] = "not covered within %g mm" % ext
    return out


def analyse(zmap, seg, K, erode=7, ext=30.0, margin=1.5, min_px=1500):
    """One frame -> cylinder, start, end, SUL (or None when there is too little urethra)."""
    fx, fy, cx, cy = K
    ok = (seg == URETHRA) & (zmap > 0)
    # fit on the eroded core: silhouette pixels mix tube and background depth
    core = cv2.erode(ok.astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8)).astype(bool)
    if ok.sum() < min_px or core.sum() < min_px // 2:
        return None

    def pts(mask, cap):
        v, u = np.nonzero(mask)
        if len(u) > cap:
            keep = np.linspace(0, len(u) - 1, cap).astype(int)
            u, v = u[keep], v[keep]
        z = zmap[v, u]
        return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1), np.stack([u, v], 1) * 1.0

    P, uv = pts(core, 20000)
    p, d, r, res, r_sil = fit_cylinder(P)
    d, rule = orient(p, d, P, uv, seg)
    t_start = float(np.percentile((pts(ok, 40000)[0] - p) @ d, 1))
    # The start is only a measurement if the tube really ENDS there. An instrument lying over the
    # proximal urethra truncates the mask and drags the start distally -- on 5e27 SUL tracked the
    # mask's size at rho 0.92. Same test as the roof, run backwards: something in front = hidden.
    # ponytail: on an end-on stump (5e27) this also fires on clean cut ends -- the cylinder is
    # ill-posed there, so no end check is trustworthy; the method needs the urethra side-on.
    back, _ = gap_along(p, d, r, t_start - np.arange(0.5, 3.01, 0.25), zmap, seg, K)
    start_ok = bool(np.isfinite(back).any() and np.nanmedian(back) <= margin)
    out = dict(p=p, d=d, r=r, r_sil=r_sil, res=res, rule=rule, t_start=t_start, n=len(P),
               start_ok=start_ok,
               **march(p, d, r, float(np.median((P - p) @ d)), zmap, seg, K, ext, margin))
    out["sul"] = out["t_end"] - t_start if out["found"] else float("nan")
    return out


# ------------------------------------------------------------------ drawing

def tube_line(fr, t0, t1, side, K, n=80):
    """side 0 = top line, +-1 = the two silhouette edges."""
    ts = np.linspace(t0, t1, n)
    A = fr["p"] + ts[:, None] * fr["d"]
    tw = toward_camera(A, fr["d"])
    X = A + fr["r"] * (tw if side == 0 else side * unit(np.cross(fr["d"], tw)))
    return np.round(proj(X, K)).astype(np.int32)


def draw(img, seg, fr, K):
    out = img.copy()
    m = seg == URETHRA
    out[m] = (0.55 * out[m] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
    if fr is None:
        return out
    t_end = fr["t_end"] if fr["found"] else fr["ts"][-1]
    for side in (-1, 1):
        cv2.polylines(out, [tube_line(fr, fr["t_start"], t_end, side, K)], False,
                      (255, 255, 0), 3, cv2.LINE_AA)
    far = tube_line(fr, t_end, t_end + 15, 0, K, 40)
    for a, b in zip(far[:-1:2], far[1::2]):             # dashed: the tube carrying on underneath
        cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), (230, 230, 230), 2, cv2.LINE_AA)
    cv2.circle(out, tuple(map(int, tube_line(fr, fr["t_start"], fr["t_start"], 0, K, 1)[0])),
               11, (0, 255, 0), -1 if fr["start_ok"] else 3)      # hollow = start hidden
    if fr["found"]:
        cv2.circle(out, tuple(map(int, tube_line(fr, t_end, t_end, 0, K, 1)[0])), 11, (0, 0, 255), -1)
    return out


def label(panel, text):
    t = cv2.copyMakeBorder(panel, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(t, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def profile(fr, w, h, margin, gmin=-3.0, gmax=12.0):
    img = np.full((h, w, 3), 24, np.uint8)
    if fr is None:
        cv2.putText(img, "no urethra in this frame", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return img
    L, R, T, B = 70, 20, 34, 30
    x0, xmax = fr["t_start"], max(fr["ts"][-1] - fr["t_start"], 1.0)
    X = lambda t: int(L + (t - x0) / xmax * (w - L - R))
    Y = lambda g: int(T + (gmax - np.clip(g, gmin, gmax)) / (gmax - gmin) * (h - T - B))
    cv2.line(img, (L, Y(0)), (w - R, Y(0)), (120, 120, 120), 1)
    for xx in range(L, w - R, 12):
        cv2.line(img, (xx, Y(margin)), (xx + 5, Y(margin)), (0, 140, 255), 1)
    for g in (0, 5, 10):
        cv2.putText(img, "%d" % g, (L - 30, Y(g) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    for mm in range(0, int(xmax) + 1, 5):
        cv2.putText(img, "%d" % mm, (X(x0 + mm) - 6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
    ok = np.isfinite(fr["gap"])
    pts = np.array([[X(t), Y(np.nan_to_num(g))] for t, g in zip(fr["ts"], fr["gap"])], np.int32)
    for s, e in zip(*[np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]) == k) for k in (1, -1)]):
        cv2.polylines(img, [pts[s:e]], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (X(fr["t_start"]), T), (X(fr["t_start"]), h - B), (0, 255, 0), 2)
    if fr["found"]:
        cv2.line(img, (X(fr["t_end"]), T), (X(fr["t_end"]), h - B), (0, 0, 255), 2)
    msg = ("SUL %.1f mm   (start -> roof, along the axis)   r %.1f mm" % (fr["sul"], fr["r"])
           if fr["found"] else "end not found: %s   r %.1f mm" % (fr["status"], fr["r"]))
    if not fr["start_ok"]:
        msg += "   START HIDDEN - frame not counted"
    cv2.putText(img, "gap = tube top - observed surface (mm) vs distance from start (mm);  orange = "
                "margin.   " + msg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def fig_3d(img, zmap, seg, fr, K, path):
    """Left: a longitudinal SECTION through the top of the tube -- the literal picture of the
    question. Tube surface points follow the cylinder's top line; past the end the observed
    points (the roof) rise above it. Right: the same slab plus the tube's flanks in 3D."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fx, fy, cx, cy = K
    v, u = np.mgrid[0:zmap.shape[0]:2, 0:zmap.shape[1]:2]
    v, u = v.ravel(), u.ravel()
    k = zmap[v, u] > 0
    v, u = v[k], u[k]
    z = zmap[v, u]
    P = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1)
    p, d, r, t0 = fr["p"], fr["d"], fr["r"], fr["t_start"]
    t1 = fr["t_end"] if fr["found"] else fr["ts"][-1]
    n = toward_camera(p + 0.5 * (t0 + t1) * d, d)
    m = np.cross(d, n)
    q = P - p
    t, h, l = q @ d - t0, q @ n, q @ m
    ure = seg[v, u] == URETHRA
    col = img[v, u, ::-1] / 255.0
    span = (t > -8) & (t < t1 - t0 + 15)
    fig = plt.figure(figsize=(17, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    sl = span & (np.abs(l) < 1.5)                   # thin slice along the tube's top line
    ax.scatter(t[sl & ~ure], h[sl & ~ure], s=4, c=col[sl & ~ure], label="observed surface")
    ax.scatter(t[sl & ure], h[sl & ure], s=4, c="gold", label="urethra mask")
    ax.axhline(r, color="c", lw=2, label="cylinder top line (r %.1f mm)" % r)
    ax.axhline(0, color="c", lw=1, ls=":", label="cylinder axis")
    ax.axhline(-r, color="c", lw=1, ls="--")
    ax.axvline(0, color="lime", lw=2, label="start (prostate side)")
    ax.axvline(t1 - t0, color="red", lw=2, label="end: roof meets the tube")
    ax.set_xlabel("distance along the urethra from the start (mm)")
    ax.set_ylabel("height toward the camera (mm)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("section through the top of the tube:  SUL %.1f mm" % fr["sul"])
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    sl = span & (np.abs(l) < r + 4) & (np.abs(h) < r + 12)
    idx = np.flatnonzero(sl)[::3]
    ax.scatter(t[idx], l[idx], h[idx], s=2, c=np.where(ure[idx, None], [1, 0.85, 0], col[idx]),
               depthshade=False, alpha=0.5)
    ts_ = np.linspace(0, t1 - t0, 25)
    ang = np.linspace(0, 2 * np.pi, 40)
    T, A = np.meshgrid(ts_, ang)
    ax.plot_wireframe(T, r * np.sin(A), r * np.cos(A), color="c", lw=0.4, alpha=0.8)
    for x, c in ((0, "lime"), (t1 - t0, "red")):
        ax.scatter([x], [0], [r], c=c, s=150, edgecolors="k", depthshade=False)
    ax.set_xlabel("along urethra mm"), ax.set_ylabel("lateral mm"), ax.set_zlabel("toward camera mm")
    ax.set_box_aspect((np.ptp(t[idx]), np.ptp(l[idx]), np.ptp(h[idx])))
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_time(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["time_s"] for r in rows])
    sul = np.array([r["sul_mm"] for r in rows], float)
    rad = np.array([r["radius_mm"] for r in rows], float)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ok = np.array([r["start_ok"] for r in rows]) & np.isfinite(sul)
    ax[0].plot(t[ok], sul[ok], "o", ms=4, label="both ends observed")
    ax[0].plot(t[~ok], sul[~ok], "o", ms=4, mfc="none", color="grey",
               label="start hidden (not counted)")
    if ok.any():
        m = np.median(sul[ok])
        ax[0].axhline(m, color="r", label="median %.1f mm (n=%d)" % (m, ok.sum()))
    ax[0].legend()
    ax[0].set_ylabel("SUL mm")
    ax[1].plot(t, rad, "o", ms=4, color="tab:green")
    ax[1].set_ylabel("fitted radius mm")
    ax[1].set_xlabel("time in window (s)")
    fig.suptitle("per-frame SUL and cylinder radius -- a length is camera-motion invariant, so the "
                 "spread IS the method's noise")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ self-test

def render(K, W, H, p, d, r, t_roof, beta=1.0, bg=75.0, t_len=60.0):
    """Ray-cast a tube (cut at t=0) on a background plane, with a roof that meets the tube's top
    line at t_roof and rises toward the camera past it."""
    fx, fy, cx, cy = K
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    D = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones(u.shape)], -1)
    a, b = np.cross(D, d), np.cross(p, d)                  # |lambda*a - b| = r
    A2, B, C = (a * a).sum(-1), (a * b).sum(-1), (b * b).sum() - r * r
    disc = B * B - A2 * C
    lc = np.where(disc > 0, (B - np.sqrt(np.maximum(disc, 0))) / A2, np.inf)
    tc = (np.nan_to_num(lc, posinf=0)[..., None] * D - p) @ d
    lc[(tc < 0) | (tc > t_len) | (lc <= 0)] = np.inf
    Ar = p + t_roof * d
    tw = toward_camera(Ar, d)
    N = unit(np.cross(d + beta * tw, np.cross(d, tw)))
    lr = ((Ar + r * tw) @ N) / (D @ N)
    lr[~(((lr[..., None] * D - Ar) @ d) > 0) | (lr <= 0)] = np.inf
    z = np.minimum(np.minimum(lc, lr), bg)
    return z.astype(np.float32), np.where(lc <= np.minimum(lr, bg), URETHRA, 0).astype(np.uint8)


def self_test():
    K, W, H = (285.8, 285.8, 150.9, 133.5), 335, 268        # the rectified K at quarter size
    d_true, p_true, r_true, t_roof = unit(np.array([0.15, -1.0, 0.3])), np.array([2.0, 12, 55]), 4.0, 18.0
    z, seg = render(K, W, H, p_true, d_true, r_true, t_roof)
    z += np.random.default_rng(0).normal(0, 0.15, z.shape).astype(np.float32)   # ~stereo noise
    fr = analyse(z, seg, K, erode=2, min_px=200)
    ang = np.degrees(np.arccos(np.clip(fr["d"] @ d_true, -1, 1)))
    print("self-test: r %.2f (true %.1f)  axis error %.2f deg  SUL %.2f (true %.1f)  %s  rule=%s"
          % (fr["r"], r_true, ang, fr["sul"], t_roof, fr["status"], fr["rule"]))
    assert abs(fr["r"] - r_true) < 0.4, "radius"
    assert ang < 4, "axis direction (or orientation flipped)"
    assert fr["found"] and abs(fr["sul"] - t_roof) < 1.5, "roof end point"
    assert fr["start_ok"], "a clean cut end was flagged as hidden"
    z3, seg3 = z.copy(), seg.copy()
    z3[172:], seg3[172:] = 40.0, 0          # an instrument across the proximal ~5 mm of the tube
    fr3 = analyse(z3, seg3, K, erode=2, min_px=200)
    print("self-test: instrument over the start -> start_ok=%s (its SUL would read %.1f)"
          % (fr3["start_ok"], fr3["sul"]))
    assert not fr3["start_ok"], "an instrument over the start was not flagged"
    z2, seg2 = render(K, W, H, p_true, d_true, r_true, t_roof=500)      # no roof in view
    fr2 = analyse(z2, seg2, K, erode=2, min_px=200)
    assert not fr2["found"], "found a roof that is not there"
    print("self-test OK (no-roof case: %s)" % fr2["status"])


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="temporal_stereo_clip.py --save-depth output dir")
    ap.add_argument("--calib", default="calib/stereo_calib.json")
    ap.add_argument("--checkpoint", default="outputs/ureth_fn/best.pth")
    ap.add_argument("--img-size", type=int, default=512, help="feed size ureth_fn was trained at")
    ap.add_argument("--erode", type=int, default=7, help="px shaved off the mask before fitting")
    ap.add_argument("--margin", type=float, default=1.5,
                    help="mm the surface must sit in front of the tube to count as covered")
    ap.add_argument("--ext", type=float, default=30.0, help="mm to follow the axis past mid-tube")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.run:
        raise SystemExit("need --run (or --self-test)")

    import torch
    from PIL import Image
    from overlay_dir import MetaFormerFPN, _keep_largest, predict
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model = MetaFormerFPN(num_classes=sd["FPN.segmentation_head.0.bias"].shape[0],
                          pretrained="ImageNet", pretrained_weights=None)
    model.load_state_dict(sd)
    model.to(dev).eval()

    with open(a.calib) as fh:
        P1 = np.array(json.load(fh)["P1"])
    K = (P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2])
    out = os.path.join(a.run, "urethra_cyl")
    os.makedirs(out, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(a.run, "images", "*.png")))
    if not imgs:
        raise SystemExit("no %s/images/*.png -- run temporal_stereo_clip.py with --save-depth" % a.run)

    frames, rows = [], []
    for k, pth in enumerate(imgs):
        img = cv2.imread(pth)
        zmap = cv2.imread(pth.replace(os.sep + "images" + os.sep, os.sep + "depth" + os.sep),
                          cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
        seg = _keep_largest(predict(model, Image.fromarray(img[..., ::-1]),
                                    (a.img_size, a.img_size), dev), URETHRA)
        fr = analyse(zmap, seg, K, a.erode, a.ext, a.margin)
        frames.append((img, zmap, seg, fr))
        rows.append(dict(frame=os.path.basename(pth)[:-4], time_s=round(k / a.fps, 3),
                         urethra_px=int((seg == URETHRA).sum()),
                         status="no urethra" if fr is None else fr["status"],
                         start_ok=fr is not None and fr["start_ok"],
                         sul_mm=float("nan") if fr is None else round(fr["sul"], 2),
                         radius_mm=float("nan") if fr is None else round(fr["r"], 2),
                         # half the 3D width across the tube: what a CIRCULAR tube would need
                         radius_silhouette_mm=float("nan") if fr is None else round(fr["r_sil"], 2),
                         fit_resid_mm=float("nan") if fr is None else round(fr["res"], 3),
                         orient=None if fr is None else fr["rule"]))
        print("  %s  px %6d  %-26s  start %-5s  SUL %s  r %s  resid %s" % (
            rows[-1]["frame"], rows[-1]["urethra_px"], rows[-1]["status"], rows[-1]["start_ok"],
            rows[-1]["sul_mm"], rows[-1]["radius_mm"], rows[-1]["fit_resid_mm"]), flush=True)

    with open(os.path.join(out, "frames.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    sul = np.array([r["sul_mm"] for r in rows], float)
    found = np.isfinite(sul)
    good = found & np.array([r["start_ok"] for r in rows])      # both ends actually observed
    summary = dict(run=a.run, frames=len(rows), fitted=sum(f[3] is not None for f in frames),
                   end_found=int(found.sum()), both_ends_observed=int(good.sum()),
                   sul_median_all_found_mm=float(np.median(sul[found])) if found.any() else None,
                   sul_median_mm=float(np.median(sul[good])) if good.any() else None,
                   sul_iqr_mm=[float(x) for x in np.percentile(sul[good], [25, 75])] if good.any() else None,
                   radius_median_mm=float(np.nanmedian([r["radius_mm"] for r in rows])),
                   margin_mm=a.margin, erode_px=a.erode)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

    zs = np.concatenate([f[1][f[1] > 0][::97] for f in frames])
    lo, hi = np.percentile(zs, [2, 97])
    vw = None
    for img, zmap, seg, fr in frames:
        heat = colorize(np.where(zmap > 0, 1.0 / np.maximum(zmap, 1e-6), np.nan), 1 / hi, 1 / lo,
                        cv2.COLORMAP_TURBO)
        heat[zmap <= 0] = (90, 90, 90)
        small = lambda x: cv2.resize(x, (670, 536), interpolation=cv2.INTER_AREA)
        top = np.hstack([label(small(draw(img, seg, fr, K)), "urethra mask (yellow), cylinder (cyan),"
                                                              " start (green), roof end (red)"),
                         label(small(draw(heat, seg, fr, K)), "stereo depth %.0f-%.0f mm, grey = none"
                               % (lo, hi))])
        frame = np.vstack([top, profile(fr, top.shape[1], 240, a.margin)])
        if vw is None:
            vw = cv2.VideoWriter(os.path.join(out, "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 a.fps, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    vw.release()

    if good.any():                          # the 3D figure on the frame closest to the median SUL
        i = int(np.argmin(np.where(good, np.abs(sul - np.median(sul[good])), np.inf)))
        img, zmap, seg, fr = frames[i]
        fig_3d(img, zmap, seg, fr, K, os.path.join(out, "fig_3d.png"))
        panel = label(cv2.resize(draw(img, seg, fr, K), (1005, 804)), "frame %s" % rows[i]["frame"])
        cv2.imwrite(os.path.join(out, "frame_median.png"),
                    np.vstack([panel, profile(fr, panel.shape[1], 240, a.margin)]))
    fig_time(rows, os.path.join(out, "sul_time.png"))
    print("wrote %s/{overlay.mp4, fig_3d.png, frame_median.png, sul_time.png, frames.csv, summary.json}"
          % out)


if __name__ == "__main__":
    main()
