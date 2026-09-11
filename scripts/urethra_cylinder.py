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

def fit_cylinder(P, init=None, r_range=(2.5, 8.0)):
    """Robust least squares on (distance to axis - r).

    Only the camera-facing part of the tube is ever seen, so the start point matters: axis along
    the points' principal direction, radius from their spread ACROSS it, and the axis one radius
    BEHIND the visible surface. init=(p, d, r) restarts from a previous fit instead.
    The radius is held to r_range: a partial arc fits a fatter, flatter cylinder almost as well,
    and the end-on 5e27 views ran to 8.5-11 mm that way. 2.5-8 mm brackets a urethra with room.
    ponytail: straight cylinder; a curved urethra biases the far end.
    """
    c = P.mean(0)
    if init is None:
        d0 = np.linalg.svd(P - c, full_matrices=False)[2][0]
        across = unit(np.cross(d0, unit(c)))
        r_sil = max(0.5, np.subtract(*np.percentile((P - c) @ across, [97, 3])) / 2)
        off, r0 = r_sil * unit(c), r_sil
    else:
        (p0, d0, r0), r_sil = init, init[2]
        off = p0 - c
    e1, e2 = basis(d0)
    r0 = float(np.clip(r0, r_range[0] + 1e-3, r_range[1] - 1e-3))

    def model(x):
        return unit(d0 + x[0] * e1 + x[1] * e2), c + x[2] * e1 + x[3] * e2

    def res(x):
        d, p = model(x)
        return np.linalg.norm(np.cross(P - p, d), axis=1) - x[4]

    s = least_squares(res, [0, 0, off @ e1, off @ e2, r0], loss="soft_l1", f_scale=0.3,
                      bounds=([-np.inf] * 4 + [r_range[0]], [np.inf] * 4 + [r_range[1]]))
    d, p = model(s.x)
    return p, d, float(s.x[4]), float(np.median(np.abs(res(s.x)))), r_sil


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
    """Tube top line minus observed surface (mm) at axial positions ts (NaN where unknown), and
    the segmentation class under each sample (-1 = outside the frame)."""
    H, W = zmap.shape
    A = p + ts[:, None] * d
    S = A + r * toward_camera(A, d)
    uv = proj(S, K)
    inside = ((uv[:, 0] >= patch) & (uv[:, 0] < W - patch - 1) &
              (uv[:, 1] >= patch) & (uv[:, 1] < H - patch - 1))
    zobs = np.full(len(ts), np.nan)
    cls = np.full(len(ts), -1)
    for i in np.flatnonzero(inside):
        ui, vi = int(round(uv[i, 0])), int(round(uv[i, 1]))
        cls[i] = 0 if seg is None else seg[vi, ui]
        if cls[i] == NONANAT:
            continue                     # an instrument in front says nothing about the tissue
        z = zmap[vi - patch:vi + patch + 1, ui - patch:ui + patch + 1]
        z = z[z > 0]
        if z.size:
            zobs[i] = np.median(z)
    return S[:, 2] - zobs, inside, cls


def march(p, d, r, t0, zmap, seg, K, ext, margin, sign=1, step=0.2, zero_tol=0.5, run_mm=1.0):
    """Walk the tube's top line from t0 and find where the observed surface stops being the tube.

    distal (sign +1): only a surface IN FRONT counts -- the roof.
    proximal (-1):    in front (the prostate base: the same kind of step as the roof), BEHIND (an
                      open, cut end), or the catheter, which leaves the cut end and would otherwise
                      read as more tube.
    If the tube was not seen in the last mm before the crossing (instrument / no depth), the
    crossing is not where the tube ends but where it reappears: that end is HIDDEN.
    """
    ts = t0 + sign * np.arange(0, ext, step)
    raw, inside, cls = gap_along(p, d, r, ts, zmap, seg, K)
    gap = nanmedian_filter(raw)
    g = np.nan_to_num(gap, nan=0.0)
    hit = (g > margin) if sign > 0 else ((np.abs(g) > margin) | (cls == CATHETER))
    k = max(1, int(round(run_mm / step)))
    run = np.flatnonzero(np.convolve(hit, np.ones(k), "valid") == k)
    out = dict(ts=ts, gap=gap, found=False, hidden=False)
    if run.size:
        j = i = run[0]
        t_cross = float(ts[i])
        if cls[i] != CATHETER:
            # the margin only makes the detection robust; the end is where the gap LEAVES zero.
            # Walk back to ~0, then put the crossing where the line through that sample and the
            # first sustained one hits zero -- a threshold alone lands ~zero_tol/slope past it,
            # and with both ends measured this way the two offsets would add up in SUL
            while j > 0 and ((g[j - 1] > zero_tol) if sign > 0 else (abs(g[j - 1]) > zero_tol)):
                j -= 1
            t_cross = float(ts[j])
            if i > j and g[i] != g[j]:
                t_cross = float(ts[j] + np.clip(g[j] / (g[j] - g[i]), -1, 1) * (ts[i] - ts[j]))
        front = bool(np.isfinite(gap[i:i + k]).any() and np.nanmedian(gap[i:i + k]) > 0)
        kind = ("catheter" if cls[i] == CATHETER else
                ("roof" if sign > 0 else "base") if front else "open end")
        if kind == "base" and cls[i] == PROSTATE:
            kind = "base (prostate)"
        out.update(found=True, t=t_cross, kind=kind, status=kind,
                   hidden=bool(j >= k and not np.isfinite(raw[j - k:j]).any()))
    elif not inside.all():
        out["status"] = "left frame at %+.1f mm" % (sign * (ts[np.argmin(inside)] - t0))
    else:
        out["status"] = "nothing within %g mm" % ext
    return out


def analyse(zmap, seg, K, erode=7, ext=30.0, margin=1.5, min_px=1500, roi_px=40, inlier_mm=1.0):
    """One frame -> cylinder, start, end, SUL (or None when there is too little urethra).

    The mask is only a rough WHERE. A first fit runs on its eroded core; then every depth point near
    the mask (instrument and catheter excluded) that lies on that surface is re-selected and the fit
    repeated, so the DEPTH decides what the tube is. Both ends come from the depth as well: walk the
    tube's top line each way until the surface stops being the tube.
    """
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

    P, puv = pts(core, 20000)
    p, d, r, res, r_sil = fit_cylinder(P)
    roi = (cv2.dilate(ok.astype(np.uint8), np.ones((2 * roi_px + 1,) * 2, np.uint8)).astype(bool)
           & (zmap > 0) & ~np.isin(seg, (NONANAT, CATHETER)))
    Q, quv = pts(roi, 40000)
    S, suv = P, puv
    for _ in range(3):
        on = np.abs(np.linalg.norm(np.cross(Q - p, d), axis=1) - r) < inlier_mm
        if on.sum() < min_px // 2:
            break
        S, suv = Q[on], quv[on]
        p, d, r, res, _ = fit_cylinder(S, init=(p, d, r))
    d, rule = orient(p, d, S, suv, seg)
    t0 = float(np.median((S - p) @ d))
    end = march(p, d, r, t0, zmap, seg, K, ext, margin, sign=+1)
    beg = march(p, d, r, t0, zmap, seg, K, ext, margin, sign=-1)
    t_mask = float(np.percentile((P - p) @ d, 1))      # where the MASK starts (eroded core)
    out = dict(p=p, d=d, r=r, r_sil=r_sil, res=res, rule=rule, n=len(S),
               t_start=beg["t"] if beg["found"] else t_mask, t_mask_start=t_mask,
               t_end=end.get("t"), found=end["found"], status=end["status"],
               ts=end["ts"], gap=end["gap"], ts_back=beg["ts"], gap_back=beg["gap"],
               start_found=beg["found"], start_kind=beg["status"],
               start_ok=beg["found"] and not beg["hidden"] and not end["hidden"])
    out["sul"] = out["t_end"] - out["t_start"] if end["found"] and beg["found"] else float("nan")
    out["sul_mask"] = out["t_end"] - t_mask if end["found"] else float("nan")
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
    for c, col in ((URETHRA, (0, 255, 255)), (PROSTATE, (255, 0, 255))):   # yellow, magenta
        m = seg == c
        out[m] = (0.55 * out[m] + 0.45 * np.array(col)).astype(np.uint8)
    if fr is None:
        return out
    t_end = fr["t_end"] if fr["found"] else fr["ts"][-1]
    for side in (-1, 1):
        cv2.polylines(out, [tube_line(fr, fr["t_start"], t_end, side, K)], False,
                      (255, 255, 0), 3, cv2.LINE_AA)
    far = tube_line(fr, t_end, t_end + 15, 0, K, 40)
    for a, b in zip(far[:-1:2], far[1::2]):             # dashed: the tube carrying on underneath
        cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), (230, 230, 230), 2, cv2.LINE_AA)
    dot = lambda t: tuple(map(int, tube_line(fr, t, t, 0, K, 1)[0]))
    cv2.circle(out, dot(fr["t_mask_start"]), 8, (255, 255, 255), 2)       # where the MASK starts
    cv2.circle(out, dot(fr["t_start"]), 11, (0, 255, 0), -1 if fr["start_ok"] else 3)
    if fr["found"]:
        cv2.circle(out, dot(t_end), 11, (0, 0, 255), -1)
    return out


def label(panel, text):
    t = cv2.copyMakeBorder(panel, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(t, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def profile(fr, w, h, margin, gmin=-6.0, gmax=12.0):
    img = np.full((h, w, 3), 24, np.uint8)
    if fr is None:
        cv2.putText(img, "no urethra in this frame", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return img
    L, R, T, B = 70, 20, 34, 30
    x0, x1 = float(fr["ts_back"].min()), float(fr["ts"].max())
    X = lambda t: int(L + (t - x0) / max(x1 - x0, 1.0) * (w - L - R))
    Y = lambda g: int(T + (gmax - np.clip(g, gmin, gmax)) / (gmax - gmin) * (h - T - B))
    cv2.line(img, (L, Y(0)), (w - R, Y(0)), (120, 120, 120), 1)
    for mg in (margin, -margin):
        for xx in range(L, w - R, 12):
            cv2.line(img, (xx, Y(mg)), (xx + 5, Y(mg)), (0, 140, 255), 1)
    for g in (-5, 0, 5, 10):
        cv2.putText(img, "%d" % g, (L - 34, Y(g) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    ts0 = fr["t_start"]
    for mm in range(int(np.ceil((x0 - ts0) / 5)) * 5, int(x1 - ts0) + 1, 5):
        cv2.putText(img, "%d" % mm, (X(ts0 + mm) - 6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
    for tt, gg in ((fr["ts_back"], fr["gap_back"]), (fr["ts"], fr["gap"])):
        ok = np.isfinite(gg)
        pts = np.array([[X(t), Y(np.nan_to_num(g))] for t, g in zip(tt, gg)], np.int32)
        for s, e in zip(*[np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]) == k) for k in (1, -1)]):
            cv2.polylines(img, [pts[s:e]], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (X(fr["t_mask_start"]), T), (X(fr["t_mask_start"]), h - B), (160, 160, 160), 1)
    cv2.line(img, (X(ts0), T), (X(ts0), h - B), (0, 255, 0), 2)
    if fr["found"]:
        cv2.line(img, (X(fr["t_end"]), T), (X(fr["t_end"]), h - B), (0, 0, 255), 2)
    if fr["found"] and fr["start_found"]:
        msg = ("SUL %.1f mm   start: %s   (mask start would give %.1f)   r %.1f mm"
               % (fr["sul"], fr["start_kind"], fr["sul_mask"], fr["r"]))
        if not fr["start_ok"]:
            msg += "   HIDDEN - not counted"
    else:
        msg = "end: %s   start: %s   r %.1f mm" % (fr["status"], fr["start_kind"], fr["r"])
    cv2.putText(img, msg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "gap = tube top - surface (mm); x = mm from start; grey = mask start",
                (L + 6, T + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
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
               label="an end hidden (not counted)")
    ax[0].plot(t, [r["sul_mask_mm"] for r in rows], "x", ms=4, color="0.6",
               label="with the mask's start instead")
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

def render(K, W, H, p, d, r, t_roof, beta=1.0, bg=75.0, t_len=60.0, base=False):
    """Ray-cast a tube on a background plane, with a roof meeting the tube's top line at t_roof and
    rising toward the camera past it. base=False: the tube is cut at t=0 (an open end, background
    behind). base=True: it carries on under a 'prostate' surface that meets the top line at t=0
    and rises toward the camera proximally."""
    fx, fy, cx, cy = K
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    D = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones(u.shape)], -1)
    a, b = np.cross(D, d), np.cross(p, d)                  # |lambda*a - b| = r
    A2, B, C = (a * a).sum(-1), (a * b).sum(-1), (b * b).sum() - r * r
    disc = B * B - A2 * C
    lc = np.where(disc > 0, (B - np.sqrt(np.maximum(disc, 0))) / A2, np.inf)
    tc = (np.nan_to_num(lc, posinf=0)[..., None] * D - p) @ d
    lc[(tc < (-30 if base else 0)) | (tc > t_len) | (lc <= 0)] = np.inf

    def plane(t_at, sgn):          # meets the top line at t_at, rises toward the camera along sgn*d
        A = p + t_at * d
        tw = toward_camera(A, d)
        N = unit(np.cross(sgn * d + beta * tw, np.cross(d, tw)))
        lam = ((A + r * tw) @ N) / (D @ N)
        lam[~((sgn * ((lam[..., None] * D - A) @ d)) > 0) | (lam <= 0)] = np.inf
        return lam

    lr = plane(t_roof, 1)
    lb = plane(0.0, -1) if base else np.full(lc.shape, np.inf)
    z = np.minimum.reduce([lc, lr, lb, np.full(lc.shape, bg)])
    seg = np.where(lc == z, URETHRA, np.where(lb == z, PROSTATE, 0)).astype(np.uint8)
    return z.astype(np.float32), seg


def self_test():
    K, W, H = (285.8, 285.8, 150.9, 133.5), 335, 268        # the rectified K at quarter size
    d_true, p_true, r_true, t_roof = unit(np.array([0.15, -1.0, 0.3])), np.array([2.0, 12, 55]), 4.0, 18.0
    rng = np.random.default_rng(0)
    noisy = lambda z: z + rng.normal(0, 0.15, z.shape).astype(np.float32)   # ~stereo noise
    run = lambda z, seg: analyse(z, seg, K, erode=2, min_px=200, roi_px=10)
    z, seg = render(K, W, H, p_true, d_true, r_true, t_roof)
    z = noisy(z)
    fr = run(z, seg)
    ang = np.degrees(np.arccos(np.clip(fr["d"] @ d_true, -1, 1)))
    print("self-test cut end : r %.2f (true %.1f)  axis error %.2f deg  SUL %.2f (true %.1f)  start=%s"
          % (fr["r"], r_true, ang, fr["sul"], t_roof, fr["start_kind"]))
    assert abs(fr["r"] - r_true) < 0.4, "radius"
    assert ang < 4, "axis direction (or orientation flipped)"
    assert fr["found"] and fr["start_ok"] and fr["start_kind"] == "open end", "cut end"
    assert abs(fr["sul"] - t_roof) < 1.5, "SUL, cut end"
    zb, segb = render(K, W, H, p_true, d_true, r_true, t_roof, base=True)
    frb = run(noisy(zb), segb)
    print("self-test prostate: SUL %.2f (true %.1f)  start=%s  rule=%s"
          % (frb["sul"], t_roof, frb["start_kind"], frb["rule"]))
    assert frb["start_ok"] and frb["start_kind"].startswith("base"), "prostate base as the start"
    assert abs(frb["sul"] - t_roof) < 1.5, "SUL, prostate base"
    z3, seg3 = z.copy(), seg.copy()
    z3[172:], seg3[172:] = 40.0, NONANAT     # an instrument across the proximal ~5 mm
    fr3 = run(z3, seg3)
    print("self-test instrument over the start -> start_ok=%s (%s)" % (fr3["start_ok"], fr3["start_kind"]))
    assert not fr3["start_ok"], "an instrument over the start was not caught"
    z2, seg2 = render(K, W, H, p_true, d_true, r_true, t_roof=500)      # no roof in view
    fr2 = run(z2, seg2)
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
                         start=None if fr is None else fr["start_kind"],
                         start_ok=fr is not None and fr["start_ok"],
                         sul_mm=float("nan") if fr is None else round(fr["sul"], 2),
                         sul_mask_mm=float("nan") if fr is None else round(fr["sul_mask"], 2),
                         radius_mm=float("nan") if fr is None else round(fr["r"], 2),
                         # half the 3D width across the tube: what a CIRCULAR tube would need
                         radius_silhouette_mm=float("nan") if fr is None else round(fr["r_sil"], 2),
                         fit_resid_mm=float("nan") if fr is None else round(fr["res"], 3),
                         orient=None if fr is None else fr["rule"]))
        print("  %s  px %6d  end %-22s  start %-18s ok %-5s  SUL %s (mask %s)  r %s" % (
            rows[-1]["frame"], rows[-1]["urethra_px"], rows[-1]["status"], rows[-1]["start"],
            rows[-1]["start_ok"], rows[-1]["sul_mm"], rows[-1]["sul_mask_mm"],
            rows[-1]["radius_mm"]), flush=True)

    with open(os.path.join(out, "frames.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    sul = np.array([r["sul_mm"] for r in rows], float)
    sulm = np.array([r["sul_mask_mm"] for r in rows], float)
    good = np.isfinite(sul) & np.array([r["start_ok"] for r in rows])   # both ends seen, not hidden
    summary = dict(run=a.run, frames=len(rows), fitted=sum(f[3] is not None for f in frames),
                   end_found=int(sum(r["status"] == "roof" for r in rows)),
                   both_ends_observed=int(good.sum()),
                   sul_median_mm=float(np.median(sul[good])) if good.any() else None,
                   sul_iqr_mm=[float(x) for x in np.percentile(sul[good], [25, 75])] if good.any() else None,
                   sul_mask_start_median_mm=float(np.nanmedian(sulm)) if np.isfinite(sulm).any() else None,
                   start_kinds={str(k): sum(r["start"] == k for r in rows) for k in set(r["start"] for r in rows)},
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
