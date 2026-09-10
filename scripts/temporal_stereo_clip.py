"""Temporal stereo: fill the holes ONE pair cannot see, using the neighbouring frames.

A single stereo pair leaves ~18% of the frame undetermined (measured: C-Fast-FoundationStereo
plus the left-right check, on the da Vinci SBS stills). Those holes are not noise -- they are
occlusions, specular highlights and textureless blood-covered tissue, i.e. places where *this*
pair carries no correspondence at all. But the endoscope and the tissue move, so a pixel with no
answer now very often had one 100 ms ago. This script transfers it along optical flow:

    20 fps window -> per-frame disparity (with holes) -> DIS flow chains -> temporal fusion

    python scripts/temporal_stereo_clip.py --video ../data/3D_ProxyGT/<clip>.mp4 \
        --start 12 --seconds 5 --matcher ffs --out outputs/temporal_stereo/demo

WHAT IS ASSUMED. Warping a neighbour's disparity into this frame assumes the tracked scene
point's DEPTH is unchanged over |dt| <= window/fps. That is false under camera motion (a 1 mm
push moves every depth by 1 mm), which is why --align-median exists: the median own-vs-warped
disparity offset is removed before fusing, so a global push/pull cannot bias the fill. What
survives is the local *shape*, which is exactly what the holes are missing.

WHY NOT JUST INPAINT THE HOLES. Because that invents geometry. Every filled pixel here comes
from a real stereo correspondence in another frame, gated by (a) forward-backward flow
consistency, (b) at least --min-support independent frames agreeing, (c) their spread being
tight. A filled pixel is a measurement moved, not a guess.

HOW THE QUALITY IS MEASURED (leave-one-out). The fusion is computed from neighbours ONLY, never
from the frame itself. Where the own-frame stereo is *also* valid the two can be compared -- the
"agree" column. It is an OPTIMISTIC bound on fill quality: pixels where own stereo succeeded
are, by construction, easier than the pixels being filled.
"""
import argparse
import hashlib
import json
import os
import sys
import warnings

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_stereo_charuco import OUT_W, OUT_H                              # noqa: E402
from make_stereo_proxy_gt import (DEPTH_SCALE, colorize, ffs_matcher, match,   # noqa: E402
                                  rect_maps, sgbm_matcher, specular_mask)


# ------------------------------------------------------------------ per-frame stereo

def cache_key(video, fields):
    """Cache identity = the clip plus everything that changes the DISPARITY. Fusion knobs are
    deliberately not in it: sweeping them is the whole point of having the cache."""
    blob = json.dumps([os.path.basename(video)] + [round(f, 6) if isinstance(f, float) else f
                                                   for f in fields], sort_keys=True)
    return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


def load_cache(d):
    man = os.path.join(d, "manifest.json")
    if not os.path.isfile(man):
        return None
    with open(man) as fh:
        m = json.load(fh)
    lefts, disps = [], []
    for i in m["idx"]:
        p = os.path.join(d, "%07d" % i)
        if not (os.path.isfile(p + "_left.png") and os.path.isfile(p + "_disp.npy")):
            return None
        lefts.append(cv2.imread(p + "_left.png"))
        disps.append(np.load(p + "_disp.npy").astype(np.float32))
    return lefts, disps, m["idx"], m["src_fps"], m["stride"]


def save_cache(d, lefts, disps, idx, src_fps, stride):
    os.makedirs(d, exist_ok=True)
    for l, dp, i in zip(lefts, disps, idx):
        p = os.path.join(d, "%07d" % i)
        cv2.imwrite(p + "_left.png", l)
        # float16: disparity here is 16..208 px, so the step is <=0.125 px, which at the
        # shortest working distance is 0.015 mm of depth -- far below the ~1 mm calibration.
        np.save(p + "_disp.npy", dp.astype(np.float16))
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump(dict(idx=idx, src_fps=src_fps, stride=stride), fh)


def stereo_pass(video, start, seconds, fps, maps, min_disp, num_disp, scale, matcher):
    """Sample a window at `fps` and run the ordinary single-pair pipeline on every frame.

    Frames are read SEQUENTIALLY (grab-and-skip) rather than by seeking each index: on H.264 a
    per-frame seek re-decodes from the previous keyframe, and this window is contiguous anyway.
    """
    (mxL, myL, vL), (mxR, myR, vR) = maps
    geom = vL & vR
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    stride = max(1, int(round(src_fps / fps)))
    i0 = int(round(start * src_fps))
    n = max(1, int(round(seconds * src_fps / stride)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    lefts, disps, idx = [], [], []
    for k in range(n):
        if k:
            for _ in range(stride - 1):
                cap.grab()
        ok, fr = cap.read()
        if not ok or fr.shape[1] != 1920:
            break
        left = cv2.remap(fr, mxL, myL, cv2.INTER_CUBIC)
        right = cv2.remap(fr, mxR, myR, cv2.INTER_CUBIC)
        d = match(matcher, left, right, min_disp, num_disp, scale)
        d[~geom] = np.nan
        d[specular_mask(left)] = np.nan
        lefts.append(left)
        disps.append(d.astype(np.float32))
        idx.append(i0 + k * stride)
        if (k + 1) % 10 == 0:
            print("    stereo %3d/%d" % (k + 1, n), flush=True)
    cap.release()
    return lefts, disps, idx, src_fps, stride, geom


# ------------------------------------------------------------------ flow plumbing

def _grid(shape):
    h, w = shape[:2]
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def dense_flows(grays, scale):
    """Consecutive DIS flows, forward and backward, at `scale` of full resolution.

    Half resolution on purpose: the flow FIELD only has to say which scene point a pixel is, and
    it is smooth; the disparity it carries is sampled at full resolution. DIS rather than
    Farneback -- same opencv main module, no new dependency, and several times faster at this
    size, which matters because a 5 s window needs 2*(N-1) flows.
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    sm = [cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for g in grays]
    fwd = [dis.calc(sm[i], sm[i + 1], None) for i in range(len(sm) - 1)]
    bwd = [dis.calc(sm[i + 1], sm[i], None) for i in range(len(sm) - 1)]
    return fwd, bwd


def compose(f1, f2):
    """flow a->b composed with flow b->c  ->  flow a->c, i.e. f1(p) + f2(p + f1(p))."""
    gx, gy = _grid(f1.shape)
    return f1 + cv2.remap(f2, gx + f1[..., 0], gy + f1[..., 1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def fb_error(f, r):
    """Round-trip error of a chain: f is t->s, r is s->t. Also returns the in-bounds mask.

    This is the occlusion detector. Where the scene point genuinely is not visible in frame s,
    the flow lands on whatever occluded it, and the way back does not return to where it
    started -- so the disparity that would be carried over is rejected before it is used.
    """
    h, w = f.shape[:2]
    gx, gy = _grid(f.shape)
    mx, my = gx + f[..., 0], gy + f[..., 1]
    e = f + cv2.remap(r, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inb = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return np.hypot(e[..., 0], e[..., 1]), inb


def upflow(flow, w, h):
    f = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    f[..., 0] *= w / float(flow.shape[1])
    f[..., 1] *= h / float(flow.shape[0])
    return f


def sample_nan(field, flow):
    """Sample `field` at p + flow(p). NaN-safe: normalised interpolation, and a pixel is only
    accepted when ALL FOUR source neighbours were valid -- otherwise a hole's edge bleeds a
    plausible-looking value one pixel outwards on every hop."""
    gx, gy = _grid(flow.shape)
    mx, my = gx + flow[..., 0], gy + flow[..., 1]
    ok = np.isfinite(field).astype(np.float32)
    v = cv2.remap(np.nan_to_num(field), mx, my, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w = cv2.remap(ok, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where(w > 0.999, v / np.maximum(w, 1e-6), np.nan)


# ------------------------------------------------------------------ temporal fusion

def chain_candidates(t, disps, fwd, bwd, k_max, fb_tol, align):
    """Every neighbour's disparity warped into frame t, gated by flow consistency.

    Chains are built INCREMENTALLY -- t->t+k is (t->t+k-1) composed with the next one-step flow
    -- so a +-k window costs O(k) composes, not O(k^2). Both directions of every chain are kept
    because the round-trip check in `fb_error` is what rejects occlusions.
    """
    n = len(disps)
    h, w = disps[t].shape
    cands, offs = [], []
    raw = np.zeros((h, w), np.uint8)        # candidates that HAD a value, before the flow gate
    for step, ahead, back in ((+1, fwd, bwd), (-1, bwd, fwd)):
        f = r = None
        for k in range(1, k_max + 1):
            s = t + step * k
            if not 0 <= s < n:
                break
            j = t + k - 1 if step > 0 else t - k        # index of the one-step flow to append
            f = ahead[j] if k == 1 else compose(f, ahead[j])
            r = back[j] if k == 1 else compose(back[j], r)
            err, inb = fb_error(f, r)
            keep = cv2.resize(((err <= fb_tol) & inb).astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST) > 0
            d = sample_nan(disps[s], upflow(f.copy(), w, h))
            raw += np.isfinite(d)
            d[~keep] = np.nan
            both = np.isfinite(d) & np.isfinite(disps[t])
            off = float(np.median(disps[t][both] - d[both])) if both.sum() > 1000 else 0.0
            offs.append(off)
            cands.append(d + off if align else d)
    return cands, offs, raw


def fuse_frame(own, cands, min_support, mad_tol, temporal_median=False):
    """-> (fused, filled, med, checkable, support, tight).

    Median, not mean: a chain that quietly tracked the wrong point produces an outlier, and with
    4-8 candidates the median ignores it where a mean would split the difference. The MAD gate
    then discards pixels whose candidates never agreed in the first place -- no support, no fill.

    `support` (how many candidates survived the flow gate) and `tight` (did they agree) come back
    so the REMAINING holes can be attributed: no neighbour had an answer / too few / they
    disagreed. Without that split there is no way to tell a hard occlusion from a tuning problem.
    """
    own_ok = np.isfinite(own)
    if not cands:
        z = np.zeros(own.shape, bool)
        return own.copy(), z, np.full_like(own, np.nan), z, np.zeros(own.shape, np.uint8), z
    stack = np.stack(cands)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        support = np.isfinite(stack).sum(0).astype(np.uint8)
        med = np.nanmedian(stack, 0)
        mad = np.nanmedian(np.abs(stack - med), 0)
    tight = np.isfinite(med) & (np.nan_to_num(mad, nan=1e9) <= mad_tol)
    good = (support >= min_support) & tight
    fused = np.where(own_ok, own, np.where(good, med, np.nan))
    if temporal_median:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            allmed = np.nanmedian(np.concatenate([stack, own[None]]), 0)
        fused = np.where(own_ok & good, allmed, fused)
    return fused, good & ~own_ok, med, good & own_ok, support, tight


# ------------------------------------------------------------------ drawing

def label(panel, text, size=0.62):
    t = cv2.copyMakeBorder(panel, 38, 8, 8, 8, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(t, text, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1,
                cv2.LINE_AA)
    return t


def color_bar(lo, hi, width, cmap, height=26):
    pad = 90
    grad = np.linspace(1 / lo, 1 / hi, max(40, width - 2 * pad))[None, :].repeat(height, 0)
    bar = colorize(grad, 1 / hi, 1 / lo, cmap)
    bar = cv2.copyMakeBorder(bar, 8, 24, pad, pad, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        mm = 1.0 / (1 / lo + frac * (1 / hi - 1 / lo))
        txt = "%.0f mm" % mm if frac == 1.0 else "%.0f" % mm
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        x = int(pad + frac * (width - 2 * pad) - (tw if frac == 1.0 else tw // 2))
        cv2.putText(bar, txt, (x, height + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return bar if bar.shape[1] == width else cv2.resize(bar, (width, bar.shape[0]))


def depth_panel(z, lo, hi, cmap, tint=None):
    """Colour is INVERSE depth (near = the warm end, matching gui_depth_measure) on a range
    FIXED for the whole clip -- a per-frame percentile stretch makes the video flicker and hides
    real depth change behind renormalisation. Unsolved pixels stay black."""
    ok = z > 0
    heat = colorize(np.where(ok, 1.0 / np.maximum(z, 1e-6), np.nan), 1 / hi, 1 / lo, cmap)
    if tint is not None and tint.any():
        heat[tint] = (0.35 * heat[tint] + 0.65 * np.array([255, 255, 255])).astype(np.uint8)
    return heat


def mosaic(panels, scale, lo, hi, cmap):
    row = np.hstack([label(cv2.resize(p, (int(OUT_W * scale), int(OUT_H * scale))), txt)
                     for p, txt in panels])
    return np.vstack([row, color_bar(lo, hi, row.shape[1], cmap)])


def to_depth(disp, fB):
    with np.errstate(divide="ignore", invalid="ignore"):
        z = fB / disp
    return np.where(np.isfinite(z) & (z > 0), z, 0.0)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calib/stereo_calib.json")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/temporal_stereo/demo")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the clip")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--min-disp", type=int, default=16)
    ap.add_argument("--num-disp", type=int, default=192)
    ap.add_argument("--scale", type=float, default=0.5, help="matcher resolution, see proxy GT")
    ap.add_argument("--matcher", default="ffs", choices=["sgbm", "ffs"])
    ap.add_argument("--ffs-root", default="~/Fast-FoundationStereo")
    ap.add_argument("--ffs-model",
                    default="~/Fast-FoundationStereo/weights/c-fast/model_best_bp2_serialize.pth")
    ap.add_argument("--ffs-iters", type=int, default=8)
    ap.add_argument("--window", type=int, default=4,
                    help="+-k frames fused (4 = +-0.2 s at 20 fps)")
    ap.add_argument("--flow-scale", type=float, default=0.5)
    ap.add_argument("--fb-tol", type=float, default=1.5, help="px round-trip flow error allowed")
    ap.add_argument("--min-support", type=int, default=2, help="neighbours that must agree")
    ap.add_argument("--mad-tol", type=float, default=1.5, help="px spread allowed among them")
    ap.add_argument("--no-align", dest="align", action="store_false",
                    help="do NOT remove the median own-vs-warped disparity offset")
    ap.add_argument("--temporal-median", action="store_true",
                    help="also replace VALID own pixels by the temporal median (de-flicker)")
    ap.add_argument("--panel-scale", type=float, default=0.46)
    ap.add_argument("--cmap", default="turbo", choices=["turbo", "magma"],
                    help="turbo separates the mid range this scene actually occupies; magma is "
                         "the gui_depth_measure convention")
    ap.add_argument("--pct", type=float, nargs=2, default=(2.0, 97.0),
                    help="percentiles of the clip's own depths that set the fixed colour range")
    ap.add_argument("--cache-dir", default="outputs/temporal_stereo/_disp_cache")
    ap.add_argument("--no-cache", dest="cache", action="store_false")
    ap.add_argument("--save-depth", action="store_true",
                    help="write uint16 mm*16 PNGs in the proxy-GT layout")
    a = ap.parse_args()

    with open(a.calib) as fh:
        cal = json.load(fh)
    fB = cal["f_times_B_rectified"]
    maps = (rect_maps(cal, False), rect_maps(cal, True))
    os.makedirs(a.out, exist_ok=True)
    cmap = dict(turbo=cv2.COLORMAP_TURBO, magma=cv2.COLORMAP_MAGMA)[a.cmap]
    geom = maps[0][2] & maps[1][2]

    # The matcher is ~95% of the runtime and depends on none of the fusion knobs, so its output
    # is cached: a window/tolerance sweep then costs seconds on a CPU node instead of a GPU job.
    cdir = os.path.join(a.cache_dir, cache_key(a.video, [a.start, a.seconds, a.fps, a.matcher,
                                                         a.scale, a.min_disp, a.num_disp,
                                                         a.ffs_iters if a.matcher == "ffs" else 0]))
    hit = load_cache(cdir) if a.cache else None
    if hit:
        lefts, disps, idx, src_fps, stride = hit
        print("stereo pass: %d frames from cache %s" % (len(disps), cdir), flush=True)
    else:
        matcher = sgbm_matcher
        if a.matcher == "ffs":
            matcher = ffs_matcher(a.ffs_root, a.ffs_model, a.ffs_iters, a.num_disp)
        print("stereo pass: %.1f s from %.1f s at %g fps" % (a.seconds, a.start, a.fps),
              flush=True)
        lefts, disps, idx, src_fps, stride, geom = stereo_pass(
            a.video, a.start, a.seconds, a.fps, maps, a.min_disp, a.num_disp, a.scale, matcher)
        if a.cache:
            save_cache(cdir, lefts, disps, idx, src_fps, stride)
            print("  cached to %s" % cdir, flush=True)
    n = len(disps)
    print("  %d frames (source %.2f fps, stride %d -> %.2f fps)"
          % (n, src_fps, stride, src_fps / stride), flush=True)
    # Ceiling. Outside the rectified overlap, and in the GUI banner, NO method can produce a
    # depth -- so every percentage below is also reported over `geom`, where the fraction of
    # holes closed is a fraction of the holes that were ever closable.
    print("  geometrically valid area: %.1f%% of the frame (the ceiling)"
          % (100 * geom.mean()), flush=True)
    if n < 3:
        raise SystemExit("need at least 3 frames")

    print("flow pass: %d DIS flows at scale %g" % (2 * (n - 1), a.flow_scale), flush=True)
    fwd, bwd = dense_flows([cv2.cvtColor(l, cv2.COLOR_BGR2GRAY) for l in lefts], a.flow_scale)

    print("fusion: window +-%d, fb-tol %.1f px, min-support %d, mad-tol %.1f px, align=%s"
          % (a.window, a.fb_tol, a.min_support, a.mad_tol, a.align), flush=True)
    rows, fused_all, offs_all, flick, pool = [], [], [], [], []
    for t in range(n):
        cands, offs, raw = chain_candidates(t, disps, fwd, bwd, a.window, a.fb_tol, a.align)
        fused, filled, med, checkable, support, tight = fuse_frame(
            disps[t], cands, a.min_support, a.mad_tol, a.temporal_median)
        own_ok = np.isfinite(disps[t])
        # Why each surviving hole survived. `blind` is the honest floor -- no frame in the
        # window had an answer at that scene point, so no amount of tuning reaches it. The
        # other three are knobs: the flow gate, --min-support, --mad-tol.
        rem = geom & ~own_ok & ~filled
        why = dict(blind=float((rem & (raw == 0)).mean()),
                   flow=float((rem & (raw > 0) & (support == 0)).mean()),
                   thin=float((rem & (support >= 1) & (support < a.min_support)).mean()),
                   disagree=float((rem & (support >= a.min_support) & ~tight).mean()))
        agree = np.nan
        if checkable.any():
            zo = to_depth(disps[t][checkable], fB)
            zm = to_depth(np.maximum(med[checkable], 1e-6), fB)
            agree = float(np.median(np.abs(zo - zm)))
        if t:                                   # 1-frame depth jitter, measured along the flow
            prev = sample_nan(disps[t - 1], upflow(bwd[t - 1].copy(), OUT_W, OUT_H))
            b = np.isfinite(prev) & own_ok
            if b.sum() > 1000:
                flick.append(float(np.median(np.abs(to_depth(disps[t][b], fB)
                                                    - to_depth(prev[b], fB)))))
        z = to_depth(fused, fB)
        if (z > 0).any():
            pool.append(z[z > 0][::37])
        fused_all.append(fused.astype(np.float32))
        offs_all.append(offs)
        rows.append(dict(frame=idx[t], own=float(own_ok.mean()),
                         fused=float(np.isfinite(fused).mean()), filled=float(filled.mean()),
                         own_g=float(own_ok[geom].mean()),
                         fused_g=float(np.isfinite(fused)[geom].mean()), agree_mm=agree,
                         support=float(support[filled].mean()) if filled.any() else 0.0,
                         **{"hole_" + k: v for k, v in why.items()}))
        print("  f%05d  own %5.1f%%  fused %5.1f%%  filled %4.1f%%  agree %s mm"
              "   left: %4.1f%% blind %4.1f%% flow %4.1f%% thin %4.1f%% disagree"
              % (idx[t], 100 * rows[-1]["own_g"], 100 * rows[-1]["fused_g"],
                 100 * rows[-1]["filled"], "%5.2f" % agree if np.isfinite(agree) else "   --",
                 100 * why["blind"], 100 * why["flow"], 100 * why["thin"],
                 100 * why["disagree"]), flush=True)

    lo, hi = np.percentile(np.concatenate(pool), list(a.pct))
    print("depth range for colour: %.0f..%.0f mm" % (lo, hi), flush=True)

    stem = os.path.splitext(os.path.basename(a.video))[0][-24:]
    two, three = os.path.join(a.out, "depth.mp4"), os.path.join(a.out, "compare.mp4")
    w2 = w3 = None
    for t in range(n):
        z_own, z_fus = to_depth(disps[t], fB), to_depth(fused_all[t], fB)
        filled = (z_fus > 0) & ~(z_own > 0)
        tsec = (idx[t] - idx[0]) / src_fps
        vo, vf = 100 * (z_own > 0)[geom].mean(), 100 * (z_fus > 0)[geom].mean()
        m2 = mosaic([(lefts[t], "rectified LEFT  %s  t=%.2fs" % (stem, tsec)),
                     (depth_panel(z_fus, lo, hi, cmap),
                      "temporal stereo depth   %.1f%% of the frame solved" % vf)],
                    a.panel_scale, lo, hi, cmap)
        m3 = mosaic([(lefts[t], "rectified LEFT   t=%.2fs" % tsec),
                     (depth_panel(z_own, lo, hi, cmap), "single pair   %.1f%% solved" % vo),
                     (depth_panel(z_fus, lo, hi, cmap, tint=filled),
                      "+ temporal (white = filled)   %.1f%% solved" % vf)],
                    a.panel_scale, lo, hi, cmap)
        if w2 is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w2 = cv2.VideoWriter(two, fourcc, a.fps, (m2.shape[1], m2.shape[0]))
            w3 = cv2.VideoWriter(three, fourcc, a.fps, (m3.shape[1], m3.shape[0]))
            cv2.imwrite(os.path.join(a.out, "figure.png"), m3)
        w2.write(m2)
        w3.write(m3)
        if a.save_depth:
            for sub in ("images", "depth"):
                os.makedirs(os.path.join(a.out, sub), exist_ok=True)
            cv2.imwrite(os.path.join(a.out, "images", "%07d.png" % idx[t]), lefts[t])
            cv2.imwrite(os.path.join(a.out, "depth", "%07d.png" % idx[t]),
                        np.clip(z_fus * DEPTH_SCALE, 0, 65535).astype(np.uint16))
    w2.release()
    w3.release()

    own = np.array([r["own_g"] for r in rows])
    fus = np.array([r["fused_g"] for r in rows])
    ag = np.array([r["agree_mm"] for r in rows], float)
    off = np.abs(np.concatenate([o for o in offs_all if o]))
    summary = dict(
        video=os.path.basename(a.video), start=a.start, seconds=a.seconds, fps=a.fps,
        frames=n, src_fps=src_fps, stride=stride, matcher=a.matcher, scale=a.scale,
        window=a.window, flow_scale=a.flow_scale, fb_tol=a.fb_tol, min_support=a.min_support,
        mad_tol=a.mad_tol, align=a.align, temporal_median=a.temporal_median, f_times_B=fB,
        colour_range_mm=[float(lo), float(hi)], geom_frac=float(geom.mean()),
        note="valid_* are fractions of the geometrically usable area, not of the whole frame",
        valid_single=float(own.mean()), valid_single_p10=float(np.percentile(own, 10)),
        valid_fused=float(fus.mean()), valid_fused_p10=float(np.percentile(fus, 10)),
        valid_gain=float(fus.mean() - own.mean()),
        holes_closed=float((fus.mean() - own.mean()) / max(1e-9, 1 - own.mean())),
        agree_mm_median=float(np.nanmedian(ag)),
        holes_left_blind=float(np.mean([r["hole_blind"] for r in rows])),
        holes_left_flow=float(np.mean([r["hole_flow"] for r in rows])),
        holes_left_thin=float(np.mean([r["hole_thin"] for r in rows])),
        holes_left_disagree=float(np.mean([r["hole_disagree"] for r in rows])),
        align_offset_px_median=float(np.median(off)) if off.size else None,
        align_offset_px_p95=float(np.percentile(off, 95)) if off.size else None,
        flicker_mm_median=float(np.median(flick)) if flick else None,
        per_frame=rows)
    with open(os.path.join(a.out, "stats.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n(percentages below are of the %.1f%% of the frame that is geometrically usable)"
          % (100 * geom.mean()))
    print("single pair  valid %.1f%% (p10 %.1f%%)"
          % (100 * own.mean(), 100 * np.percentile(own, 10)))
    print("+ temporal   valid %.1f%% (p10 %.1f%%)  -> %.0f%% of the holes closed"
          % (100 * fus.mean(), 100 * np.percentile(fus, 10), 100 * summary["holes_closed"]))
    print("holes left (%% of frame): %.1f blind (no frame in the window saw it), "
          "%.1f flow-gated, %.1f too few, %.1f disagreed"
          % (100 * summary["holes_left_blind"], 100 * summary["holes_left_flow"],
             100 * summary["holes_left_thin"], 100 * summary["holes_left_disagree"]))
    print("leave-one-out agreement (optimistic bound): median %.2f mm"
          % summary["agree_mm_median"])
    if summary["align_offset_px_median"] is not None:
        print("own-vs-warped disparity offset: median %.3f px, p95 %.3f px"
              % (summary["align_offset_px_median"], summary["align_offset_px_p95"]))
    if summary["flicker_mm_median"] is not None:
        print("frame-to-frame depth jitter along the flow: median %.2f mm"
              % summary["flicker_mm_median"])
    print("wrote %s, %s, figure.png, stats.json" % (two, three))


if __name__ == "__main__":
    main()
