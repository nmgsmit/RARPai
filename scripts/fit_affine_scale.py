"""
Calibrate a FROZEN depth model to metric, without training it.

EndoDAC is affine-invariant, so its output is only correct up to two unknowns. This fits them on
the known-size objects in scale_objects.json and reports what the depth map is then worth, in mm.

Two things are varied:

  PARAMETERISATION -- `scale` (one multiplier, what eval_scale_table.py reports), `affine-depth`
    (Z' = a*Z + b, the form written in SUL_10_week_plan Phase 3) and `affine-disp`
    (1/Z' = a/Z + b, affine in DISPARITY -- the convention MiDaS/DepthAnything-family models,
    including EndoDAC's backbone, are actually invariant under). Which of the two affine forms
    wins is itself the answer to "where does the residual error live".

  FIT LEVEL -- one fit globally / per video / per clip / per frame. Local fits absorb drift but
    have fewer anchors to constrain them.

Every LOCAL fit is scored leave-one-object-out: the held-out object's length is predicted from a
calibration fitted on the OTHER anchors only. Fitting and scoring on the same object is circular,
and LOO is also what deployment does -- calibrate on the catheter/instrument, measure the urethra.
`--cross-class` goes further and calibrates on the catheter+arm only, then predicts the ruler.

Stage 1 (GPU) dumps per-object rays/depths to an .npz; stage 2 (CPU, seconds) does the fitting,
so the grid can be re-run without touching a GPU.

    python scripts/fit_affine_scale.py --dump outputs/affine_test.npz --split test     # stage 1+2
    python scripts/fit_affine_scale.py --dump outputs/affine_test.npz --fit-only       # stage 2
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

CLASS_NAMES = {1: "Ruler", 2: "Catheter tip", 3: "Robot arm"}


# ------------------------------------------------------------------ stage 1: dump
def dump_objects(args, out_path):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from finetune_depth import (RARPTriplets, build_depth_model, _filter_load, disp_to_depth,
                                round14, round32, split_by_video)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_shape = (round14(args.image_shape[0]), round14(args.image_shape[1]))
    hw = (round32(args.image_shape[0]), round32(args.image_shape[1]))
    tr, va, te = split_by_video(args.data_root, *args.video_split)
    dirs = {"train": tr, "val": va, "test": te, "all": tr + va + te}[args.split]

    ds = RARPTriplets(None, hw, tuple(args.intrinsics), args.frame_stride, clip_dirs=dirs,
                      anchors_max=4, mask_overlay=False)
    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.workers)
    model = build_depth_model(model_shape, device)
    _filter_load(model, args.ckpt, "frozen")
    model.eval()

    rays_l, z_l, mm_l, cls_l, clip_l, vid_l, frm_l = [], [], [], [], [], [], []
    base = 0
    with torch.no_grad():
        for batch in loader:
            bs = batch["anch_w"].shape[0]
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            disp = F.interpolate(model(b[("color", 0)])[("disp", 0)], hw,
                                 mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, args.min_depth, args.max_depth)
            pts = b["anch_pts"]
            B, N, P, _ = pts.shape
            g = torch.stack([pts[..., 0] / (hw[1] - 1) * 2 - 1,
                             pts[..., 1] / (hw[0] - 1) * 2 - 1], -1)
            z = F.grid_sample(depth, g.view(B, N * P, 1, 2), align_corners=True,
                              padding_mode="border").view(B, N, P)
            hom = torch.cat([pts, torch.ones_like(pts[..., :1])], -1)
            rays = torch.einsum("bij,bnpj->bnpi", b["inv_K"][:, :3, :3], hom)
            for i in range(bs):                          # shuffle=False -> index is positional
                frames, c = ds.samples[base + i]
                clip = str(frames[0].parent)
                for j in range(N):
                    if b["anch_w"][i, j] <= 0:
                        continue
                    rays_l.append(rays[i, j].cpu().numpy())
                    z_l.append(z[i, j].cpu().numpy())
                    mm_l.append(float(b["anch_mm"][i, j]))
                    cls_l.append(int(b["anch_cls"][i, j]))
                    clip_l.append(clip)
                    vid_l.append(frames[0].parent.parent.parent.name)
                    frm_l.append(f"{clip}#{c}")
            base += bs
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, rays=np.array(rays_l, np.float64), z=np.array(z_l, np.float64),
                        mm=np.array(mm_l, np.float64), cls=np.array(cls_l, np.int64),
                        clip=np.array(clip_l), video=np.array(vid_l), frame=np.array(frm_l))
    print(f"[dump] {len(mm_l)} objects -> {out_path}", flush=True)


# ------------------------------------------------------------------- stage 2: fit
def lengths(rays, z, a, b, space):
    """3D polyline length of each object under the calibration (a, b)."""
    if space == "depth":
        zp = a * z + b                                   # Z' = a*Z + b
    else:
        zp = 1.0 / np.maximum(a / np.maximum(z, 1e-9) + b, 1e-9)   # 1/Z' = a/Z + b
    p = rays * zp[..., None]
    return np.linalg.norm(np.diff(p, axis=1), axis=2).sum(1)


def fit(rays, z, mm, space, affine):
    """Robust fit of the calibration (a, b). numpy only -- no scipy in the Snellius venv.

    Both parameterisations are positively homogeneous once the RATIO r = b/a is fixed:
      depth:  Z' = a*Z + b = a*(Z + r)        ->  L(a, a*r) = a * L(1, r)
      disp:   1/Z' = a/Z + b = a*(1/Z + r)    ->  L(a, a*r) = L(1, r) / a
    So for any r the best scale is closed form in log space -- the MEDIAN of log(L/mm), which is
    the L1-optimal (hence outlier-robust) offset of a log residual. That collapses the 2-D fit to
    a 1-D search over r: a coarse grid plus shrinking refinements.

    ponytail: shorter and more robust than a generic least_squares, and it drops a dependency.
    """
    v = z if space == "depth" else 1.0 / np.maximum(z, 1e-9)
    vmin = float(v.min())

    def solve(r):
        """-> (log-offset, robust cost) for this ratio r."""
        L = lengths(rays, z, 1.0, r, space)
        g = np.isfinite(L) & (L > 1e-9)
        if not g.any():
            return 0.0, np.inf
        res = np.log(L[g] / mm[g])
        off = float(np.median(res))
        return off, float(np.median(np.abs(res - off)))

    def to_a(off):
        return float(np.exp(-off if space == "depth" else off))

    if not affine:
        return to_a(solve(0.0)[0]), 0.0

    floor = -0.98 * vmin                       # keep Z' (or the disparity) strictly positive
    lo, hi = floor, 5.0 * float(np.median(v))
    best_r, best_c = 0.0, np.inf
    for _ in range(4):
        for r in np.linspace(lo, hi, 61):
            c = solve(float(r))[1]
            if c < best_c:
                best_c, best_r = c, float(r)
        span = (hi - lo) / 12.0
        lo, hi = max(best_r - span, floor), best_r + span
    a = to_a(solve(best_r)[0])
    return a, a * best_r


def evaluate(d, level, space, affine, cross_class=False, calib_classes=None,
             min_spread=0.0):
    """Predicted length for every object, calibrated leave-one-out within its group.

    A global fit on 2 parameters over hundreds of objects is effectively free, so it is scored
    directly; local fits are scored LOO because there the fit could otherwise memorise its own
    target. Objects whose group cannot support the fit (too few anchors left, or -- for the
    affine forms -- no depth spread among them, which leaves the shift undetermined) are skipped
    and reported as coverage.
    """
    rays, z, mm, cls = d["rays"], d["z"], d["mm"], d["cls"]
    need = (2 if affine else 1)
    calib = np.isin(cls, calib_classes) if calib_classes else None
    groups = {"global": np.zeros(len(mm), np.int64)}.get(level)
    if groups is None:
        _, groups = np.unique(d[level], return_inverse=True)
    pred = np.full(len(mm), np.nan)
    cache = {}
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for j in idx:
            if calib is not None:
                # DEPLOYMENT CASE: only one object class is ever visible, so calibrate on it and
                # measure something else. Source and target are disjoint by class, so no LOO
                # needed -- and objects OF the calibration class are not scored at all.
                if calib[j]:
                    continue
                src = idx[calib[idx]]
            elif level == "global" and not cross_class:
                src = idx                                # 2 params over ~700 objects: no leakage
            elif cross_class:
                src = idx[cls[idx] != cls[j]]            # calibrate on the OTHER object classes
            else:
                src = idx[idx != j]                      # leave-one-object-out
            if len(src) < need:
                continue
            # The shift is only identifiable if the calibration anchors sit at DIFFERENT
            # depths. Pooling one object class over time supplies that spread only if the object
            # actually moves in depth -- min_spread makes the requirement explicit.
            if affine and np.ptp(z[src]) < max(min_spread, 1e-3):
                continue
            key = src.tobytes()
            if key not in cache:
                cache[key] = fit(rays[src], z[src], mm[src], space, affine)
            a, b = cache[key]
            pred[j] = lengths(rays[j:j + 1], z[j:j + 1], a, b, space)[0]
    return pred


def diagnose(d):
    """Depth spread is the whole story for the shift, so measure where it can come from.

    WITHIN-object spread is what a long cylinder gives you inside a single frame (the annotated
    segment recedes in depth along the shaft). BETWEEN-object spread is what pooling over time
    gives you (the anchor moves). The shift b is conditioned on spread RELATIVE to mean depth --
    a few mm of spread at a ~40mm working distance is nearly degenerate however many samples
    you stack up.
    """
    print(chr(10) + "[spread] depth diversity available to an affine fit")
    print("{:<14}{:>7}{:>12}{:>12}{:>12}{:>12}".format(
        "class", "n", "within-obj", "in-frame", "in-clip", "in-video"))
    for c in sorted(set(d["cls"].tolist())):
        m = d["cls"] == c
        zc, zm = d["z"][m], d["z"][m].mean(1)
        within = float(np.median(np.ptp(zc, axis=1)))          # along the annotated segment

        def pooled(key):
            v = [np.ptp(zm[d[key][m] == k]) for k in sorted(set(d[key][m]))]
            return float(np.median(v)) if v else 0.0

        mean_z = float(np.median(zm))
        print("{:<14}{:>7}{:>11.1f}{:>12.1f}{:>12.1f}{:>12.1f}   ({:.0f}% of {:.0f}mm)".format(
            CLASS_NAMES[c], int(m.sum()), within, pooled("frame"), pooled("clip"),
            pooled("video"), 100 * pooled("clip") / max(mean_z, 1e-9), mean_z))


def report(d, rows):
    # ponytail: plain .format(), not nested f-string expressions -- Snellius runs Python 3.11,
    # where a multi-line expression inside f-string braces is a SyntaxError (PEP 701 is 3.12+).
    # Both a mean AND a median mm column: an under-determined fit (few anchors, no depth spread)
    # can extrapolate to absurd lengths, and the gap between the two columns is the tell.
    mm, cls = d["mm"], d["cls"]
    classes = sorted(set(cls.tolist()))
    per_hdr = "   | per class median %  (" + ", ".join(CLASS_NAMES[c] for c in classes) + ")"
    hdr = "{:<11}{:<20}{:<7}{:>6}{:>6}{:>8}{:>9}{:>10}".format(
        "fit level", "params", "space", "n", "cov", "med %", "med mm", "mean mm") + per_hdr
    print(chr(10) + hdr)
    print("-" * len(hdr))
    for level, space, affine, cross, pred in rows:
        ok = np.isfinite(pred)
        name = (cross if isinstance(cross, str) else ("cross-class " if cross else "")) \
            + ("affine" if affine else "scale")
        if ok.sum() < 3:
            print("{:<11}{:<20}{:<7}{:>6}{:>6}   (no group could support this fit)".format(
                level, name, space, 0, "--"))
            continue
        rel = np.abs(pred[ok] / mm[ok] - 1.0)
        dev = np.abs(pred[ok] - mm[ok])
        cells = []
        for c in classes:
            m = ok & (cls == c)
            cells.append("{:>7.1f}".format(100 * np.median(np.abs(pred[m] / mm[m] - 1.0)))
                         if m.sum() >= 3 else "{:>7}".format("--"))
        print("{:<11}{:<20}{:<7}{:>6}{:>5.0f}%{:>8.1f}{:>9.2f}{:>10.2f}   |{}".format(
            level, name, space, int(ok.sum()), 100 * ok.mean(),
            100 * np.median(rel), np.median(dev), dev.mean(), "".join(cells)))


def _selfcheck():
    """Plant a known (a, b) in synthetic data and check the fit recovers the resulting lengths.

    (a, b) itself is only identifiable up to what the geometry constrains, so the assertion is on
    the thing we actually report -- the predicted length -- not on the raw parameters.
    """
    rng = np.random.default_rng(0)
    n, p = 40, 5
    z = rng.uniform(30.0, 90.0, (n, p))
    z = np.sort(z, axis=1)
    rays = np.stack([rng.uniform(-0.3, 0.3, (n, p)), rng.uniform(-0.3, 0.3, (n, p)),
                     np.ones((n, p))], -1)
    for space, (a, b) in (("depth", (0.7, 12.0)), ("disp", (1.4, 0.004))):
        mm = lengths(rays, z, a, b, space)
        assert np.all(mm > 0)
        af, bf = fit(rays, z, mm, space, affine=True)
        got = lengths(rays, z, af, bf, space)
        err = np.median(np.abs(got / mm - 1.0))
        assert err < 0.02, (space, err, af, bf)
        # scale-only must NOT reproduce a length field that genuinely needed a shift
        a1, b1 = fit(rays, z, mm, space, affine=False)
        assert b1 == 0.0
        assert np.median(np.abs(lengths(rays, z, a1, b1, space) / mm - 1.0)) > err
    print("[selfcheck] affine fit recovers planted calibrations in both spaces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="outputs/affine_test.npz")
    ap.add_argument("--fit-only", action="store_true", help="reuse an existing --dump (CPU only)")
    ap.add_argument("--ckpt", default="../backbones/EndoDAC/depth_model.pth")
    ap.add_argument("--data-root", default="../data/processed/depthclips_ruler_NoGUI")
    ap.add_argument("--video-split", type=int, nargs=2, default=[4, 5])
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--image-shape", type=int, nargs=2, default=[392, 490])
    ap.add_argument("--intrinsics", type=float, nargs=4, default=[0.82, 1.02, 0.5, 0.5])
    ap.add_argument("--frame-stride", type=int, default=1)
    # 20/200 mm, NOT the 0.1/150 KITTI-unit defaults: those compress the 35-120mm endoscopic
    # band into 0.2% of the sigmoid range and make every metric number an artifact.
    # See CLAUDE_NOTES 2026-09-02 "--min-depth/--max-depth WAS THE BUG".
    ap.add_argument("--min-depth", type=float, default=20.0)
    ap.add_argument("--max-depth", type=float, default=200.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--calib-classes", type=int, nargs="+", default=None, metavar="ID",
                    help="DEPLOYMENT case: calibrate using ONLY these class_ids (1=Ruler, "
                         "2=Catheter tip, 3=Robot arm) and score only the other classes. "
                         "Use --calib-classes 3 when the robot arm is the only anchor in frame.")
    ap.add_argument("--selfcheck", action="store_true",
                    help="verify the fit on synthetic data with a planted calibration, then exit")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    if not args.fit_only:
        dump_objects(args, args.dump)
    d = dict(np.load(args.dump, allow_pickle=False))
    print(f"[fit] {len(d['mm'])} objects | {len(set(d['video']))} videos "
          f"{len(set(d['clip']))} clips {len(set(d['frame']))} frames", flush=True)

    # How often is a lone anchor class even present? A calibration you cannot compute is a
    # deployment failure, so coverage matters as much as the error when it works.
    for c in sorted(set(d["cls"].tolist())):
        m = d["cls"] == c
        fr = len(set(d["frame"][m])) / len(set(d["frame"]))
        cl = len(set(d["clip"][m])) / len(set(d["clip"]))
        zs = d["z"][m].mean(1)
        spread = [np.ptp(d["z"][m & (d["clip"] == k)].mean(1))
                  for k in sorted(set(d["clip"][m]))]
        print("[avail] {:<13} n={:<4} in {:>5.0f}% of frames, {:>5.0f}% of clips | "
              "depth {:.0f}-{:.0f}mm, median within-clip spread {:.1f}mm".format(
                  CLASS_NAMES[c], int(m.sum()), 100 * fr, 100 * cl,
                  zs.min(), zs.max(), float(np.median(spread)) if spread else 0.0))

    diagnose(d)

    rows = []
    for level in ("global", "video", "clip", "frame"):
        for space, affine in (("depth", False), ("depth", True), ("disp", True)):
            rows.append((level, space, affine, False, evaluate(d, level, space, affine)))
    for level in ("global", "clip", "frame"):          # calibrate on cath+arm, predict ruler
        for space, affine in (("depth", False), ("disp", True)):
            rows.append((level, space, affine, True,
                         evaluate(d, level, space, affine, cross_class=True)))
    if args.calib_classes:                             # e.g. --calib-classes 3 = arm only
        tag = "arm-only " if args.calib_classes == [3] else "calib%s " % args.calib_classes
        for level in ("global", "video", "clip", "frame"):
            for space, affine in (("depth", False), ("depth", True), ("disp", True)):
                rows.append((level, space, affine, tag,
                             evaluate(d, level, space, affine,
                                      calib_classes=args.calib_classes)))
        # Does MORE depth spread rescue the shift? Same fit, but only on groups whose anchors
        # actually span this many mm. If affine never beats scale even at the top of the range
        # the data has, pooling one anchor class cannot identify the shift here.
        for ms in (3.0, 6.0, 10.0, 15.0):
            for space in ("depth", "disp"):
                rows.append(("clip", space, True, tag + ">=%.0fmm " % ms,
                             evaluate(d, "clip", space, True,
                                      calib_classes=args.calib_classes, min_spread=ms)))
    report(d, rows)


if __name__ == "__main__":
    main()
