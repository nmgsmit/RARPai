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
    """Least-squares (robust) on log(predicted / true) -- scale-free, same objective as training."""
    from scipy.optimize import least_squares
    l1 = lengths(rays, z, 1.0, 0.0, space)
    a0 = np.median(mm / np.maximum(l1, 1e-9)) if space == "depth" \
        else np.median(np.maximum(l1, 1e-9) / mm)

    def res(p):
        L = lengths(rays, z, p[0], p[1] if affine else 0.0, space)
        return np.log(np.maximum(L, 1e-9) / mm)

    p0 = [a0, 0.0] if affine else [a0]
    r = least_squares(res, p0, loss="soft_l1", f_scale=0.3, max_nfev=300)
    return r.x[0], (r.x[1] if affine else 0.0)


def evaluate(d, level, space, affine, cross_class=False):
    """Predicted length for every object, calibrated leave-one-out within its group.

    A global fit on 2 parameters over hundreds of objects is effectively free, so it is scored
    directly; local fits are scored LOO because there the fit could otherwise memorise its own
    target. Objects whose group cannot support the fit (too few anchors left, or -- for the
    affine forms -- no depth spread among them, which leaves the shift undetermined) are skipped
    and reported as coverage.
    """
    rays, z, mm, cls = d["rays"], d["z"], d["mm"], d["cls"]
    need = (2 if affine else 1)
    groups = {"global": np.zeros(len(mm), np.int64)}.get(level)
    if groups is None:
        _, groups = np.unique(d[level], return_inverse=True)
    pred = np.full(len(mm), np.nan)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for j in idx:
            if level == "global" and not cross_class:
                src = idx                                # 2 params over ~700 objects: no leakage
            elif cross_class:
                src = idx[cls[idx] != cls[j]]            # calibrate on the OTHER object classes
            else:
                src = idx[idx != j]                      # leave-one-object-out
            if len(src) < need:
                continue
            if affine and np.ptp(z[src].mean(1)) < 1e-3:  # no depth spread -> shift unidentifiable
                continue
            a, b = fit(rays[src], z[src], mm[src], space, affine)
            pred[j] = lengths(rays[j:j + 1], z[j:j + 1], a, b, space)[0]
    return pred


def report(d, rows):
    # ponytail: plain .format(), not nested f-string expressions -- Snellius runs Python 3.11,
    # where a multi-line expression inside f-string braces is a SyntaxError (PEP 701 is 3.12+).
    mm, cls = d["mm"], d["cls"]
    classes = sorted(set(cls.tolist()))
    per_hdr = "   | per class median %  (" + ", ".join(CLASS_NAMES[c] for c in classes) + ")"
    hdr = "{:<11}{:<14}{:<7}{:>6}{:>6}{:>8}{:>9}".format(
        "fit level", "params", "space", "n", "cov", "med %", "MAE mm") + per_hdr
    print(chr(10) + hdr)
    print("-" * len(hdr))
    for level, space, affine, cross, pred in rows:
        ok = np.isfinite(pred)
        name = ("cross-class " if cross else "") + ("affine" if affine else "scale")
        if ok.sum() < 3:
            print("{:<11}{:<14}{:<7}{:>6}{:>6}   (no group could support this fit)".format(
                level, name, space, 0, "--"))
            continue
        rel = np.abs(pred[ok] / mm[ok] - 1.0)
        cells = []
        for c in classes:
            m = ok & (cls == c)
            cells.append("{:>7.1f}".format(100 * np.median(np.abs(pred[m] / mm[m] - 1.0)))
                         if m.sum() >= 3 else "{:>7}".format("--"))
        print("{:<11}{:<14}{:<7}{:>6}{:>5.0f}%{:>8.1f}{:>9.2f}   |{}".format(
            level, name, space, int(ok.sum()), 100 * ok.mean(),
            100 * np.median(rel), np.abs(pred[ok] - mm[ok]).mean(), "".join(cells)))


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
    args = ap.parse_args()

    if not args.fit_only:
        dump_objects(args, args.dump)
    d = dict(np.load(args.dump, allow_pickle=False))
    print(f"[fit] {len(d['mm'])} objects | {len(set(d['video']))} videos "
          f"{len(set(d['clip']))} clips {len(set(d['frame']))} frames", flush=True)

    rows = []
    for level in ("global", "video", "clip", "frame"):
        for space, affine in (("depth", False), ("depth", True), ("disp", True)):
            rows.append((level, space, affine, False, evaluate(d, level, space, affine)))
    for level in ("global", "clip", "frame"):          # calibrate on cath+arm, predict ruler
        for space, affine in (("depth", False), ("disp", True)):
            rows.append((level, space, affine, True,
                         evaluate(d, level, space, affine, cross_class=True)))
    report(d, rows)


if __name__ == "__main__":
    main()
