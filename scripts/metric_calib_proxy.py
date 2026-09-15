#!/usr/bin/env python3
"""Absolute (mm) depth: calibrate each model ONCE on the ruler set, test on the stereo proxy-GT.

1. RULER PASS. Every annotated object in depthclips_ruler_NoGUI (ruler / catheter tip / robot arm,
   known mm) gives, at full res with the fixed da Vinci K, its in-plane ray length R at unit depth.
   The object's TRUE distance is z_true = mm / R; the model's is z_pred = mean predicted depth
   over the object's points. In-plane (not the 3D polyline) so depth roughness along a thin segment
   cannot bias the fit (CLAUDE_NOTES 2026-09-07).
2. FIT, frozen afterwards:
     scale+shift in inverse depth:  1/z_cal = s / z_pred + b   (robust least squares, log space)
     scale only:                    z_cal   = z_pred / s'      (s' = median z_pred / z_true)
3. PROXY PASS. Apply both frozen calibrations to all 83 proxy-GT frames and score the depth as
   ABSOLUTE mm against stereo (no per-frame scaling at all). The proxy patients are not in the
   ruler set, so this is a held-out test of "calibrate once, then measure".

Also reported from the ruler objects: the fit residual per class and the distance-tracking slope
(log z_cal vs log z_true; 1 = tracks distance, 0 = one constant distance).

Each model runs in its own subprocess with its own PYTHONPATH (zeroshot_proxy_gt.MODELS).

    python scripts/metric_calib_proxy.py --all                  # 6 models, then --summarize
    python scripts/metric_calib_proxy.py --model unidepth_v2_vitl
    python scripts/metric_calib_proxy.py --summarize
    python scripts/metric_calib_proxy.py --selftest
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

from zeroshot_proxy_gt import MODELS, boot_ci

SIX = ["endodac_warmstart", "unidepth_v2_vitl", "depth_anything_v2_large", "moge2_vitl",
       "metric3d_v2_vit_giant2", "depth_anything_v2_metric_indoor_large"]
K_NORM = (0.82, 1.02, 0.5, 0.5)                 # finetune_depth.DEFAULT_K_NORM, on the 5:4 content
HW = (1072, 1340)
CLASSES = {1: "ruler", 2: "catheter", 3: "arm"}


# ------------------------------------------------------------------------------ geometry / fit
def ray_length(pts, hw=HW, k=K_NORM):
    """Summed in-plane distance between consecutive rays at unit depth (pts: Nx2 full-res px)."""
    fx, fy, cx, cy = k[0] * hw[1], k[1] * hw[0], k[2] * hw[1], k[3] * hw[0]
    rays = np.stack([(pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy], 1)
    return float(np.linalg.norm(np.diff(rays, axis=0), axis=1).sum())


def fit_calibration(z_pred, z_true):
    """-> (s, b) with 1/z_cal = s/z_pred + b, and s_only with z_cal = z_pred/s_only."""
    from scipy.optimize import least_squares
    x, y = 1.0 / z_pred, 1.0 / z_true
    s_only = float(np.median(z_pred / z_true))
    s0 = float(np.median(y / x))

    def resid(p):                                   # log ratio of calibrated to true distance
        inv = p[0] * x + p[1]
        return np.log(np.clip(inv, 1e-9, None) / y)
    r = least_squares(resid, [s0, 0.0], loss="soft_l1", f_scale=0.1)
    s, b = map(float, r.x)
    if np.any(s * x + b <= 0):                      # shift made some depth negative: fall back
        s, b = 1.0 / s_only, 0.0
    return s, b, s_only


def apply_affine(z_pred, s, b):
    return 1.0 / np.clip(s / np.clip(z_pred, 1e-6, None) + b, 1e-6, None)


def tracking_slope(z_cal, z_true):
    lx, ly = np.log(z_true), np.log(z_cal)
    return float(np.polyfit(lx, ly, 1)[0])


# ------------------------------------------------------------------------------ data
def ruler_objects(root, exclude):
    """[(image path, pts Nx2, mm, class_id, source, video)] for every annotated object."""
    import json as js
    from finetune_depth import load_scale_anchors
    out = []
    for so in sorted(Path(root).glob("*/clip_*/scale_objects.json")):
        clip, video = so.parent, so.parent.parent.name
        if any(video.startswith(e) for e in exclude):
            continue
        imgs = sorted(p for p in (clip / "images").iterdir() if p.suffix in (".jpg", ".png"))
        raw = js.loads(so.read_text())["frames"]
        anchors = load_scale_anchors(clip, HW)
        for fi, objs in anchors.items():
            srcs = [o.get("source") or "manual" for o in raw[str(fi)] if len(o.get("points") or []) >= 2 and o.get("mm")]
            for (pts, mm, conf, cid), src in zip(objs, srcs):
                out.append((imgs[int(fi)], pts, mm, cid, src, video))
    return out


def sample(depth, pts):
    from scipy.ndimage import map_coordinates
    d = np.nan_to_num(depth, nan=np.inf, posinf=np.inf)
    v = map_coordinates(d, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
    return v[np.isfinite(v)]


# ------------------------------------------------------------------------------ one model
def run_model(name, ruler_root, proxy_dir, exclude, out):
    import torch
    from PIL import Image
    from eval_scared import EVAL_MAX, EVAL_MIN, compute_errors
    from zeroshot_proxy_gt import load_frames, make_predictor
    kind, src, _ = MODELS[name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    objs = ruler_objects(ruler_root, exclude)
    by_img = {}
    for i, o in enumerate(objs):
        by_img.setdefault(o[0], []).append(i)
    print(f"[{name}] {len(objs)} ruler objects on {len(by_img)} frames", flush=True)

    rec = np.full((len(objs), 2), np.nan)              # (z_pred, z_true)
    with torch.no_grad():
        f = make_predictor(kind, src, device)
        for n, (img, idx) in enumerate(by_img.items()):
            depth = f(Image.open(img).convert("RGB"), HW)
            for i in idx:
                _, pts, mm, _, _, _ = objs[i]
                z = sample(depth, pts)
                R = ray_length(pts)
                if len(z) == len(pts) and R > 0:
                    rec[i] = (float(np.mean(z)), mm / R)
            if n % 250 == 0:
                print(f"[{name}] ruler {n}/{len(by_img)}", flush=True)
        ok = np.isfinite(rec).all(1) & (rec[:, 0] > 0)
        s, b, s_only = fit_calibration(rec[ok, 0], rec[ok, 1])
        print(f"[{name}] calibration: 1/z = {s:.4g}/z_pred + {b:.4g} | scale-only z_pred/{s_only:.4g} "
              f"| {ok.sum()}/{len(objs)} objects used", flush=True)

        frames, gt = load_frames(proxy_dir)
        err = {"affine": [], "scale": []}
        for fp, g in zip(frames, gt):
            pred = f(Image.open(fp).convert("RGB"), g.shape)
            mask = (g > EVAL_MIN) & (g < EVAL_MAX)
            t = g[mask]
            for key, cal in (("affine", apply_affine(pred, s, b)), ("scale", pred / s_only)):
                p = np.clip(np.nan_to_num(cal[mask], nan=EVAL_MAX, posinf=EVAL_MAX), EVAL_MIN, EVAL_MAX)
                err[key].append(compute_errors(t, p))
    cls = np.array([o[3] for o in objs]); srcs = np.array([o[4] for o in objs])
    np.savez(out / "per_model" / f"{name}.npz", rec=rec, ok=ok, cls=cls, src=srcs,
             video=np.array([o[5] for o in objs]), s=s, b=b, s_only=s_only,
             affine=np.array(err["affine"]), scale=np.array(err["scale"]),
             frames=np.array([p.name for p in frames]))
    a = np.array(err["affine"])
    print(f"[{name}] proxy ABSOLUTE (scale+shift cal): abs_rel={a[:, 0].mean():.4f} "
          f"rmse={a[:, 2].mean():.2f}mm a1={a[:, 4].mean():.4f}", flush=True)


# ------------------------------------------------------------------------------ summary
def summarize(out, names):
    from eval_scared import METRIC_NAMES
    R = {n: np.load(out / "per_model" / f"{n}.npz") for n in names if (out / "per_model" / f"{n}.npz").exists()}
    base = "endodac_warmstart"
    res = {}
    for n, r in R.items():
        rec, ok, cls = r["rec"], r["ok"], r["cls"]
        zp, zt = rec[ok, 0], rec[ok, 1]
        za, zs = apply_affine(zp, float(r["s"]), float(r["b"])), zp / float(r["s_only"])
        fit = {}
        for key, zc in (("affine", za), ("scale", zs)):
            ratio = zc / zt
            fit[key] = {"abs_rel_all": float(np.median(np.abs(ratio - 1))),
                        **{f"abs_rel_{CLASSES[c]}": float(np.median(np.abs(ratio[cls[ok] == c] - 1)))
                           for c in CLASSES if (cls[ok] == c).any()},
                        "track_slope": tracking_slope(zc, zt)}
        res[n] = {"calibration": {"s": float(r["s"]), "b": float(r["b"]), "s_only": float(r["s_only"])},
                  "ruler_fit": fit,
                  **{f"proxy_{key}_{m}": float(v) for key in ("affine", "scale")
                     for m, v in zip(METRIC_NAMES, r[key].mean(0))}}
    order = sorted(res, key=lambda n: res[n]["proxy_affine_abs_rel"])
    print(f"\nPROXY-GT, ABSOLUTE mm after ONE ruler calibration (no per-frame scaling)")
    print(f"{'model':40s} {'abs_rel':>8s} {'rmse mm':>8s} {'a1':>6s} | {'scale-only abs_rel':>18s} | "
          f"d abs_rel vs EndoDAC [95% CI]")
    for n in order:
        d = R[n]["affine"][:, 0] - R[base]["affine"][:, 0] if base in R else np.zeros(1)
        ci = boot_ci(d) if n != base and base in R else (0.0, 0.0, 0.0)
        res[n]["d_proxy_affine_abs_rel_vs_endodac"] = ci
        print(f"{n:40s} {res[n]['proxy_affine_abs_rel']:8.4f} {res[n]['proxy_affine_rmse']:8.2f} "
              f"{res[n]['proxy_affine_a1']:6.3f} | {res[n]['proxy_scale_abs_rel']:18.4f} | "
              f"{ci[0]:+.4f} [{ci[1]:+.4f},{ci[2]:+.4f}]")
    print(f"\nRULER calibration fit (median |ratio-1| on the calibration objects) + distance tracking")
    print(f"{'model':40s} {'all':>6s} {'ruler':>6s} {'cath':>6s} {'arm':>6s} {'slope':>6s} | scale-only all / slope")
    for n in order:
        fa, fs = res[n]["ruler_fit"]["affine"], res[n]["ruler_fit"]["scale"]
        print(f"{n:40s} {fa['abs_rel_all']:6.3f} {fa.get('abs_rel_ruler', np.nan):6.3f} "
              f"{fa.get('abs_rel_catheter', np.nan):6.3f} {fa.get('abs_rel_arm', np.nan):6.3f} "
              f"{fa['track_slope']:6.3f} | {fs['abs_rel_all']:.3f} / {fs['track_slope']:.3f}")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out / 'results.json'}")


def selftest():
    rng = np.random.default_rng(0)
    zt = rng.uniform(30, 120, 400)
    zp = 1.0 / (0.5 / zt + 0.004)                       # affine in inverse depth
    s, b, s_only = fit_calibration(zp, zt)
    assert np.allclose(apply_affine(zp, s, b), zt, rtol=1e-3), (s, b)
    zc = 3.0 * zt
    assert abs(fit_calibration(zc, zt)[2] - 3.0) < 1e-9
    assert abs(tracking_slope(zt * 2, zt) - 1.0) < 1e-9 and abs(tracking_slope(np.full_like(zt, 50), zt)) < 1e-9
    pts = np.array([[670.0, 536.0], [670.0 + 0.82 * 1340 * 0.1, 536.0]])
    assert abs(ray_length(pts) - 0.1) < 1e-9          # 0.1 units of ray at unit depth
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler-root", default="../data/processed/depthclips_ruler_NoGUI/NOgui")
    ap.add_argument("--proxy-gt-dir", default="../data/processed/proxy_gt_nogui_ffs")
    ap.add_argument("--out", default="outputs/metric_calib_proxy")
    ap.add_argument("--exclude-videos", default="", help="comma-separated ruler video prefixes to skip (zoomed)")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", default=",".join(SIX))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = Path(a.out)
    (out / "per_model").mkdir(parents=True, exist_ok=True)
    exclude = [e for e in a.exclude_videos.split(",") if e]
    if a.model:
        return run_model(a.model, a.ruler_root, a.proxy_gt_dir, exclude, out)
    names = a.only.split(",")
    assert set(names) <= set(MODELS), set(names) - set(MODELS)
    if a.all:
        failed = []
        for n in names:
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(MODELS[n][2] + [os.environ.get("PYTHONPATH", "")])}
            r = subprocess.run([sys.executable, __file__, "--model", n, "--ruler-root", a.ruler_root,
                                "--proxy-gt-dir", a.proxy_gt_dir, "--out", a.out,
                                "--exclude-videos", a.exclude_videos], env=env)
            if r.returncode:
                failed.append(n)
                print(f"[FAILED] {n} (exit {r.returncode})", flush=True)
        print(f"[all] failed: {failed or 'none'}", flush=True)
    if a.all or a.summarize:
        summarize(out, names)


if __name__ == "__main__":
    main()
