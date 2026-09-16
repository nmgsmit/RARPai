#!/usr/bin/env python3
"""How long does each depth model MEASURE a ruler of known length? Per annotated Ruler: model mm vs true mm.

Every Ruler object (class 1) in depthclips_ruler_NoGUI -- annotator-typed mm, 5 points a..b along it -- is
measured on each model's depth map three ways:
  chord     3D distance between the two END points, each at its own 5x5 patch-median depth
            (= a line drawn in gui_depth_measure)
  polyline  summed 3D distance over the 5 points, bilinear depth (= finetune_depth.scale_loss, what sw05 trained on)
  inplane   a..b with every point at the object's mean depth (= distance only; what metric_calib_proxy scores)
and two depth scalings (a depth model is never used unscaled):
  scale     z / s_only      fitted LEAVE-ONE-SURGERY-OUT on all classes' distances (as metric_calib_proxy)
  affine    1/z = s/z + b   same LOO fit
Deviation = measured - true mm per ruler -> mean, mean |dev|, median |dev|, IQR, min, max.
K = da Vinci K_NORM on the 1340x1072 content frame, as the annotations and the calibration.

    sbatch jobs/ruler_length_models.sh --model unidepth_v2_vitl     # CPU (genoa), one job per model
    python scripts/ruler_length_models.py --summarize               # CPU seconds, runs anywhere
    python scripts/ruler_length_models.py --overlays --only unidepth_v2_vitl,depth_anything_v2_metric_indoor_large
    python scripts/ruler_length_models.py --selftest
"""
import argparse, csv, os, subprocess, sys
from pathlib import Path

import numpy as np

from metric_calib_proxy import HW, K_NORM, apply_affine, fit_calibration, ruler_objects

RULER = "../data/processed/depthclips_ruler_NoGUI/NOgui"
OUT = Path("outputs/ruler_length_models")
MODELS = ["unidepth_v2_vitl", "metric3d_v2_vit_giant2", "moge2_vitl", "depth_anything_v2_large",
          "depth_anything_v2_metric_indoor_large", "endodac_warmstart", "ft_ruler_sw05",
          "vda_metric_large", "vda_large", "da3_large_mv"]
SW05_TEST = ("9192f353", "84f43102", "494b85c9", "68ab379c", "4c4ae254")   # sw05's held-out surgeries
P, R = 5, 2                                                                # points per object, patch radius


# ------------------------------------------------------------------------------ geometry
def backproject(pts, z, k=K_NORM, hw=HW):
    """pts (..., 2) full-res px, z (...) mm -> (..., 3) camera coords in mm."""
    fx, fy, cx, cy = k[0] * hw[1], k[1] * hw[0], k[2] * hw[1], k[3] * hw[0]
    z = np.broadcast_to(z, pts.shape[:-1])
    return np.stack([(pts[..., 0] - cx) / fx * z, (pts[..., 1] - cy) / fy * z, z], -1)


def lengths(pts, zb, ze):
    """pts (N,P,2), zb (N,P) bilinear depth, ze (N,2) end patch depth -> dict of (N,) mm."""
    chord = np.linalg.norm(backproject(pts[:, -1], ze[:, 1]) - backproject(pts[:, 0], ze[:, 0]), axis=-1)
    X = backproject(pts, zb)
    poly = np.linalg.norm(np.diff(X, axis=1), axis=-1).sum(1)
    zm = np.mean(zb, 1)
    inplane = np.linalg.norm(backproject(pts[:, -1], zm) - backproject(pts[:, 0], zm), axis=-1)
    return {"chord": chord, "polyline": poly, "inplane": inplane}


def unit_ray(pts):
    """In-plane a..b length at unit depth -> true distance = mm / unit_ray."""
    return np.linalg.norm(backproject(pts[:, -1], 1.0) - backproject(pts[:, 0], 1.0), axis=-1)


# ------------------------------------------------------------------------------ CPU/GPU: depth at the points
def predict(name, out):
    import torch
    from PIL import Image
    from scipy.ndimage import map_coordinates
    import video_depth_proxy_gt as vd                   # image AND video models behind one f(frames)
    objs = [o for o in ruler_objects(RULER, []) if len(o[1]) == P]
    by_clip = {}
    for i, o in enumerate(objs):
        by_clip.setdefault(o[0].parent.parent, {}).setdefault(o[0], []).append(i)
    pts = np.stack([o[1] for o in objs]).astype(np.float32)
    zb = np.full((len(objs), P), np.nan, np.float32)
    ze = np.full((len(objs), 2), np.nan, np.float32)
    video = name in vd.MODELS
    print(f"[{name}] {len(objs)} objects, {sum(map(len, by_clip.values()))} frames, {len(by_clip)} clips, "
          f"{'video' if video else 'image'} model, {torch.get_num_threads()} threads", flush=True)
    with torch.no_grad():
        f = vd.make_seq_predictor(name, "cuda" if torch.cuda.is_available() else "cpu")
        for n, (clip, by_img) in enumerate(by_clip.items()):
            imgs = (sorted(p for p in (clip / "images").iterdir() if p.suffix in (".jpg", ".png"))
                    if video else sorted(by_img))       # video models need the whole clip as context
            for p, d in zip(imgs, f([np.asarray(Image.open(p).convert("RGB")) for p in imgs])):
                d = np.asarray(d, np.float32)
                if d.shape != HW:
                    d = torch.nn.functional.interpolate(torch.from_numpy(d)[None, None], size=HW, mode="bilinear",
                                                        align_corners=False)[0, 0].numpy()
                d = np.where(np.isfinite(d) & (d > 0), d, np.nan)
                for i in by_img.get(p, []):
                    q = pts[i]
                    zb[i] = map_coordinates(np.nan_to_num(d, nan=np.inf), [q[:, 1], q[:, 0]], order=1, mode="nearest")
                    for e, (u, v) in enumerate(q[[0, -1]].round().astype(int)):
                        patch = d[max(v - R, 0):v + R + 1, max(u - R, 0):u + R + 1]
                        ze[i, e] = np.nanmedian(patch) if np.isfinite(patch).any() else np.nan
            print(f"[{name}] clip {n + 1}/{len(by_clip)}", flush=True)
    zb[~np.isfinite(zb)] = np.nan
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.npz", pts=pts, zb=zb, ze=ze, mm=np.array([o[2] for o in objs]),
             cls=np.array([o[3] for o in objs]), video=np.array([o[5] for o in objs]),
             image=np.array([str(o[0]) for o in objs]))
    print(f"[{name}] wrote {out / f'{name}.npz'}", flush=True)


# ------------------------------------------------------------------------------ CPU: lengths + stats
def stats(dev, mm):
    p25, p75 = np.percentile(dev, [25, 75])
    return dict(n=len(dev), mean=dev.mean(), mean_abs=np.abs(dev).mean(), median_abs=np.median(np.abs(dev)),
                p25=p25, p75=p75, min=dev.min(), max=dev.max(), mean_abs_pct=100 * np.mean(np.abs(dev) / mm))


def calibrate(name, out):
    """-> (npz, ok, surgery, {"scale": (zb, ze), "affine": (zb, ze)}) with LEAVE-ONE-SURGERY-OUT fits."""
    r = np.load(out / f"{name}.npz")
    pts, zb, ze, mm = r["pts"], r["zb"], r["ze"], r["mm"]
    surg = np.array([v[:8] for v in r["video"]])
    ok = np.isfinite(zb).all(1) & np.isfinite(ze).all(1) & (zb > 0).all(1)
    zp, zt = zb.mean(1), mm / unit_ray(pts)
    depths = {}
    sc_b, sc_e, af_b, af_e = (np.full_like(a, np.nan) for a in (zb, ze, zb, ze))
    for s_ in np.unique(surg):                          # calibration never sees the surgery it measures
        te, tr = surg == s_, (surg != s_) & ok
        s, b, s_only = fit_calibration(zp[tr], zt[tr])
        sc_b[te], sc_e[te] = zb[te] / s_only, ze[te] / s_only
        af_b[te], af_e[te] = apply_affine(zb[te], s, b), apply_affine(ze[te], s, b)
    depths["scale"], depths["affine"] = (sc_b, sc_e), (af_b, af_e)
    return r, ok, surg, depths


def evaluate(name, out):
    r, ok, surg, depths = calibrate(name, out)
    pts, mm, cls = r["pts"], r["mm"], r["cls"]
    rows = []
    for subset, sel in (("all", np.ones_like(ok)), ("sw05_test", np.isin(surg, SW05_TEST))):
        m = ok & sel & (cls == 1)
        for cal, (b_, e_) in depths.items():
            L = lengths(pts[m], b_[m], e_[m])
            for meth, l in L.items():
                good = np.isfinite(l)
                rows.append(dict(model=name, subset=subset, calib=cal, method=meth,
                                 **stats(l[good] - mm[m][good], mm[m][good])))
    return rows


LABEL = {"unidepth_v2_vitl": "UniDepth", "depth_anything_v2_metric_indoor_large": "DAv2-mi",
         "depth_anything_v2_large": "DAv2-L", "ft_ruler_sw05": "EndoDAC-sw05"}
COLOR = {"unidepth_v2_vitl": (80, 220, 255), "depth_anything_v2_metric_indoor_large": (255, 200, 60),
         "depth_anything_v2_large": (255, 110, 200), "ft_ruler_sw05": (170, 170, 170)}


def overlays(out, names, calib="scale", worst=6):
    """Test-surgery rulers drawn on their frame with each model's GUI-line (chord) length, calibrated LOO.
    One frame per test clip + the `worst` frames by largest |chord - true| over the models."""
    from PIL import Image, ImageDraw, ImageFont
    L, base = {}, None
    for n in names:
        r, ok, surg, depths = calibrate(n, out)
        if base is None:
            base = r
            keep, bsurg = ok & (r["cls"] == 1) & np.isin(surg, SW05_TEST), surg
        assert np.array_equal(r["image"], base["image"]) and np.allclose(r["pts"], base["pts"]), f"{n}: object order differs"
        keep &= ok
        L[n] = lengths(r["pts"], *depths[calib])["chord"]
    img, pts, mm = base["image"], base["pts"], base["mm"]
    idx = np.flatnonzero(keep & np.all([np.isfinite(L[n]) for n in names], 0))
    clip = np.array([str(Path(i).parents[1]) for i in img])
    pick = {img[i] for i in {clip[i]: i for i in idx[::-1]}.values()}          # first ruler frame per clip
    err = np.max([np.abs(L[n][idx] - mm[idx]) for n in names], 0)
    pick |= {img[i] for i in idx[np.argsort(-err)][:worst * 3]}
    pick = sorted(pick)[:200]
    d = out / f"overlays_{calib}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default(size=22)
    for f in pick:
        im = Image.open(f).convert("RGB")
        dr = ImageDraw.Draw(im)
        for i in idx[img[idx] == f]:
            a, b = pts[i, 0], pts[i, -1]
            dr.line([tuple(a), tuple(b)], fill=(40, 255, 120), width=4)
            for q in (a, b):
                dr.ellipse([q[0] - 7, q[1] - 7, q[0] + 7, q[1] + 7], outline=(40, 255, 120), width=3)
            lines = [(f"true {mm[i]:.1f} mm", (40, 255, 120))] +                     [(f"{LABEL.get(n, n)} {L[n][i]:.1f} ({L[n][i] - mm[i]:+.1f})", COLOR.get(n, (255, 255, 255))) for n in names]
            x, y = max(b[0], a[0]) + 16, min(a[1], b[1])
            x = min(x, im.width - 380)
            box = [x - 6, y - 6, x + 374, y + 28 * len(lines) + 2]
            dr.rectangle(box, fill=(0, 0, 0))
            for k, (t, c) in enumerate(lines):
                dr.text((x, y + 28 * k), t, fill=c, font=font)
        dr.rectangle([0, 0, 900, 34], fill=(0, 0, 0))
        dr.text((8, 5), f"{Path(f).parents[2].name[:8]} {Path(f).parents[1].name} {Path(f).stem} | chord (GUI line), "
                        f"{calib} calib LOO | label: mm (measured - true)", fill=(255, 255, 255), font=ImageFont.load_default(size=18))
        im.save(d / f"{Path(f).parents[2].name[:8]}_{Path(f).parents[1].name.replace('.mp4', '')}_{Path(f).stem}.jpg", quality=90)
    print(f"wrote {len(pick)} overlays ({len(idx)} test rulers) -> {d}")


def summarize(out, names):
    rows = [row for n in names if (out / f"{n}.npz").exists() for row in evaluate(n, out)]
    keys = list(rows[0])
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    for subset in ("all", "sw05_test"):
        for meth in ("chord", "polyline", "inplane"):
            sel = sorted((r for r in rows if r["subset"] == subset and r["method"] == meth), key=lambda r: r["mean_abs"])
            print(f"\n== RULERS, {subset}, {meth}: deviation = measured - true (mm); n rulers = {sel[0]['n'] if sel else 0}")
            print(f"{'model':38s} {'calib':6s} {'mean':>6s} {'mean|d|':>7s} {'med|d|':>6s} {'IQR':>13s} "
                  f"{'min':>6s} {'max':>6s} {'mean|d|%':>8s}")
            for r in sel:
                print(f"{r['model']:38s} {r['calib']:6s} {r['mean']:6.2f} {r['mean_abs']:7.2f} {r['median_abs']:6.2f} "
                      f"{r['p25']:6.2f}..{r['p75']:<5.2f} {r['min']:6.1f} {r['max']:6.1f} {r['mean_abs_pct']:8.1f}")
    print(f"\nwrote {out / 'summary.csv'}")


def selftest():
    # a 10 mm ruler on a fronto-parallel plane at 50 mm: every method must read 10 mm
    z, fx = 50.0, K_NORM[0] * HW[1]
    u0 = K_NORM[2] * HW[1]
    pts = np.stack([np.linspace(u0, u0 + 10.0 / z * fx, P), np.full(P, 400.0)], 1)[None].astype(np.float64)
    L = lengths(pts, np.full((1, P), z), np.full((1, 2), z))
    for k, v in L.items():
        assert abs(v[0] - 10.0) < 1e-6, (k, v)
    assert abs(10.0 / unit_ray(pts)[0] - z) < 1e-6
    # a tilted ruler: chord / polyline see the depth step, inplane does not
    L = lengths(pts, np.linspace(50, 56, P)[None], np.array([[50.0, 56.0]]))
    assert L["chord"][0] > 11.0 and L["polyline"][0] >= L["chord"][0] - 1e-9 and abs(L["inplane"][0] - 10.6) < 1e-6
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--overlays", action="store_true", help="draw test rulers with each --only model's length")
    ap.add_argument("--only", default=",".join(MODELS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.summarize:
        return summarize(Path(a.out), a.only.split(","))
    if a.overlays:
        return overlays(Path(a.out), a.only.split(","))
    if a.model and not os.environ.get("RULER_LEN_CHILD"):   # each model needs its own code dirs on PYTHONPATH
        import video_depth_proxy_gt as vd
        from zeroshot_proxy_gt import MODELS as ZS
        extra = vd.MODELS[a.model][1] if a.model in vd.MODELS else ZS[a.model][2]
        env = {**os.environ, "RULER_LEN_CHILD": "1",
               "PYTHONPATH": os.pathsep.join(extra + [os.environ.get("PYTHONPATH", "")])}
        sys.exit(subprocess.run([sys.executable, __file__] + sys.argv[1:], env=env).returncode)
    if a.model:
        return predict(a.model, Path(a.out))
    ap.print_help()


if __name__ == "__main__":
    main()
