"""GoodRulerTest: ruler-read urethra length vs UniDepth 3D length, one global scale factor.

Layout (transfer_atlas_mod workspace): <root>/<clip>-seg1.mp4/{images,masks,scale_objects.json}.
Images are already the 1340x1072 content frame. scale_objects.json keys = index into the sorted
images; each "Ruler" line runs along the urethra and its `mm` is the length read off the ruler.

The masks are the RULER. It is excluded from every depth read (mask dilated 5 px): UniDepth sees the
whole frame, but a depth sample is the median of the non-ruler pixels around the point, the window
growing until it holds enough tissue. The ruler lies on top of the urethra, so its depth is not the
urethra's.
Per line: UniDepth V2 depth (da Vinci K, no crop) -> endpoints back-projected -> 3D chord mm.
Also the length along the line (50 samples, depth median-filtered).
Then ONE global scale factor g = median(true / est) over all lines, reported in-sample and
leave-one-case-out. With --calib, also the frozen ruler-set calibration (metric_calib_proxy,
scale mode, inferred without K as in that fit) -- a global factor fitted on other data.

    python scripts/goodruler_unidepth.py --root ../data/GoodRulerTest
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

K_NORM = (0.82, 1.02, 0.5, 0.5)   # da Vinci Xi, same as gui_depth_measure DEFAULT_K_NORM


def patch_z(z, p, r=4, excl=None, min_px=20):
    """Median depth around p over non-excluded pixels; the window grows until min_px remain."""
    x, y = int(round(p[0])), int(round(p[1]))
    while True:
        sl = np.s_[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        v = z[sl] if excl is None else z[sl][~excl[sl]]
        if v.size >= min_px or r > 60:
            return float(np.median(v)) if v.size else np.nan
        r *= 2


def backproject(p, zz, K):
    return np.array([(p[0] - K[0, 2]) / K[0, 0] * zz, (p[1] - K[1, 2]) / K[1, 1] * zz, zz])


def lengths(z, a, b, K, excl):
    chord = np.linalg.norm(backproject(b, patch_z(z, b, 4, excl), K) - backproject(a, patch_z(z, a, 4, excl), K))
    P = a + np.linspace(0, 1, 50)[:, None] * (b - a)
    zs = median_filter(np.array([patch_z(z, p, 2, excl, 10) for p in P]), size=7, mode="nearest")
    X = np.array([backproject(p, zz, K) for p, zz in zip(P, zs)])
    return chord, np.linalg.norm(np.diff(X, axis=0), axis=1).sum()


def summarize(tag, true, est, case):
    ok = ~np.isnan(est)
    t, e, c = true[ok], est[ok], case[ok]
    g_in = np.median(t / e)
    loo = np.array([e[i] * np.median((t / e)[c != c[i]]) for i in range(len(e))])
    out = {}
    for name, v in [("raw", e), ("global_in", e * g_in), ("global_loo", loo)]:
        err = v - t
        out[name] = dict(mae=float(np.abs(err).mean()), med=float(np.median(err)),
                         mape=float(100 * np.mean(np.abs(err) / t)), r=float(np.corrcoef(v, t)[0, 1]))
    print(f"[{tag}] n={ok.sum()} global factor {g_in:.3f}")
    for k, s in out.items():
        print(f"   {k:11s} MAE {s['mae']:5.2f} mm  median signed {s['med']:+5.2f}  MAPE {s['mape']:4.0f}%  r {s['r']:.2f}")
    return g_in, loo, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="lpiccinelli/unidepth-v2-vitl14")
    ap.add_argument("--calib", help="metric_calib_proxy results.json (frozen ruler-set calibration)")
    ap.add_argument("--out", default="outputs/goodruler_unidepth")
    args = ap.parse_args()

    import torch
    from unidepth.models import UniDepthV2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UniDepthV2.from_pretrained(args.model).to(dev).eval()
    cal = json.loads(Path(args.calib).read_text())["unidepth_v2_vitl"]["calibration"] if args.calib else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(Path(args.root).glob("*-seg1.mp4")):
        so = json.loads((d / "scale_objects.json").read_text())["frames"]
        imgs = sorted((d / "images").iterdir())
        for key, objs in sorted(so.items(), key=lambda t: int(t[0])):
            img = imgs[int(key)]
            rgb = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
            ruler = cv2.imread(str(d / "masks" / f"{img.stem}.png"), cv2.IMREAD_GRAYSCALE) > 0
            ruler = cv2.dilate(ruler.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
            h, w = rgb.shape[:2]
            fx, fy, cx, cy = K_NORM
            K = np.array([[fx * w, 0, cx * w], [0, fy * h, cy * h], [0, 0, 1]], dtype=np.float32)
            t = torch.from_numpy(rgb).permute(2, 0, 1).to(dev)
            with torch.no_grad():
                z = model.infer(t, torch.from_numpy(K).to(dev))["depth"][0, 0].float().cpu().numpy() * 1000
                zc = None
                if cal:
                    zc = model.infer(t)["depth"][0, 0].float().cpu().numpy() / cal["s_only"]
            (d / "depth").mkdir(exist_ok=True)
            np.savez(d / "depth" / f"{img.stem}_unidepth.npz", depth=z.astype(np.float32),
                     **({"depth_calib": zc.astype(np.float32)} if cal else {}))
            for o in objs:
                a, b = np.array(o["a"]), np.array(o["b"])
                ch, al = lengths(z, a, b, K, ruler)
                chc, alc = lengths(zc, a, b, K, ruler) if cal else (np.nan, np.nan)
                rows.append(dict(case=d.name[:8], key=key, image=img.name, true_mm=o["mm"],
                                 line_px=float(np.linalg.norm(b - a)), z_mm=patch_z(z, (a + b) / 2, 4, ruler),
                                 z_ruler_mm=float(np.median(z[ruler])) if ruler.any() else np.nan,
                                 chord_mm=ch, along_mm=al, chord_calib_mm=chc, along_calib_mm=alc))
                print(f"{d.name[:8]} {key} {img.name} true {o['mm']:.0f}  z {rows[-1]['z_mm']:.0f} mm  "
                      f"chord {ch:.1f} along {al:.1f}" + (f"  calib chord {chc:.1f}" if cal else ""))

    with open(out / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    true = np.array([r["true_mm"] for r in rows], float)
    case = np.array([r["case"] for r in rows])
    summ = {}
    for col in ["chord_mm", "along_mm"] + (["chord_calib_mm", "along_calib_mm"] if cal else []):
        g, loo, s = summarize(col, true, np.array([r[col] for r in rows], float), case)
        summ[col] = dict(global_factor=float(g), **s)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
