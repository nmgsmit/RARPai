"""Arch tip (urethra end point) from UniDepth depth: fit a parametric arch to the depth map.

Model: the arch is the rim of a deeper cavity, so the arch curve should sit on an edge where log-depth
changes across the curve. A candidate arch = apex (x, y), half-chord d, height h, tilt, power -- the
same curve the annotation tool draws. Its score is the mean log-depth gradient along the curve's
inward normal (GUI pixels excluded), normalised per frame, minus lam * Mahalanobis^2 of the apex under
the training tip distribution. Coarse grid (apex every 12 px x 27 training-quantile shapes), then a
local refine. The predicted tip is the best apex + a training bias.

Learned from the 4 TRAIN patients only: shape grid, apex prior, gradient sign, smoothing sigma, lam,
bias. Reported on the 3 TEST patients vs the consensus tip (mean of Nick/Veerle/Aron), next to a
constant baseline (train mean tip) and each annotator vs the mean of the other two.

    python scripts/arch_tip_fit.py --selfcheck
    python scripts/arch_tip_fit.py                        # data from arch_tip_prep + arch_tip_unidepth
"""
import argparse
import csv
import itertools
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = 4                 # depth stored at 1/4 res (arch_tip_unidepth.py)
FX = 0.82 * 1340       # assumed fx (gui_depth_measure DEFAULT_K_NORM) on the 1340 px frame -> approximate mm only
K = 33                 # samples per curve
STEP = 12              # coarse apex grid, px


def curve(c):
    """c (N, 6) = ax, ay, d, h, tilt, p -> x, y, inward normal nx, ny; each (N, K), full-res px."""
    ax, ay, d, h, th, p = (c[:, i, None] for i in range(6))
    u = np.linspace(-1, 1, K)[None]
    tx, ty = np.cos(th), np.sin(th)
    nx, ny = ty, -tx                                    # h > 0 -> apex above the chord
    s = 1 - np.abs(u) ** p
    x = ax - h * nx + u * d * tx + s * h * nx
    y = ay - h * ny + u * d * ty + s * h * ny
    ds = -p * np.abs(u) ** np.maximum(p - 1, 0) * np.sign(u)
    Tx, Ty = d * tx + ds * h * nx, d * ty + ds * h * ny
    n = np.hypot(Tx, Ty) + 1e-9
    return x, y, -Ty / n, Tx / n                        # (0, 1) = down at the apex of an upright arch


def score(c, gx, gy, valid):
    """Mean inward log-depth gradient along each candidate; -inf when < half the curve is usable."""
    x, y, nx, ny = curve(c)
    H, W = gx.shape
    ix, iy = np.rint(x / DS).astype(int), np.rint(y / DS).astype(int)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy = ix.clip(0, W - 1), iy.clip(0, H - 1)
    ok &= valid[iy, ix]
    g = (gx[iy, ix] * nx + gy[iy, ix] * ny) * ok
    n = ok.sum(1)
    return np.where(n >= K // 2, g.sum(1) / np.maximum(n, 1), -np.inf)


def grads(depth, gui, sigma):
    L = cv2.GaussianBlur(np.log(np.clip(depth.astype(np.float32), 1, None)), (0, 0), sigma)
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0) / 8, cv2.Sobel(L, cv2.CV_32F, 0, 1) / 8
    r = int(3 * sigma) + 1
    valid = cv2.dilate(gui.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) == 0
    m = np.hypot(gx, gy)[valid]
    scale = np.median(m) + 1e-6 if m.size else 1.0     # per-frame normalisation: contrast varies a lot
    return gx / scale, gy / scale, valid


class Prior:
    """Everything the fitter takes from the training arches."""

    def __init__(self, rows, margin=150):
        tips = np.array([r["tip"] for r in rows])
        self.mu, self.icov = tips.mean(0), np.linalg.inv(np.cov(tips.T) + 1e-3 * np.eye(2))
        self.box = (max(tips[:, 0].min() - margin, 0), min(tips[:, 0].max() + margin, 1340),
                    max(tips[:, 1].min() - margin, 0), min(tips[:, 1].max() + margin, 1072))
        g = np.array([arch_params(r["arch"]) for r in rows])          # d, h, tilt, p
        q = lambda i: np.quantile(g[:, i], [0.2, 0.5, 0.8])            # noqa: E731
        self.shapes = np.array([(d, h, t, np.median(g[:, 3]))
                                for d, h, t in itertools.product(q(0), q(1), q(2))])
        x0, x1, y0, y1 = self.box
        ax, ay = np.meshgrid(np.arange(x0, x1, STEP), np.arange(y0, y1, STEP))
        A = np.stack([ax.ravel(), ay.ravel()], 1)
        self.coarse = np.concatenate([np.repeat(A, len(self.shapes), 0), np.tile(self.shapes, (len(A), 1))], 1)
        self.coarse_mahal = self.mahal(self.coarse[:, :2])

    def mahal(self, a):
        z = a - self.mu
        return np.einsum("ni,ij,nj->n", z, self.icov, z)


def arch_params(a):
    """annotation arch dict -> (d, h, tilt, p); apex via the tool's formula."""
    l, r = np.array(a["left"]), np.array(a["right"])
    ch = r - l
    return np.linalg.norm(ch) / 2, a["height"], np.arctan2(ch[1], ch[0]), a["power"]


def apex_of_arch(a):
    l, r = np.array(a["left"]), np.array(a["right"])
    t = (r - l) / (np.linalg.norm(r - l) + 1e-9)
    return (l + r) / 2 + a["height"] * np.array([t[1], -t[0]])


def refine(best, gx, gy, valid, prior, sign, lam):
    ax, ay, d, h, th, p = best
    c = np.array([(ax + dx, ay + dy, d * fd, h * fh, th + dt, p)
                  for dx, dy in itertools.product((-8, -4, 0, 4, 8), repeat=2)
                  for fd, fh, dt in itertools.product((0.8, 1, 1.25), (0.8, 1, 1.25), (-0.1, 0, 0.1))])
    tot = sign * score(c, gx, gy, valid) - lam * prior.mahal(c[:, :2])
    return c[np.argmax(tot)]


def fit(gx, gy, valid, prior, sign, lam, raw=None):
    raw = score(prior.coarse, gx, gy, valid) if raw is None else raw
    tot = sign * raw - lam * prior.coarse_mahal
    return refine(prior.coarse[np.argmax(tot)], gx, gy, valid, prior, sign, lam)


# ------------------------------------------------------------------------------------ data
def load(data, r):
    depth = np.load(data / "depth" / f"{r['stem']}.npz")["depth"]
    H, W = depth.shape
    gui = cv2.resize(cv2.imread(str(data / "images" / f"{r['stem']}_mask.png"), cv2.IMREAD_GRAYSCALE),
                     (W, H), interpolation=cv2.INTER_AREA) > 0
    return depth, gui


def summarize(err):
    e = np.asarray(err)
    return dict(n=len(e), mean=float(e.mean()), median=float(np.median(e)), p90=float(np.quantile(e, 0.9)),
                within25=float((e <= 25).mean()), within50=float((e <= 50).mean()),
                within100=float((e <= 100).mean()))


def draw(img, arch_c, color):
    x, y, _, _ = curve(np.asarray(arch_c, float)[None])
    cv2.polylines(img, [np.stack([x[0], y[0]], 1).astype(np.int32)], False, color, 4, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "processed" / "arch_tip_besthalf"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "arch_tip_depth"))
    ap.add_argument("--test", nargs="*", help="test patient ids (default: 3 seeded from those with >= 50 frames)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune-every", type=int, default=4, help="train frame subsampling for the config search")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((data / "labels.json").read_text())["rows"]
    rows = [r for r in rows if (data / "depth" / f"{r['stem']}.npz").exists()]
    counts = {p: sum(r["patient"] == p for r in rows) for p in sorted({r["patient"] for r in rows})}
    test = set(args.test or random.Random(args.seed).sample(sorted(p for p, n in counts.items() if n >= 50), 3))
    train_rows = [r for r in rows if r["patient"] not in test]
    test_rows = [r for r in rows if r["patient"] in test]
    print("frames per patient:", counts)
    print(f"TRAIN {sorted(set(counts) - test)} ({len(train_rows)} frames)  TEST {sorted(test)} ({len(test_rows)} frames)")

    prior = Prior(train_rows)
    print(f"apex prior mean {prior.mu.round(0)}, box {np.round(prior.box)}, {len(prior.shapes)} shapes, "
          f"{len(prior.coarse)} coarse candidates")

    # ---- tune sign / sigma / lam / bias on train (raw coarse scores cached per sigma) -----------
    tune = train_rows[::args.tune_every]
    cfgs = []
    for sigma in (1.5, 3.0):
        feats = [grads(*load(data, r), sigma) for r in tune]
        raws = [score(prior.coarse, *f) for f in feats]
        for sign, lam in itertools.product((1, -1), (0, 0.03, 0.1, 0.3, 1, 3)):
            pred = np.array([fit(*f, prior, sign, lam, raw)[:2] for f, raw in zip(feats, raws)])
            gt = np.array([r["tip"] for r in tune])
            bias = np.median(gt - pred, 0)
            err = np.linalg.norm(pred + bias - gt, axis=1)
            cfgs.append(dict(sigma=sigma, sign=sign, lam=lam, bias=bias.tolist(), train_mean_px=float(err.mean())))
            print(f"  sigma {sigma} sign {sign:+d} lam {lam:<4} bias {bias.round(0)} train mean {err.mean():6.1f} px")
    cfg = min(cfgs, key=lambda c: c["train_mean_px"])
    print("chosen:", cfg)

    # ---- test ---------------------------------------------------------------------------------
    per, bias = [], np.array(cfg["bias"])
    for r in test_rows:
        depth, gui = load(data, r)
        best = fit(*grads(depth, gui, cfg["sigma"]), prior, cfg["sign"], cfg["lam"])
        gt, pred = np.array(r["tip"]), best[:2] + bias
        chord = np.linalg.norm(np.subtract(r["arch"]["right"], r["arch"]["left"]))
        gx, gy = int(np.clip(gt[0] / DS, 0, depth.shape[1] - 1)), int(np.clip(gt[1] / DS, 0, depth.shape[0] - 1))
        z = float(np.median(depth[max(gy - 1, 0):gy + 2, max(gx - 1, 0):gx + 2]))
        tips = {n: np.array(t) for n, t in r["tips"].items()}
        human = np.mean([np.linalg.norm(t - np.mean([o for m, o in tips.items() if m != n], 0))
                         for n, t in tips.items()])
        err = float(np.linalg.norm(pred - gt))
        per.append(dict(patient=r["patient"], stem=r["stem"], gt_x=gt[0], gt_y=gt[1], pred_x=pred[0], pred_y=pred[1],
                        err_px=err, err_pct_chord=100 * err / chord, err_mm=err * z / FX, depth_mm=z,
                        prior_err_px=float(np.linalg.norm(prior.mu - gt)), human_loo_px=float(human),
                        fit=best.tolist()))

    with open(out / "test_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in per[0] if k != "fit"], extrasaction="ignore")
        w.writeheader()
        w.writerows(per)
    res = dict(train=sorted(set(counts) - test), test=sorted(test), frames=counts, config=cfg, tuning=cfgs,
               pooled={k: summarize([p[k] for p in per]) for k in ("err_px", "prior_err_px", "human_loo_px")},
               pooled_pct_chord=summarize([p["err_pct_chord"] for p in per]),
               pooled_mm=summarize([p["err_mm"] for p in per]),
               per_patient={pt: {k: summarize([p[k] for p in per if p["patient"] == pt])
                                 for k in ("err_px", "prior_err_px", "human_loo_px", "err_mm", "err_pct_chord")}
                            for pt in sorted(test)})
    res["patient_macro_mean"] = {k: float(np.mean([v[k]["mean"] for v in res["per_patient"].values()]))
                                 for k in ("err_px", "prior_err_px", "human_loo_px", "err_mm", "err_pct_chord")}
    (out / "results.json").write_text(json.dumps(res, indent=2))

    print(f"\n=== TEST ({len(per)} frames, {len(test)} patients) : tip error vs consensus ===")
    print(f"{'':<22}{'mean':>7}{'median':>8}{'p90':>7}{'<=25':>6}{'<=50':>6}{'<=100':>7}")
    for k, name in (("err_px", "depth arch fit (px)"), ("prior_err_px", "train-mean tip (px)"),
                    ("human_loo_px", "annotator vs others")):
        s = res["pooled"][k]
        print(f"{name:<22}{s['mean']:7.1f}{s['median']:8.1f}{s['p90']:7.1f}{s['within25']:6.0%}{s['within50']:6.0%}{s['within100']:7.0%}")
    print(f"depth arch fit: {res['pooled_pct_chord']['mean']:.1f}% of chord, ~{res['pooled_mm']['mean']:.1f} mm (assumed fx)")
    for pt, v in res["per_patient"].items():
        print(f"  {pt:<38} n={v['err_px']['n']:4d}  fit {v['err_px']['mean']:6.1f}  prior {v['prior_err_px']['mean']:6.1f}"
              f"  human {v['human_loo_px']['mean']:5.1f} px   fit ~{v['err_mm']['mean']:.1f} mm")
    print("patient-macro mean:", {k: round(v, 1) for k, v in res["patient_macro_mean"].items()})

    # ---- figure: per test patient the 10th / 50th / 90th percentile error frame ---------------------
    cells = []
    for pt in sorted(test):
        pp = sorted((p for p in per if p["patient"] == pt), key=lambda p: p["err_px"])
        for q in (0.1, 0.5, 0.9):
            p = pp[int(q * (len(pp) - 1))]
            r = next(r for r in test_rows if r["stem"] == p["stem"])
            img = cv2.imread(str(data / "images" / f"{p['stem']}.jpg"))
            depth = np.load(data / "depth" / f"{p['stem']}.npz")["depth"].astype(np.float32)
            inv = 1 / depth
            col = cv2.applyColorMap((np.clip((inv - np.percentile(inv, 2)) / (np.ptp(np.percentile(inv, [2, 98])) + 1e-9), 0, 1)
                                     * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            img = cv2.addWeighted(img, 0.6, cv2.resize(col, (img.shape[1], img.shape[0])), 0.4, 0)
            gt_c = [*apex_of_arch(r["arch"]), *arch_params(r["arch"])]
            draw(img, gt_c, (0, 255, 0))
            draw(img, p["fit"], (0, 0, 255))
            cv2.drawMarker(img, (int(p["gt_x"]), int(p["gt_y"])), (0, 255, 0), cv2.MARKER_STAR, 40, 4)
            cv2.drawMarker(img, (int(p["pred_x"]), int(p["pred_y"])), (0, 0, 255), cv2.MARKER_CROSS, 40, 4)
            cv2.putText(img, f"{pt[:8]} {p['stem'][-5:]}  p{int(q * 100)}  {p['err_px']:.0f}px ~{p['err_mm']:.1f}mm",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
            cells.append(cv2.resize(img, (670, 536)))
    grid = np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)])
    cv2.imwrite(str(out / "test_examples.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {out}/results.json, test_frames.csv, test_examples.jpg (green = consensus, red = depth fit)")


def selfcheck():
    """A far cavity under a known arch must pull the fit onto that arch's apex."""
    H, W = 268, 335
    c0 = np.array([670, 250, 400, 300, 0.0, 2.0])
    X, Y = np.meshgrid(np.arange(W) * DS, np.arange(H) * DS)
    u = (X - c0[0]) / c0[2]
    inside = (np.abs(u) < 1) & (Y > c0[1] + c0[3] - (1 - u ** 2) * c0[3])
    depth = np.where(inside, 80.0, 50.0) + np.random.default_rng(0).normal(0, 0.5, (H, W))
    gui = np.zeros((H, W), bool)
    gui[250:] = True                                     # a HUD band must not break anything
    rows = [dict(tip=[670 + dx, 250 + dy], arch=dict(left=[270, 550], right=[1070, 550], height=300, power=2.0))
            for dx, dy in itertools.product((-100, 100), repeat=2)]
    prior = Prior(rows)
    f = grads(depth, gui, 1.5)
    best = fit(*f, prior, sign=1, lam=0)
    assert np.linalg.norm(best[:2] - c0[:2]) <= 12, best
    wrong = fit(*f, prior, sign=-1, lam=0)
    assert score(best[None], *f)[0] > 0 > score(wrong[None], *f)[0], "far-inside must score positive"
    assert np.allclose(apex_of_arch(rows[0]["arch"]), [670, 250])
    print(f"selfcheck ok: apex {best[:2]} vs {c0[:2]}")


if __name__ == "__main__":
    main()
