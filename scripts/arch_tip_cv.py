"""Arch tip, round 2: improved arch fit, head refinement and temporal smoothing over the 20 patient splits.

Genoa CPU job after arch_tip_head.py. Every method is scored like round 1: tip error vs the 3-way consensus on
the BEST-HALF frames of the 3 test patients, per split; reported as the patient-macro mean over the 20 splits.
  prior            train-mean tip
  fit_depth        arch fit on log-depth edges. vs round 1: tool pixels (rarp_nick_fullres catheter |
                   non-anatomical) excluded, apex searched up to 400 px ABOVE the frame, fixed a-priori shape
                   grid; gradient sign, prior weight lam and bias tuned on the train patients
  fit_rgb          same on Lab-lightness edges
  fit_depth_rgb    weighted sum of both (weights + signs tuned on train)
  head_rgb/_rgbd   arch_tip_head.py
  refine_<head>    local arch fit (fit_depth_rgb's tuned weights) around the head's arch, pulled to the head tip
                   with sigma 40 px, lam 1 (fixed a priori, not tuned)
  <m>_smooth       running median of the tip over +-7 frames (0.23 s at 59.94 fps)
  human            each annotator vs the mean of the other two

    python scripts/arch_tip_cv.py --selfcheck
    sbatch jobs/arch_tip_cv.sh
"""
import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_tip_data import A, FrameStore, cv_splits  # noqa: E402
from arch_tip_fit import K, curve, grads, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AX, AY = np.arange(200, 1141, 16), np.arange(-400, 701, 16)
SHAPES = np.array([(d, h, t, 2.0) for d in (250, 350, 450, 550) for h in (150, 250, 350) for t in (-0.25, 0, 0.25)])
COARSE = np.concatenate([np.repeat(np.stack(np.meshgrid(AX, AY), -1).reshape(-1, 2), len(SHAPES), 0),
                         np.tile(SHAPES, (len(AX) * len(AY), 1))], 1).astype(np.float32)
SIGMA, MINV, WIN = 2.0, K // 3, 7
REF_SIGMA, REF_LAM = 40.0, 1.0
# (depth weight, depth sign, rgb weight, rgb sign, lam)
BASE = ([(1, s, 0, 1) for s in (1, -1)] + [(0, 1, 1, s) for s in (1, -1)]
        + [(1, ds, wr, rs) for ds in (1, -1) for wr in (0.5, 1, 2) for rs in (1, -1)])
CFGS = [(*b, lam) for b in BASE for lam in (0.1, 0.3, 1, 3)]
FIT = {"fit_depth": lambda c: c[2] == 0, "fit_rgb": lambda c: c[0] == 0, "fit_depth_rgb": lambda c: True}
OFF8 = list(itertools.product((-8, -4, 0, 4, 8), repeat=2))
OFF_REF = list(itertools.product(range(-48, 49, 8), repeat=2))
G = {}


# ------------------------------------------------------------------------------------ fitting
def local(b, offs, fd, fh, dt):
    return np.array([(b[0] + dx, b[1] + dy, b[2] * a, b[3] * h, b[4] + t, b[5])
                     for dx, dy in offs for a in fd for h in fh for t in dt], np.float32)


def comb(cfg, sd, sr):
    ok = np.isfinite(sd)                                  # same valid mask for both fields
    return np.where(ok, cfg[0] * cfg[1] * np.where(ok, sd, 0) + cfg[2] * cfg[3] * np.where(ok, sr, 0), -np.inf)


def mahal(a, mu, icov):
    z = a - mu
    return np.einsum("ni,ij,nj->n", z, icov, z)


def fit_cfg(cfg, s, gd, gr, rd, rr):
    mu, icov = G["prior"][s]
    tot = comb(cfg, rd, rr) - cfg[4] * G["M"][s]
    if not np.isfinite(tot).any():
        return np.array([*mu, 0, 0, 0, 0])
    c = local(COARSE[np.argmax(tot)], OFF8, (0.8, 1, 1.25), (0.8, 1, 1.25), (-0.1, 0, 0.1))
    t = comb(cfg, score(c, *gd, min_valid=MINV), score(c, *gr, min_valid=MINV)) - cfg[4] * mahal(c[:, :2], mu, icov)
    return c[np.argmax(t)] if np.isfinite(t).any() else COARSE[np.argmax(tot)]


def head_arch(tip, left, right):
    """Head output (tip, two ends) -> arch params (apex x, apex y, half-chord, height, tilt, power)."""
    l, r = (left, right) if left[0] <= right[0] else (right, left)
    ch = r - l
    d = max(np.linalg.norm(ch) / 2, 50)
    t = ch / (2 * d)
    h = max(float(np.dot(tip - (l + r) / 2, [t[1], -t[0]])), 50)
    return np.array([tip[0], tip[1], d, h, np.arctan2(ch[1], ch[0]), 2.0], np.float32)


def frame_maps(short, frame):
    fs, k = G["fs"][short], G["idx"][short][frame]
    depth = np.asarray(G["depth"][short][k], np.float32)
    H, W = depth.shape
    gui = cv2.resize(fs.mask(k).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA) > 0
    bad = gui | (np.asarray(G["tools"][short][k]) > 0)
    img = cv2.resize(fs.jpg(k), (W, H), interpolation=cv2.INTER_AREA)
    gd = grads(depth, bad, SIGMA)
    gr = grads(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0], bad, SIGMA, log=False)
    return gd, gr, score(COARSE, *gd, min_valid=MINV), score(COARSE, *gr, min_valid=MINV)


# ------------------------------------------------------------------------------------ workers
def _init(state):
    G.update(state)
    G["tools"] = {s: np.load(A / s / "tools.npy", mmap_mode="r") for s in G["shorts"]}
    G["depth"] = {s: np.load(A / s / "depth.npy", mmap_mode="r") for s in G["shorts"]}
    G["fs"] = {s: FrameStore(s) for s in G["shorts"]}
    G["idx"] = {s: fs.pos for s, fs in G["fs"].items()}
    G["M"] = [mahal(COARSE[:, :2], *p) for p in G["prior"]]
    G["head"] = {}
    for (v, s, short), f in G.get("head_files", {}).items():
        z = np.load(f)
        G["head"][(v, s, short)] = ({int(x): k for k, x in enumerate(z["frames"])}, z["tip"], z["left"], z["right"])


def tune_job(job):
    short, frame, patient = job
    gd, gr, rd, rr = frame_maps(short, frame)
    return {(s, ci): fit_cfg(cfg, s, gd, gr, rd, rr)[:2]
            for s, test in enumerate(G["splits"]) if patient not in test for ci, cfg in enumerate(CFGS)}


def test_job(job):
    short, frame, patient = job
    gd, gr, rd, rr = frame_maps(short, frame)
    out = {}
    for s, test in enumerate(G["splits"]):
        if patient not in test:
            continue
        ch = G["chosen"][s]
        for m in FIT:
            out[(m, s)] = (fit_cfg(ch[m]["cfg"], s, gd, gr, rd, rr)[:2] + ch[m]["bias"]).tolist()
        w = ch["fit_depth_rgb"]["cfg"]
        for v in G["variants"]:
            pos, tip, left, right = G["head"][(v, s, short)]
            k = pos[frame]
            c = local(head_arch(tip[k], left[k], right[k]), OFF_REF, (0.85, 1, 1.15), (0.85, 1, 1.15), (-0.1, 0, 0.1))
            t = (comb(w, score(c, *gd, min_valid=MINV), score(c, *gr, min_valid=MINV))
                 - REF_LAM * ((c[:, 0] - tip[k][0]) ** 2 + (c[:, 1] - tip[k][1]) ** 2) / REF_SIGMA ** 2)
            out[(f"refine_{v}", s)] = (c[np.argmax(t), :2] if np.isfinite(t).any() else tip[k]).tolist()
    return job, out


# ------------------------------------------------------------------------------------ evaluation
def smooth(series, frames, win=WIN):
    return {f: np.median([series[g] for g in range(f - win, f + win + 1) if g in series], 0) for f in frames}


def summarize(E):
    """E: per split {patient: [errors]} -> macro over splits, pooled rates, per-patient means."""
    macro = [np.mean([np.mean(v) for v in d.values()]) for d in E]
    pooled = np.concatenate([np.concatenate(list(d.values())) for d in E])
    per = {}
    for d in E:
        for p, v in d.items():
            per.setdefault(p, []).append(np.mean(v))
    return dict(macro_mean=float(np.mean(macro)), macro_sd=float(np.std(macro)), macro_min=float(np.min(macro)),
                macro_max=float(np.max(macro)), median=float(np.median(pooled)),
                within25=float((pooled <= 25).mean()), within50=float((pooled <= 50).mean()),
                within100=float((pooled <= 100).mean()), per_patient={p: float(np.mean(v)) for p, v in per.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--tune-every", type=int, default=12)
    ap.add_argument("--variants", nargs="+", default=["rgb", "rgbd"])
    ap.add_argument("--head", default=str(ROOT / "outputs" / "arch_tip_cv" / "head"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "arch_tip_cv"))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    rows = json.loads((A / "labels_all.json").read_text())["rows"]
    splits = cv_splits(rows)
    short_of = {r["patient"]: r["short"] for r in rows}
    cands = sorted(set().union(*splits))
    BH = {p: {r["frame"]: np.array(r["tip"]) for r in rows if r["patient"] == p and r["best_half"]} for p in cands}
    prior = []
    for test in splits:
        tips = np.array([r["tip"] for r in rows if r["patient"] not in test])
        prior.append((tips.mean(0), np.linalg.inv(np.cov(tips.T) + 1e-3 * np.eye(2))))
    state = dict(shorts=sorted(set(short_of.values())), splits=splits, prior=prior, variants=a.variants)
    print(f"{len(splits)} splits over {cands}; {len(COARSE)} coarse arches x {len(CFGS)} configs", flush=True)

    # ---- tune the three fitters per split on train patients --------------------------------------------------
    tune = [r for r in rows if r["frame"] % a.tune_every == 0]
    with Pool(a.workers, _init, (state,)) as pool:
        res = pool.map(tune_job, [(r["short"], r["frame"], r["patient"]) for r in tune], chunksize=1)
    chosen = []
    for s, test in enumerate(splits):
        js = [j for j, r in enumerate(tune) if r["patient"] not in test]
        gt = np.array([tune[j]["tip"] for j in js])
        errs = {}
        for ci in range(len(CFGS)):
            pred = np.array([res[j][(s, ci)] for j in js])
            bias = np.median(gt - pred, 0)
            errs[ci] = (float(np.linalg.norm(pred + bias - gt, axis=1).mean()), bias)
        ch = {}
        for m, keep in FIT.items():
            ci = min((c for c in errs if keep(CFGS[c])), key=lambda c: errs[c][0])
            ch[m] = dict(cfg=list(CFGS[ci]), bias=errs[ci][1].tolist(), train_px=errs[ci][0])
        chosen.append(ch)
        print(f"s{s:02d} " + "  ".join(f"{m} {c['cfg']} train {c['train_px']:.0f}px" for m, c in ch.items()), flush=True)
    del res

    # ---- fit + refine every best-half test frame and its +-WIN neighbours --------------------------------------
    state.update(chosen=chosen, head_files={(v, s, short_of[p]): Path(a.head) / v / f"s{s:02d}" / f"{short_of[p]}.npz"
                                            for v in a.variants for s, test in enumerate(splits) for p in test})
    _init(state)                                            # main process needs frames lists + head preds too
    jobs = sorted({(short_of[p], g, p) for p in cands for f in BH[p] for g in range(f - WIN, f + WIN + 1)
                   if g in G["idx"][short_of[p]]})
    P = {}
    with Pool(a.workers, _init, (state,)) as pool:
        for (short, frame, _), out in pool.imap_unordered(test_job, jobs, chunksize=2):
            for (m, s), xy in out.items():
                P.setdefault((m, s, short), {})[frame] = np.array(xy)
    for s, test in enumerate(splits):
        for p in test:
            sh = short_of[p]
            P[("prior", s, sh)] = {f: prior[s][0] for f in BH[p]}
            for v in a.variants:
                pos, tip, _, _ = G["head"][(v, s, sh)]
                P[(f"head_{v}", s, sh)] = {f: tip[k] for f, k in pos.items()}

    methods = ["prior", *FIT, *[f"head_{v}" for v in a.variants], *[f"refine_{v}" for v in a.variants]]
    methods += [f"{m}_smooth" for m in methods[1:]]
    results = {}
    for m in methods:
        base = m.removesuffix("_smooth")
        E = []
        for s, test in enumerate(splits):
            d = {}
            for p in sorted(test):
                ser = P[(base, s, short_of[p])]
                ser = smooth(ser, BH[p]) if m.endswith("_smooth") else ser
                d[p] = np.array([np.linalg.norm(ser[f] - t) for f, t in BH[p].items()])
            E.append(d)
        results[m] = summarize(E)
    hum = {p: np.array([np.mean([np.linalg.norm(np.array(t) - np.mean([o for n2, o in r["tips"].items() if n2 != n], 0))
                                 for n, t in r["tips"].items()]) for r in rows if r["patient"] == p and r["best_half"]])
           for p in cands}
    results["human"] = summarize([{p: hum[p] for p in sorted(test)} for test in splits])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(dict(splits=[sorted(t) for t in splits], chosen=chosen,
                                                      results=results), indent=1))
    print(f"\n=== tip error vs consensus, best-half frames of the test patients, {len(splits)} splits ===")
    print(f"{'method':<22}{'macro mean':>11}{'sd':>6}{'min':>6}{'max':>6}{'median':>8}{'<=25':>6}{'<=50':>6}{'<=100':>7}  "
          + " ".join(f"{p[:8]:>8}" for p in cands))
    for m, r in sorted(results.items(), key=lambda kv: kv[1]["macro_mean"]):
        print(f"{m:<22}{r['macro_mean']:11.1f}{r['macro_sd']:6.1f}{r['macro_min']:6.1f}{r['macro_max']:6.1f}"
              f"{r['median']:8.1f}{r['within25']:6.0%}{r['within50']:6.0%}{r['within100']:7.0%}  "
              + " ".join(f"{r['per_patient'][p]:8.1f}" for p in cands))
    print(f"wrote {out / 'results.json'}")


def selfcheck():
    tip, left, right = np.array([670., 250.]), np.array([270., 550.]), np.array([1070., 550.])
    c = head_arch(tip, left, right)
    assert np.allclose(c[:5], [670, 250, 400, 300, 0], atol=1e-3), c
    x, y, _, _ = curve(c[None])
    assert np.allclose([x[0, 0], y[0, 0], x[0, -1], y[0, -1]], [270, 550, 1070, 550], atol=1e-2)
    assert np.allclose(head_arch(tip, right, left), c)                      # end order must not matter
    ser = {f: np.array([5.0, 7.0]) for f in range(30) if f != 12}           # a gap must not break it
    ser[10] = np.array([999.0, 999.0])                                      # one-frame outlier
    assert np.allclose(smooth(ser, [10, 29])[10], [5, 7]) and np.allclose(smooth(ser, [29])[29], [5, 7])
    sd, sr = np.array([1.0, -np.inf, 0.5]), np.array([2.0, -np.inf, -1.0])
    t = comb((0, 1, 1, -1), sd, sr)                                         # rgb only, flipped: no 0 * inf nan
    assert np.isneginf(t[1]) and np.allclose(t[[0, 2]], [-2, 1])
    print("selfcheck ok")


if __name__ == "__main__":
    main()
