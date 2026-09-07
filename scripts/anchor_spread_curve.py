"""
How much DEPTH SPREAD must a calibration anchor be annotated across?

The per-video arm calibration lands at 16.8% while the same fit on richer anchors reaches 7.3%
(CLAUDE_NOTES 2026-09-07). The difference is not the object, it is that the arm only ever appears
in a ~6 mm depth band, and an affine fit (Z' = a*Z + b) cannot separate the gain from the offset
without spread. This turns that into a curve an annotation protocol can be written from.

Method: take the anchor class that DOES have spread (the ruler), restrict it to a depth band of
width W inside each video, fit on that band only, and predict the OTHER classes in the same video.
Source and target are disjoint by class, so nothing is scored on its own fit. Sweeping W says what
a band of that width buys; sweeping the anchor COUNT at full spread says how many you need.

Reads the stage-1 dump written by fit_affine_scale.py --dump (CPU only, seconds):

    python scripts/anchor_spread_curve.py --dump outputs/adapt_base_test.npz
"""
from __future__ import annotations
import argparse

import numpy as np

from fit_affine_scale import fit, lengths


def band_trial(d, calib_cls, width, rng, affine=True, space="depth", n_cap=None, reps=40):
    """Median |pred-true|/true over non-calib objects, calibrating on a `width`-mm band per video.

    width=None means "no restriction" (all of that video's calib anchors). Repeated with random
    band centres because WHERE the band sits is itself a nuisance the protocol cannot control.
    """
    rays, z, mm, cls, vid = d["rays"], d["z"], d["mm"], d["cls"], d["video"]
    errs = []
    for _ in range(reps):
        pred, true = [], []
        for v in np.unique(vid):
            inv = vid == v
            src_all = np.where(inv & (cls == calib_cls))[0]
            tgt = np.where(inv & (cls != calib_cls))[0]
            if len(src_all) < 2 or not len(tgt):
                continue
            if width is None:
                src = src_all
            else:                              # a band of the requested width, centred anywhere
                lo, hi = z[src_all].min(), z[src_all].max()
                c = rng.uniform(lo, max(lo, hi - width)) if hi - lo > width else lo
                src = src_all[(z[src_all] >= c) & (z[src_all] <= c + width)]
            if n_cap is not None and len(src) > n_cap:
                src = rng.choice(src, n_cap, replace=False)
            if len(src) < 2 or np.ptp(z[src]) < 1e-3:
                continue                        # degenerate: the shift is not identifiable
            a, b = fit(rays[src], z[src], mm[src], space, affine)
            p = lengths(rays[tgt], z[tgt], a, b, space)
            pred.append(p); true.append(mm[tgt])
        if pred:
            p, t = np.concatenate(pred), np.concatenate(true)
            ok = np.isfinite(p)
            if ok.sum():
                errs.append(float(np.median(np.abs(p[ok] - t[ok]) / t[ok])))
    return (float(np.median(errs)), float(np.percentile(errs, 90)), len(errs)) if errs else (np.nan,) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="outputs/adapt_base_test.npz")
    ap.add_argument("--calib-class", type=int, default=1, help="1=Ruler 2=Catheter 3=Robot arm")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = dict(np.load(a.dump, allow_pickle=True))
    rng = np.random.default_rng(a.seed)
    z, cls = d["z"], d["cls"]
    print(f"[dump] {len(cls)} objects | calibrate on class {a.calib_class} "
          f"(n={int((cls == a.calib_class).sum())}, depth {z[cls == a.calib_class].min():.0f}-"
          f"{z[cls == a.calib_class].max():.0f} mm), score the others")

    print("\nDEPTH SPREAD of the calibration anchor (all of them, affine in depth, per video)")
    print(" band width   median err   p90      (mm at a 60mm working distance)")
    for w in (2, 5, 10, 15, 20, 30, 40, None):
        med, p90, n = band_trial(d, a.calib_class, w, rng)
        tag = "unrestricted" if w is None else f"{w:>4} mm    "
        print(f" {tag}   {med:8.1%}   {p90:6.1%}")

    print("\nNUMBER of annotated objects (full spread, affine in depth, per video)")
    print(" n per video  median err   p90")
    for n_cap in (2, 3, 5, 10, 20, 50, None):
        med, p90, _ = band_trial(d, a.calib_class, None, rng, n_cap=n_cap)
        tag = "unrestricted" if n_cap is None else f"{n_cap:>4}        "
        print(f" {tag}   {med:8.1%}   {p90:6.1%}")

    print("\nONE MULTIPLIER instead of affine (why the offset term is not optional)")
    for w in (5, 20, None):
        med, _, _ = band_trial(d, a.calib_class, w, rng, affine=False)
        print(f"  scale-only, band {'unrestricted' if w is None else f'{w} mm':>12}: {med:.1%}")


if __name__ == "__main__":
    main()
