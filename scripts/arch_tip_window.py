"""Tip deviation on the cutting windows only: per test video the 5 s around the urethra cut (Nick, 2026-09-22).

Windows are seconds into each test clip (the 59.94 fps InterAnnotator mp4s, frame = second * fps). Two test sets:
  best-half  the earlier test frames (per video the better-agreeing half of its 3-annotator frames) inside the window
  all-3      every frame all three annotators marked inside the window
Scored against the 3-way consensus tip, tip only. Reads the per-frame predictions that `arch_tip_pure.py consistency`
saved per training set (outputs/<set>/pred_all/<short>.npz: frames, tip = 3-seed mean). Human = each annotator vs the
mean of the other two. CPU only.

    python scripts/arch_tip_window.py --sets arch_tip_pure arch_tip_pure40 arch_tip_pure_v2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_tip_data import A  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FPS = 59.94005994005994
WINDOWS = {"4db28d2e": (5, 10), "749c8234": (26, 31), "46867a8e": (6, 11), "acd22d98": (12, 17), "cada5bef": (6, 11),
           "RARP_062": (18, 23), "RARP_063": (36, 41)}
LABEL = {"arch_tip_pure": "14 videos", "arch_tip_pure40": "36 videos", "arch_tip_pure_v2": "36 re-annotated"}


def window_frames(rows):
    """{short: {'all-3': rows, 'best-half': rows}} inside each window."""
    out = {}
    for s, (t0, t1) in WINDOWS.items():
        rs = sorted((r for r in rows if r["short"] == s and r["n_annot"] == 3), key=lambda r: r["frame"])
        med = np.median([r["agree_px"] for r in rs])
        inw = [r for r in rs if t0 * FPS <= r["frame"] < t1 * FPS]
        anyw = sorted((r for r in rows if r["short"] == s and t0 * FPS <= r["frame"] < t1 * FPS), key=lambda r: r["frame"])
        # "any": every annotated frame in the window vs the mean of whoever annotated it; 749c8234 and RARP_062 only
        # have Nick inside their windows, so this is the only set that covers all 7 videos
        out[s] = {"all-3": inw, "best-half": [r for r in inw if r["agree_px"] <= med], "any": anyw}
    return out


def human(rs):
    return np.array([np.mean([np.linalg.norm(np.array(t) - np.mean([o for m, o in r["tips"].items() if m != n], 0))
                              for n, t in r["tips"].items()]) for r in rs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["arch_tip_pure", "arch_tip_pure40", "arch_tip_pure_v2"])
    a = ap.parse_args()
    rows = json.loads((A / "labels_all.json").read_text())["rows"]
    W = window_frames(rows)
    print("frames per window (all-3 / best-half / any):  "
          + "  ".join(f"{s} {len(w['all-3'])}/{len(w['best-half'])}/{len(w['any'])}" for s, w in W.items()))
    res = {}
    for ts in ("best-half", "all-3", "any"):
        res[ts] = {"human": {s: float(human(W[s][ts]).mean()) if W[s][ts] and all(len(r["tips"]) > 1 for r in W[s][ts])
                             else float("nan") for s in WINDOWS}}
        for name in a.sets:
            per = {}
            for s in WINDOWS:
                f = ROOT / "outputs" / name / "pred_all" / f"{s}.npz"
                if not f.exists() or not W[s][ts]:
                    per[s] = float("nan")
                    continue
                z = np.load(f)
                pos = {int(fr): k for k, fr in enumerate(z["frames"])}
                gt = np.array([r["tip"] for r in W[s][ts]])
                pred = z["tip"][[pos[r["frame"]] for r in W[s][ts]]]
                per[s] = float(np.linalg.norm(pred - gt, axis=1).mean())
            res[ts][LABEL.get(name, name)] = per
        print(f"\n=== tip error in the cutting windows, test set '{ts}' (px, 3-seed average; mean = over videos) ===")
        print(f"{'model':<18}{'mean':>7}  " + " ".join(f"{s:>9}" for s in WINDOWS))
        for m, per in res[ts].items():
            v = [x for x in per.values() if not np.isnan(x)]
            print(f"{m:<18}{np.mean(v) if v else float('nan'):7.1f}  " + " ".join(f"{per[s]:9.1f}" for s in WINDOWS))
    out = ROOT / "outputs" / "arch_tip_window.json"
    out.write_text(json.dumps(dict(windows=WINDOWS, fps=FPS, results=res,
                                   n_frames={s: {k: len(v) for k, v in w.items()} for s, w in W.items()}), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    assert int(5 * FPS) == 299 and int(10 * FPS) == 599
    main()
