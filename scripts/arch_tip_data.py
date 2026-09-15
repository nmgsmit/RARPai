"""All frames + GUI masks + per-frame labels for the 10 arch-annotated videos -> ../data/processed/arch_tip_all.

Snellius genoa (jobs/arch_tip_data.sh). `arch_tip_all` is a symlink to /scratch-shared (home is ~93% full).
    <A>/<short>/images/<short>_<frame:05d>.jpg (+ _mask.png)   1340x1072 content crop, GUI blacked + mask
    <A>/labels_all.json   every frame with >= 1 annotator: per-annotator tips (offset-corrected into Nick's crop,
                          as compare_arch_multi), consensus tip/arch = their mean, n_annot, agree_px (mean
                          pairwise tip distance), best_half (all 3 annotators and agree_px <= the median over all
                          such frames = the 2957 frames of arch_tip_prep / round 1)
cv_splits() = the 20 leave-3-patients-out splits shared by arch_tip_head.py and arch_tip_cv.py.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_tip_prep import oriented  # noqa: E402
from compare_arch_multi import ANNOTATORS, DATA, REFERENCE, apex_of, find_arches, load_frames  # noqa: E402
from prep_sharpest_clips import patient  # noqa: E402
from visualize_multi_annotators import offset_for  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
A = ROOT.parent / "data" / "processed" / "arch_tip_all"


def cv_splits(rows, n_test=3, min_best=50):
    """Every 3-of-6 choice of the patients with >= 50 best-half frames (acd22d98 has 3, RARP_075 / d7419222 /
    de5f0c6b have no 3-annotator frames: always train)."""
    n = {}
    for r in rows:
        n[r["patient"]] = n.get(r["patient"], 0) + bool(r["best_half"])
    return [frozenset(c) for c in itertools.combinations(sorted(p for p, k in n.items() if k >= min_best), n_test)]


def video_rows(video):
    names = list(ANNOTATORS)
    ann = {}
    for n in names:
        path = find_arches(ANNOTATORS[n], video)
        frames = load_frames(path)[0] if path else None
        ann[n] = {int(k): v for k, v in (frames or {}).items()}
    off = {REFERENCE: np.zeros(2)}
    for n in names:                    # an annotator sharing no frame with Nick cannot be placed in his crop
        if n != REFERENCE and set(ann[n]) & set(ann[REFERENCE]):
            off[n] = offset_for(video, n, {(m, video): ann[m] for m in names})
    short, rows = patient(video)[:8], []
    for i in sorted(set().union(*(ann[n] for n in off))):
        have = [n for n in off if i in ann[n]]
        tips = {n: apex_of(ann[n][i])[0] - off[n] for n in have}
        arcs = [oriented(ann[n][i], off[n]) for n in have]
        T = np.array(list(tips.values()))
        pair = [np.linalg.norm(a - b) for a, b in itertools.combinations(T, 2)]
        rows.append(dict(stem=f"{short}_{i:05d}", short=short, patient=patient(video), video=video, frame=i,
                         n_annot=len(have), tips={n: t.tolist() for n, t in tips.items()}, tip=T.mean(0).tolist(),
                         arch=dict(left=np.mean([a[0] for a in arcs], 0).tolist(),
                                   right=np.mean([a[1] for a in arcs], 0).tolist(),
                                   height=float(np.mean([a[2] for a in arcs])),
                                   power=float(np.mean([a[3] for a in arcs]))),
                         agree_px=float(np.mean(pair)) if pair else None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--labels-only", action="store_true")
    args = ap.parse_args()
    from arch_tip_render import extract

    rows = []
    for v in sorted(p.name for p in DATA.glob("*.mp4")):
        if not args.labels_only:
            extract(v, A / patient(v)[:8] / "images", args.workers)
        rows += video_rows(v)
        print(f"{v}: {sum(r['video'] == v for r in rows)} annotated frames", flush=True)
    three = [r for r in rows if r["n_annot"] == 3]
    med = float(np.median([r["agree_px"] for r in three]))
    for r in rows:
        r["best_half"] = r["n_annot"] == 3 and r["agree_px"] <= med
    nb = sum(r["best_half"] for r in rows)
    print(f"{len(rows)} annotated frames, {len(three)} with all 3, median {med:.2f} px, {nb} best-half, "
          f"{len(cv_splits(rows))} splits")
    assert len(three) == 5914 and nb == 2957, "must reproduce compare_arch_multi / arch_tip_prep"
    A.mkdir(parents=True, exist_ok=True)
    (A / "labels_all.json").write_text(json.dumps(dict(median_px=med, rows=rows)))


if __name__ == "__main__":
    main()
