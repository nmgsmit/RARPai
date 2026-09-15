"""All frames + GUI masks + per-frame labels for the 10 arch-annotated videos -> ../data/processed/arch_tip_all.

Snellius genoa (jobs/arch_tip_data.sh). `arch_tip_all` is a symlink to /scratch-shared (home is ~93% full).
    <A>/<short>/frames.bin + frames.npy   every frame's 1340x1072 GUI-blacked crop (JPEG q95) + GUI mask (PNG, 255 =
                          GUI), packed per video; read with FrameStore. NOT one file per frame: from a compute node,
                          creating small files on /scratch-shared ran at 9-43 frames/s (home ~1000, but no quota)
    <A>/labels_all.json   every frame with >= 1 annotator: per-annotator tips (offset-corrected into Nick's crop,
                          as compare_arch_multi), consensus tip/arch = their mean, n_annot, agree_px (mean
                          pairwise tip distance), best_half (all 3 annotators and agree_px <= the median over all
                          such frames = the 2957 frames of arch_tip_prep / round 1)
cv_splits() = the 20 leave-3-patients-out splits shared by arch_tip_head.py and arch_tip_cv.py.

    python scripts/arch_tip_data.py --selfcheck
"""
import argparse
import itertools
import json
import sys
import tempfile
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_tip_prep import oriented  # noqa: E402
from compare_arch_multi import ANNOTATORS, DATA, REFERENCE, apex_of, find_arches, load_frames  # noqa: E402
from prep_sharpest_clips import patient  # noqa: E402
from visualize_multi_annotators import offset_for  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
A = ROOT.parent / "data" / "processed" / "arch_tip_all"


class FrameStore:
    """One video's crops + GUI masks. frames.npy rows: (frame, jpg offset, jpg length, png offset, png length);
    methods take the row position k (pos[frame] -> k)."""

    def __init__(self, short, root=A):
        d = Path(root) / short
        self.index = np.load(d / "frames.npy")
        self.frames = self.index[:, 0]
        self.pos = {int(f): k for k, f in enumerate(self.frames)}
        self.bin = np.memmap(d / "frames.bin", np.uint8, "r")

    def __len__(self):
        return len(self.frames)

    def jpg(self, k):
        """BGR uint8, 1072x1340."""
        _, o, n, _, _ = self.index[k]
        return cv2.imdecode(np.asarray(self.bin[o:o + n]), cv2.IMREAD_COLOR)

    def mask(self, k):
        """bool, True = GUI."""
        _, _, _, o, n = self.index[k]
        return cv2.imdecode(np.asarray(self.bin[o:o + n]), cv2.IMREAD_GRAYSCALE) > 0


def append(fh, index, frame, jpg, png):
    o = fh.tell()
    fh.write(jpg)
    fh.write(png)
    index.append((frame, o, len(jpg), o + len(jpg), len(png)))


def pack_video(video, d, workers):
    """Sequential decode (exact frame indices), GUI masking + encoding in a spawn pool, one appending writer.
    frames.npy is written last, so it marks the video done; a killed run leaves only frames.bin.part."""
    from arch_tip_render import _init, _mask
    from cut_cue_clips import content_box

    if (d / "frames.npy").exists():
        print(f"{d.name}: already packed", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    cap, short = cv2.VideoCapture(str(DATA / video)), patient(video)[:8]
    ok, f = cap.read()
    box, i, index = content_box(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)), 0, []
    with open(d / "frames.bin.part", "wb") as fh, get_context("spawn").Pool(workers, _init, (box, d)) as pool:
        while ok:
            batch = []
            while ok and len(batch) < 2 * workers:
                batch.append((f"{short}_{i:05d}", f))
                ok, f = cap.read()
                i += 1
            for stem, jpg, png in pool.imap(_mask, batch):     # ordered
                append(fh, index, int(stem.rsplit("_", 1)[1]), jpg, png)
            print(f"{short}: {i} frames", flush=True)
    (d / "frames.bin.part").replace(d / "frames.bin")
    np.save(d / "frames.npy", np.array(index, np.int64))


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
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()

    rows = []
    for v in sorted(p.name for p in DATA.glob("*.mp4")):
        if not args.labels_only:
            pack_video(v, A / patient(v)[:8], args.workers)
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


def selfcheck():
    """Packed frames must come back at the right index: same mask exactly, near-same image."""
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "vid"
        d.mkdir()
        index, truth = [], {}
        with open(d / "frames.bin", "wb") as fh:
            for frame in (0, 1, 7):                                      # a gap in the frame numbers
                img = np.full((1072, 1340, 3), 20 + 30 * frame, np.uint8)
                m = rng.random((1072, 1340)) > 0.9
                truth[frame] = (img, m)
                append(fh, index, frame, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(),
                       cv2.imencode(".png", m.astype(np.uint8) * 255)[1].tobytes())
        np.save(d / "frames.npy", np.array(index, np.int64))
        fs = FrameStore("vid", root=tmp)
        assert list(fs.frames) == [0, 1, 7] and len(fs) == 3 and fs.pos[7] == 2
        for frame, (img, m) in truth.items():
            k = fs.pos[frame]
            assert np.array_equal(fs.mask(k), m), frame
            assert np.abs(fs.jpg(k).astype(int) - img).max() <= 2, frame
        del fs
    print("selfcheck ok")


if __name__ == "__main__":
    main()
