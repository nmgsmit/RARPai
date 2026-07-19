#!/usr/bin/env python
"""Sweep marker/bar thresholds for cue detection and score each combination.

A cue is a striped bar with a digit marker at each end. Matching only the digits
also fires on popup-dialog lettering, so a pair is accepted only when enough
BAR SEGMENT templates (GrayBar/GraybarV/YellowH/YellowV) sit on the line between
the two markers. Digit thresholds go low (catch every cue), bar thresholds
higher (the segments are distinctive).

Two phases, because matching is slow and thresholding is not:

    collect  one pass per video, stores every candidate above a floor score
    sweep    re-thresholds the cached candidates, instant per combination

    python scripts/sweep_cue_thresholds.py --videos "data/raw/*.mp4"

Score = clips containing a frame with no connected bar. Padding adds pad_after
frames that legitimately lack a cue, so <= pad_after such frames is expected;
more means the bar went undetected mid-cue and would leak GUI into training.
"""
import argparse
import glob
import os
import pickle
import sys
import time
from multiprocessing.pool import ThreadPool

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cut_cue_clips import (  # noqa: E402
    BAR_TEMPLATES, collect_frame, content_box, frame_has_cue, runs,
)


# Detection logic is canonical in cut_cue_clips so the sweep and the pipeline
# can never drift apart; this script only re-thresholds cached candidates.


def collect(path, markers, bars, workers):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f0 = cap.read()
    cap.release()
    box = content_box(cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY))

    chunk = max(1, -(-n // max(1, workers)))
    jobs = [(s, min(chunk, n - s)) for s in range(0, n, chunk)]

    def do(job):
        start, count = job
        c = cv2.VideoCapture(path)
        c.set(cv2.CAP_PROP_POS_FRAMES, start)
        out = []
        for i in range(count):
            good, fr = c.read()
            if not good:
                break
            out.append((start + i, *collect_frame(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY),
                                                  markers, bars, box)))
        c.release()
        return out

    with ThreadPool(max(1, workers)) as pool:
        parts = pool.map(do, jobs)
    return n, {i: (mk, br) for part in parts for i, mk, br in part}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--templates", default="data/Move_Que")
    p.add_argument("--cache", default="data/sweep_cache")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--pad-after", type=int, default=2)
    p.add_argument("--pad-before", type=int, default=0)
    p.add_argument("--gap", type=int, default=3)
    p.add_argument("--min-clip", type=int, default=15)
    p.add_argument("--marker-thresh", nargs="+", type=float,
                   default=[0.50, 0.55, 0.60, 0.65])
    p.add_argument("--bar-thresh", nargs="+", type=float,
                   default=[0.60, 0.70, 0.80])
    p.add_argument("--min-bars", nargs="+", type=int, default=[2])
    p.add_argument("--recollect", action="store_true")
    args = p.parse_args()

    cv2.setNumThreads(1)
    allt = {os.path.splitext(os.path.basename(f))[0]: cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            for f in sorted(glob.glob(os.path.join(args.templates, "*.png")))}
    markers = {k: v for k, v in allt.items() if k not in BAR_TEMPLATES}
    bars = {k: v for k, v in allt.items() if k in BAR_TEMPLATES}
    print(f"markers: {sorted(markers)}\nbars   : {sorted(bars)}")
    os.makedirs(args.cache, exist_ok=True)

    data = {}
    for path in sorted([q for v in args.videos for q in (glob.glob(v) or [v])]):
        stem = os.path.splitext(os.path.basename(path))[0]
        cf = os.path.join(args.cache, stem + ".pkl")
        if os.path.exists(cf) and not args.recollect:
            with open(cf, "rb") as fh:
                data[stem] = pickle.load(fh)
            print(f"cached  {stem[:40]:42s} {data[stem][0]} frames")
            continue
        t0 = time.time()
        n, cand = collect(path, markers, bars, args.workers)
        with open(cf, "wb") as fh:
            pickle.dump((n, cand), fh)
        data[stem] = (n, cand)
        print(f"collect {stem[:40]:42s} {n} frames in {time.time() - t0:.0f}s")

    print(f"\n{'m_thr':>6} {'b_thr':>6} {'minbar':>6} {'clips':>6} "
          f"{'bad_clips':>10} {'worst':>6}   (bad = clip with > pad_after cue-less frames)")
    grid = []
    for mt in args.marker_thresh:
        for bt in args.bar_thresh:
            for mb in args.min_bars:
                tot_clips = bad = worst = 0
                for stem, (n, cand) in data.items():
                    flags = [frame_has_cue(*cand.get(i, ([], [])), mt, bt, mb)
                             for i in range(n)]
                    spans = runs(flags, args.pad_before, args.pad_after, n,
                                 args.min_clip, args.gap)
                    tot_clips += len(spans)
                    for s, e in spans:
                        miss = sum(1 for i in range(s, e + 1) if not flags[i])
                        worst = max(worst, miss)
                        if miss > args.pad_after:
                            bad += 1
                grid.append((mt, bt, mb, tot_clips, bad, worst))
                print(f"{mt:6.2f} {bt:6.2f} {mb:6d} {tot_clips:6d} {bad:10d} {worst:6d}",
                      flush=True)

    ok = [g for g in grid if g[4] == 0]
    best = max(ok or grid, key=lambda g: (g[3], -g[4]))
    print(f"\nmost clips with zero bad: m_thr={best[0]} b_thr={best[1]} "
          f"min_bars={best[2]} -> {best[3]} clips, {best[4]} bad")


if __name__ == "__main__":
    main()
