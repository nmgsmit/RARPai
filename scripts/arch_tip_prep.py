"""Best-half-agreement arch frames -> cropped images + GUI masks + consensus labels.

Reads outputs/arch_tip_comparison_multi_triple.json (compare_arch_multi.py): the 5914 frames all
three annotators (Nick, Veerle, Aron) share, 7 videos = 7 patients. Keeps frames whose 3-way mean
pairwise tip distance is <= the global median ("best half"), and writes per frame:
    <out>/images/<video_short>_<frame:05d>.jpg        1340x1072 content crop, GUI blacked
    <out>/images/<video_short>_<frame:05d>_mask.png   255 = GUI
    <out>/labels.json   one row per frame: consensus tip (mean of the 3 offset-corrected apexes,
                        Nick's crop frame), each annotator's tip, consensus arch (left/right/height/power)

    python scripts/arch_tip_prep.py            # local: the mp4s live next to the annotations
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_arch_multi import ANNOTATORS, DATA, REFERENCE, apex_of, find_arches, load_frames  # noqa: E402
from prep_sharpest_clips import CUE, _crop, load_gui_templates, patient  # noqa: E402
from cut_cue_clips import content_box, full_gui_mask  # noqa: E402
from visualize_multi_annotators import offset_for  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def oriented(entry, off):
    """(left, right, height, power) with left.x < right.x, shifted by -off. Swapping the ends flips
    the chord normal, so height changes sign to keep the same apex."""
    l, r, h = np.array(entry["left"]) - off, np.array(entry["right"]) - off, float(entry["height"])
    if l[0] > r[0]:
        l, r, h = r, l, -h
    return l, r, h, float(entry.get("power", 2.0))


def do_video(job):
    """One video: read sequentially, write the kept frames + masks, return their label rows."""
    video, want, out = job
    names = list(ANNOTATORS)
    T = load_gui_templates()
    ann = {n: {int(k): v for k, v in load_frames(find_arches(ANNOTATORS[n], video))[0].items()}
           for n in names}
    off = {n: (np.zeros(2) if n == REFERENCE else offset_for(video, n, {(m, video): ann[m] for m in names}))
           for n in names}
    short = patient(video)[:8]
    rows, cap, box, i = [], cv2.VideoCapture(str(DATA / video)), None, -1
    while want and i < max(want):
        ok, f = cap.read()
        i += 1
        if not ok:
            raise SystemExit(f"{video}: video ended at frame {i}, still need {sorted(want)[:5]}")
        if i not in want:
            continue
        box = box or content_box(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        stem = f"{short}_{i:05d}"
        if not (out / "images" / f"{stem}_mask.png").exists():   # resumable
            m = full_gui_mask(f, *T, box, CUE["m_thr"], CUE["b_thr"], CUE["min_bars"], CUE["dilate"])
            f[m] = 0                                            # black pixels AND a mask file
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), _crop(f), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / "images" / f"{stem}_mask.png"), _crop(m).astype(np.uint8) * 255)
        arcs = [oriented(ann[n][i], off[n]) for n in names]
        tips = {n: (apex_of(ann[n][i])[0] - off[n]).tolist() for n in names}
        rows.append(dict(stem=stem, video=video, patient=patient(video), frame=i,
                         tip=np.mean(list(tips.values()), axis=0).tolist(), tips=tips,
                         arch=dict(left=np.mean([a[0] for a in arcs], 0).tolist(),
                                   right=np.mean([a[1] for a in arcs], 0).tolist(),
                                   height=float(np.mean([a[2] for a in arcs])),
                                   power=float(np.mean([a[3] for a in arcs]))),
                         agree_px=want[i]["mean_pairwise_px"]))
    cap.release()
    print(f"{video}: {len(rows)} frames", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "data" / "processed" / "arch_tip_besthalf"))
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    triple = json.loads((ROOT / "outputs" / "arch_tip_comparison_multi_triple.json").read_text())
    med = float(np.median([r["mean_pairwise_px"] for r in triple]))
    keep = [r for r in triple if r["mean_pairwise_px"] <= med]
    print(f"{len(keep)}/{len(triple)} frames with 3-way mean pairwise tip distance <= median {med:.1f}px")

    jobs = [(v, {r["frame"]: r for r in keep if r["video"] == v}, out) for v in sorted({r["video"] for r in keep})]
    with Pool(args.workers) as pool:                            # GUI template matching is the bottleneck
        rows = [r for rs in pool.map(do_video, jobs) for r in rs]
    (out / "labels.json").write_text(json.dumps(dict(median_px=med, rows=rows), indent=1))
    print(f"wrote {len(rows)} rows -> {out / 'labels.json'}")


if __name__ == "__main__":
    main()
