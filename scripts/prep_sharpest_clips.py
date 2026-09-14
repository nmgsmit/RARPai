#!/usr/bin/env python3
"""Sharpest cue clip per patient -> a finetune_depth dataset (relative depth, no scale anchors).

    <src>/<video>/<video>_clipNNN_f*.mp4       (cut_cue_clips output, GUI already black)
 -> <dst>/sharpness_rank.csv                   every clip > --min-mb, ranked by var(Laplacian)
 -> <dst>/{Train,Validation,Test}/rarp/<video>/clip_000/{images/<i>.jpg + <i>_mask.png, source_crop.json}

One clip per PATIENT (uuid or RARP_NNN prefix; a patient can have several <video> dirs), the one
with the highest `sharp` from rank_sharpness.py. Frames are cropped once to the 5:4 content
(x289 y4 1340x1072, same as the ruler dumps).

    python scripts/prep_sharpest_clips.py --src /home/nsmit2/data/depth_clips_staging \
        --dst ../data/processed/depthclips_sharpest
    python scripts/prep_sharpest_clips.py --selftest
"""
import argparse, csv, json, os, random, re
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from rank_sharpness import score_video

CROP = dict(x=289, y=4, w=1340, h=1072)          # source_crop.json of the 1920x1080 console
PATIENT = re.compile(r"^(RARP_\d+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")


def best_per_patient(rows):
    """rows: dicts with video(dir name), clip, sharp -> {patient: best row}."""
    best = {}
    for r in rows:
        p = PATIENT.match(r["video"]).group(1)
        if p not in best or r["sharp"] > best[p]["sharp"]:
            best[p] = r
    return best


def _score(path):
    return path, score_video(path, 15, 512, 50.0)


def extract(clip, out):
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        assert f.shape[:2] == (1080, 1920), (clip, f.shape)
        frames.append(f[CROP["y"]:CROP["y"] + CROP["h"], CROP["x"]:CROP["x"] + CROP["w"]])
    cap.release()
    # ponytail: GUI = pixels black in EVERY frame of the clip. The GUI was blacked at cut time and
    # data/templates is gone on Snellius, so full_gui_mask can't be re-run; this misses transient
    # popups. Restore the templates + recompute from the source video if a consumer needs them.
    # Real clips: this area is ~5.3% and identical for thr 2..20 (crisp edge); dilate 3 = cut_cue_clips.
    gui = cv2.dilate((np.stack(frames).max(axis=(0, 3)) <= 2).astype(np.uint8),
                     np.ones((7, 7), np.uint8)) * 255
    (out / "images").mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):                 # <stem>_mask.png next to the image (CLAUDE.md)
        cv2.imwrite(str(out / "images" / f"{i:07d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "images" / f"{i:07d}_mask.png"), gui)
    (out / "source_crop.json").write_text(json.dumps(
        {"version": 1, "source_size": [1920, 1080], "crop": CROP,
         "output_size": [CROP["w"], CROP["h"]]}, indent=2))
    return len(frames)


def selftest():
    rows = [{"video": "RARP_083-010-00.38", "clip": "a", "sharp": 5.0},
            {"video": "RARP_083-011-01.00", "clip": "b", "sharp": 9.0},   # same patient, sharper
            {"video": "31e2c520-bf13-4370-8807-50c3f8af3fd0-01.15-seg1", "clip": "c", "sharp": 1.0}]
    b = best_per_patient(rows)
    assert set(b) == {"RARP_083", "31e2c520-bf13-4370-8807-50c3f8af3fd0"}, b
    assert b["RARP_083"]["clip"] == "b", b
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/nsmit2/data/depth_clips_staging")
    ap.add_argument("--dst", default="../data/processed/depthclips_sharpest")
    ap.add_argument("--min-mb", type=float, default=1.0, help="skip clips at or below this size")
    ap.add_argument("--n-val", type=int, default=5, help="patients held out as Validation")
    ap.add_argument("--n-test", type=int, default=5, help="patients held out as Test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    src, dst = Path(a.src), Path(a.dst)
    assert not dst.exists(), f"{dst} exists; pick a new --dst"
    clips = [c for c in sorted(src.glob("*/*.mp4")) if c.stat().st_size > a.min_mb * 2**20]
    print(f"{len(clips)} clips > {a.min_mb} MB under {src}", flush=True)
    cv2.setNumThreads(1)
    with Pool(a.workers) as pool:
        scored = pool.map(_score, clips)
    rows = [{"video": c.parent.name, "clip": c.name, "size_mb": round(c.stat().st_size / 2**20, 2),
             **{k: float(v) for k, v in s.items()}} for c, s in scored if s]
    rows.sort(key=lambda r: r["sharp"], reverse=True)
    best = best_per_patient(rows)

    patients = sorted(best)
    random.Random(a.seed).shuffle(patients)
    split = {p: "Validation" if i < a.n_val else "Test" if i < a.n_val + a.n_test else "Train"
             for i, p in enumerate(patients)}
    dst.mkdir(parents=True)
    with open(dst / "sharpness_rank.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["video", "clip", "size_mb", "sharp", "nsharp", "blur_frac",
                               "patient", "selected"])
        w.writeheader()
        for r in rows:
            p = PATIENT.match(r["video"]).group(1)
            w.writerow({**r, "patient": p, "selected": split[p] if best[p] is r else ""})

    for p in sorted(best):
        r = best[p]
        n = extract(src / r["video"] / r["clip"], dst / split[p] / "rarp" / r["video"] / "clip_000")
        print(f"{split[p]:10s} {p[:8]}  sharp={r['sharp']:7.1f}  {n:3d} frames  {r['clip']}",
              flush=True)
    print(f"{len(best)} patients -> {sum(v == 'Train' for v in split.values())} Train / "
          f"{a.n_val} Validation / {a.n_test} Test; ranking in {dst / 'sharpness_rank.csv'}")


if __name__ == "__main__":
    main()
