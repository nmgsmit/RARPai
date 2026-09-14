#!/usr/bin/env python3
"""Sharpest cue clip per patient -> a finetune_depth dataset (relative depth, no scale anchors).

    <src>/<video>/<video>_clipNNN_fSSSSSS-EEEEEE.mp4   (cut_cue_clips output, GUI already black)
 -> <dst>/sharpness_rank.csv                   every clip > --min-mb, ranked by var(Laplacian)
 -> <dst>/{Train,Validation,Test}/rarp/<video>/clip_000/{images/<i>.jpg + <i>_mask.png, source_crop.json}

One clip per PATIENT (uuid or RARP_NNN prefix; a patient can have several <video> dirs), the one
with the highest `sharp` from rank_sharpness.py. Frames are cropped once to the 5:4 content
(x289 y4 1340x1072, same as the ruler dumps).

GUI masks come from cut_cue_clips.full_gui_mask on the SOURCE video (--gui-src, frames S..E from
the clip name): the staging clips have the GUI blacked, so the templates have nothing to match
there. The UMCdissectionvidNOgui sources only lack the fixed HUD, which full_gui_mask draws from
geometry anyway; cue bars and popups are still visible in them.

    python scripts/prep_sharpest_clips.py --src /home/nsmit2/data/depth_clips_staging \
        --dst ../data/processed/depthclips_sharpest
    python scripts/prep_sharpest_clips.py --dst ../data/processed/depthclips_sharpest --masks-only
    python scripts/prep_sharpest_clips.py --selftest
"""
import argparse, csv, glob, json, os, random, re
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np

from cut_cue_clips import (BAR_PREFIX, CONNECT_TEMPLATES, GUI_TEMPLATE_DIR, content_box,
                           full_gui_mask, load_templates)
from rank_sharpness import score_video

CROP = dict(x=289, y=4, w=1340, h=1072)          # source_crop.json of the 1920x1080 console
PATIENT = re.compile(r"^(RARP_\d+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
CUE = dict(m_thr=0.55, b_thr=0.90, min_bars=4, dilate=3)   # cut_cue_clips defaults
_T = {}


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


def _crop(a):
    return a[CROP["y"]:CROP["y"] + CROP["h"], CROP["x"]:CROP["x"] + CROP["w"]]


def load_gui_templates():
    mq = os.path.join(GUI_TEMPLATE_DIR, "Move_Que")
    allt = {Path(f).stem: cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            for f in sorted(glob.glob(os.path.join(mq, "*.png")))}
    markers = {k: v for k, v in allt.items() if not k.startswith(BAR_PREFIX)}
    bars = {k: v for k, v in allt.items() if k.startswith(BAR_PREFIX)}
    panels = {k: v for k, v in load_templates(GUI_TEMPLATE_DIR).items() if k not in CONNECT_TEMPLATES}
    assert markers and bars and panels, f"GUI templates missing under {GUI_TEMPLATE_DIR}"
    return panels, markers, bars


def gui_masks(video, start, n):
    """Cropped uint8 masks (255 = GUI) for source frames start..start+n-1."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)       # same seek cut_cue_clips used to cut the clip
    out, box = [], None
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        assert f.shape[:2] == (1080, 1920), (video, f.shape)
        if box is None:
            box = content_box(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        m = full_gui_mask(f, *_T["t"], box, CUE["m_thr"], CUE["b_thr"], CUE["min_bars"], CUE["dilate"])
        out.append(_crop(m).astype(np.uint8) * 255)
    cap.release()
    assert len(out) == n, (video, start, len(out), n)
    return out


def write_masks(imgdir, masks):
    """Write <i>_mask.png; return (mask frac, frac of near-black jpg px the mask covers)."""
    fr, cov = [], []
    for i, m in enumerate(masks):
        cv2.imwrite(str(imgdir / f"{i:07d}_mask.png"), m)
        black = cv2.imread(str(imgdir / f"{i:07d}.jpg")).max(axis=2) <= 10
        fr.append((m > 0).mean())
        cov.append((black & (m > 0)).sum() / max(black.sum(), 1))
    return float(np.median(fr)), float(np.median(cov))


def _start(clip_name):
    return int(re.search(r"_f(\d+)-\d+\.mp4$", clip_name).group(1))


def _masks_job(job):
    imgdir, video, start = job
    n = len(list(imgdir.glob("[0-9]*[0-9].jpg")))
    return imgdir, write_masks(imgdir, gui_masks(video, start, n))


def extract(clip, out):
    cap = cv2.VideoCapture(str(clip))
    (out / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        assert f.shape[:2] == (1080, 1920), (clip, f.shape)
        cv2.imwrite(str(out / "images" / f"{n:07d}.jpg"), _crop(f), [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    (out / "source_crop.json").write_text(json.dumps(
        {"version": 1, "source_size": [1920, 1080], "crop": CROP,
         "output_size": [CROP["w"], CROP["h"]]}, indent=2))
    return n


def run_masks(dst, gui_src, workers):
    """(Re)write every selected clip's masks from its source video, in place."""
    _T["t"] = load_gui_templates()
    rows = [r for r in csv.DictReader(open(dst / "sharpness_rank.csv")) if r["selected"]]
    jobs = [(dst / r["selected"] / "rarp" / r["video"] / "clip_000" / "images",
             Path(gui_src) / f"{r['video']}.mp4", _start(r["clip"])) for r in rows]
    cv2.setNumThreads(1)
    with ThreadPool(workers) as pool:                # cv2 frees the GIL; _T is shared
        res = pool.map(_masks_job, jobs)
    for imgdir, (fr, cov) in res:
        print(f"[mask] {imgdir.parent.parent.name[:8]}  gui={fr:.3f}  covers_black={cov:.3f}", flush=True)
    cov = np.array([c for _, (_, c) in res])
    print(f"[mask] {len(res)} clips, covers_black median {np.median(cov):.3f} min {cov.min():.3f}")


def selftest():
    rows = [{"video": "RARP_083-010-00.38", "clip": "a", "sharp": 5.0},
            {"video": "RARP_083-011-01.00", "clip": "b", "sharp": 9.0},   # same patient, sharper
            {"video": "31e2c520-bf13-4370-8807-50c3f8af3fd0-01.15-seg1", "clip": "c", "sharp": 1.0}]
    b = best_per_patient(rows)
    assert set(b) == {"RARP_083", "31e2c520-bf13-4370-8807-50c3f8af3fd0"}, b
    assert b["RARP_083"]["clip"] == "b", b
    assert _start("x-01.25.01.339_clip010_f007014-007031.mp4") == 7014
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/nsmit2/data/depth_clips_staging")
    ap.add_argument("--dst", default="../data/processed/depthclips_sharpest")
    ap.add_argument("--gui-src", default="/home/nsmit2/data/UMCdissectionvidNOgui",
                    help="source videos (<video>.mp4) the clips were cut from, for GUI masks")
    ap.add_argument("--masks-only", action="store_true", help="rewrite masks of an existing --dst")
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
    if a.masks_only:
        return run_masks(dst, a.gui_src, a.workers)

    assert not dst.exists(), f"{dst} exists; pick a new --dst (or --masks-only)"
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
    run_masks(dst, a.gui_src, a.workers)


if __name__ == "__main__":
    main()
