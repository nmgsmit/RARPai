#!/usr/bin/env python3
"""Cue clips from depth_clips_staging -> a finetune_depth dataset (relative depth, no scale anchors).

    <src>/<video>/<video>_clipNNN_fSSSSSS-EEEEEE.mp4   (cut_cue_clips output, GUI already black)
 -> <dst>/sharpness_rank.csv                   every clip > --min-mb, ranked by var(Laplacian)
 -> <dst>/{Train,Validation,Test}/rarp/<video>/clip_NNN/{images/<i>.jpg + <i>_mask.png, source_crop.json}

Default: one clip per PATIENT (uuid or RARP_NNN prefix), the one with the highest `sharp` from
rank_sharpness.py, split by patient. `--all` keeps EVERY clip > --min-mb instead. `--zoom-csv`
(zoomdet output: clip,zoom) keeps only 1x clips. `--val-test-from` links an existing dataset's
Validation/Test (so runs share them) and keeps those patients out of Train. Frames are cropped once
to the 5:4 content (x289 y4 1340x1072, same as the ruler dumps).

GUI masks come from cut_cue_clips.full_gui_mask on the SOURCE video (--gui-src, frames S..E from
the clip name): the staging clips have the GUI blacked, so the templates have nothing to match
there. The UMCdissectionvidNOgui sources only lack the fixed HUD, which full_gui_mask draws from
geometry anyway; cue bars and popups are still visible in them.

    python scripts/prep_sharpest_clips.py --src /home/nsmit2/data/depth_clips_staging \
        --dst ../data/processed/depthclips_sharpest
    python scripts/prep_sharpest_clips.py --all --min-mb 1.5 --zoom-csv ~/zoomcrop/zoom_staging15.csv \
        --val-test-from ../data/processed/depthclips_sharpest_1x --dst ../data/processed/depthclips_all15_1x
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


def patient(video):
    return PATIENT.match(video).group(1)


def best_per_patient(rows):
    """rows: dicts with video(dir name), clip, sharp -> {patient: best row}."""
    best = {}
    for r in rows:
        p = patient(r["video"])
        if p not in best or r["sharp"] > best[p]["sharp"]:
            best[p] = r
    return best


def _score(path):
    return path, score_video(path, 15, 512, 50.0)


def _crop(a):
    return a[CROP["y"]:CROP["y"] + CROP["h"], CROP["x"]:CROP["x"] + CROP["w"]]


def write_crop_json(out):
    (out / "source_crop.json").write_text(json.dumps(
        {"version": 1, "source_size": [1920, 1080], "crop": CROP,
         "output_size": [CROP["w"], CROP["h"]]}, indent=2))


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


def _clip_index(clip_name):
    m = re.search(r"_clip(\d+)_", clip_name)
    return int(m.group(1)) if m else 0


def _masks_job(job):
    imgdir, video, start = job
    jpgs = sorted(imgdir.glob("[0-9]*[0-9].jpg"))
    # The templates miss some popups/cue panels in the source frame (solid 334-474 px rectangles on
    # 5/77 clips) that cut_cue_clips DID black at cut time. Black in EVERY frame of the clip is that
    # cut-time GUI, so OR it in (dilate 3 = cut_cue_clips) -- tissue is never black in all frames.
    cut = (np.stack([cv2.imread(str(j)) for j in jpgs]).max(axis=(0, 3)) <= 10).astype(np.uint8)
    cut = cv2.dilate(cut, np.ones((7, 7), np.uint8)) * 255
    return imgdir, write_masks(imgdir, [m | cut for m in gui_masks(video, start, len(jpgs))])


def extract(job):
    clip, out = job
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
    write_crop_json(out)
    return n


def run_masks(dst, gui_src, workers):
    """(Re)write the masks of every clip this script wrote (rows with a `dir`), in place."""
    _T["t"] = load_gui_templates()
    rows = list(csv.DictReader(open(dst / "sharpness_rank.csv")))
    if rows and "dir" not in rows[0]:              # datasets built before --all: one clip_000 each
        for r in rows:
            r["dir"] = f"{r['selected']}/rarp/{r['video']}/clip_000" if r["selected"] else ""
    jobs = [(dst / r["dir"] / "images", Path(gui_src) / f"{r['video']}.mp4", _start(r["clip"]))
            for r in rows if r["dir"]]
    cv2.setNumThreads(1)
    with ThreadPool(workers) as pool:                # cv2 frees the GIL; _T is shared
        res = pool.map(_masks_job, jobs)
    for imgdir, (fr, cov) in res:
        print(f"[mask] {imgdir.parent.parent.name[:8]}/{imgdir.parent.name}  gui={fr:.3f}  "
              f"covers_black={cov:.3f}", flush=True)
    cov = np.array([c for _, (_, c) in res])
    print(f"[mask] {len(res)} clips, covers_black median {np.median(cov):.3f} min {cov.min():.3f}")


def link_val_test(src_root, dst):
    """Symlink src_root/{Validation,Test}/rarp/<video> into dst; return their patients."""
    held = set()
    for sp in ("Validation", "Test"):
        (dst / sp / "rarp").mkdir(parents=True, exist_ok=True)
        for v in sorted((Path(src_root) / sp / "rarp").iterdir()):
            os.symlink(v.resolve(), dst / sp / "rarp" / v.name)
            held.add(patient(v.name))
    return held


def read_zoom(path):
    return {r["clip"]: r["zoom"] for r in csv.DictReader(open(os.path.expanduser(path)))}


def selftest():
    rows = [{"video": "RARP_083-010-00.38", "clip": "a", "sharp": 5.0},
            {"video": "RARP_083-011-01.00", "clip": "b", "sharp": 9.0},   # same patient, sharper
            {"video": "31e2c520-bf13-4370-8807-50c3f8af3fd0-01.15-seg1", "clip": "c", "sharp": 1.0}]
    b = best_per_patient(rows)
    assert set(b) == {"RARP_083", "31e2c520-bf13-4370-8807-50c3f8af3fd0"}, b
    assert b["RARP_083"]["clip"] == "b", b
    assert _start("x-01.25.01.339_clip010_f007014-007031.mp4") == 7014
    assert _clip_index("x-01.25.01.339_clip010_f007014-007031.mp4") == 10
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/nsmit2/data/depth_clips_staging")
    ap.add_argument("--dst", default="../data/processed/depthclips_sharpest")
    ap.add_argument("--gui-src", default="/home/nsmit2/data/UMCdissectionvidNOgui",
                    help="source videos (<video>.mp4) the clips were cut from, for GUI masks")
    ap.add_argument("--masks-only", action="store_true", help="rewrite masks of an existing --dst")
    ap.add_argument("--all", action="store_true",
                    help="keep EVERY clip > --min-mb, not only the sharpest per patient")
    ap.add_argument("--zoom-csv", help="zoomdet csv (clip,zoom): keep only 1x clips")
    ap.add_argument("--val-test-from",
                    help="existing dataset: symlink its Validation/Test, keep their patients out of Train")
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

    zoom = read_zoom(a.zoom_csv) if a.zoom_csv else {}
    cand = [r for r in rows if not zoom or zoom.get(r["clip"]) == "1x"]
    dst.mkdir(parents=True)
    held = link_val_test(a.val_test_from, dst) if a.val_test_from else set()
    cand = [r for r in cand if patient(r["video"]) not in held]
    keep = cand if a.all else list(best_per_patient(cand).values())

    pats = sorted({patient(r["video"]) for r in keep})
    if a.val_test_from:
        psplit = dict.fromkeys(pats, "Train")
    else:
        random.Random(a.seed).shuffle(pats)
        psplit = {p: "Validation" if i < a.n_val else "Test" if i < a.n_val + a.n_test else "Train"
                  for i, p in enumerate(pats)}
    kept = {id(r) for r in keep}
    for r in rows:
        r["zoom"] = zoom.get(r["clip"], "")
        r["selected"] = psplit[patient(r["video"])] if id(r) in kept else ""
        r["dir"] = f"{r['selected']}/rarp/{r['video']}/clip_{_clip_index(r['clip']):03d}" if id(r) in kept else ""
    with open(dst / "sharpness_rank.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["video", "clip", "size_mb", "sharp", "nsharp", "blur_frac", "zoom",
                               "selected", "dir"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    with ThreadPool(a.workers) as pool:
        nframes = pool.map(extract, [(src / r["video"] / r["clip"], dst / r["dir"]) for r in keep])
    for sp in ("Train", "Validation", "Test"):
        idx = [i for i, r in enumerate(keep) if r["selected"] == sp]
        print(f"{sp:10s} {len({patient(keep[i]['video']) for i in idx}):3d} patients "
              f"{len(idx):4d} clips {sum(nframes[i] for i in idx):6d} frames written", flush=True)
    print(f"{len(rows)} ranked, {len(rows) - len(cand)} dropped by zoom/held-out patients, "
          f"{len(keep)} kept ({'all' if a.all else 'sharpest per patient'}); "
          f"val/test {'linked from ' + a.val_test_from if held else 'split here'}", flush=True)
    run_masks(dst, a.gui_src, a.workers)


if __name__ == "__main__":
    main()
