#!/usr/bin/env python3
"""Manually selected clips (GUI still visible) -> a finetune_depth Train set with fixed val/test.

    <src>/<video>-HH.MM.SS.mmm-HH.MM.SS.mmm[-segN].mp4        (1920x1080, GUI on screen)
 -> <dst>/Train/rarp/<clip stem>/clip_000/{images/<i>.jpg + <i>_mask.png, source_crop.json}
 -> <dst>/{Validation,Test}/rarp/*    symlinked from --val-test-from (shared across runs)
 -> <dst>/clips.csv                   clip, patient, zoom, frames, gui fraction

The GUI is still visible here, so cut_cue_clips.full_gui_mask runs on each clip's OWN frames; masked
pixels are blacked AND saved as <i>_mask.png (CLAUDE.md). `--zoom-csv` (zoomdet: clip,zoom) keeps
only 1x clips. Clips of a --val-test-from patient are refused (they would leak into Train).

    python scripts/prep_manual_clips.py --src ~/data/depthclips_manual_raw \
        --dst ../data/processed/depthclips_manual --zoom-csv ~/zoomcrop/zoom_manual.csv \
        --val-test-from ../data/processed/depthclips_sharpest_1x
    python scripts/prep_manual_clips.py --selftest
"""
import argparse, csv, os, re
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np

from cut_cue_clips import content_box, full_gui_mask
from prep_sharpest_clips import (CUE, _crop, link_val_test, load_gui_templates, patient, read_zoom,
                                 write_crop_json)

NAME = re.compile(r"^(?P<video>.+?)-(\d\d\.\d\d\.\d\d\.\d{3})-(\d\d\.\d\d\.\d\d\.\d{3})(-seg\d+)?\.mp4$")


def cut(job):
    clip, out, T = job
    (out / "images").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip))
    box, fr, n = None, [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        assert f.shape[:2] == (1080, 1920), (clip, f.shape)
        if box is None:
            box = content_box(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        m = full_gui_mask(f, *T, box, CUE["m_thr"], CUE["b_thr"], CUE["min_bars"], CUE["dilate"])
        f[m] = 0                                       # black pixels AND a mask file
        cv2.imwrite(str(out / "images" / f"{n:07d}.jpg"), _crop(f), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "images" / f"{n:07d}_mask.png"), _crop(m).astype(np.uint8) * 255)
        fr.append(_crop(m).mean())
        n += 1
    cap.release()
    write_crop_json(out)
    return n, float(np.median(fr)) if fr else float("nan")


def selftest():
    a = NAME.match("001320e9-9a68-4b93-9445-8ad26d3beca1-01.08.24.494-01.09.07.142-seg1-"
                   "00.00.00.750-00.00.01.222-seg1.mp4")
    assert a and a["video"] == "001320e9-9a68-4b93-9445-8ad26d3beca1-01.08.24.494-01.09.07.142-seg1", a
    b = NAME.match("0da0b610-32e7-48a8-96c8-8c9d7a152ec6-01.11.14.429-01.12.55.167-"
                   "00.00.05.968-00.00.06.558.mp4")                  # no -segN suffix
    assert b and b["video"] == "0da0b610-32e7-48a8-96c8-8c9d7a152ec6-01.11.14.429-01.12.55.167", b
    assert patient(b["video"]) == "0da0b610-32e7-48a8-96c8-8c9d7a152ec6"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/data/depthclips_manual_raw"))
    ap.add_argument("--dst", default="../data/processed/depthclips_manual")
    ap.add_argument("--val-test-from", default="../data/processed/depthclips_sharpest_1x")
    ap.add_argument("--zoom-csv", help="zoomdet csv (clip,zoom): keep only 1x clips")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    src, dst = Path(a.src), Path(a.dst)
    assert not dst.exists(), f"{dst} exists; pick a new --dst"
    clips = sorted(src.glob("*.mp4"))
    bad = [c.name for c in clips if not NAME.match(c.name)]
    assert not bad, f"unparsed clip names: {bad}"
    zoom = read_zoom(a.zoom_csv) if a.zoom_csv else {}
    keep = [c for c in clips if not zoom or zoom.get(c.name) == "1x"]
    print(f"{len(clips)} clips, {len(clips) - len(keep)} dropped as not 1x", flush=True)

    dst.mkdir(parents=True)
    held = link_val_test(a.val_test_from, dst)
    leak = [c.name for c in keep if patient(NAME.match(c.name)["video"]) in held]
    assert not leak, f"clips from val/test patients would leak into Train: {leak}"

    T = load_gui_templates()
    cv2.setNumThreads(1)
    jobs = [(c, dst / "Train" / "rarp" / c.stem / "clip_000", T) for c in keep]
    with ThreadPool(a.workers) as pool:
        res = pool.map(cut, jobs)
    with open(dst / "clips.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "patient", "zoom", "frames", "gui_frac"])
        for c, (n, g) in zip(keep, res):
            w.writerow([c.name, patient(NAME.match(c.name)["video"]), zoom.get(c.name, ""), n, round(g, 4)])
    g = np.array([x for _, x in res])
    print(f"Train: {len({patient(NAME.match(c.name)['video']) for c in keep})} patients, {len(keep)} clips, "
          f"{sum(n for n, _ in res)} frames | gui frac median {np.median(g):.3f} min {g.min():.3f} "
          f"max {g.max():.3f} | val/test linked from {a.val_test_from}", flush=True)


if __name__ == "__main__":
    main()
