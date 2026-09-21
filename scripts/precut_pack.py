#!/usr/bin/env python3
"""Pack the footage BEFORE an arch-annotated Pure Arch clip, for surgical_map --context (zoomed-out context frames).

The Pure Arch clips are sub-clips of the UMCdissectionvid dissection clips: the video name carries the sub-clip start
("<parent>-HH.MM.SS.mmm-HH.MM.SS.mmm.mp4"). This decodes the parent from --before s ahead of the first annotated frame
to the last one, keeps every --stride-th frame plus every annotated frame, crops + GUI-masks them exactly as
arch_tip_data.pack_video does, and writes a FrameStore pack with PARENT frame numbers:
    <out>/<short>/frames.bin + frames.npy     (FrameStore)
    <out>/<short>/labels.json                 {"video", "offset_frames", "shift_xy", "labels": {parent frame: {"tip"}}}
The annotated frame f of the sub-clip is parent frame round(offset * fps) + f + dt; dt and the pixel shift of the
annotation tool's crop against ours are MEASURED on annotated frames (packed jpg vs decoded parent crop) and the tips
are moved by that shift, so a tip lands on the same tissue in our crop.

    python scripts/precut_pack.py --short RARP_079 RARP_088 e97cc145
"""
import argparse, json, re, sys
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_tip_data import FrameStore, append  # noqa: E402
from prep_sharpest_clips import _crop  # noqa: E402

PURE = [Path("/scratch-shared/nsmit2/arch_tip/pure"), Path("/scratch-shared/nsmit2/arch_tip/pure40")]
PARENTS = Path.home() / "data" / "UMCdissectionvid"


def parent_of(video):
    """'<parent>-HH.MM.SS.mmm-HH.MM.SS.mmm.mp4' -> (parent file name, sub-clip start in s)."""
    m = re.match(r"(.+?-\d\d\.\d\d\.\d\d\.\d+-\d\d\.\d\d\.\d\d\.\d+)-(\d\d)\.(\d\d)\.(\d\d)\.(\d+)-", video)
    if not m:
        return None, None
    h, mi, s, ms = map(int, m.groups()[1:])
    return m.group(1) + ".mp4", h * 3600 + mi * 60 + s + ms / 1000


def align(ref, cand):
    """Mean |diff| of two crops on pixels non-black in both, and (dx, dy): a point at x in ref is at x - (dx, dy) in cand."""
    g = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (ref, cand)]
    ok = (g[0] > 8) & (g[1] > 8)
    (dx, dy), _ = cv2.phaseCorrelate(g[1] * ok, g[0] * ok)
    return float(np.abs(g[0] - g[1])[ok].mean()), (dx, dy)


def pack(short, out, before, stride, workers):
    from arch_tip_render import _init, _mask
    from cut_cue_clips import content_box
    src = next(p / short for p in PURE if (p / short / "labels.json").exists())
    lab = json.loads((src / "labels.json").read_text())
    par, t0 = parent_of(lab["video"])
    if par is None or not (PARENTS / par).exists():
        raise SystemExit(f"{short}: no parent clip for {lab['video']}")
    tips = {int(f): v["tip"] for f, v in lab["labels"].items()}
    cap = cv2.VideoCapture(str(PARENTS / par))
    fps = cap.get(cv2.CAP_PROP_FPS)
    off = int(round(t0 * fps))
    fs = FrameStore(short, src.parent)

    # --- alignment on 5 annotated frames: dt in -4..4 by the smallest mean |diff|, then the pixel shift
    probe = sorted(tips)[:: max(1, len(tips) // 5)][:5]
    want = {off + f + dt: (f, dt) for f in probe for dt in range(-4, 5)}
    got, i = {}, 0
    ok, fr = cap.read()
    box = content_box(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    while ok and i <= max(want):
        if i in want:
            got[i] = _crop(fr)
        ok, fr = cap.read()
        i += 1
    dts, shifts = [], []
    for f in probe:
        ref = fs.jpg(fs.pos[f])
        errs = {dt: align(ref, got[off + f + dt]) for dt in range(-4, 5) if off + f + dt in got}
        dt = min(errs, key=lambda k: errs[k][0])
        dts.append(dt); shifts.append(errs[dt][1])
        print(f"  frame {f}: dt {dt:+d}, |diff| {errs[dt][0]:.1f} (dt 0: {errs[0][0]:.1f}), shift {np.round(errs[dt][1], 1)}")
    dt, sh = int(np.median(dts)), np.median(shifts, 0)
    if len(set(dts)) > 1:
        print(f"  WARNING: dt not constant {dts} -> median {dt}")

    # --- decode [first annotated - before, last annotated], keep the stride grid + every annotated frame
    ann = {off + f + dt: f for f in tips}
    lo, hi = max(0, min(ann) - int(before * fps)), max(ann)
    keep = set(range(lo, hi + 1, stride)) | set(ann)
    d = out / short
    d.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(PARENTS / par))
    index, i, batch = [], 0, []
    with open(d / "frames.bin.part", "wb") as fh, get_context("spawn").Pool(workers, _init, (box, d)) as pool:
        def flush():
            for stem, jpg, png in pool.imap(_mask, batch):
                append(fh, index, int(stem.rsplit("_", 1)[1]), jpg, png)
            batch.clear()
        while i <= hi:
            if i in keep:
                ok, fr = cap.read()
                if not ok:
                    break
                batch.append((f"{short}_{i:06d}", fr))
                if len(batch) >= 2 * workers:
                    flush()
            elif not cap.grab():
                break
            i += 1
        flush()
    (d / "frames.bin.part").replace(d / "frames.bin")
    np.save(d / "frames.npy", np.array(index, np.int64))
    labels = {str(p): {"tip": [tips[f][0] - sh[0], tips[f][1] - sh[1]]} for p, f in ann.items()}   # into OUR crop
    (d / "labels.json").write_text(json.dumps(dict(video=lab["video"], parent=par, fps=fps, offset_frames=off + dt,
                                                   shift_xy=sh.tolist(), labels=labels)))
    print(f"{short}: parent {par} @ {fps:.2f} fps, offset {off}+{dt}, shift {np.round(sh, 1)}, "
          f"packed {len(index)} frames {lo}..{hi} ({(min(ann) - lo) / fps:.0f} s before the first annotated)", flush=True)


def self_test():
    p, t = parent_of("RARP_064-02.10.38.140-02.12.05.128-00.00.22.363-00.01.27.154.mp4")
    assert p == "RARP_064-02.10.38.140-02.12.05.128.mp4" and abs(t - 22.363) < 1e-9, (p, t)
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (200, 240, 3), np.uint8), (9, 9), 0)
    moved = np.roll(img, (3, -5), (0, 1))                 # content moved 3 down, 5 left
    e, (dx, dy) = align(img, moved)
    assert abs(dx - 5) < 0.5 and abs(dy + 3) < 0.5, (dx, dy)
    y0, x0 = 100, 120                                     # a point in img sits at (x0 - dx, y0 - dy) in moved
    assert (moved[int(round(y0 - dy)), int(round(x0 - dx))] == img[y0, x0]).all()
    print("self-test ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--short", nargs="+")
    ap.add_argument("--out", default="/scratch-shared/nsmit2/arch_tip/precut")
    ap.add_argument("--before", type=float, default=60, help="s of footage ahead of the first annotated frame")
    ap.add_argument("--stride", type=int, default=6, help="keep every n-th parent frame (60 fps -> 10 fps)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    else:
        for s in a.short:
            pack(s, Path(a.out), a.before, a.stride, a.workers)
