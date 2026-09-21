#!/usr/bin/env python3
"""Pack the footage BEFORE an arch-annotated Pure Arch clip, for surgical_map --context (zoomed-out context frames).

The Pure Arch clips are sub-clips of the UMCdissectionvid dissection clips: the video name carries the sub-clip start
("<parent>-HH.MM.SS.mmm-HH.MM.SS.mmm.mp4"). This decodes the parent from --before s ahead of the first annotated frame
to the last one, keeps every --stride-th frame plus every annotated frame, crops + GUI-masks them exactly as
arch_tip_data.pack_video does, and writes a FrameStore pack with PARENT frame numbers:
    <out>/<short>/frames.bin + frames.npy     (FrameStore)
    <out>/<short>/labels.json                 {"video", "offset_frames", "shift_xy", "labels": {parent frame: {"tip"}}}
The annotated frame f of the sub-clip is parent frame round(offset * fps + f * r) + dt. The annotation tool's frame
numbers do NOT run at the parent's fps (first run: frame 0 matched, later frames matched nothing within +-4), so the
rate ratio r (parent fps / a common rate), dt and the pixel shift of the tool's crop against ours are all MEASURED on
5 annotated frames (packed jpg vs decoded parent crop, 1/4 res); nothing is packed unless every probe matches.

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


def pack(short, out, before, stride, workers, a_max_diff=15.0):
    from arch_tip_render import _init, _mask
    from cut_cue_clips import content_box
    if (out / short / "frames.npy").exists():
        print(f"{short}: already packed", flush=True)
        return
    src = next(p / short for p in PURE if (p / short / "labels.json").exists())
    lab = json.loads((src / "labels.json").read_text())
    par, t0 = parent_of(lab["video"])
    if par is None or not (PARENTS / par).exists():
        print(f"{short}: no parent clip for {lab['video']} - NOT packed")
        return
    tips = {int(f): v["tip"] for f, v in lab["labels"].items()}
    cap = cv2.VideoCapture(str(PARENTS / par))
    fps = cap.get(cv2.CAP_PROP_FPS)
    off = int(round(t0 * fps))
    fs = FrameStore(short, src.parent)

    # --- alignment on 5 annotated frames: rate ratio r x dt in -6..6 by the smallest mean |diff| (1/4 res)
    probe = sorted(tips)[:: max(1, len(tips) // 5)][:5]
    rates = sorted({1.0} | {fps / c for c in (59.94, 50, 30, 29.97, 25, 24, 20, 15, 12, 10)})
    at = lambda f, r: int(round(off + f * r))
    want = {at(f, r) + dt for r in rates for f in probe for dt in range(-6, 7)}
    small = lambda x: cv2.resize(x, (335, 268), interpolation=cv2.INTER_AREA)
    got, i = {}, 0
    ok, fr = cap.read()
    box = content_box(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    while ok and i <= max(want):
        if i in want:
            got[i] = small(_crop(fr))
        ok, fr = cap.read() if i < max(want) else (False, None)
        i += 1
    refs = {f: small(fs.jpg(fs.pos[f])) for f in probe}
    best = {}                                            # r -> (mean best |diff|, [(dt, |diff|, shift) per probe])
    for r in rates:
        per = []
        for f in probe:
            e = {dt: align(refs[f], got[at(f, r) + dt]) for dt in range(-6, 7) if at(f, r) + dt in got}
            if not e:
                break
            dt = min(e, key=lambda k: e[k][0])
            per.append((dt, *e[dt]))
        if len(per) == len(probe):
            best[r] = (float(np.mean([x[1] for x in per])), per)
    r = min(best, key=lambda k: best[k][0])
    err, per = best[r]
    for f, (dt_f, e_f, sh_f) in zip(probe, per):
        print(f"  frame {f}: r {r:.4f}, dt {dt_f:+d}, |diff| {e_f:.1f}, shift {np.round(np.array(sh_f) * 4, 1)} px")
    print("  other rates: " + ", ".join(f"{k:.3f}:{v[0]:.0f}" for k, v in sorted(best.items())))
    med = float(np.median([x[1] for x in per]))
    second = min(v[0] for k, v in best.items() if abs(k - r) > 0.01)
    if med > a_max_diff or err > second / 2:            # one blurred / occluded probe may miss; the rate must not
        print(f"{short}: no clear rate (median |diff| {med:.1f}, mean {err:.1f} vs next rate {second:.1f}) - NOT packed")
        return
    dts = [x[0] for x in per]
    dt, sh = int(np.median(dts)), np.median([x[2] for x in per], 0) * 4      # 1/4 res -> full-res px
    if max(dts) - min(dts) > 1:
        print(f"  WARNING: dt drifts {dts} -> median {dt}")

    # --- decode [first annotated - before, last annotated], keep the stride grid + every annotated frame
    ann = {at(f, r) + dt: f for f in tips}
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
    (d / "labels.json").write_text(json.dumps(dict(video=lab["video"], parent=par, fps=fps, rate=r, offset_frames=off + dt,
                                                   shift_xy=sh.tolist(), labels=labels)))
    print(f"{short}: parent {par} @ {fps:.2f} fps, rate {r:.4f}, offset {off}+{dt}, shift {np.round(sh, 1)}, "
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
