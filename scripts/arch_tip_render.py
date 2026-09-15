"""Render one FULL video with the depth arch fit, the curve samples it scored, and the annotators' arches.

Same pipeline as the test in arch_tip_fit.py (GUI-masked crop -> ruler-calibrated UniDepth at 1/4 res ->
fit with the train-tuned config/prior from outputs/arch_tip_depth/results.json), but on every frame, also
the ones nobody annotated.

    python scripts/arch_tip_render.py --video 46867a8e --extract          # frames + GUI masks (local)
    sbatch jobs/arch_tip_unidepth.sh --dir ../data/processed/arch_tip_video_46867a8e/images \
                                     --out ../data/processed/arch_tip_video_46867a8e/depth
    python scripts/arch_tip_render.py --video 46867a8e                    # -> outputs/arch_tip_depth/<id>_fit.mp4

Left = frame, right = inverse depth (turbo, GUI grey). Both carry: annotator arches (Nick cyan, Veerle
magenta, Aron green, offset-corrected into Nick's crop), consensus tip (white star, only when all three
annotated), fitted arch (red) with its predicted tip (red x = apex + train bias), the 33 curve samples
(white dot = log-depth deeper inward, i.e. supports the fit under the chosen sign; black dot = against;
grey x = excluded: GUI or outside the frame), and the train apex prior (box = search region, ellipse = 2 sigma).
"""
import argparse
import json
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_tip_fit import DS, Prior, curve, fit, grads, score  # noqa: E402
from compare_arch_multi import ANNOTATORS, DATA, REFERENCE, find_arches, load_frames  # noqa: E402
from cut_cue_clips import content_box, full_gui_mask  # noqa: E402
from prep_sharpest_clips import CUE, _crop, load_gui_templates, patient  # noqa: E402
from visualize_multi_annotators import COLORS, arch_points, offset_for  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BEST = ROOT.parent / "data" / "processed" / "arch_tip_besthalf"
_W = {}


# ------------------------------------------------------------------------------ extract
def _init(box, out):
    _W.update(T=load_gui_templates(), box=box, out=out)


def _mask(job):
    stem, f = job
    m = full_gui_mask(f, *_W["T"], _W["box"], CUE["m_thr"], CUE["b_thr"], CUE["min_bars"], CUE["dilate"])
    f[m] = 0                                                    # black pixels AND a mask file
    cv2.imwrite(str(_W["out"] / f"{stem}.jpg"), _crop(f), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(_W["out"] / f"{stem}_mask.png"), _crop(m).astype(np.uint8) * 255)


def extract(video, out, workers):
    """Sequential decode (exact frame indices), GUI masking in a pool, 64 frames in memory at a time."""
    out.mkdir(parents=True, exist_ok=True)
    cap, short = cv2.VideoCapture(str(DATA / video)), patient(video)[:8]
    ok, f = cap.read()
    box, i = content_box(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)), 0
    with Pool(workers, _init, (box, out)) as pool:
        while ok:
            batch = []
            while ok and len(batch) < max(64, 2 * workers):
                stem = f"{short}_{i:05d}"
                if not (out / f"{stem}_mask.png").exists():
                    batch.append((stem, f))
                ok, f = cap.read()
                i += 1
            pool.map(_mask, batch)
            print(f"{i} frames", flush=True)


# ------------------------------------------------------------------------------- render
def bgr(name):
    return tuple(int(c) for c in COLORS[name][::-1])


def poly(img, pts, color, t=3):
    cv2.polylines(img, [np.round(pts).astype(np.int32)], False, color, t, cv2.LINE_AA)


def render(video, d, out_path):
    res = json.loads((ROOT / "outputs" / "arch_tip_depth" / "results.json").read_text())
    cfg, sign, bias = res["config"], res["config"]["sign"], np.array(res["config"]["bias"])
    rows = json.loads((BEST / "labels.json").read_text())["rows"]
    prior = Prior([r for r in rows if r["patient"] in res["train"]])
    best_half = {r["frame"] for r in rows if r["video"] == video}
    in_train = patient(video) in res["train"]

    names = list(ANNOTATORS)
    ann = {n: {int(k): v for k, v in load_frames(find_arches(ANNOTATORS[n], video))[0].items()} for n in names}
    off = {n: (np.zeros(2) if n == REFERENCE else offset_for(video, n, {(m, video): ann[m] for m in names}))
           for n in names}

    cap = cv2.VideoCapture(str(DATA / video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    stems = sorted(p.stem for p in (d / "depth").glob("*.npz"))
    N = len(list((d / "images").glob("*.jpg")))                # whole video, not just frames with depth
    evals, evecs = np.linalg.eigh(np.linalg.inv(prior.icov))
    ell_axes = tuple(int(2 * np.sqrt(e)) for e in evals)
    ell_ang = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))

    # annotators put the apex OUTSIDE the frame on some videos -> grey margin so off-frame tips stay visible
    apx = np.array([arch_points(e)[1] - off[n] for n in names for e in ann[n].values()])
    PT = int(np.clip(-apx[:, 1].min() + 40, 0, 1000))
    PL = int(np.clip(-apx[:, 0].min() + 40, 0, 1000))
    PR = int(np.clip(apx[:, 0].max() - 1340 + 40, 0, 1000))
    o = np.array([PL, PT])                                      # crop coords -> canvas coords
    x0, y0 = (np.array(prior.box[::2]) + o).astype(int)
    x1, y1 = (np.array(prior.box[1::2]) + o).astype(int)
    pix = lambda q: tuple(int(v) for v in np.asarray(q) + o)   # noqa: E731
    STRIP = 120
    CW, CH = 2 * (1340 + PL + PR), 1072 + PT + 2 * STRIP
    OW = 1920
    OH = int(round(CH * OW / CW / 2)) * 2

    tmp = out_path.with_suffix(".mp4v.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OW, OH))
    errs = {"all3": [], "best_half": []}
    for k, stem in enumerate(stems):
        i = int(stem.rsplit("_", 1)[1])
        img = cv2.imread(str(d / "images" / f"{stem}.jpg"))
        H, W = img.shape[:2]
        depth = np.load(d / "depth" / f"{stem}.npz")["depth"].astype(np.float32)
        gui_full = cv2.imread(str(d / "images" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        gui = cv2.resize(gui_full.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_AREA) > 0
        gx, gy, valid = grads(depth, gui, cfg["sigma"])
        best = fit(gx, gy, valid, prior, sign, cfg["lam"])
        tip = best[:2] + bias
        s_raw = sign * score(best[None], gx, gy, valid)[0]
        pen = cfg["lam"] * prior.mahal(best[None, :2])[0]

        inv = 1 / np.clip(depth, 1, None)
        lo, hi = np.percentile(inv, [2, 98])
        col = cv2.applyColorMap((np.clip((inv - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col = cv2.resize(col, (W, H), interpolation=cv2.INTER_NEAREST)
        col[gui_full] = 90

        x, y, nx, ny = (a[0] for a in curve(best[None]))
        ix, iy = np.rint(x / DS).astype(int), np.rint(y / DS).astype(int)
        inb = (ix >= 0) & (ix < depth.shape[1]) & (iy >= 0) & (iy < depth.shape[0])
        ixc, iyc = ix.clip(0, depth.shape[1] - 1), iy.clip(0, depth.shape[0] - 1)
        used = inb & valid[iyc, ixc]
        g = sign * (gx[iyc, ixc] * nx + gy[iyc, ixc] * ny)

        tips = {}
        for n in names:
            if i in ann[n]:
                pts, apex = arch_points(ann[n][i])
                tips[n] = apex - off[n]
        consensus = np.mean(list(tips.values()), 0) if len(tips) == 3 else None
        if consensus is not None:
            e = float(np.linalg.norm(tip - consensus))
            errs["all3"].append(e)
            if i in best_half:
                errs["best_half"].append(e)

        pad = dict(top=PT, bottom=0, left=PL, right=PR, borderType=cv2.BORDER_CONSTANT, value=(45, 45, 45))
        panels = [cv2.copyMakeBorder(img, **pad), cv2.copyMakeBorder(col, **pad)]
        for p in panels:
            cv2.rectangle(p, (PL, PT), (PL + W - 1, PT + H - 1), (120, 120, 120), 2)   # the real frame edge
            cv2.rectangle(p, (x0, y0), (x1, y1), (200, 200, 200), 1)
            cv2.ellipse(p, pix(prior.mu), ell_axes, ell_ang, 0, 360, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.drawMarker(p, pix(prior.mu), (200, 200, 200), cv2.MARKER_CROSS, 24, 2)
            for n in names:
                if i in ann[n]:
                    pts, apex = arch_points(ann[n][i])
                    poly(p, pts - off[n] + o, bgr(n), 3)
                    cv2.circle(p, pix(apex - off[n]), 9, bgr(n), -1, cv2.LINE_AA)
            if consensus is not None:
                cv2.drawMarker(p, pix(consensus), (255, 255, 255), cv2.MARKER_STAR, 50, 3)
            poly(p, np.stack([x, y], 1) + o, (0, 0, 255), 3)
            for j in range(len(x)):
                c = pix((x[j], y[j]))
                if used[j]:
                    r = 5 + int(min(7, abs(g[j]) * 2))
                    cv2.circle(p, c, r, (255, 255, 255) if g[j] > 0 else (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(p, c, r, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.drawMarker(p, c, (150, 150, 150), cv2.MARKER_TILTED_CROSS, 14, 3)
            cv2.drawMarker(p, pix(tip), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 50, 6)

        frame = np.hstack(panels)
        head, foot = (np.zeros((STRIP, frame.shape[1], 3), np.uint8) for _ in range(2))
        line1 = (f"{patient(video)[:8]} ({'TRAIN' if in_train else 'TEST'})   frame {i}/{N - 1}   t {i / fps:6.2f}s   "
                 f"annotated: {' '.join(n for n in names if n in tips) or 'none'}"
                 f"{'   [best-half test frame]' if i in best_half else ''}")
        line2 = (f"fit: depth score {s_raw:+.2f}   prior penalty {pen:.2f}   samples used {used.sum()}/{len(x)}"
                 + (f"   tip error vs consensus {np.linalg.norm(tip - consensus):.0f} px" if consensus is not None else ""))
        cv2.putText(head, line1, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(head, line2, (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(foot, "cyan Nick | magenta Veerle | green Aron (dot = their tip) | white star = consensus tip | "
                    "red arch + red x = depth fit + its tip | grey margin = outside the camera frame",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(foot, "dots on red arch = the 33 depth samples scored: white = deeper inside the arch (supports), "
                    "black = against, grey x = excluded (GUI / off-frame) | box + ellipse = train tip prior (2 sigma)",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(cv2.resize(np.vstack([head, frame, foot]), (OW, OH), interpolation=cv2.INTER_AREA))
        if k % 100 == 0:
            print(f"{k}/{N}", flush=True)
    vw.release()

    ff = shutil.which("ffmpeg")
    if ff:                                                      # H.264 so any player / browser opens it
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", str(out_path)], check=True)
        tmp.unlink()
    else:
        tmp.replace(out_path)
    for key, e in errs.items():
        if e:
            print(f"tip error vs consensus, {key} frames (n={len(e)}): mean {np.mean(e):.1f} median {np.median(e):.1f} px")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="prefix of the mp4 name, e.g. 46867a8e")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    video = next(p.name for p in sorted(DATA.glob(f"{args.video}*.mp4")))
    d = ROOT.parent / "data" / "processed" / f"arch_tip_video_{args.video}"
    if args.extract:
        extract(video, d / "images", args.workers)
    else:
        render(video, d, ROOT / "outputs" / "arch_tip_depth" / f"{args.video}_fit.mp4")


if __name__ == "__main__":
    main()
