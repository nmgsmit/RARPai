"""STANDARD SUL (Nick, 2026-09-23): start = cylinder depth step, end = arch model tip, UniDepth mm.

Per frame of a labelling-tool workspace folder (images/*.png = 1920x1080 console frames, masks/*.png =
hand masks, palette ids):
  crop   1340x1072 content frame (source_crop x289 y4); GUI blacked and kept as a mask (full_gui_mask)
  depth  UniDepth V2 ViT-L, no K, frozen ruler calibration (metric_calib_proxy, scale mode: z_mm = z_m / s_only),
         as arch_tip_unidepth -> <root>/depth/<stem>.npz ('depth' float32 mm, 'gui' bool), reused when present
  tube   urethra_cylinder.analyse on the hand mask (axis="mask"), K = da Vinci K_NORM on the crop
  START  analyse's t_start: the depth step near the mask's proximal end (knee rule, outward only)
  END    arch model tip (outputs/arch_tip_pure40, surgical DINOv3 + arc7, 3-seed mean), matched IN THE IMAGE to
         the nearest point of the tube's top line (as hand_measure / the catheter start)
  SUL    t_end - t_start along the axis
Out: <root>/sul.csv, <root>/visualization/<stem>.jpg; with --truth the error vs the real length.

    sbatch jobs/sul_arch_cyl.sh                       # ../data/GoodRulerTest/SUL
    python scripts/sul_arch_cyl.py --self-test        # synthetic tube, no models
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cut_cue_clips import content_box, full_gui_mask  # noqa: E402
from prep_sharpest_clips import CUE, _crop, load_gui_templates  # noqa: E402
from urethra_cylinder import analyse, draw, load_mask, profile, proj, toward_camera  # noqa: E402

W, H = 1340, 1072
K = (0.82 * W, 1.02 * H, 0.5 * W, 0.5 * H)       # da Vinci K_NORM on the crop, as the ruler calibration
PT = 250                                          # grey margin on top: arch tips can sit above the frame


def frame_inputs(img_path, mask_path, T):
    """-> GUI-blacked crop (BGR), GUI mask, hand-mask ids (GUI = background), all 1340x1072."""
    full = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    m = full_gui_mask(full, *T, content_box(cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)),
                      CUE["m_thr"], CUE["b_thr"], CUE["min_bars"], CUE["dilate"])
    full[m] = 0
    seg = _crop(load_mask(str(mask_path), full.shape[:2]))
    gui = _crop(m)
    seg[gui] = 0
    return _crop(full), gui, seg


def end_on_tube(fr, uv, K=K, span=(-15.0, 80.0)):
    """Image point uv -> (t on the axis whose top-line projection is nearest, that distance in px)."""
    tt = np.arange(fr["t_start"] + span[0], fr["t_start"] + span[1], 0.1)
    A = fr["p"] + tt[:, None] * fr["d"]
    S = A + fr["r"] * toward_camera(A, fr["d"])
    front = S[:, 2] > 1.0
    tt, dd = tt[front], np.linalg.norm(proj(S[front], K) - uv, axis=1)
    i = int(np.argmin(dd))
    return float(tt[i]), float(dd[i])


def measure(zmap, seg, tip):
    """-> (analyse output with t_end/found replaced by the arch end, SUL mm, tip-to-tube px), or (None, nan, nan)."""
    fr = analyse(zmap, seg, K)
    if fr is None:
        return None, float("nan"), float("nan")
    t_end, off = end_on_tube(fr, tip)
    fr = dict(fr, t_end_roof=fr["t_end"] if fr["found"] else float("nan"), t_end=t_end, found=True,
              sul=t_end - fr["t_start"], sul_mask=t_end - fr["t_mask_start"])
    return fr, t_end - fr["t_start"], off


def panel(img, zmap, seg, fr, tip, title):
    """[hand mask + tube (green start, red arch end) + arch tip star | depth] over the gap profile, top padded."""
    Kp = (K[0], K[1], K[2], K[3] + PT)
    pad = lambda a, v: cv2.copyMakeBorder(a, PT, 0, 0, 0, cv2.BORDER_CONSTANT, value=v)
    ok = zmap > 0
    lo, hi = np.percentile(zmap[ok], [2, 98])
    heat = cv2.applyColorMap((np.clip((hi - zmap) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~ok] = 90
    q = tuple(int(round(v)) for v in (tip[0], tip[1] + PT))
    cols = []
    for base in (img, heat):
        p = draw(pad(base, (45, 45, 45)), pad(seg, 0), fr, Kp) if fr is not None else pad(base, (45, 45, 45))
        cv2.drawMarker(p, q, (255, 0, 255), cv2.MARKER_STAR, 50, 4, cv2.LINE_AA)
        cols.append(p)
    cv2.putText(cols[1], "UniDepth %.0f-%.0f mm" % (lo, hi), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(cols[0], title, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    top = cv2.resize(np.hstack(cols), (W, (H + PT) // 2), interpolation=cv2.INTER_AREA)
    return np.vstack([top, profile(fr, W, 260, 1.5)])


def read_truth(path):
    """realsizeurethra.txt: header, then '<8-char case> <mm>' per line."""
    out = {}
    for line in Path(path).read_text().splitlines()[1:]:
        f = line.split()
        if len(f) >= 2:
            out[f[0][:8]] = float(f[1])
    return out


def self_test():
    """A tip on the fitted tube's top line (visible, then under the roof), and 30 px beside it, comes back at its t."""
    from urethra_cylinder import render, unit
    Ks = (285.8, 285.8, 150.9, 133.5)                   # quarter-size K, as urethra_cylinder.self_test
    z, seg = render(Ks, 335, 268, np.array([2.0, 12, 55]), unit(np.array([0.15, -1.0, 0.3])), 4.0, 18.0, base=True)
    fr = analyse(z, seg, Ks, erode=2, min_px=200, roi_px=10)
    for t_true, side in ((8.0, 0), (22.0, 0), (12.0, 30)):
        A = fr["p"] + (fr["t_start"] + t_true) * fr["d"]
        uv = proj(A + fr["r"] * toward_camera(A[None], fr["d"])[0], Ks)
        nxt = proj(A + fr["d"] + fr["r"] * toward_camera(A[None], fr["d"])[0], Ks)
        n = np.array([-(nxt - uv)[1], (nxt - uv)[0]]) / np.linalg.norm(nxt - uv)   # image normal to the line
        t, off = end_on_tube(fr, uv + side * n, Ks)
        print("self-test tip at %+.1f mm, %d px beside the line -> %+.2f mm, %.1f px off"
              % (t_true, side, t - fr["t_start"], off))
        assert abs(t - fr["t_start"] - t_true) < 0.3 and abs(off - side) < 1.0, "end_on_tube"
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/GoodRulerTest/SUL")
    ap.add_argument("--truth", default="../data/GoodRulerTest/realsizeurethra.txt")
    ap.add_argument("--calib", default="outputs/metric_calib_proxy/results.json")
    ap.add_argument("--arch", default="outputs/arch_tip_pure40/model/arch_tip_arc7_dinov3_surg.pt")
    ap.add_argument("--model", default="lpiccinelli/unidepth-v2-vitl14")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    import torch
    from arch_tip_pure import backbone, load_arch_model, predict_tips, prep
    from unidepth.models import UniDepthV2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(a.root)
    (root / "depth").mkdir(exist_ok=True)
    (root / "visualization").mkdir(exist_ok=True)
    s_only = json.loads(Path(a.calib).read_text())["unidepth_v2_vitl"]["calibration"]["s_only"]
    truth = read_truth(a.truth) if a.truth and Path(a.truth).exists() else {}
    T = load_gui_templates()
    arch = load_arch_model(a.arch, dev)
    feat, _ = backbone(arch["backbone"], dev)
    ud = UniDepthV2.from_pretrained(a.model).to(dev).eval()
    print(f"arch model {a.arch} ({arch['train_set']}, {arch['method']}), UniDepth s_only {s_only:.4g}, {dev}")

    rows = []
    for ip in sorted((root / "images").glob("*.png")):
        mp = root / "masks" / ip.name
        if not mp.exists():
            print(f"{ip.name}: no mask, skipped")
            continue
        img, gui, seg = frame_inputs(ip, mp, T)
        dp = root / "depth" / f"{ip.stem}.npz"
        with torch.no_grad():
            if not dp.exists():
                rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).to(dev)
                z = ud.infer(rgb)["depth"][0, 0].float().cpu().numpy() / s_only
                np.savez(dp, depth=z.astype(np.float32), gui=gui, s_only=s_only)
            zmap = np.load(dp)["depth"].copy()
            zmap[gui] = 0
            tip = predict_tips(arch, feat(torch.from_numpy(prep(img))[None].to(dev)).float(), dev)[0][0]
        fr, sul, off = measure(zmap, seg, tip)
        case = ip.name[:8]
        tr = truth.get(case, float("nan"))
        row = dict(case=case, image=ip.name, true_mm=tr, sul_mm=round(sul, 2), err_mm=round(sul - tr, 2),
                   start=None if fr is None else fr["start_kind"], start_ok=fr is not None and fr["start_ok"],
                   tip_u=round(float(tip[0]), 1), tip_v=round(float(tip[1]), 1), tip_to_tube_px=round(off, 1),
                   radius_mm=None if fr is None else round(fr["r"], 2),
                   sul_mask_start_mm=None if fr is None else round(fr["t_end"] - fr["t_mask_start"], 2),
                   sul_roof_end_mm=None if fr is None else round(fr["t_end_roof"] - fr["t_start"], 2),
                   depth_median_mm=round(float(np.median(zmap[zmap > 0])), 1))
        rows.append(row)
        print(row, flush=True)
        cv2.imwrite(str(root / "visualization" / f"{ip.stem}.jpg"),
                    panel(img, zmap, seg, fr, tip, f"{case}  SUL {sul:.1f} mm  (real {tr:g})"),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    with open(root / "sul.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    e = np.array([r["err_mm"] for r in rows], float)
    e = e[np.isfinite(e)]
    print(f"\n{'case':<10}{'real':>6}{'SUL':>7}{'err':>7}  start")
    for r in rows:
        print(f"{r['case']:<10}{r['true_mm']:6g}{r['sul_mm']:7.1f}{r['err_mm']:+7.1f}  {r['start']} ok={r['start_ok']}")
    if e.size:
        print(f"MAE {np.abs(e).mean():.1f} mm, mean signed {e.mean():+.1f} mm, n={e.size}")
    print(f"wrote {root}/sul.csv, depth/, visualization/")


if __name__ == "__main__":
    main()
