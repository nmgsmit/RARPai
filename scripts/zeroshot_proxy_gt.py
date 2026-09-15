#!/usr/bin/env python3
"""Zero-shot depth models vs the stereo proxy-GT: can ANY model beat EndoDAC's shape here?

Every model sees the same 83 rectified left frames and is scored on the same GT pixels
(depth16 in (EVAL_MIN, EVAL_MAX), as eval_scared.run_proxy_gt_eval), two ways:
  ms_*   per-frame median scaling in DEPTH -- identical to training's proxy_gt/ms_abs_rel
  ssi_*  per-frame least-squares scale + shift in INVERSE depth (the MiDaS / Depth Anything
         protocol): fair to relative-disparity models whose output has an unknown shift
Per-frame ms_abs_rel differences vs the EndoDAC warm start get a bootstrap 95% CI, so "better"
means better than the frame-to-frame noise of this 83-frame / 2-patient set.

EndoDAC checkpoints use the training feed (392x490, disp_to_depth 20..200). HF models use their
own processors. Output: outputs/zeroshot_proxy_gt/{results.json, per_frame.csv, grid.png}.

    PYTHONPATH=~/pylibs/bench HF_HUB_OFFLINE=1 python scripts/zeroshot_proxy_gt.py
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from eval_scared import EVAL_MAX, EVAL_MIN, METRIC_NAMES, compute_errors, run_proxy_gt_eval
from finetune_depth import _filter_load, colorize, disp_to_depth, make_endodac
from make_stereo_proxy_gt import DEPTH_SCALE

ENDODAC_SHAPE, MIN_D, MAX_D = (392, 490), 20.0, 200.0      # the training runs' feed + depth range
CKPTS = {                                                   # name -> state_dict (skipped if absent)
    "endodac_warmstart": "../backbones/EndoDAC/depth_model.pth",
    "ft_manual_ep5": "outputs/depth_manual_ep12/best.pth",
    "ft_sharpest1x_ep5": "outputs/depth_sharpest_1x_ep12/best.pth",
    "ft_ruler_sw05": "outputs/depth_ruler_range_sw05/best.pth",
}
HF = {                                                      # name -> (repo, output kind)
    "depth_anything_v2_large": ("depth-anything/Depth-Anything-V2-Large-hf", "disp"),
    "depth_anything_v2_metric_indoor_large": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "depth"),
    "depth_pro": ("apple/DepthPro-hf", "depth"),
}


def load_endodac(path, device):
    m = make_endodac(ENDODAC_SHAPE).to(device)
    _filter_load(m, path, Path(path).parent.name)
    return m.eval()


def endodac_predict(m, img, hw, device):
    feed = transforms.ToTensor()(img.resize(ENDODAC_SHAPE[::-1], Image.BILINEAR)).unsqueeze(0).to(device)
    _, depth = disp_to_depth(m(feed)[("disp", 0)], MIN_D, MAX_D)
    return F.interpolate(depth, size=hw, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def load_hf(repo, device):
    from transformers import (AutoImageProcessor, DepthAnythingForDepthEstimation,
                              DepthProForDepthEstimation)
    cls = DepthProForDepthEstimation if "DepthPro" in repo else DepthAnythingForDepthEstimation
    return AutoImageProcessor.from_pretrained(repo), cls.from_pretrained(repo).to(device).eval()


def hf_predict(proc_model, kind, img, hw, device):
    proc, m = proc_model
    inp = proc(images=img, return_tensors="pt").to(device)
    out = proc.post_process_depth_estimation(m(**inp), target_sizes=[hw])[0]["predicted_depth"]
    a = out.float().cpu().numpy()
    return 1.0 / np.clip(a, 1e-6, None) if kind == "disp" else a   # always hand back DEPTH-like


def frame_metrics(pred, g):
    """(ms errors, ssi errors) on the GT-valid pixels; pred is depth up to scale (disp models: 1/disp)."""
    mask = (g > EVAL_MIN) & (g < EVAL_MAX) & np.isfinite(g)
    p = np.clip(np.nan_to_num(pred[mask], nan=EVAL_MAX, posinf=EVAL_MAX), EVAL_MIN, None)
    t = g[mask]
    ms = compute_errors(t, np.clip(p * (np.median(t) / np.median(p)), EVAL_MIN, EVAL_MAX))
    x, y = 1.0 / p, 1.0 / t
    s, b = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)[0]
    ssi = compute_errors(t, 1.0 / np.clip(s * x + b, 1.0 / EVAL_MAX, 1.0 / EVAL_MIN))
    return ms, ssi, (s, b)


def boot_ci(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-gt-dir", default="../data/processed/proxy_gt_nogui_ffs")
    ap.add_argument("--out", default="outputs/zeroshot_proxy_gt")
    ap.add_argument("--grid-frames", type=int, default=4)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    gts = sorted(Path(a.proxy_gt_dir).glob("*_depth16.png"))
    assert gts, f"no *_depth16.png in {a.proxy_gt_dir}"
    pairs = [(p.with_name(p.name[:-len("_depth16.png")] + "_left.png"), p) for p in gts]
    gt = [np.asarray(Image.open(p), np.float32) / DEPTH_SCALE for _, p in pairs]
    imgs = [Image.open(f).convert("RGB") for f, _ in pairs]
    grid_idx = np.linspace(0, len(pairs) - 1, a.grid_frames).astype(int)
    print(f"[data] {len(pairs)} frames, {gt[0].shape}", flush=True)

    models = [(n, "endodac", p) for n, p in CKPTS.items() if Path(p).exists()]
    models += [(n, "hf", r) for n, r in HF.items()]
    res, per_frame, vis = {}, {}, {}
    for name, typ, src in models:
        with torch.no_grad():
            if typ == "endodac":
                m = load_endodac(src, device)
                pred_fn = lambda img, hw: endodac_predict(m, img, hw, device)
            else:
                pm = load_hf(src[0], device)
                pred_fn = lambda img, hw: hf_predict(pm, src[1], img, hw, device)
            ms, ssi, v = [], [], []
            for i, (img, g) in enumerate(zip(imgs, gt)):
                pred = pred_fn(img, g.shape)
                e_ms, e_ssi, (s, b) = frame_metrics(pred, g)
                ms.append(e_ms); ssi.append(e_ssi)
                if i in grid_idx:                           # ssi-aligned inverse depth, GT-valid px
                    mask = (g > EVAL_MIN) & (g < EVAL_MAX)
                    inv = s / np.clip(pred, EVAL_MIN, None) + b
                    v.append(colorize(np.clip(inv, 1.0 / EVAL_MAX, None), mask))
        if typ == "endodac" and name == "endodac_warmstart":
            # self-check: this script's ms metric == the training eval's, on the same model
            ref = run_proxy_gt_eval(m, a.proxy_gt_dir, ENDODAC_SHAPE, device, MIN_D, MAX_D, num_vis=0)[0]
            mine = float(np.mean([e[0] for e in ms]))
            assert abs(mine - ref["ms_abs_rel"]) < 1e-4, (mine, ref["ms_abs_rel"])
            print(f"[check] ms_abs_rel {mine:.4f} == run_proxy_gt_eval {ref['ms_abs_rel']:.4f}", flush=True)
        m = pm = pred_fn = None                             # free this model's GPU memory
        torch.cuda.empty_cache()
        ms, ssi = np.array(ms), np.array(ssi)
        res[name] = {**{f"ms_{k}": float(x) for k, x in zip(METRIC_NAMES, ms.mean(0))},
                     **{f"ssi_{k}": float(x) for k, x in zip(METRIC_NAMES, ssi.mean(0))}}
        per_frame[name] = (ms[:, 0], ssi[:, 0])
        vis[name] = v
        print(f"[{name}] ms_abs_rel={res[name]['ms_abs_rel']:.4f} ms_a1={res[name]['ms_a1']:.4f} | "
              f"ssi_abs_rel={res[name]['ssi_abs_rel']:.4f} ssi_a1={res[name]['ssi_a1']:.4f}", flush=True)

    base = "endodac_warmstart"
    print(f"\n{'model':40s} {'ms_abs_rel':>10s} {'ssi_abs_rel':>11s} {'ms_a1':>6s}   "
          f"d(ms_abs_rel) vs warm-start [95% CI over frames]")
    for name in res:
        d = per_frame[name][0] - per_frame[base][0]
        mean, lo, hi = boot_ci(d) if name != base else (0.0, 0.0, 0.0)
        res[name]["d_ms_abs_rel_vs_warm"] = [mean, lo, hi]
        print(f"{name:40s} {res[name]['ms_abs_rel']:10.4f} {res[name]['ssi_abs_rel']:11.4f} "
              f"{res[name]['ms_a1']:6.3f}   {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    (out / "results.json").write_text(json.dumps(res, indent=2))
    with open(out / "per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + [f"{n}_{k}" for n in res for k in ("ms_abs_rel", "ssi_abs_rel")])
        for i, (fp, _) in enumerate(pairs):
            w.writerow([fp.name] + [round(float(per_frame[n][j][i]), 5) for n in res for j in (0, 1)])
    rows = []
    for k, i in enumerate(grid_idx):                        # [rgb | GT | model...] per frame, 1/4 res
        g = gt[i]
        mask = (g > EVAL_MIN) & (g < EVAL_MAX)
        cells = [np.asarray(imgs[i].resize(g.shape[::-1])), colorize(1.0 / np.clip(g, EVAL_MIN, None), mask)]
        cells += [vis[n][k] for n in res]
        rows.append(np.concatenate(cells, 1)[::4, ::4])
    Image.fromarray(np.concatenate(rows, 0)).save(out / "grid.png")
    print(f"\ngrid columns: rgb | stereo GT | {' | '.join(res)}\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
