#!/usr/bin/env python3
"""Zero-shot depth models vs the stereo proxy-GT: which pretrained model gets the SHAPE right here?

Every model sees the same 83 rectified left frames and is scored on the same GT pixels
(depth16 in (EVAL_MIN, EVAL_MAX), as eval_scared.run_proxy_gt_eval), two ways:
  ms_*   per-frame median scaling in DEPTH -- identical to training's proxy_gt/ms_abs_rel
  ssi_*  per-frame least-squares scale + shift in INVERSE depth (the MiDaS / Depth Anything
         protocol): fair to relative-disparity models whose output has an unknown shift
The summary adds bootstrap 95% CIs vs the EndoDAC warm start, per-patient means and a grid.

Each model runs in its OWN subprocess (--model NAME) with only its code dirs on PYTHONPATH: the
model packages clash (numpy pins, EndoUFM's top-level `networks`). Code lives in ~/pylibs/<dir>
(pip --target --no-deps, never in the training venv); weights are pre-downloaded, so jobs run offline.

    python scripts/zeroshot_proxy_gt.py --all          # every model, then --summarize
    python scripts/zeroshot_proxy_gt.py --model da3_large
    python scripts/zeroshot_proxy_gt.py --summarize
"""
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

H = os.path.expanduser("~")
PL = f"{H}/pylibs"
ENDODAC = {"endodac_warmstart": "../backbones/EndoDAC/depth_model.pth",
           "ft_manual_ep5": "outputs/depth_manual_ep12/best.pth",
           "ft_sharpest1x_ep5": "outputs/depth_sharpest_1x_ep12/best.pth",
           "ft_ruler_sw05": "outputs/depth_ruler_range_sw05/best.pth"}
MODELS = {  # name -> (kind, source, extra PYTHONPATH dirs)
    **{n: ("endodac", p, []) for n, p in ENDODAC.items()},
    "depth_anything_v2_large": ("hf_disp", "depth-anything/Depth-Anything-V2-Large-hf", [f"{PL}/bench"]),
    "depth_anything_v2_metric_indoor_large": ("hf_depth", "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", [f"{PL}/bench"]),
    "depth_pro": ("hf_depth", "apple/DepthPro-hf", [f"{PL}/bench"]),
    "da3_large": ("da3", "depth-anything/DA3-LARGE", [f"{PL}/da3"]),
    "da3_giant": ("da3", "depth-anything/DA3-GIANT", [f"{PL}/da3"]),
    "da3_metric_large": ("da3", "depth-anything/DA3METRIC-LARGE", [f"{PL}/da3"]),
    "moge2_vitl": ("moge2", "Ruicheng/moge-2-vitl", [f"{PL}/moge"]),
    "moge1_vitl": ("moge1", "Ruicheng/moge-vitl", [f"{PL}/moge"]),
    "unidepth_v2_vitl": ("unidepth", "lpiccinelli/unidepth-v2-vitl14", [f"{PL}/unidepth"]),
    "metric3d_v2_vit_large": ("metric3d", "metric3d_vit_large", [f"{PL}/metric3d_mmcvstub", f"{PL}/metric3d"]),
    "metric3d_v2_vit_giant2": ("metric3d", "metric3d_vit_giant2", [f"{PL}/metric3d_mmcvstub", f"{PL}/metric3d"]),
    "endoufm": ("endoufm", "../backbones/EndoUFM/depth_model.pth", [f"{PL}/endoufm_deps"]),
}
BASE = "endodac_warmstart"


# ---------------------------------------------------------------------------------- metrics
def frame_metrics(pred, g):
    """(ms errors, ssi errors, (s, b)) on GT-valid px; pred = depth up to scale (disp models: 1/disp)."""
    from eval_scared import EVAL_MAX, EVAL_MIN, compute_errors
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


def load_frames(proxy_dir):
    from make_stereo_proxy_gt import DEPTH_SCALE
    gts = sorted(Path(proxy_dir).glob("*_depth16.png"))
    assert gts, f"no *_depth16.png in {proxy_dir}"
    pairs = [(p.with_name(p.name[:-len("_depth16.png")] + "_left.png"), p) for p in gts]
    return ([f for f, _ in pairs], [np.asarray(Image.open(p), np.float32) / DEPTH_SCALE for _, p in pairs])


# --------------------------------------------------------------------------------- adapters
def make_predictor(kind, src, device):
    """-> f(PIL image, (h, w)) -> HxW numpy DEPTH up to scale (disparity outputs are inverted)."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    def up(t, hw):                                          # (h', w') tensor -> hw numpy
        return F.interpolate(t.float()[None, None], size=hw, mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()

    if kind == "endodac":
        from finetune_depth import _filter_load, disp_to_depth, make_endodac
        m = make_endodac((392, 490)).to(device)
        _filter_load(m, src, Path(src).parent.name)
        m.eval()

        def f(img, hw):                                     # training feed + depth range
            feed = to_tensor(img.resize((490, 392), Image.BILINEAR))[None].to(device)
            _, depth = disp_to_depth(m(feed)[("disp", 0)], 20.0, 200.0)
            return up(depth[0, 0], hw)
        f.model = m
        return f

    if kind in ("hf_disp", "hf_depth"):
        from transformers import (AutoImageProcessor, DepthAnythingForDepthEstimation,
                                  DepthProForDepthEstimation)
        cls = DepthProForDepthEstimation if "DepthPro" in src else DepthAnythingForDepthEstimation
        proc, m = AutoImageProcessor.from_pretrained(src), cls.from_pretrained(src).to(device).eval()

        def f(img, hw):
            inp = proc(images=img, return_tensors="pt").to(device)
            a = proc.post_process_depth_estimation(m(**inp), target_sizes=[hw])[0]["predicted_depth"]
            a = a.float().cpu().numpy()
            return 1.0 / np.clip(a, 1e-6, None) if kind == "hf_disp" else a
        return f

    if kind == "da3":
        # DA3's sky clamp (model/da3.py _process_mono_sky_estimation) is a no-op for these
        # checkpoints: none has a sky head. Verified: patching it out changed nothing (4 decimals).
        from depth_anything_3.api import DepthAnything3
        m = DepthAnything3.from_pretrained(src).to(device).eval()

        def f(img, hw):
            pred = m.inference([np.asarray(img)], process_res=504)
            return up(torch.from_numpy(np.asarray(pred.depth[0], np.float32)), hw)
        return f

    if kind in ("moge2", "moge1"):
        if kind == "moge2":
            from moge.model.v2 import MoGeModel
        else:
            from moge.model.v1 import MoGeModel
        m = MoGeModel.from_pretrained(src).to(device).eval()

        def f(img, hw):
            d = m.infer(to_tensor(img).to(device))["depth"]
            return up(torch.nan_to_num(d, nan=float("inf")), hw)
        return f

    if kind == "unidepth":
        from unidepth.models import UniDepthV2
        m = UniDepthV2.from_pretrained(src).to(device).eval()

        def f(img, hw, k=None):                     # k = normalised (fx, fy, cx, cy); None = UniDepth guesses
            rgb = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)   # uint8, infer() normalises
            K = None
            if k is not None:
                w, h = img.size
                K = torch.tensor([[k[0] * w, 0, k[2] * w], [0, k[1] * h, k[3] * h], [0, 0, 1]],
                                 dtype=torch.float32, device=device)
            o = m.infer(rgb, K)
            if "intrinsics" in o:                   # what focal it used, normalised like k
                Ki = o["intrinsics"][0].float().cpu().numpy()
                f.last_k = (Ki[0, 0] / img.size[0], Ki[1, 1] / img.size[1])
            return up(o["depth"][0, 0], hw)
        return f

    if kind == "metric3d":
        import cv2
        repo = f"{H}/.cache/torch/hub/yvanyin_metric3d_main"
        m = torch.hub.load(repo, src, source="local", pretrain=True).to(device).eval()
        mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]

        def f(img, hw):                                     # hubconf recipe; focal scale irrelevant here
            rgb = np.asarray(img)
            ih, iw = 616, 1064
            sc = min(ih / rgb.shape[0], iw / rgb.shape[1])
            r = cv2.resize(rgb, (int(rgb.shape[1] * sc), int(rgb.shape[0] * sc)), interpolation=cv2.INTER_LINEAR)
            ph, pw = ih - r.shape[0], iw - r.shape[1]
            t, l = ph // 2, pw // 2
            r = cv2.copyMakeBorder(r, t, ph - t, l, pw - l, cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
            x = ((torch.from_numpy(r.transpose(2, 0, 1)).float() - mean) / std)[None].to(device)
            d, _, _ = m.inference({"input": x})
            d = d[0, 0, t:ih - (ph - t), l:iw - (pw - l)]
            return up(d, hw)
        return f

    if kind == "endoufm":
        # EndoUFM uses top-level `from layers import *` / `utils` / `networks`, which clash with
        # third_party/endodac (already imported via eval_scared). Import it against its own repo with
        # ours evicted, then put ours back -- its classes keep the functions they already bound.
        repo = f"{H}/pylibs/EndoUFM"
        clash = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k.split(".")[0] in ("utils", "layers", "networks", "options", "datasets")}
        sys.path.insert(0, repo)
        try:
            import networks.endoufm as eu
        finally:
            sys.path.remove(repo)
            sys.modules.update(clash)
        # rvlora's random_1/random_2 are built but unused in RVLinear.forward (only lora_A/B/U/V,
        # all in the checkpoint), so the load is deterministic without the training seed.
        m = eu.endoufm(backbone_size="base", r=4, lora_type="rvlora", image_shape=(224, 280),
                       pretrained_path=None, residual_block_indexes=[2, 5, 8, 11], include_cls_token=True)
        sd = torch.load(src, map_location="cpu")
        md = m.state_dict()
        keep = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
        print(f"[endoufm] loaded {len(keep)}/{len(md)} tensors ({len(sd)} in checkpoint)", flush=True)
        m.load_state_dict(keep, strict=False)
        m.to(device).eval()

        def f(img, hw):                                         # evaluate_depth_new.py: 256x320 in [0,1], 0.1..150
            feed = to_tensor(img.resize((320, 256), Image.BILINEAR))[None].to(device)
            disp = m(feed)[("disp", 0)]
            min_disp, max_disp = 1 / 150.0, 1 / 0.1
            return up(1.0 / (min_disp + (max_disp - min_disp) * disp[0, 0]), hw)
        return f

    raise ValueError(kind)


# ------------------------------------------------------------------------------ one model
def run_model(name, proxy_dir, out, grid_frames):
    import torch
    from eval_scared import EVAL_MAX, EVAL_MIN
    from finetune_depth import colorize
    kind, src, _ = MODELS[name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, gt = load_frames(proxy_dir)
    grid_idx = set(np.linspace(0, len(frames) - 1, grid_frames).astype(int).tolist())
    ms, ssi, vis = [], [], []
    with torch.no_grad():
        f = make_predictor(kind, src, device)
        for i, (fp, g) in enumerate(zip(frames, gt)):
            pred = f(Image.open(fp).convert("RGB"), g.shape)
            e_ms, e_ssi, (s, b) = frame_metrics(pred, g)
            ms.append(e_ms); ssi.append(e_ssi)
            if i in grid_idx:                                   # ssi-aligned inverse depth on GT-valid px
                mask = (g > EVAL_MIN) & (g < EVAL_MAX)
                inv = s / np.clip(np.nan_to_num(pred, nan=EVAL_MAX, posinf=EVAL_MAX), EVAL_MIN, None) + b
                vis.append(colorize(np.clip(inv, 1.0 / EVAL_MAX, None), mask)[::4, ::4])
    if name == BASE:                                            # self-check vs the training eval
        from eval_scared import run_proxy_gt_eval
        ref = run_proxy_gt_eval(f.model, proxy_dir, (392, 490), device, 20.0, 200.0, num_vis=0)[0]
        mine = float(np.mean([e[0] for e in ms]))
        assert abs(mine - ref["ms_abs_rel"]) < 1e-4, (mine, ref["ms_abs_rel"])
        print(f"[check] ms_abs_rel {mine:.4f} == run_proxy_gt_eval {ref['ms_abs_rel']:.4f}", flush=True)
    ms, ssi = np.array(ms), np.array(ssi)
    np.savez(out / "per_model" / f"{name}.npz", ms=ms, ssi=ssi, vis=np.stack(vis),
             frames=np.array([p.name for p in frames]))
    print(f"[{name}] ms_abs_rel={ms[:, 0].mean():.4f} ms_a1={ms[:, 4].mean():.4f} | "
          f"ssi_abs_rel={ssi[:, 0].mean():.4f} ssi_a1={ssi[:, 4].mean():.4f}", flush=True)


# ------------------------------------------------------------------------------- summary
def summarize(proxy_dir, out, grid_frames):
    from eval_scared import EVAL_MAX, EVAL_MIN, METRIC_NAMES
    from finetune_depth import colorize
    R = {n: np.load(out / "per_model" / f"{n}.npz") for n in MODELS if (out / "per_model" / f"{n}.npz").exists()}
    assert BASE in R, "the EndoDAC warm start is the reference; run it first"
    frames = [str(x) for x in R[BASE]["frames"]]
    pat = np.array([x[:8] for x in frames])
    res = {}
    print(f"\n{'model':40s} {'ms_abs_rel':>10s} {'ssi_abs_rel':>11s} {'ssi_a1':>6s}  "
          f"{'d_ssi vs warm [95% CI]':>28s}  per-patient ssi  frames better")
    for n, r in sorted(R.items(), key=lambda kv: kv[1]["ssi"][:, 0].mean()):
        d = r["ssi"][:, 0] - R[BASE]["ssi"][:, 0]
        dm = r["ms"][:, 0] - R[BASE]["ms"][:, 0]
        ci, cim = (boot_ci(d), boot_ci(dm)) if n != BASE else ((0.0, 0.0, 0.0),) * 2
        pp = {p: float(r["ssi"][pat == p, 0].mean()) for p in np.unique(pat)}
        res[n] = {**{f"ms_{k}": float(v) for k, v in zip(METRIC_NAMES, r["ms"].mean(0))},
                  **{f"ssi_{k}": float(v) for k, v in zip(METRIC_NAMES, r["ssi"].mean(0))},
                  "d_ssi_abs_rel_vs_warm": ci, "d_ms_abs_rel_vs_warm": cim, "ssi_abs_rel_per_patient": pp,
                  "frames_better_ssi": int((d < 0).sum())}
        print(f"{n:40s} {res[n]['ms_abs_rel']:10.4f} {res[n]['ssi_abs_rel']:11.4f} {res[n]['ssi_a1']:6.3f}  "
              f"{ci[0]:+.4f} [{ci[1]:+.4f},{ci[2]:+.4f}]  "
              f"{' / '.join(f'{v:.3f}' for v in pp.values())}  {res[n]['frames_better_ssi']}/{len(d)}")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    order = list(res)
    with open(out / "per_frame.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame"] + [f"{n}_{k}" for n in order for k in ("ms_abs_rel", "ssi_abs_rel")])
        for i, fr in enumerate(frames):
            w.writerow([fr] + [round(float(R[n][k][i, 0]), 5) for n in order for k in ("ms", "ssi")])
    fr_paths, gt = load_frames(proxy_dir)
    idx = np.linspace(0, len(fr_paths) - 1, grid_frames).astype(int)
    rows = []
    for k, i in enumerate(idx):                                 # [rgb | GT | models sorted by ssi]
        g = gt[i]
        mask = (g > EVAL_MIN) & (g < EVAL_MAX)
        cells = [np.asarray(Image.open(fr_paths[i]).convert("RGB"))[::4, ::4],
                 colorize(1.0 / np.clip(g, EVAL_MIN, None), mask)[::4, ::4]]
        cells += [R[n]["vis"][k] for n in order]
        hmin = min(c.shape[0] for c in cells); wmin = min(c.shape[1] for c in cells)
        rows.append(np.concatenate([c[:hmin, :wmin] for c in cells], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out / "grid.png")
    print(f"\ngrid columns: rgb | stereo GT | {' | '.join(order)}\nwrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-gt-dir", default="../data/processed/proxy_gt_nogui_ffs")
    ap.add_argument("--out", default="outputs/zeroshot_proxy_gt")
    ap.add_argument("--grid-frames", type=int, default=4)
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--all", action="store_true", help="every model in its own subprocess, then summarize")
    ap.add_argument("--only", help="comma-separated model names to run with --all (summary still uses "
                                   "every per_model/*.npz already in --out)")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "per_model").mkdir(parents=True, exist_ok=True)
    if a.model:
        return run_model(a.model, a.proxy_gt_dir, out, a.grid_frames)
    if a.all:
        failed = []
        names = a.only.split(",") if a.only else list(MODELS)
        assert set(names) <= set(MODELS), f"unknown models: {set(names) - set(MODELS)}"
        for n in names:
            extra = MODELS[n][2]
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(extra + [os.environ.get("PYTHONPATH", "")])}
            r = subprocess.run([sys.executable, __file__, "--model", n, "--proxy-gt-dir", a.proxy_gt_dir,
                                "--out", a.out, "--grid-frames", str(a.grid_frames)], env=env)
            if r.returncode:
                failed.append(n)
                print(f"[FAILED] {n} (exit {r.returncode})", flush=True)
        print(f"[all] failed: {failed or 'none'}", flush=True)
    if a.all or a.summarize:
        summarize(a.proxy_gt_dir, out, a.grid_frames)


if __name__ == "__main__":
    main()
