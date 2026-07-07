"""
SCARED depth evaluation for a trained EndoDAC depth_model (outputs/<run>/best.pth).

SCARED is the metric-GT endoscopic benchmark used by AF-SfMLearner / EndoDAC. Self-supervised
depth is scale-ambiguous, so we apply per-frame MEDIAN SCALING (standard) before the 7 Monodepth2
metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3).

GT: the challenge *test* release (ds8/9) has no structured-light GT, so we use calibrated-STEREO
depth built by scripts/export_scared_stereo_gt.py -> <dir>/{frames/, gt_depths.npz}. Metric (mm),
the model never sees the right image (fair), noisier than structured light, N=10 keyframes.

Standalone:
    python scripts/eval_scared.py --ckpt outputs/depth_s1/best.pth \
        --rgb-dir ../data/SCARED/stereo_gt/frames --gt-npz ../data/SCARED/stereo_gt/gt_depths.npz \
        --image-shape 392 490 --run-name depth_s1-scared

Reused by finetune_depth.py via run_scared_eval(model, scared_dir, ...) so every training run
also reports SCARED metrics to its wandb run.

ponytail: reuses finetune_depth's model builders + vendored compute_errors/disp_to_depth.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))          # import sibling script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "endodac"))
os.environ.setdefault("XFORMERS_DISABLED", "1")

from finetune_depth import (build_depth_model, _filter_load, round14,   # noqa: E402
                            colorize, disp_to_depth)
from utils.utils import compute_errors                                   # noqa: E402

# AF-SfM / EndoDAC SCARED eval convention: mm, depths outside this band are ignored.
EVAL_MIN, EVAL_MAX = 1e-3, 150.0
METRIC_NAMES = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")


def load_gt(path):
    """gt_depths.npz -> (N,H,W) array of depth maps in mm. Accepts npz(key 'data'/first) or npy."""
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        obj = obj["data"] if "data" in obj.files else obj[obj.files[0]]
    return obj


def _crop(img, g, side, bottom):
    """Crop side-frac off each of L/R and bottom-frac off the bottom, on BOTH the PIL image and
    the (H,W) GT array identically (they're pixel-aligned) so a crop-trained model is eval'd fairly."""
    if side <= 0 and bottom <= 0:
        return img, g
    W, H = img.size
    img = img.crop((int(W * side), 0, int(W * (1 - side)), int(H * (1 - bottom))))
    gh, gw = g.shape[:2]
    g = g[0:int(gh * (1 - bottom)), int(gw * side):int(gw * (1 - side))]
    return img, g


def _eval_pairs(model, frames, gt, image_shape, device, min_depth, max_depth,
                num_vis=8, side_crop=0.0, bottom_crop=0.0):
    """Core loop: per frame -> pred depth, median-scale to GT, compute_errors. Returns
    (mean_errors[7], ratios[N], vis[list of [rgb|pred|gt] uint8 rows])."""
    mh, mw = round14(image_shape[0]), round14(image_shape[1])
    to_tensor = transforms.ToTensor()
    errors, ratios, vis = [], [], []
    model.eval()
    for fp, g in zip(frames, gt):
        g = np.asarray(g, np.float32)
        img = Image.open(fp).convert("RGB")
        img, g = _crop(img, g, side_crop, bottom_crop)
        feed = to_tensor(img.resize((mw, mh), Image.BILINEAR)).unsqueeze(0).to(device)
        with torch.no_grad():
            disp = model(feed)[("disp", 0)]
            _, depth = disp_to_depth(disp, min_depth, max_depth)
            depth = F.interpolate(depth, size=g.shape[:2], mode="bilinear", align_corners=False)
        pred_full = depth[0, 0].cpu().numpy()

        mask = (g > EVAL_MIN) & (g < EVAL_MAX) & np.isfinite(g)
        if mask.sum() == 0:
            continue
        p, gt_v = pred_full[mask], g[mask]
        ratio = np.median(gt_v) / np.median(p)                # per-frame median scaling
        ratios.append(ratio)
        p = np.clip(p * ratio, EVAL_MIN, EVAL_MAX)
        errors.append(compute_errors(gt_v, p))

        if len(vis) < num_vis:                                # [rgb | pred | gt] disparity color
            rgb_np = np.asarray(img.resize(g.shape[:2][::-1])).astype(np.uint8)
            vis.append(np.concatenate([rgb_np,
                                       colorize(1.0 / np.clip(pred_full * ratio, EVAL_MIN, None), mask),
                                       colorize(1.0 / np.clip(g, EVAL_MIN, None), mask)], axis=1))
    if not errors:
        raise SystemExit("no valid GT pixels in any frame -- check --gt-npz units/pairing")
    return np.array(errors).mean(0), np.array(ratios), vis


def run_scared_eval(model, scared_dir, image_shape, device, min_depth=0.1, max_depth=150.0,
                    num_vis=8, side_crop=0.0, bottom_crop=0.0):
    """Convenience wrapper for the fixed exporter layout <dir>/{frames/, gt_depths.npz}.
    Returns (metrics dict incl scale-ratio, ratios, vis) or None if GT not present."""
    scared_dir = Path(scared_dir)
    npz, fdir = scared_dir / "gt_depths.npz", scared_dir / "frames"
    if not npz.exists() or not fdir.exists():
        return None
    gt = load_gt(npz)
    frames = sorted(p for e in IMG_EXTS for p in fdir.rglob(e))
    if len(frames) != len(gt):
        print(f"[scared] pairing mismatch {len(frames)} frames vs {len(gt)} GT -- skipping")
        return None
    mean_errors, ratios, vis = _eval_pairs(model, frames, gt, image_shape, device,
                                           min_depth, max_depth, num_vis, side_crop, bottom_crop)
    metrics = dict(zip(METRIC_NAMES, mean_errors.tolist()))
    metrics["scale_ratio_median"] = float(np.median(ratios))
    metrics["scale_ratio_std"] = float(np.std(ratios))
    return metrics, ratios, vis


def list_frames(args):
    if args.list:
        root = Path(args.data_path) if args.data_path else Path(".")
        return [root / ln.strip() for ln in Path(args.list).read_text().splitlines() if ln.strip()]
    return sorted(p for e in IMG_EXTS for p in Path(args.rgb_dir).rglob(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained depth_model state_dict (best.pth)")
    ap.add_argument("--gt-npz", help="ground-truth depths (N,H,W) in mm; key 'data'")
    ap.add_argument("--rgb-dir", help="dir of frames (sorted <-> gt order)")
    ap.add_argument("--list", help="alt: file of RGB paths, one per line, rel to --data-path")
    ap.add_argument("--data-path", default="", help="root prepended to --list paths")
    ap.add_argument("--image-shape", type=int, nargs=2, default=[392, 490],
                    help="model feed res (H W), /14 -- match the training run")
    ap.add_argument("--side-crop-frac", type=float, default=0.0,
                    help="crop this frac off EACH of L/R (match a crop-trained model, e.g. depth_crop)")
    ap.add_argument("--bottom-crop-frac", type=float, default=0.0, help="crop this frac off the bottom")
    ap.add_argument("--min-depth", type=float, default=0.1)   # disp_to_depth range (scale absorbed
    ap.add_argument("--max-depth", type=float, default=150.0) #   by median scaling; kept for parity)
    ap.add_argument("--run-name", default="endodac-scared")
    ap.add_argument("--num-vis", type=int, default=8)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="metric-math self-check, no data/GPU")
    args = ap.parse_args()

    if args.smoke:                                            # pred==gt (post-scale) -> perfect
        gt = np.random.uniform(10, 100, (4, 32, 40)).astype(np.float32)
        errs = np.array([compute_errors(g[g > 0], (g * 0.5)[g > 0] * (np.median(g) / np.median(g * 0.5)))
                         for g in gt]).mean(0)
        assert errs[0] < 1e-5 and errs[4] > 0.999, errs
        print(f"[smoke] ok: abs_rel={errs[0]:.2e} a1={errs[4]:.4f} (median scaling cancels the 0.5x)")
        return

    assert args.gt_npz and (args.rgb_dir or args.list), "need --gt-npz and (--rgb-dir or --list)"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = load_gt(args.gt_npz)
    frames = list_frames(args)
    assert len(frames) == len(gt), \
        f"{len(frames)} frames vs {len(gt)} GT maps -- pairing mismatch (check --rgb-dir / order)"
    print(f"[setup] {len(frames)} frames | GT {np.shape(gt[0])} | feed={tuple(args.image_shape)} "
          f"| crop(side={args.side_crop_frac},bot={args.bottom_crop_frac}) | device={device}\n"
          f"  first: {frames[0]}\n  last : {frames[-1]}", flush=True)

    model_shape = (round14(args.image_shape[0]), round14(args.image_shape[1]))
    model = build_depth_model(model_shape, device)
    _filter_load(model, args.ckpt, "depth")

    mean_errors, ratios, vis = _eval_pairs(model, frames, gt, args.image_shape, device,
                                           args.min_depth, args.max_depth, args.num_vis,
                                           args.side_crop_frac, args.bottom_crop_frac)
    metrics = dict(zip(METRIC_NAMES, mean_errors.tolist()))
    print("\n  " + " ".join(f"{n:>9}" for n in METRIC_NAMES))
    print("  " + " ".join(f"{v:9.4f}" for v in mean_errors))
    print(f"  median scaling ratio: {np.median(ratios):.3f} +- {np.std(ratios):.3f} "
          f"(n={len(ratios)})", flush=True)

    if not args.no_wandb:
        import wandb
        wandb.init(project=os.getenv("WANDB_PROJECT", "rarp"),
                   entity=os.getenv("WANDB_ENTITY", "nmgtue"), name=args.run_name,
                   job_type="eval", config=dict(task="depth-eval", benchmark="SCARED-stereo",
                       ckpt=str(args.ckpt), image_shape=model_shape, n_frames=len(ratios),
                       median_scaling=True, side_crop=args.side_crop_frac, bottom_crop=args.bottom_crop_frac))
        wandb.log({f"scared/{k}": v for k, v in metrics.items()})
        wandb.run.summary.update({f"scared/{k}": v for k, v in metrics.items()})
        wandb.run.summary["scared/scale_ratio_median"] = float(np.median(ratios))
        wandb.run.summary["scared/scale_ratio_std"] = float(np.std(ratios))
        if vis:
            wandb.log({"scared/overlays": wandb.Image(
                np.concatenate(vis, axis=0), caption="[rgb | pred | gt] (disparity color)")})
        wandb.finish()
    print("[done]", {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
