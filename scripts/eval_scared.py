"""
SCARED depth evaluation for a trained EndoDAC depth_model (outputs/<run>/best.pth).

SCARED is the structured-light ground-truth benchmark used by AF-SfMLearner / EndoDAC,
so the numbers here are directly comparable to those papers. Self-supervised depth is
scale-ambiguous, so we apply per-frame MEDIAN SCALING (the standard convention) before
computing the 7 Monodepth2 metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3).

You must supply SCARED yourself -- it's license-gated (Intuitive Surgical EULA), NOT
downloadable without signing the data-sharing agreement. Once you have the test
keyframes (dataset_8/9), give me the ground-truth depth as an npz of HxW depth maps
in mm (AF-SfM's `export_gt_depth.py` produces exactly this: key "data", shape (N,H,W)).

Two ways to point at the data (pick one):
  (a) parallel folders (simplest, recommended):
        --rgb-dir  <dir of extracted left frames, sorted>   --gt-npz gt_depths.npz
      i-th sorted frame  <->  i-th GT map. The script prints the first/last pairing
      so you can eyeball that they line up before trusting the metrics.
  (b) explicit list:
        --list  test_files.txt (one RGB path per line, relative to --data-path)  --gt-npz ...

Run (Snellius, gpu_h100):
    python scripts/eval_scared.py \
        --ckpt outputs/rarp_depth/best.pth \
        --rgb-dir ../data/SCARED/test_left --gt-npz ../data/SCARED/gt_depths.npz \
        --image-shape 392 490 --run-name endodac-rarp-scared

ponytail: reuses finetune_depth's model builders + vendored compute_errors/disp_to_depth;
no reimplementation. Depth-only eval -> intrinsics irrelevant (median scaling absorbs scale).
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


def list_frames(args):
    if args.list:
        root = Path(args.data_path) if args.data_path else Path(".")
        return [root / ln.strip() for ln in Path(args.list).read_text().splitlines() if ln.strip()]
    return sorted(p for e in IMG_EXTS for p in Path(args.rgb_dir).rglob(e))


def eval_scared(model, frames, gt, args, device):
    mh, mw = round14(args.image_shape[0]), round14(args.image_shape[1])
    to_tensor = transforms.ToTensor()
    errors, ratios, vis = [], [], []
    model.eval()
    for i, (fp, g) in enumerate(zip(frames, gt)):
        g = np.asarray(g, np.float32)
        img = Image.open(fp).convert("RGB").resize((mw, mh), Image.BILINEAR)
        with torch.no_grad():
            disp = model(to_tensor(img).unsqueeze(0).to(device))[("disp", 0)]
            _, depth = disp_to_depth(disp, args.min_depth, args.max_depth)
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

        if len(vis) < args.num_vis:                           # [rgb | pred | gt] overlay row
            rgb_np = (np.asarray(img.resize(g.shape[:2][::-1])) ).astype(np.uint8)
            vmask = mask
            vis.append(np.concatenate([rgb_np,
                                       colorize(1.0 / np.clip(pred_full * ratio, EVAL_MIN, None), vmask),
                                       colorize(1.0 / np.clip(g, EVAL_MIN, None), vmask)], axis=1))
    if not errors:
        raise SystemExit("no valid GT pixels in any frame -- check --gt-npz units/pairing")
    return np.array(errors).mean(0), np.array(ratios), vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained depth_model state_dict (best.pth)")
    ap.add_argument("--gt-npz", help="ground-truth depths (N,H,W) in mm; key 'data'")
    ap.add_argument("--rgb-dir", help="dir of extracted left frames (sorted <-> gt order)")
    ap.add_argument("--list", help="alt: file of RGB paths, one per line, rel to --data-path")
    ap.add_argument("--data-path", default="", help="root prepended to --list paths")
    ap.add_argument("--image-shape", type=int, nargs=2, default=[392, 490],
                    help="model feed res (H W), /14 -- match the training run")
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
          f"| device={device}\n  first: {frames[0]}\n  last : {frames[-1]}", flush=True)

    model_shape = (round14(args.image_shape[0]), round14(args.image_shape[1]))
    model = build_depth_model(model_shape, device)
    _filter_load(model, args.ckpt, "depth")

    mean_errors, ratios, vis = eval_scared(model, frames, gt, args, device)
    metrics = dict(zip(METRIC_NAMES, mean_errors.tolist()))
    print("\n  " + " ".join(f"{n:>9}" for n in METRIC_NAMES))
    print("  " + " ".join(f"{v:9.4f}" for v in mean_errors))
    print(f"  median scaling ratio: {np.median(ratios):.3f} +- {np.std(ratios):.3f} "
          f"(n={len(ratios)})", flush=True)

    if not args.no_wandb:
        import wandb
        wandb.init(project=os.getenv("WANDB_PROJECT", "rarp"),
                   entity=os.getenv("WANDB_ENTITY", "nmgtue"), name=args.run_name,
                   job_type="eval", config=dict(task="depth-eval", benchmark="SCARED",
                       ckpt=str(args.ckpt), image_shape=model_shape, n_frames=len(ratios),
                       median_scaling=True, eval_min=EVAL_MIN, eval_max=EVAL_MAX))
        wandb.log({f"scared/{k}": v for k, v in metrics.items()})
        wandb.run.summary.update({f"scared/{k}": v for k, v in metrics.items()})
        wandb.run.summary["scared/scale_ratio_median"] = float(np.median(ratios))
        wandb.run.summary["scared/scale_ratio_std"] = float(np.std(ratios))
        if vis:
            wandb.log({"scared/overlays": wandb.Image(
                np.concatenate(vis, axis=0), caption="[rgb | pred | gt] (disparity color)")})
        wandb.finish()
    print("[done]", metrics)


if __name__ == "__main__":
    main()
