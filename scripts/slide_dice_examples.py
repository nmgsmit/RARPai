"""
One presentation slide (16:9 PNG): best / median / worst urethra predictions.

Per-frame urethra dice over the held-out test clips, then 2 best + 2 median + 2 worst,
**at most one frame per clip** -- consecutive frames of a clip are near-duplicates, so
without that rule the "worst" column is the same bad frame six times and the slide
says nothing.

Dice is computed on the 512 grid the model is scored on, so the numbers on the slide
are the same ones in the eval table. The picture is the ORIGINAL frame (prediction
upsampled NEAREST) because a 512 square looks squashed on a slide.

    python scripts/slide_dice_examples.py --checkpoint outputs/ureth_fn/best.pth \
        --keep-largest --out outputs/slide_urethra.png
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Patch, Rectangle      # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "third_party" / "surgenet"))
_spec = importlib.util.spec_from_file_location("ft", _HERE / "finetune_seg_tversky.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)
_spec2 = importlib.util.spec_from_file_location("eu", _HERE / "eval_urethra.py")
eu = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(eu)
from metaformer import MetaFormerFPN                 # noqa: E402

PRED_RGB = (1.0, 0.85, 0.0)      # prediction fill
GT_RGB   = "#00E5FF"             # ground-truth outline


def crop_box(gt, pred, shape, aspect, pad=1.9, min_h=300):
    """Crop around the union of GT and prediction at the PANEL's aspect ratio, so the
    anatomy fills the tile. A square crop in a wide panel wastes half the slide."""
    h, w = shape
    m = gt | pred
    if not m.any():
        cy, cx, bh, bw = h / 2, w / 2, h, w
    else:
        ys, xs = np.nonzero(m)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        bh, bw = (ys.max() - ys.min()) * pad, (xs.max() - xs.min()) * pad
    ch = max(bh, bw / aspect, min_h)
    cw = ch * aspect
    if cw > w:                                   # frame is not wide enough: clamp both
        cw, ch = w, w / aspect
    if ch > h:
        ch, cw = h, h * aspect
    y0 = int(np.clip(cy - ch / 2, 0, h - ch))
    x0 = int(np.clip(cx - cw / 2, 0, w - cw))
    return y0, x0, int(round(ch)), int(round(cw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clip-root", default="../data/processed/Segmentation/Nick")
    ap.add_argument("--keep-classes", default="1,2,4,5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--per-group", type=int, default=2)
    ap.add_argument("--keep-largest", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default="Urethra segmentation on held-out surgeries")
    args = ap.parse_args()

    keep = [int(c) for c in args.keep_classes.split(",")]
    nc, u = len(keep) + 1, 1                          # urethra is first in keep -> compact 1
    assert keep[0] == 1, "this slide assumes urethra is the first kept class"
    remap = ft.build_remap(keep)
    size = args.img_size

    splits, _ = ft.clip_pairs(Path(args.clip_root), args.seed)
    pairs = splits["Test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MetaFormerFPN(num_classes=nc, pretrained="ImageNet", pretrained_weights=None)
    model.load_state_dict(sd)
    model.eval().to(device)
    print(f"[model] {args.checkpoint} device={device} frames={len(pairs)}")

    rows = []
    with torch.no_grad():
        for i, (imgp, mskp) in enumerate(pairs):
            im = Image.open(imgp).convert("RGB")
            x = np.array(im.resize((size, size), Image.BICUBIC)) / 255.0
            x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            x = torch.from_numpy(x).permute(2, 0, 1).float()[None].to(device)
            pr = model(x).argmax(1).cpu().numpy()
            if args.keep_largest:
                pr = eu.keep_largest(pr, u)
            pr = pr[0]

            mk = Image.open(mskp)
            if mk.mode != "P":
                mk = mk.convert("L")
            gt = remap[np.array(mk.resize((size, size), Image.NEAREST))]

            g, p = (gt == u), (pr == u)
            if not g.any():
                continue                              # no GT urethra -> dice undefined
            dice = 2 * (g & p).sum() / (g.sum() + p.sum())
            rows.append(dict(dice=float(dice), img=imgp, gt=g, pr=p,
                             clip=imgp.parent.parent.name))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(pairs)}", flush=True)

    rows.sort(key=lambda r: r["dice"])
    print(f"[scored] {len(rows)} frames with GT urethra, "
          f"dice min={rows[0]['dice']:.3f} med={rows[len(rows)//2]['dice']:.3f} "
          f"max={rows[-1]['dice']:.3f}")

    n = args.per_group
    used_clips, chosen = set(), {}

    def take(order, k):
        out = []
        for r in order:
            if len(out) == k:
                break
            if r["clip"] in used_clips:
                continue                              # max ONE frame per clip
            used_clips.add(r["clip"])
            out.append(r)
        return out

    mid = len(rows) // 2
    # order matters: best and worst claim their clips first, median fills from the
    # centre outwards with whatever clips are left
    chosen["Best"] = take(rows[::-1], n)
    chosen["Worst"] = take(rows, n)
    centre = sorted(rows, key=lambda r: abs(rows.index(r) - mid))
    chosen["Typical"] = take(centre, n)

    cols = ["Best", "Typical", "Worst"]
    # panel aspect drives the crop; keep the two in step or the tiles letterbox
    FIG_W, FIG_H = 13.333, 7.5
    L, R, TOP, BOT, WS, HS = 0.015, 0.985, 0.845, 0.105, 0.02, 0.045
    panel_aspect = ((FIG_W * (R - L) / (3 + 2 * WS)) /
                    (FIG_H * (TOP - BOT) / (n + (n - 1) * HS)))
    fig, axes = plt.subplots(n, 3, figsize=(FIG_W, FIG_H), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)

    for c, name in enumerate(cols):
        for r in range(n):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r >= len(chosen[name]):
                ax.axis("off")
                continue
            rec = chosen[name][r]
            im = np.array(Image.open(rec["img"]).convert("RGB"))
            H, W = im.shape[:2]
            gt = np.array(Image.fromarray(rec["gt"].astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
            pr = np.array(Image.fromarray(rec["pr"].astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
            y0, x0, ch, cw = crop_box(gt, pr, (H, W), panel_aspect)
            sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))
            im, gt, pr = im[sl], gt[sl], pr[sl]

            shown = im.astype(float) / 255.0
            shown[pr] = 0.45 * shown[pr] + 0.55 * np.array(PRED_RGB)
            ax.imshow(shown)
            if gt.any():
                ax.contour(gt.astype(float), levels=[0.5], colors=[GT_RGB], linewidths=1.6)
            ax.text(0.03, 0.955, f"Dice {rec['dice']:.2f}", transform=ax.transAxes,
                    fontsize=13, fontweight="bold", color="white", va="top",
                    bbox=dict(boxstyle="round,pad=0.28", fc="#00000099", ec="none"))
            if r == 0:
                ax.set_title(name, fontsize=21, fontweight="bold", pad=8,
                             color={"Best": "#1a7f37", "Typical": "#8a6d00",
                                    "Worst": "#b42318"}[name])

    med = rows[len(rows) // 2]["dice"]
    fig.suptitle(args.title, fontsize=26, fontweight="bold", y=0.975)
    fig.text(0.5, 0.895, f"median Dice {med:.2f} across {len(rows)} frames  |  "
                         f"{len(used_clips)} different surgeries, one frame each",
             ha="center", fontsize=13, color="#444444")
    fig.legend(handles=[Patch(fc=PRED_RGB, ec="none", label="Model prediction"),
                        Patch(fc="none", ec=GT_RGB, lw=2.5, label="Ground truth (outline)")],
               loc="lower center", ncol=2, frameon=False, fontsize=15,
               bbox_to_anchor=(0.5, 0.012))
    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT, wspace=WS, hspace=HS)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"[done] {args.out}")
    for name in cols:
        for rec in chosen[name]:
            print(f"  {name:8s} dice={rec['dice']:.3f}  {rec['clip'][:44]}")


if __name__ == "__main__":
    main()
