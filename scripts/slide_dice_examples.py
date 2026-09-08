"""
Presentation slides (16:9 PNG): best / typical / worst urethra predictions.

Frame choice obeys three rules, in this order:

  1. CONTENT FILTERS, read off the ground truth, not guessed: no catheter labelled,
     instruments below --max-instrument of the frame, urethra at least --min-elong
     long-to-wide. That is what "a generic prostate/urethra view" means concretely.
  2. one frame per clip -- consecutive frames are near-duplicates, so without this
     the Worst column is the same bad frame repeated.
  3. within each band, --pick-seeds picks different eligible frames, so several
     candidate slides come out of ONE inference pass and you choose.

Bands are percentile ranges over the ELIGIBLE pool, not the global extremes: with
filters on, "Best" means best-of-this-view, and the subtitle reports both medians so
the slide never overstates the model.

NOTE --seed is the clip-split seed and must stay at the value the model trained with
(42), or the "held-out" frames are ones it saw. --pick-seeds is the cosmetic one.

    python scripts/slide_dice_examples.py --checkpoint outputs/ureth_fn/best.pth \
        --keep-largest --pick-seeds 0,1,2,3 --out outputs/slide_urethra.png
"""
from __future__ import annotations
import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Patch                 # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "third_party" / "surgenet"))
_spec = importlib.util.spec_from_file_location("ft", _HERE / "finetune_seg_tversky.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)
_spec2 = importlib.util.spec_from_file_location("eu", _HERE / "eval_urethra.py")
eu = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(eu)
from metaformer import MetaFormerFPN                 # noqa: E402

PRED_RGB = (1.0, 0.85, 0.0)
GT_RGB = "#00E5FF"
COLS = ["Best", "Typical", "Worst"]
COL_C = {"Best": "#1a7f37", "Typical": "#8a6d00", "Worst": "#b42318"}


def elongation(mask):
    """Long/short axis ratio of the mask, via PCA on its pixel coordinates. A thin
    tube gives a big number; a blob gives ~1. Bounded below by 1."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 8:
        return 0.0
    pts = np.stack([ys - ys.mean(), xs - xs.mean()])
    ev = np.linalg.eigvalsh(np.cov(pts))
    return float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))


def crop_box(gt, pred, shape, aspect, pad=1.9, min_h=300):
    """Crop around the union of GT and prediction at the PANEL's aspect ratio -- a
    square crop in a wide panel wastes half the slide."""
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
    if cw > w:
        cw, ch = w, w / aspect
    if ch > h:
        ch, cw = h, h * aspect
    y0 = int(np.clip(cy - ch / 2, 0, h - ch))
    x0 = int(np.clip(cx - cw / 2, 0, w - cw))
    return y0, x0, int(round(ch)), int(round(cw))


def choose(pool, n, pick_seed, band=0.15):
    """One frame per clip, sampled from the top/middle/bottom band of `pool`
    (already sorted ascending by dice). Best and Worst claim clips first."""
    rng = random.Random(pick_seed)
    N = len(pool)
    k = max(n, int(round(N * band)))
    bands = {"Best": pool[-k:][::-1],
             "Worst": pool[:k],
             "Typical": pool[max(0, N // 2 - k // 2):N // 2 + max(1, k // 2)]}
    used, out = set(), {}
    for name in ("Best", "Worst", "Typical"):        # extremes pick before the middle
        cand = bands[name][:]
        rng.shuffle(cand)
        sel = []
        for r in cand:
            if len(sel) == n:
                break
            if r["clip"] in used:
                continue
            used.add(r["clip"])
            sel.append(r)
        out[name] = sorted(sel, key=lambda r: -r["dice"])
    return out, used


def render(chosen, used, n, med_all, med_pool, n_all, n_pool, args, out_path):
    FIG_W, FIG_H = 13.333, 7.5
    L, R, TOP, BOT, WS, HS = 0.015, 0.985, 0.845, 0.105, 0.02, 0.045
    panel_aspect = ((FIG_W * (R - L) / (3 + 2 * WS)) /
                    (FIG_H * (TOP - BOT) / (n + (n - 1) * HS)))
    fig, axes = plt.subplots(n, 3, figsize=(FIG_W, FIG_H), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)

    for c, name in enumerate(COLS):
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
                ax.set_title(name, fontsize=21, fontweight="bold", pad=8, color=COL_C[name])

    fig.suptitle(args.title, fontsize=26, fontweight="bold", y=0.975)
    sub = f"median Dice {med_all:.2f} across {n_all} held-out frames"
    if n_pool != n_all:
        sub += f"  |  panels from {n_pool} frames of this view (median {med_pool:.2f})"
    sub += f"  |  {len(used)} surgeries, one frame each"
    fig.text(0.5, 0.895, sub, ha="center", fontsize=13, color="#444444")
    fig.legend(handles=[Patch(fc=PRED_RGB, ec="none", label="Model prediction"),
                        Patch(fc="none", ec=GT_RGB, lw=2.5, label="Ground truth (outline)")],
               loc="lower center", ncol=2, frameon=False, fontsize=15,
               bbox_to_anchor=(0.5, 0.012))
    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT, wspace=WS, hspace=HS)
    fig.savefig(out_path, dpi=args.dpi, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clip-root", default="../data/processed/Segmentation/Nick")
    ap.add_argument("--keep-classes", default="1,2,4,5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--per-group", type=int, default=2)
    ap.add_argument("--keep-largest", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="CLIP-SPLIT seed; must match the run")
    ap.add_argument("--pick-seeds", default="0", help="comma list; one slide per seed")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default="Urethra segmentation on held-out surgeries")
    ap.add_argument("--no-catheter", action="store_true",
                    help="drop frames whose GT labels any catheter")
    ap.add_argument("--max-instrument", type=float, default=1.0,
                    help="drop frames where GT instruments exceed this area fraction")
    ap.add_argument("--min-elong", type=float, default=0.0,
                    help="drop frames whose GT urethra is less elongated than this")
    ap.add_argument("--min-urethra", type=float, default=0.0,
                    help="drop frames whose GT urethra is under this area fraction")
    args = ap.parse_args()

    keep = [int(c) for c in args.keep_classes.split(",")]
    scheme = ft.NICK_NAMES
    names = ["background"] + [scheme[c] for c in keep]
    nc, u = len(keep) + 1, names.index("urethra")
    i_cath = names.index("catheter") if "catheter" in names else None
    i_instr = names.index("nonanatomical") if "nonanatomical" in names else None
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
                continue
            rows.append(dict(
                dice=float(2 * (g & p).sum() / (g.sum() + p.sum())),
                img=imgp, gt=g, pr=p, clip=imgp.parent.parent.name,
                cath=float((gt == i_cath).mean()) if i_cath else 0.0,
                instr=float((gt == i_instr).mean()) if i_instr else 0.0,
                elong=elongation(g), ufrac=float(g.mean())))
            if (i + 1) % 300 == 0:
                print(f"  {i+1}/{len(pairs)}", flush=True)

    rows.sort(key=lambda r: r["dice"])
    med_all = rows[len(rows) // 2]["dice"]
    print(f"[scored] {len(rows)} frames, median dice {med_all:.3f}")
    for f in ("elong", "instr", "cath", "ufrac"):
        v = np.array([r[f] for r in rows])
        print(f"  {f:6s} p10={np.percentile(v,10):.3f} p50={np.percentile(v,50):.3f} "
              f"p90={np.percentile(v,90):.3f}  zero={100*(v==0).mean():.0f}%")

    pool = [r for r in rows
            if (not args.no_catheter or r["cath"] == 0)
            and r["instr"] <= args.max_instrument
            and r["elong"] >= args.min_elong
            and r["ufrac"] >= args.min_urethra]
    clips = {r["clip"] for r in pool}
    print(f"[filter] {len(pool)}/{len(rows)} frames eligible, {len(clips)} clips")
    if len(pool) < args.per_group * 3 or len(clips) < args.per_group * 3:
        print(f"[filter] TOO STRICT for {args.per_group*3} panels from distinct clips "
              f"-- relax --min-elong / --max-instrument", file=sys.stderr)
        sys.exit(2)
    med_pool = sorted(r["dice"] for r in pool)[len(pool) // 2]

    out = Path(args.out)
    for ps in [int(v) for v in args.pick_seeds.split(",")]:
        chosen, used = choose(pool, args.per_group, ps)
        dst = out if args.pick_seeds == "0" else out.with_name(f"{out.stem}_pick{ps}{out.suffix}")
        render(chosen, used, args.per_group, med_all, med_pool,
               len(rows), len(pool), args, dst)
        print(f"[done] {dst}")
        for name in COLS:
            for rec in chosen[name]:
                print(f"    {name:8s} dice={rec['dice']:.3f} elong={rec['elong']:.1f} "
                      f"instr={100*rec['instr']:.0f}% {rec['clip'][:38]}")


if __name__ == "__main__":
    main()
