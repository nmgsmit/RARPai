"""
Run a finetuned MetaFormerFPN on N images from a directory and write ONE contact
sheet of raw | overlay pairs. For eyeballing a checkpoint on unseen stills.

    python scripts/overlay_dir.py --images ../data/processed/UMCsulsnaps/sul_relaxed/images \
        --checkpoint outputs/rarp_nick_dice/best.pth --n 6 --out outputs/sul_relaxed_overlay.png
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "surgenet"))
from metaformer import MetaFormerFPN  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def round32(x):
    """CAFormer downsamples by 32, so the feed H and W must be multiples of 32."""
    return max(32, int(round(x / 32)) * 32)

# Colours + names straight from the labelling tool's legend
# (transfer_atlas_mod/gui/cutie/utils/palette.py), so an overlay here reads the
# same as one in the GUI. Keyed by the NEW scheme's ids.
NEW = {1: ("urethra", (255, 255, 0)), 2: ("prostate", (255, 0, 255)),
       3: ("dorsal venous plexus", (0, 0, 255)), 4: ("catheter", (0, 255, 0)),
       5: ("non-anatomical", (128, 128, 128))}
OLD = {1: ("catheter", (0, 255, 0)), 2: ("prostate", (255, 0, 255)),
       3: ("urethra", (255, 255, 0)), 4: ("apical vesicle", (0, 200, 255))}


def compact_legend(scheme: dict, keep: str | None) -> dict:
    """Model outputs COMPACT ids (1..K in --keep-classes order), which stop matching
    the raw scheme ids as soon as a class is excluded: with --keep-classes 1,2,4,5
    compact 3 is catheter, not the raw scheme's dorsal venous plexus. Rebuild the
    legend on compact ids, or the overlay is mislabelled AND miscoloured."""
    if not keep:
        return scheme                                  # compact == raw
    return {i: scheme[int(r)] for i, r in enumerate(keep.split(","), start=1)}


def predict(model, img: Image.Image, size_hw, device) -> np.ndarray:
    """Class-id map at the image's ORIGINAL size. size_hw is (H, W) -- feed the model
    the shape it was TRAINED at; a 1088x1344 model fed a 512 square sees the wrong
    scale and aspect at once."""
    h, w = size_hw
    x = np.array(img.convert("RGB").resize((w, h), Image.BICUBIC)) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None].to(device)
    with torch.no_grad():
        pred = model(x).argmax(1)[0].byte().cpu().numpy()
    # NEAREST back to full res: upsampling class IDS bilinearly would invent classes
    return np.array(Image.fromarray(pred).resize(img.size, Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=512, help="square feed size if --height/--width unset")
    ap.add_argument("--height", type=int, default=0, help="feed height (rounded to /32); match training")
    ap.add_argument("--width", type=int, default=0, help="feed width (rounded to /32); match training")
    ap.add_argument("--keep-classes", default=None,
                    help="the run's --keep-classes, so compact ids map back to the right "
                         "names/colours (e.g. 1,2,4,5). Omit if no class was excluded.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--scheme", default="new", choices=("new", "old"))
    ap.add_argument("--tile-width", type=int, default=1100, help="px per tile in the sheet")
    args = ap.parse_args()

    legend = compact_legend(NEW if args.scheme == "new" else OLD, args.keep_classes)
    if args.height and args.width:
        size_hw = (round32(args.height), round32(args.width))
    else:
        size_hw = (args.img_size, args.img_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    assert files, f"no images in {args.images}"
    picks = random.Random(args.seed).sample(files, min(args.n, len(files)))

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    nc = sd["FPN.segmentation_head.0.bias"].shape[0]
    model = MetaFormerFPN(num_classes=nc, pretrained="ImageNet", pretrained_weights=None)
    model.load_state_dict(sd)
    model.eval().to(device)
    print(f"[model] {args.checkpoint} num_classes={nc} feed={size_hw} device={device}")
    assert nc == len(legend) + 1, (
        f"checkpoint has {nc} classes but the legend has {len(legend)}+bg -- "
        f"pass the run's --keep-classes")

    rows = []
    for p in picks:
        img = Image.open(p).convert("RGB")
        m = predict(model, img, size_hw, device)
        raw = np.array(img)
        ov = raw.copy()
        for cid, (name, col) in legend.items():
            sel = m == cid
            if sel.any():
                ov[sel] = ((1 - args.alpha) * raw[sel] + args.alpha * np.array(col)).astype(np.uint8)
        present = [n for c, (n, _) in legend.items() if (m == c).any()]
        print(f"  {p.name[:52]}  ->  {', '.join(present) or 'NOTHING DETECTED'}")
        tile = Image.fromarray(np.concatenate([raw, ov], 1))
        tile = tile.resize((args.tile_width, round(args.tile_width * tile.height / tile.width)))
        d = ImageDraw.Draw(tile)
        d.text((6, 6), p.name[:46], fill=(255, 255, 255))
        rows.append(np.array(tile))

    # legend strip, so the colours mean something without reading this file
    strip = Image.new("RGB", (args.tile_width, 26), (16, 16, 16))
    d = ImageDraw.Draw(strip)
    x = 8
    for cid, (name, col) in legend.items():
        d.rectangle([x, 7, x + 12, 19], fill=col)
        d.text((x + 17, 8), name, fill=(230, 230, 230))
        x += 24 + 7 * len(name)
    rows.append(np.array(strip))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
