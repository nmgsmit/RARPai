"""
Finetune variant: background-excluded Tversky+CE loss, photometric augmentation,
and weight EMA. Built on the lowerlr config (lr=5e-5, hflip).
Optimizes catheter(1) + urethra(3); other classes remapped to background.

Why these three:
  - Loss: the old dice term averaged over ALL classes incl. background, diluting
    the gradient on the foreground. Tversky here EXCLUDES background (class 0) and
    averages only over foreground classes. alpha/beta are tunable; alpha=beta=0.5
    is exactly Dice. Default 0.4/0.6 is a mild FN nudge — urethra is ~20% of image
    width (not thin), so heavy FN weighting would over-segment.
  - Photometric aug (brightness/contrast/gamma): regularizes against the epoch-4
    overfit WITHOUT geometric distortion of directional surgical anatomy.
  - Weight EMA: evaluate/checkpoint a moving average of weights — smooths the
    noisy validation and usually nudges Dice up.

Real run:
    python scripts/finetune_seg_tversky.py \
        --data-root ../data/RARPSurgenet/fold1 \
        --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "surgenet"))
from metaformer import MetaFormerFPN  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# raw mask label ids. We keep only catheter+prostate; everything else -> background.
RAW_NAMES = {0: "background", 1: "catheter", 2: "prostate", 3: "urethra", 4: "apicalvesicle"}

# The clip-layout data (data/processed/Segmentation/Nick) was annotated with a DIFFERENT
# and NEWER scheme -- ids taken verbatim from the labelling tool's own legend,
# transfer_atlas_mod/gui/cutie/utils/palette.py `custom_names`. Do not reorder.
NICK_NAMES = {0: "background", 1: "urethra", 2: "prostate",
              3: "dorsalvenousplexus", 4: "catheter", 5: "nonanatomical"}
SCHEMES = {"rarpsurgenet": RAW_NAMES, "nick": NICK_NAMES}

# Old-scheme raw id -> new-scheme raw id, by NAME. Only the three classes present in
# both survive; apicalvesicle has no counterpart in the new scheme and maps to
# background, so the old test set cannot score it and neither can we.
OLD2NEW = np.zeros(256, dtype=np.uint8)
for _o, _n in RAW_NAMES.items():
    for _k, _v in NICK_NAMES.items():
        if _o and _v == _n:
            OLD2NEW[_o] = _k


def round32(x):
    """CAFormer downsamples by 32, so H and W must be multiples of 32."""
    return max(32, int(round(x / 32)) * 32)


def build_remap(keep):
    """LUT mapping raw label ids -> compact ids. kept classes become 1..K (in the
    given order), every other id (incl. raw background) maps to 0=background."""
    lut = np.zeros(256, dtype=np.uint8)
    for new_id, old_id in enumerate(keep, start=1):
        lut[old_id] = new_id
    return lut


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def photometric(img):
    """Brightness/contrast/gamma jitter on a uint8 HWC image. Mask untouched."""
    img = img.astype(np.float32)
    if random.random() < 0.5:                       # brightness
        img *= random.uniform(0.7, 1.3)
    if random.random() < 0.5:                       # contrast
        m = img.mean()
        img = (img - m) * random.uniform(0.7, 1.3) + m
    if random.random() < 0.5:                       # gamma
        g = random.uniform(0.7, 1.4)
        img = 255.0 * np.clip(img / 255.0, 0, 1) ** g
    return np.clip(img, 0, 255).astype(np.uint8)


def split_pairs(split_dir: Path):
    """Old RARPSurgenet layout: <split>/frames/*.png alongside <split>/masks/*.png,
    paired by sort order."""
    frames = sorted((split_dir / "frames").glob("*.png"))
    masks  = sorted((split_dir / "masks").glob("*.png"))
    assert len(frames) == len(masks) and frames, \
        f"frame/mask count mismatch or empty in {split_dir}"
    return list(zip(frames, masks))


def clip_pairs(clip_root: Path, seed: int, val_frac=0.15, test_frac=0.15):
    """Clip layout: <root>/<clip>/images/NNNNNNN.jpg + <clip>/masks/NNNNNNN.png.

    Split is by CLIP, not by frame: consecutive frames of one clip are near-duplicates,
    so a frame-level split leaks the test set into training and inflates every metric.
    Pairing is a stem INTERSECTION -- the export has masks with no image and vice versa
    (6659 images / 6635 masks / 6584 paired), so zip-by-sort-order would silently
    misalign every frame after the first gap.
    """
    clips = sorted(d for d in clip_root.iterdir() if (d / "images").is_dir())
    assert clips, f"no <clip>/images dirs under {clip_root}"
    rng = random.Random(seed)
    order = clips[:]
    rng.shuffle(order)
    n_val, n_test = round(len(order) * val_frac), round(len(order) * test_frac)
    groups = {"Test": order[:n_test],
              "Validation": order[n_test:n_test + n_val],
              "Train": order[n_test + n_val:]}
    out = {}
    for name, sel in groups.items():
        pairs = []
        for c in sel:
            imgs = {f.stem: f for f in (c / "images").glob("*.jpg")}
            msks = {f.stem: f for f in (c / "masks").glob("*.png")}
            pairs += [(imgs[s], msks[s]) for s in sorted(imgs.keys() & msks.keys())]
        assert pairs, f"split {name} is empty"
        out[name] = pairs
    return out, {k: len(v) for k, v in groups.items()}


class SegDataset(Dataset):
    def __init__(self, pairs, size_hw, remap, augment: bool = False):
        self.frames = [p[0] for p in pairs]
        self.masks  = [p[1] for p in pairs]
        assert self.frames, "empty dataset"
        self.size_hw = size_hw                           # (H, W)
        self.remap   = remap
        self.augment = augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        h, w = self.size_hw
        img = Image.open(self.frames[i]).convert("RGB").resize((w, h), Image.BICUBIC)
        msk = Image.open(self.masks[i])
        # The labelling tool writes PALETTE-mode masks: the label id is the palette
        # INDEX, and .convert("L") would push it through the palette to a luminance
        # (id 1 -> 226), which remap then drops to background -- an all-background
        # dataset that trains to loss 0 and reports dice 1.0. Old RARPSurgenet masks
        # are mode "L", where the convert was a harmless no-op. Read indices directly.
        if msk.mode != "P":
            msk = msk.convert("L")
        msk = msk.resize((w, h), Image.NEAREST)      # NEAREST keeps indices exact
        img = np.array(img)
        msk = self.remap[np.array(msk)]                 # drop unwanted classes -> bg
        if self.augment:
            if random.random() < 0.5:               # hflip (only safe geometric one)
                img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
            img = photometric(img)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = torch.from_numpy(msk).long()
        return x, y


class EMA:
    """Exponential moving average of model weights."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)  # ints (e.g. counters) just copied


def load_encoder(model: MetaFormerFPN, ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("teacher", "model", "state_dict"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            ck = ck[key]
            break
    sd = {k.replace("module.", "").replace("backbone.", ""): v
          for k, v in ck.items() if not k.startswith("head.")}
    msg = model.metaformer.load_state_dict(sd, strict=False)
    print(f"[encoder] loaded {len(sd)} tensors | missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}")


def tversky_ce_loss(logits, target, num_classes, alpha=0.4, beta=0.6, include_bg=False):
    """CE + (1 - mean Tversky). Background (class 0) is excluded from the Tversky
    term unless include_bg=True (keeps it, which can sharpen fg/bg boundaries).
    alpha weights FP, beta weights FN; alpha=beta=0.5 -> Dice."""
    ce    = F.cross_entropy(logits, target)
    probs = logits.softmax(1)
    oh    = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims  = (0, 2, 3)
    tp = (probs * oh).sum(dims)
    fp = (probs * (1 - oh)).sum(dims)
    fn = ((1 - probs) * oh).sum(dims)
    tversky = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    region = tversky if include_bg else tversky[1:]   # optionally keep background
    return ce + (1.0 - region.mean())


@torch.no_grad()
def validate(model, loader, num_classes, device, alpha, beta, include_bg=False):
    inter      = torch.zeros(num_classes)
    union      = torch.zeros(num_classes)
    dice_inter = torch.zeros(num_classes)
    dice_denom = torch.zeros(num_classes)
    total_loss = 0.0
    model.eval()
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        logits   = model(x)
        total_loss += tversky_ce_loss(logits, y, num_classes, alpha, beta, include_bg).item()
        pred    = logits.argmax(1).cpu()
        y_cpu   = y.cpu()
        for c in range(num_classes):
            p, t = pred == c, y_cpu == c
            inter[c]      += (p & t).sum()
            union[c]      += (p | t).sum()
            dice_inter[c] += (p & t).sum()
            dice_denom[c] += p.sum() + t.sum()
    present  = union > 0
    per_dice = (2 * dice_inter / dice_denom.clamp(min=1))
    val_miou = (inter / union.clamp(min=1))[present].mean().item()
    val_dice = per_dice[present].mean().item()
    return val_miou, val_dice, total_loss / len(loader), per_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",    default="../data/RARPSurgenet")
    ap.add_argument("--clip-root",    default=None,
                    help="clip-layout data root (<clip>/images/*.jpg + <clip>/masks/*.png). "
                         "Overrides --data-root; splits BY CLIP using --seed.")
    ap.add_argument("--label-scheme", default="rarpsurgenet", choices=sorted(SCHEMES),
                    help="which raw-id -> class-name legend the masks use")
    ap.add_argument("--compare-test", default=None,
                    help="extra eval on an old-scheme split dir (e.g. ../data/RARPSurgenet/Test) "
                         "with its labels mapped into the training scheme by name. Only classes "
                         "present in BOTH schemes are scorable.")
    ap.add_argument("--encoder-ckpt", default="../backbones/RARP_checkpoint_epoch0050_teacher.pth")
    ap.add_argument("--out",          default="outputs/rarp_tversky")
    ap.add_argument("--run-name",     default="tversky-ema")
    ap.add_argument("--keep-classes", default="1,3",
                    help="raw class ids to optimize; all others -> background. "
                         "Default catheter(1),urethra(3).")
    ap.add_argument("--img-size",     type=int,   default=512, help="square size if --height/--width unset")
    ap.add_argument("--height",       type=int,   default=0,   help="rectangular train height (rounded to /32)")
    ap.add_argument("--width",        type=int,   default=0,   help="rectangular train width (rounded to /32)")
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch-size",   type=int,   default=8)
    ap.add_argument("--lr",           type=float, default=5e-5)
    ap.add_argument("--alpha",        type=float, default=0.4, help="Tversky FP weight")
    ap.add_argument("--beta",         type=float, default=0.6, help="Tversky FN weight")
    ap.add_argument("--ema-decay",    type=float, default=0.999)
    ap.add_argument("--accum-steps",  type=int,   default=1,
                    help="gradient accumulation steps; effective batch = batch_size * accum_steps")
    ap.add_argument("--bg-in-loss",   action="store_true",
                    help="include background in the Tversky region term (default excluded)")
    ap.add_argument("--no-augment",   action="store_true")
    ap.add_argument("--workers",      type=int,   default=8)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--smoke",        action="store_true")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        m   = MetaFormerFPN(num_classes=12, pretrained="ImageNet", pretrained_weights=None).to(device)
        x   = torch.randn(2, 3, args.img_size, args.img_size, device=device)
        y   = torch.randint(0, 12, (2, args.img_size, args.img_size), device=device)
        out = m(x)
        assert out.shape[-2:] == x.shape[-2:], f"output {out.shape} != input HxW"
        tversky_ce_loss(out, y, 12, args.alpha, args.beta).backward()
        # EMA self-check: update moves shadow toward live weights, ints stay valid
        ema = EMA(m, decay=0.9)
        with torch.no_grad():
            list(m.parameters())[0].add_(1.0)
        ema.update(m)
        assert all(v.shape == m.state_dict()[k].shape for k, v in ema.shadow.items())

        # old-scheme id -> new-scheme id, by name (the compare-test path depends on this)
        assert (OLD2NEW[1], OLD2NEW[2], OLD2NEW[3]) == (4, 2, 1), OLD2NEW[:5]
        assert OLD2NEW[4] == 0, "apicalvesicle has no counterpart -> background"
        # composing with build_remap must land old ids on the right compact slots
        _lut = build_remap([1, 2, 3, 4, 5])[OLD2NEW]      # keep all 5 new classes
        assert (_lut[1], _lut[2], _lut[3], _lut[4]) == (4, 2, 1, 0)

        # clip split: by clip, stems intersected, no clip in two splits
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            for ci in range(20):
                for sub, ext in (("images", ".jpg"), ("masks", ".png")):
                    d = Path(td) / f"clip{ci}" / sub
                    d.mkdir(parents=True)
                    n = 5 if sub == "images" else 7      # deliberate mask surplus
                    for k in range(n):
                        Image.new("RGB" if ext == ".jpg" else "L", (8, 8)).save(d / f"{k:07d}{ext}")
            # a PALETTE mask must survive the loader as label INDICES, not luminance
            pm = Image.new("P", (8, 8), 0)
            pm.putpalette([0, 0, 0, 255, 255, 0, 255, 0, 255] + [0] * 759)
            pm.putpixel((0, 0), 1); pm.putpixel((1, 1), 2)
            pmf = Path(td) / "pal.png"; pm.save(pmf)
            assert set(np.unique(np.array(Image.open(pmf).convert("L")))) != {0, 1, 2}, \
                "test palette is degenerate -- convert(L) must differ from the indices"
            got = SegDataset([(pmf, pmf)], (8, 8), build_remap([1, 2]))[0][1].numpy()
            assert set(np.unique(got)) == {0, 1, 2}, f"palette mask mangled: {np.unique(got)}"

            sp, ncl = clip_pairs(Path(td), seed=42)
            assert sum(ncl.values()) == 20, ncl
            seen = [p[0].parent.parent.name for v in sp.values() for p in v]
            assert len(set(seen)) == 20, "a clip leaked across splits"
            assert all(len(v) == 5 * ncl[k] for k, v in sp.items()), "stems not intersected"
            assert all(a.stem == b.stem for v in sp.values() for a, b in v), "misaligned pair"
            assert clip_pairs(Path(td), seed=42)[0].keys() == sp.keys()
        print(f"[smoke] ok | out={tuple(out.shape)} device={device}")
        return

    root  = Path(args.data_root)
    scheme = SCHEMES[args.label_scheme]
    keep  = [int(c) for c in args.keep_classes.split(",")]
    names = ["background"] + [scheme.get(c, f"class{c}") for c in keep]  # compact-id -> name
    remap = build_remap(keep)
    nc    = len(keep) + 1

    if args.clip_root:
        splits, nclips = clip_pairs(Path(args.clip_root), args.seed)
        print(f"[data] clips {nclips} -> frames "
              f"{ {k: len(v) for k, v in splits.items()} }")
    else:
        splits = {s: split_pairs(root / s) for s in ("Train", "Validation", "Test")}
    if args.height and args.width:
        size_hw = (round32(args.height), round32(args.width))
    else:
        size_hw = (args.img_size, args.img_size)
    print(f"[setup] keep={keep} -> num_classes={nc} ({names}) "
          f"size_hw={size_hw} device={device}")

    import wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "rarp-segmentation"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=args.run_name,
        config=dict(
            num_classes=nc, keep_classes=keep, class_names=names,
            size_hw=size_hw, epochs=args.epochs,
            batch_size=args.batch_size, accum_steps=args.accum_steps,
            eff_batch=args.batch_size * args.accum_steps, lr=args.lr,
            alpha=args.alpha, beta=args.beta, ema_decay=args.ema_decay,
            bg_in_loss=args.bg_in_loss,
            augment=not args.no_augment, loss="tversky+ce(bg-excluded)",
            aug="hflip+photometric", seed=args.seed,
            encoder_ckpt=args.encoder_ckpt, data_root=str(root),
        ),
    )

    g  = torch.Generator()
    g.manual_seed(args.seed)
    tr = DataLoader(SegDataset(splits["Train"], size_hw, remap, augment=not args.no_augment),
                    args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, generator=g)
    va = DataLoader(SegDataset(splits["Validation"], size_hw, remap),
                    args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Label sanity: an all-background training set trains to loss 0 and reports
    # dice 1.0 (validate() averages over PRESENT classes only, and only bg is
    # present), which looks like a great run. Cost of this check is ~2 s.
    ds = tr.dataset
    hist = np.zeros(nc, dtype=np.int64)
    for j in np.linspace(0, len(ds) - 1, min(24, len(ds))).astype(int):
        hist += np.bincount(ds[int(j)][1].numpy().ravel(), minlength=nc)
    frac = hist / hist.sum()
    print("[labels] " + "  ".join(f"{names[c]}={frac[c]*100:.2f}%" for c in range(nc)), flush=True)
    missing = [names[c] for c in range(1, nc) if hist[c] == 0]
    assert not missing, (f"classes {missing} never appear in {len(ds)} training masks -- "
                         "wrong --keep-classes, --label-scheme, or mask palette mode?")

    model = MetaFormerFPN(num_classes=nc, pretrained="ImageNet", pretrained_weights=None).to(device)
    load_encoder(model, args.encoder_ckpt)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    ema   = EMA(model, decay=args.ema_decay)
    print(f"[optim] AdamW lr={args.lr} ReduceLROnPlateau(patience=3) "
          f"tversky(a={args.alpha},b={args.beta}) ema={args.ema_decay}")
    scaler = torch.amp.GradScaler(device)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    accum = max(1, args.accum_steps)
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        opt.zero_grad()
        pending = False
        for i, (x, y) in enumerate(tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device):
                loss = tversky_ce_loss(model(x), y, nc, args.alpha, args.beta, args.bg_in_loss)
            scaler.scale(loss / accum).backward()   # average grads over accum micro-batches
            run += loss.item()
            pending = True
            if (i + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                ema.update(model)
                pending = False
        if pending:                                 # flush leftover micro-batches
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            ema.update(model)
        avg_loss = run / len(tr)

        # validate on the EMA weights, then restore live weights for training
        live = deepcopy(model.state_dict())
        model.load_state_dict(ema.shadow)
        val_miou, val_dice, val_loss, per_dice = validate(model, va, nc, device, args.alpha, args.beta, args.bg_in_loss)
        model.load_state_dict(live)

        sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        track = [(c, names[c]) for c in range(1, nc)]   # every foreground class
        per_cls = "  ".join(f"{n}={per_dice[c]:.4f}" for c, n in track)
        print(f"epoch {ep+1}/{args.epochs}  train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_mIoU={val_miou:.4f}  val_dice={val_dice:.4f}  {per_cls}", flush=True)
        wandb.log({"train/loss": avg_loss, "val/loss": val_loss,
                   "val/mIoU": val_miou, "val/dice": val_dice,
                   **{f"val/dice_{n}": per_dice[c].item() for c, n in track},
                   "lr": lr, "epoch": ep + 1})
        if val_dice > best:
            best = val_dice
            torch.save(ema.shadow, outdir / "best.pth")   # save the EMA weights
            wandb.run.summary["best_val_dice"] = best
            wandb.run.summary["best_val_mIoU"] = val_miou

    # final test-set eval using best (EMA) checkpoint
    te = DataLoader(SegDataset(splits["Test"], size_hw, remap),
                    args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    model.load_state_dict(torch.load(outdir / "best.pth", map_location=device))
    te_miou, te_dice, te_loss, te_per_dice = validate(model, te, nc, device, args.alpha, args.beta, args.bg_in_loss)
    track = [(c, names[c]) for c in range(1, nc)]
    te_per = "  ".join(f"{n}={te_per_dice[c]:.4f}" for c, n in track)
    print(f"[test]  mIoU={te_miou:.4f}  dice={te_dice:.4f}  {te_per}", flush=True)
    wandb.run.summary.update({
        "test/mIoU":  te_miou,
        "test/dice":  te_dice,
        **{f"test/dice_{n}": te_per_dice[c].item() for c, n in track},
    })

    # Second eval on the OLD test set, for comparison against the pre-existing runs.
    # Its masks carry the old scheme, so compose old-raw -> new-raw -> compact. Classes
    # the old set has no label for (and old apicalvesicle) become background there, so
    # only report the shared ones -- a class absent from the GT scores a meaningless 0.
    if args.compare_test:
        shared = [(c, names[c]) for c in range(1, nc)
                  if names[c] in set(RAW_NAMES.values()) & set(NICK_NAMES.values())]
        cmp_lut = remap[OLD2NEW] if args.label_scheme == "nick" else remap
        cmp_ld = DataLoader(SegDataset(split_pairs(Path(args.compare_test)), size_hw, cmp_lut),
                            args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        c_miou, c_dice, _, c_per = validate(model, cmp_ld, nc, device,
                                            args.alpha, args.beta, args.bg_in_loss)
        c_shared = sum(c_per[c] for c, _ in shared) / len(shared)
        print(f"[compare] {args.compare_test}  shared_dice={c_shared:.4f}  "
              + "  ".join(f"{n}={c_per[c]:.4f}" for c, n in shared)
              + f"  (all-class mIoU={c_miou:.4f} dice={c_dice:.4f}, not comparable)", flush=True)
        wandb.run.summary.update({
            "compare/shared_dice": float(c_shared),
            **{f"compare/dice_{n}": c_per[c].item() for c, n in shared},
        })

    wandb.finish()
    print(f"[done] best val_dice={best:.4f} -> {outdir/'best.pth'}")


if __name__ == "__main__":
    main()
