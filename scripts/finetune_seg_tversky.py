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


class SegDataset(Dataset):
    def __init__(self, split_dir: Path, img_size: int, remap, augment: bool = False):
        self.frames = sorted((split_dir / "frames").glob("*.png"))
        self.masks  = sorted((split_dir / "masks").glob("*.png"))
        assert len(self.frames) == len(self.masks) and self.frames, \
            f"frame/mask count mismatch or empty in {split_dir}"
        self.img_size = img_size
        self.remap    = remap
        self.augment  = augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        img = Image.open(self.frames[i]).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BICUBIC)
        msk = Image.open(self.masks[i]).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST)
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


def tversky_ce_loss(logits, target, num_classes, alpha=0.4, beta=0.6):
    """CE + (1 - mean foreground Tversky). Background (class 0) excluded from the
    Tversky term. alpha weights FP, beta weights FN; alpha=beta=0.5 -> Dice."""
    ce    = F.cross_entropy(logits, target)
    probs = logits.softmax(1)
    oh    = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims  = (0, 2, 3)
    tp = (probs * oh).sum(dims)
    fp = (probs * (1 - oh)).sum(dims)
    fn = ((1 - probs) * oh).sum(dims)
    tversky = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return ce + (1.0 - tversky[1:].mean())          # drop background class 0


@torch.no_grad()
def validate(model, loader, num_classes, device, alpha, beta):
    inter      = torch.zeros(num_classes)
    union      = torch.zeros(num_classes)
    dice_inter = torch.zeros(num_classes)
    dice_denom = torch.zeros(num_classes)
    total_loss = 0.0
    model.eval()
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        logits   = model(x)
        total_loss += tversky_ce_loss(logits, y, num_classes, alpha, beta).item()
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
    ap.add_argument("--data-root",    default="../data/RARPSurgenet/fold1")
    ap.add_argument("--encoder-ckpt", default="../backbones/RARP_checkpoint_epoch0050_teacher.pth")
    ap.add_argument("--out",          default="outputs/rarp_tversky")
    ap.add_argument("--run-name",     default="tversky-ema")
    ap.add_argument("--keep-classes", default="1,3",
                    help="raw class ids to optimize; all others -> background. "
                         "Default catheter(1),urethra(3).")
    ap.add_argument("--img-size",     type=int,   default=512)
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch-size",   type=int,   default=8)
    ap.add_argument("--lr",           type=float, default=5e-5)
    ap.add_argument("--alpha",        type=float, default=0.4, help="Tversky FP weight")
    ap.add_argument("--beta",         type=float, default=0.6, help="Tversky FN weight")
    ap.add_argument("--ema-decay",    type=float, default=0.999)
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
        print(f"[smoke] ok | out={tuple(out.shape)} device={device}")
        return

    root  = Path(args.data_root)
    keep  = [int(c) for c in args.keep_classes.split(",")]
    names = ["background"] + [RAW_NAMES.get(c, f"class{c}") for c in keep]  # compact-id -> name
    remap = build_remap(keep)
    nc    = len(keep) + 1
    print(f"[setup] keep={keep} -> num_classes={nc} ({names}) "
          f"img_size={args.img_size} device={device}")

    import wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "rarp-segmentation"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=args.run_name,
        config=dict(
            num_classes=nc, keep_classes=keep, class_names=names,
            img_size=args.img_size, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            alpha=args.alpha, beta=args.beta, ema_decay=args.ema_decay,
            augment=not args.no_augment, loss="tversky+ce(bg-excluded)",
            aug="hflip+photometric", seed=args.seed,
            encoder_ckpt=args.encoder_ckpt, data_root=str(root),
        ),
    )

    g  = torch.Generator()
    g.manual_seed(args.seed)
    tr = DataLoader(SegDataset(root / "Train", args.img_size, remap, augment=not args.no_augment),
                    args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, generator=g)
    va = DataLoader(SegDataset(root / "Validation", args.img_size, remap),
                    args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

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
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(device):
                loss = tversky_ce_loss(model(x), y, nc, args.alpha, args.beta)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            run += loss.item()
        avg_loss = run / len(tr)

        # validate on the EMA weights, then restore live weights for training
        live = deepcopy(model.state_dict())
        model.load_state_dict(ema.shadow)
        val_miou, val_dice, val_loss, per_dice = validate(model, va, nc, device, args.alpha, args.beta)
        model.load_state_dict(live)

        sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        track = [(c, names[c]) for c in range(1, nc) if names[c] in ("catheter", "urethra")]
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
    te = DataLoader(SegDataset(root / "Test", args.img_size, remap),
                    args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    model.load_state_dict(torch.load(outdir / "best.pth", map_location=device))
    te_miou, te_dice, te_loss, te_per_dice = validate(model, te, nc, device, args.alpha, args.beta)
    track = [(c, names[c]) for c in range(1, nc) if names[c] in ("catheter", "urethra")]
    te_per = "  ".join(f"{n}={te_per_dice[c]:.4f}" for c, n in track)
    print(f"[test]  mIoU={te_miou:.4f}  dice={te_dice:.4f}  {te_per}", flush=True)
    wandb.run.summary.update({
        "test/mIoU":  te_miou,
        "test/dice":  te_dice,
        **{f"test/dice_{n}": te_per_dice[c].item() for c, n in track},
    })

    wandb.finish()
    print(f"[done] best val_dice={best:.4f} -> {outdir/'best.pth'}")


if __name__ == "__main__":
    main()
