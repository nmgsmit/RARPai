"""
Finetune a MetaFormerFPN (CAFormer-S18 encoder) segmentation model from a
SurgeNet pretrained encoder checkpoint (e.g. SurgeNet-RARP teacher).

Data layout (same as SurgeNet):
    <data_root>/<split>/frames/*.png
    <data_root>/<split>/masks/*.png   # single-channel, pixel value = class id
    split in {Train, Validation, Test}

Run a no-data smoke test first on the cluster:
    python scripts/finetune_segmentation.py --smoke

Real run (encoder frozen, high-res, only catheter+urethra):
    python scripts/finetune_segmentation.py \
        --data-root ../data/RARPSurgenet/fold1 \
        --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth \
        --freeze-encoder
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset

load_dotenv()  # picks up .env from cwd (repo root)

# vendored from https://github.com/timjaspers0801/surgenet
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "surgenet"))
from metaformer import MetaFormerFPN  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SegDataset(Dataset):
    def __init__(self, split_dir: Path, img_size: int, augment: bool = False):
        self.frames = sorted((split_dir / "frames").glob("*.png"))
        self.masks = sorted((split_dir / "masks").glob("*.png"))
        assert len(self.frames) == len(self.masks) and self.frames, \
            f"frame/mask count mismatch or empty in {split_dir}"
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        img = Image.open(self.frames[i]).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        # ponytail: masks assumed indexed (1 channel, value=class id). If yours are
        # RGB color-coded, map colors->ids here instead of .convert("L").
        msk = Image.open(self.masks[i]).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST)
        img = np.array(img)
        msk = np.array(msk)

        # Very light augmentation (test set ~ train set, so keep it minimal)
        if self.augment:
            if random.random() < 0.5:                       # horizontal flip
                img = img[:, ::-1].copy()
                msk = msk[:, ::-1].copy()
            if random.random() < 0.2:                       # tiny brightness jitter
                f = 1.0 + random.uniform(-0.08, 0.08)
                img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)

        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = torch.from_numpy(msk).long()
        return x, y


def detect_num_classes(split_dir: Path) -> int:
    hi = 0
    for m in sorted((split_dir / "masks").glob("*.png")):
        hi = max(hi, int(np.array(Image.open(m).convert("L")).max()))
    return hi + 1


def load_encoder(model: MetaFormerFPN, ckpt_path: str):
    """Load SurgeNet encoder weights from a LOCAL file into model.metaformer."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("teacher", "model", "state_dict"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            ck = ck[key]
            break
    sd = {}
    for k, v in ck.items():
        k = k.replace("module.", "").replace("backbone.", "")
        if k.startswith("head."):
            continue
        sd[k] = v
    msg = model.metaformer.load_state_dict(sd, strict=False)
    print(f"[encoder] loaded {len(sd)} tensors | missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}")


def dice_ce_loss(logits, target, num_classes, class_weights=None):
    """Weighted Dice + CE. Simple, stable, works well."""
    if class_weights is not None:
        ce = F.cross_entropy(logits, target, weight=class_weights)
    else:
        ce = F.cross_entropy(logits, target)
    probs = logits.softmax(1)
    oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * oh).sum(dims)
    dice = (2 * inter + 1.0) / (probs.sum(dims) + oh.sum(dims) + 1.0)
    if class_weights is not None:
        dice = (dice * class_weights).sum() / class_weights.sum()
    else:
        dice = dice.mean()
    return ce + (1.0 - dice)


@torch.no_grad()
def validate(model, loader, num_classes, device, class_weights=None):
    inter = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    dice_inter = torch.zeros(num_classes)
    dice_denom = torch.zeros(num_classes)
    total_loss = 0.0
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += dice_ce_loss(logits, y, num_classes, class_weights=class_weights).item()
        pred = logits.argmax(1).cpu()
        y_cpu = y.cpu()
        for c in range(num_classes):
            p, t = pred == c, y_cpu == c
            inter[c] += (p & t).sum()
            union[c] += (p | t).sum()
            dice_inter[c] += (p & t).sum()
            dice_denom[c] += p.sum() + t.sum()
    # Compute metrics only on classes with non-zero weight in class_weights
    if class_weights is not None:
        w_cpu = class_weights.cpu() if class_weights.device.type == 'cuda' else class_weights
        present = (union > 0) & (w_cpu > 0)
    else:
        present = union > 0
    if present.sum() > 0:
        miou = (inter / union.clamp(min=1))[present].mean().item()
        dice = (2 * dice_inter / dice_denom.clamp(min=1))[present].mean().item()
    else:
        miou, dice = 0.0, 0.0
    val_loss = total_loss / len(loader)
    return miou, dice, val_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/RARPSurgenet/fold1")
    ap.add_argument("--encoder-ckpt",
                    default="../backbones/RARP_checkpoint_epoch0050_teacher.pth")
    ap.add_argument("--out", default="outputs/rarp_finetune")
    ap.add_argument("--run-name", default=None, help="wandb run name")
    ap.add_argument("--num-classes", type=int, default=0, help="0 = auto-detect")
    ap.add_argument("--img-size", type=int, default=768, help="higher = better for thin structures")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3, help="decoder/head LR")
    ap.add_argument("--encoder-lr", type=float, default=1e-6,
                    help="encoder LR (very low; ignored if --freeze-encoder)")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="freeze pretrained encoder, train decoder only")
    ap.add_argument("--no-augment", action="store_true", help="disable light augmentation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="no-data build+step self-test")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        m = MetaFormerFPN(num_classes=12, pretrained="ImageNet", pretrained_weights=None).to(device)
        x = torch.randn(2, 3, args.img_size, args.img_size, device=device)
        y = torch.randint(0, 12, (2, args.img_size, args.img_size), device=device)
        out = m(x)
        assert out.shape[-2:] == x.shape[-2:], f"output {out.shape} != input HxW"
        dice_ce_loss(out, y, 12).backward()
        print(f"[smoke] ok | out={tuple(out.shape)} device={device}")
        return

    root = Path(args.data_root)
    nc = args.num_classes or detect_num_classes(root / "Train")
    print(f"[setup] num_classes={nc} img_size={args.img_size} device={device}")

    # Only optimize catheter (class 1) and urethra (class 3), ignore others
    class_weights = torch.ones(nc)
    class_weights[[0, 2, 4]] = 0.0  # zero out background, class 2, class 4
    class_weights = class_weights / class_weights.sum()  # normalize
    class_weights = class_weights.to(device)
    print(f"[classes] optimizing only: 1=catheter, 3=urethra | weights={class_weights.tolist()}")

    import wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "rarp-segmentation"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=args.run_name,
        config=dict(
            num_classes=nc, img_size=args.img_size, epochs=args.epochs,
            warmup_epochs=args.warmup_epochs, batch_size=args.batch_size,
            lr=args.lr, encoder_lr=args.encoder_lr, freeze_encoder=args.freeze_encoder,
            augment=not args.no_augment, loss="focal_tversky+ce",
            encoder_ckpt=args.encoder_ckpt, data_root=str(root),
        ),
    )

    tr = DataLoader(SegDataset(root / "Train", args.img_size, augment=not args.no_augment),
                    args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)
    va = DataLoader(SegDataset(root / "Validation", args.img_size),
                    args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = MetaFormerFPN(num_classes=nc, pretrained="ImageNet", pretrained_weights=None).to(device)
    load_encoder(model, args.encoder_ckpt)

    # Discriminative LR: encoder pretrained on RARP -> freeze or tiny LR; decoder is random -> higher LR
    if args.freeze_encoder:
        for p in model.metaformer.parameters():
            p.requires_grad = False
        param_groups = [{"params": model.FPN.parameters(), "lr": args.lr}]
        print("[optim] encoder FROZEN, training decoder only")
    else:
        param_groups = [
            {"params": model.metaformer.parameters(), "lr": args.encoder_lr},
            {"params": model.FPN.parameters(), "lr": args.lr},
        ]
        print(f"[optim] encoder_lr={args.encoder_lr} head_lr={args.lr}")
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    # Linear warmup -> cosine decay
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs - args.warmup_epochs)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[args.warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
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
                loss = dice_ce_loss(model(x), y, nc, class_weights=class_weights)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        avg_loss = run / len(tr)
        val_miou, val_dice, val_loss = validate(model, va, nc, device, class_weights=class_weights)
        print(f"epoch {ep+1}/{args.epochs}  train_loss={avg_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  val_dice={val_dice:.4f}", flush=True)
        wandb.log({
            "train/loss": avg_loss, "val/loss": val_loss,
            "val/mIoU": val_miou, "val/dice": val_dice,
            "lr": opt.param_groups[-1]["lr"], "epoch": ep + 1,
        })
        # Save best by DICE (the metric we want to maximize)
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), outdir / "best.pth")
            wandb.run.summary["best_val_dice"] = best
            wandb.run.summary["best_val_mIoU"] = val_miou

    wandb.finish()
    print(f"[done] best val_dice={best:.4f} -> {outdir/'best.pth'}")


if __name__ == "__main__":
    main()
