"""
Finetune variant: lower LR (5e-5), hflip-only aug — middle ground between
surgenet-paper (1e-5, D4) and higherlr-plateau (1e-4, hflip).

higherlr-plateau shows val_loss increasing after epoch 4-5 (overfitting).
This run tests if 5e-5 + hflip avoids that.

Real run:
    python scripts/finetune_seg_lowerlr.py \
        --data-root ../data/RARPSurgenet/fold1 \
        --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth
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

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "surgenet"))
from metaformer import MetaFormerFPN  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SegDataset(Dataset):
    def __init__(self, split_dir: Path, img_size: int, augment: bool = False):
        self.frames = sorted((split_dir / "frames").glob("*.png"))
        self.masks  = sorted((split_dir / "masks").glob("*.png"))
        assert len(self.frames) == len(self.masks) and self.frames, \
            f"frame/mask count mismatch or empty in {split_dir}"
        self.img_size = img_size
        self.augment  = augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        img = Image.open(self.frames[i]).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BICUBIC)
        msk = Image.open(self.masks[i]).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST)
        img = np.array(img)
        msk = np.array(msk)
        if self.augment:
            # hflip only — vflip + rot90 seem to hurt surgical anatomy
            if random.random() < 0.5:
                img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
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


def dice_ce_loss(logits, target, num_classes):
    ce    = F.cross_entropy(logits, target)
    probs = logits.softmax(1)
    oh    = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims  = (0, 2, 3)
    inter = (probs * oh).sum(dims)
    dice  = (2 * inter + 1.0) / (probs.sum(dims) + oh.sum(dims) + 1.0)
    return ce + (1.0 - dice.mean())


@torch.no_grad()
def validate(model, loader, num_classes, device):
    inter      = torch.zeros(num_classes)
    union      = torch.zeros(num_classes)
    dice_inter = torch.zeros(num_classes)
    dice_denom = torch.zeros(num_classes)
    total_loss = 0.0
    model.eval()
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        logits   = model(x)
        total_loss += dice_ce_loss(logits, y, num_classes).item()
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
    ap.add_argument("--out",          default="outputs/rarp_lowerlr")
    ap.add_argument("--run-name",     default="lowerlr-plateau")
    ap.add_argument("--num-classes",  type=int,   default=0)
    ap.add_argument("--img-size",     type=int,   default=512)
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch-size",   type=int,   default=8)
    ap.add_argument("--lr",           type=float, default=5e-5)
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
        dice_ce_loss(out, y, 12).backward()
        print(f"[smoke] ok | out={tuple(out.shape)} device={device}")
        return

    root = Path(args.data_root)
    nc   = args.num_classes or detect_num_classes(root / "Train")
    print(f"[setup] num_classes={nc} img_size={args.img_size} device={device}")

    import wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "rarp-segmentation"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=args.run_name,
        config=dict(
            num_classes=nc, img_size=args.img_size, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            augment=not args.no_augment, loss="dice+ce", seed=args.seed,
            encoder_ckpt=args.encoder_ckpt, data_root=str(root),
        ),
    )

    g  = torch.Generator()
    g.manual_seed(args.seed)
    tr = DataLoader(SegDataset(root / "Train", args.img_size, augment=not args.no_augment),
                    args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, generator=g)
    va = DataLoader(SegDataset(root / "Validation", args.img_size),
                    args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = MetaFormerFPN(num_classes=nc, pretrained="ImageNet", pretrained_weights=None).to(device)
    load_encoder(model, args.encoder_ckpt)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    print(f"[optim] AdamW lr={args.lr} ReduceLROnPlateau(patience=3, factor=0.5)")
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
                loss = dice_ce_loss(model(x), y, nc)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        avg_loss = run / len(tr)
        val_miou, val_dice, val_loss, per_dice = validate(model, va, nc, device)
        sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {ep+1}/{args.epochs}  train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_mIoU={val_miou:.4f}  val_dice={val_dice:.4f}  "
              f"catheter={per_dice[1]:.4f}  urethra={per_dice[3]:.4f}", flush=True)
        wandb.log({"train/loss": avg_loss, "val/loss": val_loss,
                   "val/mIoU": val_miou, "val/dice": val_dice,
                   "val/dice_catheter": per_dice[1].item(),
                   "val/dice_urethra":  per_dice[3].item(),
                   "lr": lr, "epoch": ep + 1})
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), outdir / "best.pth")
            wandb.run.summary["best_val_dice"] = best
            wandb.run.summary["best_val_mIoU"] = val_miou

    # final test-set eval using best checkpoint
    te = DataLoader(SegDataset(root / "Test", args.img_size),
                    args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    model.load_state_dict(torch.load(outdir / "best.pth", map_location=device))
    te_miou, te_dice, te_loss, te_per_dice = validate(model, te, nc, device)
    print(f"[test]  mIoU={te_miou:.4f}  dice={te_dice:.4f}  "
          f"catheter={te_per_dice[1]:.4f}  urethra={te_per_dice[3]:.4f}", flush=True)
    wandb.run.summary.update({
        "test/mIoU":          te_miou,
        "test/dice":          te_dice,
        "test/dice_catheter": te_per_dice[1].item(),
        "test/dice_urethra":  te_per_dice[3].item(),
    })

    wandb.finish()
    print(f"[done] best val_dice={best:.4f} -> {outdir/'best.pth'}")


if __name__ == "__main__":
    main()
