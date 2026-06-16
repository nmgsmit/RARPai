"""
Finetune a MetaFormerFPN (CAFormer-S18 encoder) segmentation model from a
SurgeNet pretrained encoder checkpoint (e.g. SurgeNet-RARP teacher).

Data layout (same as SurgeNet):
    <data_root>/<split>/frames/*.png
    <data_root>/<split>/masks/*.png   # single-channel, pixel value = class id
    split in {Train, Validation, Test}

Run a no-data smoke test first on the cluster:
    python scripts/finetune_segmentation.py --smoke

Real run:
    python scripts/finetune_segmentation.py \
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

load_dotenv()  # picks up .env from cwd (repo root)

# vendored from https://github.com/timjaspers0801/surgenet
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "surgenet"))
from metaformer import MetaFormerFPN  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SegDataset(Dataset):
    def __init__(self, split_dir: Path, img_size: int):
        self.frames = sorted((split_dir / "frames").glob("*.png"))
        self.masks = sorted((split_dir / "masks").glob("*.png"))
        assert len(self.frames) == len(self.masks) and self.frames, \
            f"frame/mask count mismatch or empty in {split_dir}"
        self.img_size = img_size

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        img = Image.open(self.frames[i]).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        # ponytail: masks assumed indexed (1 channel, value=class id). If yours are
        # RGB color-coded, map colors->ids here instead of .convert("L").
        msk = Image.open(self.masks[i]).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = torch.from_numpy(np.array(msk)).long()
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


def dice_ce_loss(logits, target, num_classes):
    ce = F.cross_entropy(logits, target)
    probs = logits.softmax(1)
    oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * oh).sum(dims)
    dice = (2 * inter + 1.0) / (probs.sum(dims) + oh.sum(dims) + 1.0)
    return ce + (1.0 - dice.mean())


@torch.no_grad()
def validate(model, loader, num_classes, device):
    inter = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    dice_inter = torch.zeros(num_classes)
    dice_denom = torch.zeros(num_classes)
    total_loss = 0.0
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += dice_ce_loss(logits, y, num_classes).item()
        pred = logits.argmax(1).cpu()
        y_cpu = y.cpu()
        for c in range(num_classes):
            p, t = pred == c, y_cpu == c
            inter[c] += (p & t).sum()
            union[c] += (p | t).sum()
            dice_inter[c] += (p & t).sum()
            dice_denom[c] += p.sum() + t.sum()
    present = union > 0
    val_miou = (inter / union.clamp(min=1))[present].mean().item()
    val_dice = (2 * dice_inter / dice_denom.clamp(min=1))[present].mean().item()
    val_loss = total_loss / len(loader)
    return val_miou, val_dice, val_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/RARPSurgenet/fold1")
    ap.add_argument("--encoder-ckpt",
                    default="../backbones/RARP_checkpoint_epoch0050_teacher.pth")
    ap.add_argument("--out", default="outputs/rarp_finetune")
    ap.add_argument("--run-name", default=None, help="wandb run name")
    ap.add_argument("--num-classes", type=int, default=0, help="0 = auto-detect")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="no-data build+step self-test")
    args = ap.parse_args()
    seed_everything(args.seed)
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

    import wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "rarp-segmentation"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=args.run_name,
        config=dict(
            num_classes=nc,
            img_size=args.img_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            encoder_ckpt=args.encoder_ckpt,
            data_root=str(root),
        ),
    )

    g = torch.Generator()
    g.manual_seed(args.seed)
    tr = DataLoader(SegDataset(root / "Train", args.img_size), args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True,
                    drop_last=True, generator=g)
    va = DataLoader(SegDataset(root / "Validation", args.img_size), args.batch_size,
                    shuffle=False, num_workers=args.workers, pin_memory=True)

    model = MetaFormerFPN(num_classes=nc, pretrained="ImageNet", pretrained_weights=None).to(device)
    load_encoder(model, args.encoder_ckpt)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
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
                loss = dice_ce_loss(model(x), y, nc)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        avg_loss = run / len(tr)
        val_miou, val_dice, val_loss = validate(model, va, nc, device)
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {ep+1}/{args.epochs}  loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_mIoU={val_miou:.4f}  val_dice={val_dice:.4f}", flush=True)
        wandb.log({"train/loss": avg_loss, "val/loss": val_loss,
                   "val/mIoU": val_miou, "val/dice": val_dice,
                   "lr": lr, "epoch": ep + 1})
        if val_miou > best:
            best = val_miou
            torch.save(model.state_dict(), outdir / "best.pth")
            wandb.run.summary["best_val_mIoU"] = best
            wandb.run.summary["best_val_dice"] = val_dice

    wandb.finish()
    print(f"[done] best val_mIoU={best:.4f} -> {outdir/'best.pth'}")


if __name__ == "__main__":
    main()
