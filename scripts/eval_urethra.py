"""
Urethra-focused evaluation. Per-class dice tells you nothing about WHICH way a mask
is wrong, and the three failures that matter here are all invisible in it:

  U->P   GT urethra pixels predicted prostate      (urethra marked as prostate)
  P->U   GT prostate pixels predicted urethra      (prostate marked as urethra)
  leak   predicted urethra landing anywhere that is NOT GT urethra, as a fraction
         of all predicted urethra = 1 - precision  ("no urethra signal elsewhere")

A model can hold urethra dice flat while trading recall for leak, so all four
numbers are reported together. Splits/remap/loader are imported from the trainer,
never re-implemented: a split that differs from the run's own is a silent lie.

    python scripts/eval_urethra.py --checkpoint outputs/rarp_nick_dice/best.pth \
        --keep-classes 1,2,4,5 --img-size 512
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "third_party" / "surgenet"))
_spec = importlib.util.spec_from_file_location("ft", _HERE / "finetune_seg_tversky.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)
from metaformer import MetaFormerFPN  # noqa: E402


def stray_analysis(model, loader, u, device, halo_px=6):
    """Split the urethra leak into HALO and STRAY, because they mean opposite things.

    A 21% leak that is a few pixels of boundary slop around an otherwise correct mask
    is annotation-grade disagreement. The same 21% spread over blobs on unrelated
    tissue is the thing we actually forbid ("no urethra signal other than on the
    urethra"). Precision cannot distinguish them; connected components can.

      stray = predicted-urethra pixels in a component with ZERO overlap of GT urethra
      halo  = the rest of the leak: wrong pixels attached to a component that did
              find real urethra (split by distance from GT)
    """
    from scipy import ndimage
    tot = dict(pred=0, hit=0, stray=0, halo_near=0, halo_far=0, stray_blobs=0, frames=0)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = (model(x.to(device)).argmax(1).cpu().numpy() == u)
            gt = (y.numpy() == u)
            for pm, gm in zip(pred, gt):
                lab, n = ndimage.label(pm)
                near = ndimage.binary_dilation(gm, iterations=halo_px) if gm.any() else gm
                tot["frames"] += 1
                tot["pred"] += int(pm.sum())
                tot["hit"] += int((pm & gm).sum())
                for k in range(1, n + 1):
                    comp = lab == k
                    if (comp & gm).any():                     # component found real urethra
                        wrong = comp & ~gm
                        tot["halo_near"] += int((wrong & near).sum())
                        tot["halo_far"] += int((wrong & ~near).sum())
                    else:                                     # component touches no GT at all
                        tot["stray"] += int(comp.sum())
                        tot["stray_blobs"] += 1
    return tot


def confusion(model, loader, nc, device):
    """cm[t, p] = pixels of true class t predicted as p."""
    cm = torch.zeros(nc, nc, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().view(-1)
            t = y.view(-1)
            cm += torch.bincount(t * nc + pred, minlength=nc * nc).reshape(nc, nc)
    return cm


def report(cm, names):
    tp = cm.diag().double()
    gt, pr = cm.sum(1).double(), cm.sum(0).double()
    dice = (2 * tp / (gt + pr).clamp(min=1)).numpy()
    rec = (tp / gt.clamp(min=1)).numpy()
    prec = (tp / pr.clamp(min=1)).numpy()
    print(f"  {'class':22s} {'dice':>7s} {'recall':>8s} {'prec':>8s} {'GT px%':>8s}")
    tot = gt.sum().item()
    for c in range(len(names)):
        print(f"  {names[c]:22s} {dice[c]:7.4f} {rec[c]:8.4f} {prec[c]:8.4f} "
              f"{100*gt[c].item()/tot:7.2f}%")
    return dice, rec, prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clip-root", default="../data/processed/Segmentation/Nick")
    ap.add_argument("--keep-classes", default="1,2,4,5")
    ap.add_argument("--label-scheme", default="nick")
    ap.add_argument("--split", default="Test", choices=("Test", "Validation", "Train"))
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42, help="MUST match the run's --seed")
    ap.add_argument("--no-stray", action="store_true", help="skip the connected-component pass")
    args = ap.parse_args()

    scheme = ft.SCHEMES[args.label_scheme]
    keep = [int(c) for c in args.keep_classes.split(",")]
    names = ["background"] + [scheme[c] for c in keep]
    nc = len(keep) + 1
    size_hw = ((ft.round32(args.height), ft.round32(args.width))
               if args.height and args.width else (args.img_size, args.img_size))

    splits, _ = ft.clip_pairs(Path(args.clip_root), args.seed)
    ld = DataLoader(ft.SegDataset(splits[args.split], size_hw, ft.build_remap(keep)),
                    args.batch_size, shuffle=False, num_workers=args.workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    got = sd["FPN.segmentation_head.0.bias"].shape[0]
    assert got == nc, f"checkpoint has {got} classes, --keep-classes implies {nc}"
    model = MetaFormerFPN(num_classes=nc, pretrained="ImageNet", pretrained_weights=None)
    model.load_state_dict(sd)
    model.to(device)

    print(f"[eval] {args.checkpoint}")
    print(f"[eval] split={args.split} n={len(ld.dataset)} feed={size_hw} device={device}")
    cm = confusion(model, ld, nc, device)
    dice, rec, prec = report(cm, names)

    u = names.index(scheme[1]) if scheme[1] in names else None   # urethra compact id
    p = names.index("prostate") if "prostate" in names else None
    assert u is not None, "urethra is not among the kept classes"
    gt_u = cm[u].sum().item()
    pred_u = cm[:, u].sum().item()
    print("\n[urethra] the numbers that decide this model")
    print(f"  dice                    {dice[u]:.4f}")
    if p is not None:
        print(f"  U->P  GT urethra called prostate   {100*cm[u,p].item()/max(gt_u,1):6.2f}%  of GT urethra")
        print(f"  P->U  GT prostate called urethra   {100*cm[p,u].item()/max(cm[p].sum().item(),1):6.2f}%  of GT prostate")
    leak = 1 - prec[u]
    print(f"  leak  predicted urethra NOT on urethra {100*leak:6.2f}%  of predicted urethra")
    if p is not None and pred_u:
        print(f"        ...of which prostate            {100*cm[p,u].item()/pred_u:6.2f}%")
        other = pred_u - cm[u, u].item() - cm[p, u].item()
        print(f"        ...of which bg/other            {100*other/pred_u:6.2f}%")
    if not args.no_stray:
        t = stray_analysis(model, ld, u, device)
        pr = max(t["pred"], 1)
        print("")
        print("[leak anatomy] where predicted urethra actually lands")
        print(f"  on GT urethra                {100*t['hit']/pr:6.2f}%")
        print(f"  halo  <=6px of GT urethra    {100*t['halo_near']/pr:6.2f}%   (boundary slop)")
        print(f"  halo  >6px, same blob        {100*t['halo_far']/pr:6.2f}%   (over-extension)")
        print(f"  STRAY blob, no GT overlap    {100*t['stray']/pr:6.2f}%   <-- the constraint")
        print(f"  stray blobs                  {t['stray_blobs']} over {t['frames']} frames "
              f"({t['stray_blobs']/max(t['frames'],1):.2f}/frame)")
        print(f"  STRAYPCT {100*t['stray']/pr:.3f}")

    # one line a loop can grep and rank on
    print(f"\nSCORE dice={dice[u]:.4f} leak={leak:.4f} "
          f"u2p={cm[u,p].item()/max(gt_u,1):.4f} p2u={cm[p,u].item()/max(cm[p].sum().item(),1):.4f} "
          f"{args.checkpoint}")


if __name__ == "__main__":
    main()
