"""Per packed video in ../data/processed/arch_tip_all (arch_tip_data.py FrameStore): features, tool mask or depth.

    --what feats  <short>/feats_s3.npy (N,320,32,40) + feats_s4.npy (N,512,16,20) float16: CAFormer-S18 stages 3/4
                  of the 512x640 GUI-blacked crop; frozen ../backbones/RARP_checkpoint_epoch0050_teacher.pth in the
                  SurgeNet (ReLU) variant, so every encoder key loads
    --what tools  <short>/tools.npy (N,268,335) uint8: rarp_nick_fullres (kept classes 1,2,4,5, fed 1088x1344 as
                  trained) compact 3 catheter | 4 non-anatomical, >= 30% of a cell, at the 1/4 depth resolution
    --what depth  <short>/depth.npy (N,268,335) float16 mm: UniDepthV2 ViT-L through the frozen ruler calibration
                  (scale mode, no K), exactly as arch_tip_unidepth.py, but one array per video, not one file per frame
Row k of every array = row k of the video's frames.npy.

    sbatch jobs/arch_tip_features.sh --what feats
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "surgenet"))
from arch_tip_data import A, FrameStore  # noqa: E402
from metaformer import MetaFormerFPN, variant_for  # noqa: E402

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
TOOL_IDS = (3, 4)
DEPTH_HW = (268, 335)


class Frames(torch.utils.data.Dataset):
    """hw=None: raw uint8 RGB at full res (UniDepth); else ImageNet-normalised float at hw."""

    def __init__(self, store, hw):
        self.s, self.hw = store, hw

    def __len__(self):
        return len(self.s)

    def __getitem__(self, k):
        img = cv2.cvtColor(self.s.jpg(k), cv2.COLOR_BGR2RGB)
        if self.hw is None:
            return torch.from_numpy(img.transpose(2, 0, 1).copy())
        img = cv2.resize(img, self.hw[::-1], interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        return torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["feats", "tools", "depth"], required=True)
    ap.add_argument("--encoder", default="../backbones/RARP_checkpoint_epoch0050_teacher.pth")
    ap.add_argument("--seg", default="outputs/rarp_nick_fullres/best.pth")
    ap.add_argument("--calib", default="outputs/metric_calib_proxy/results.json")
    ap.add_argument("--unidepth", default="lpiccinelli/unidepth-v2-vitl14")
    args = ap.parse_args()

    if args.what == "feats":
        from finetune_segmentation import load_encoder
        # "SurgeNet" = the ReLU CAFormer the teacher was trained as: all keys load (load_encoder asserts it).
        m = MetaFormerFPN(num_classes=1, pretrained="SurgeNet", pretrained_weights=None)
        load_encoder(m, args.encoder)
        net, hw, bs = m.metaformer.cuda().eval(), (512, 640), 32
    elif args.what == "tools":
        sd = torch.load(args.seg, map_location="cpu", weights_only=False)
        nc = sd["FPN.segmentation_head.0.bias"].shape[0]
        assert nc == 5, f"{args.seg}: expected bg + 4 kept classes (1,2,4,5), got {nc}"
        net = MetaFormerFPN(num_classes=nc, pretrained=variant_for(sd), pretrained_weights=None)
        net.load_state_dict(sd)
        net, hw, bs = net.cuda().eval(), (1088, 1344), 4      # full-res FPN: batch 32 asked for 39 GB more than the H100 had
    else:
        from unidepth.models import UniDepthV2
        s_only = json.loads(Path(args.calib).read_text())["unidepth_v2_vitl"]["calibration"]["s_only"]
        net, hw, bs = UniDepthV2.from_pretrained(args.unidepth).cuda().eval(), None, 1

    for d in sorted(p for p in A.iterdir() if (p / "frames.npy").exists()):
        store = FrameStore(d.name)
        N = len(store)
        dl = torch.utils.data.DataLoader(Frames(store, hw), batch_size=bs, num_workers=16)
        if args.what == "feats":
            s3 = np.lib.format.open_memmap(d / "feats_s3.npy", "w+", np.float16, (N, 320, 32, 40))
            s4 = np.lib.format.open_memmap(d / "feats_s4.npy", "w+", np.float16, (N, 512, 16, 20))
        else:
            out = np.lib.format.open_memmap(d / f"{args.what}.npy", "w+", np.uint8 if args.what == "tools" else np.float16,
                                            (N, *DEPTH_HW))
        k = 0
        with torch.no_grad():
            for x in dl:
                n = len(x)
                if args.what == "depth":                  # no autocast: same numerics as the calibration fit
                    z = net.infer(x[0].cuda())["depth"][0, 0].float().cpu().numpy()
                    out[k] = cv2.resize(z / s_only, DEPTH_HW[::-1], interpolation=cv2.INTER_AREA)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x = x.cuda(non_blocking=True)
                        if args.what == "feats":
                            f = net.forward_features(x)[1]
                            s3[k:k + n] = f[2].float().cpu().numpy()
                            s4[k:k + n] = f[3].float().cpu().numpy()
                        else:
                            t = torch.isin(net(x).argmax(1), torch.tensor(TOOL_IDS, device="cuda")).float()
                            t = torch.nn.functional.interpolate(t[:, None], size=DEPTH_HW, mode="area")[:, 0]
                            out[k:k + n] = (t >= 0.3).byte().cpu().numpy()
                k += n
        assert k == N
        extra = ""
        if args.what == "tools":
            extra = f", tool px {out.mean():.1%}"
        elif args.what == "depth":
            extra = f", median {np.median(out[::50]):.0f} mm"
        print(f"{d.name}: {N} frames{extra}", flush=True)


if __name__ == "__main__":
    main()
