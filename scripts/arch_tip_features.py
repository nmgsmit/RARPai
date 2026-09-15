"""Per video in ../data/processed/arch_tip_all: frozen SurgeNet-RARP encoder features or the tool mask.

    --what feats  <short>/feats_s3.npy (N,320,32,40) + feats_s4.npy (N,512,16,20) float16: CAFormer-S18 stages 3/4
                  of the 512x640 GUI-blacked crop; encoder built and loaded as finetune_segmentation.py does
                  (../backbones/RARP_checkpoint_epoch0050_teacher.pth)
    --what tools  <short>/tools.npy (N,268,335) uint8: rarp_nick_fullres (kept classes 1,2,4,5, fed 1088x1344 as
                  trained) compact 3 catheter | 4 non-anatomical, >= 30% of a cell, at the 1/4 depth resolution
Both write <short>/stems.json, the frame order every per-video array follows.

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
from arch_tip_data import A  # noqa: E402
from metaformer import MetaFormerFPN  # noqa: E402

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
TOOL_IDS = (3, 4)
DEPTH_HW = (268, 335)


class Frames(torch.utils.data.Dataset):
    def __init__(self, paths, hw):
        self.p, self.hw = paths, hw

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.p[i])), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.hw[::-1], interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        return torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["feats", "tools"], required=True)
    ap.add_argument("--encoder", default="../backbones/RARP_checkpoint_epoch0050_teacher.pth")
    ap.add_argument("--seg", default="outputs/rarp_nick_fullres/best.pth")
    args = ap.parse_args()

    if args.what == "feats":
        from finetune_segmentation import load_encoder
        m = MetaFormerFPN(num_classes=1, pretrained="ImageNet", pretrained_weights=None)
        load_encoder(m, args.encoder)
        net, hw = m.metaformer.cuda().eval(), (512, 640)
    else:
        sd = torch.load(args.seg, map_location="cpu", weights_only=False)
        nc = sd["FPN.segmentation_head.0.bias"].shape[0]
        assert nc == 5, f"{args.seg}: expected bg + 4 kept classes (1,2,4,5), got {nc}"
        net = MetaFormerFPN(num_classes=nc, pretrained="ImageNet", pretrained_weights=None)
        net.load_state_dict(sd)
        net, hw = net.cuda().eval(), (1088, 1344)

    for d in sorted(p for p in A.iterdir() if (p / "images").is_dir()):
        paths = sorted(p for p in (d / "images").glob("*.jpg"))
        (d / "stems.json").write_text(json.dumps([p.stem for p in paths]))
        N = len(paths)
        dl = torch.utils.data.DataLoader(Frames(paths, hw), batch_size=32, num_workers=16)
        if args.what == "feats":
            s3 = np.lib.format.open_memmap(d / "feats_s3.npy", "w+", np.float16, (N, 320, 32, 40))
            s4 = np.lib.format.open_memmap(d / "feats_s4.npy", "w+", np.float16, (N, 512, 16, 20))
        else:
            tools = np.lib.format.open_memmap(d / "tools.npy", "w+", np.uint8, (N, *DEPTH_HW))
        k = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x in dl:
                x, n = x.cuda(non_blocking=True), len(x)
                if args.what == "feats":
                    f = net.forward_features(x)[1]
                    s3[k:k + n] = f[2].float().cpu().numpy()
                    s4[k:k + n] = f[3].float().cpu().numpy()
                else:
                    t = torch.isin(net(x).argmax(1), torch.tensor(TOOL_IDS, device="cuda")).float()
                    t = torch.nn.functional.interpolate(t[:, None], size=DEPTH_HW, mode="area")[:, 0]
                    tools[k:k + n] = (t >= 0.3).byte().cpu().numpy()
                k += n
        assert k == N
        print(f"{d.name}: {N} frames" + (f", tool px {tools.mean():.1%}" if args.what == "tools" else ""), flush=True)


if __name__ == "__main__":
    main()
