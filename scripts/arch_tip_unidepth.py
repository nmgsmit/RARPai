"""UniDepth V2 metric depth (ruler-calibrated mm) for already-cropped frames, stored at 1/4 res.

Frames are the 1340x1072 content crop from arch_tip_prep.py, so no auto-crop. Depth goes through the
frozen ruler-set calibration of metric_calib_proxy (scale mode, z_mm = z_m / s_only, no K -- same
inference as that fit; see unidepth_overlays.py). Output <out>/<stem>.npz: key 'depth' float16 mm at
335x268 (a 1340x1072 frame / 4), so ~2 GB of full-res float32 becomes ~0.5 GB.

    sbatch jobs/arch_tip_unidepth.sh
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    import torch
    from unidepth.models import UniDepthV2

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../data/processed/arch_tip_besthalf/images")
    ap.add_argument("--out", default="../data/processed/arch_tip_besthalf/depth")
    ap.add_argument("--calib", default="outputs/metric_calib_proxy/results.json")
    ap.add_argument("--model", default="lpiccinelli/unidepth-v2-vitl14")
    ap.add_argument("--down", type=int, default=4)
    args = ap.parse_args()

    s = json.loads(Path(args.calib).read_text())["unidepth_v2_vitl"]["calibration"]["s_only"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = UniDepthV2.from_pretrained(args.model).to("cuda").eval()
    imgs = [p for p in sorted(Path(args.dir).glob("*.jpg")) if not (out / f"{p.stem}.npz").exists()]
    print(f"{len(imgs)} frames to do, calibration s_only={s}")
    for k, p in enumerate(imgs):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            z = model.infer(torch.from_numpy(rgb.copy()).permute(2, 0, 1).cuda())["depth"][0, 0].float().cpu().numpy()
        h, w = z.shape
        d = cv2.resize(z / s, (w // args.down, h // args.down), interpolation=cv2.INTER_AREA)
        np.savez(out / f"{p.stem}.npz", depth=d.astype(np.float16))
        if k % 200 == 0:
            print(f"{k}/{len(imgs)} {p.stem}: median {np.median(d):.0f} mm", flush=True)
    print("done")


if __name__ == "__main__":
    main()
