"""Arch tip without the robot arm: the arm is VOID, as at annotation (it is never tracked).

Same data / targets (arc7) / test set as scripts/arch_tip_pure.py, surgical DINOv3. The robot arm = the palette's grey
"non-anatomical" class (id 5 in the packed training masks; compact id 4 of outputs/rarp_nick_fullres, which keeps
classes 1,2,4,5). The 7 test videos have no masks, so there the arm is PREDICTED.

  arm      rarp_nick_fullres on every training and test frame -> <PURE>/arm/{train,test}/<short>.npy (N,268,335) uint8
           (arm in >= 30% of a 4x4 cell). Prints how well the prediction matches the drawn masks on the training videos.
  feats    void = predicted arm (training frames: OR the drawn arm mask), grown by ARM_DILATE cells (~12 px) so edges
           and shadows go too. Arm pixels set to black (like the GUI) BEFORE the backbone -> feats/dinov3_surg_noarm, and
           the void fraction on the 32x40 grid -> arm/{split}/<short>_void32.npy.
  compare  arc7, 3 seeds: baseline (unmasked, = the 61 px model) | masked input | masked input + void channel + void
           cells may not vote for any point (offsets still reach under the arm). Errors split by whether the consensus
           tip itself lies under the (predicted) arm.
  all      arm -> feats -> compare

    sbatch jobs/arch_tip_noarm.sh
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_tip_data import FrameStore  # noqa: E402
from arch_tip_pure import (GH, GW, MEAN, PURE, ROOT, STD, TEST, backbone, fit, infer_with, norm_stats, prep,  # noqa: E402
                           test_frames, train_data)

ARM_ID, SEG_ARM = 5, 4          # packed palette id / rarp_nick_fullres compact id
QH, QW = 268, 335               # 1/4 of the 1072x1340 crop
ARM_DILATE = 3                  # cells at 1/4 res -> ~12 px
ARM = PURE / "arm"


def jobs():
    out = [("train", s, FrameStore(s, root=PURE), None) for s in sorted(p.name for p in PURE.iterdir() if (p / "frames.npy").exists())]
    for short, (fr, _, _) in test_frames().items():
        fs = FrameStore(short)
        out.append(("test", short, fs, [fs.pos[f] for f in fr]))
    return [(sp, s, fs, list(range(len(fs))) if ks is None else ks) for sp, s, fs, ks in out]


def quarter(full):
    return cv2.resize(full.astype(np.float32), (QW, QH), interpolation=cv2.INTER_AREA) >= 0.3


def void_of(split, fs, k, pred_row):
    q = pred_row > 0
    if split == "train":
        q |= quarter(fs.labels(k) == ARM_ID)
    return cv2.dilate(q.astype(np.uint8), np.ones((2 * ARM_DILATE + 1,) * 2, np.uint8)) > 0


def arm(args):
    import torch
    sys.path.insert(0, str(ROOT / "third_party" / "surgenet"))
    from metaformer import MetaFormerFPN
    sd = torch.load(ROOT / "outputs" / "rarp_nick_fullres" / "best.pth", map_location="cpu", weights_only=False)
    nc = sd["FPN.segmentation_head.0.bias"].shape[0]
    assert nc == 5, nc
    net = MetaFormerFPN(num_classes=nc, pretrained="ImageNet", pretrained_weights=None)
    net.load_state_dict(sd)
    net = net.cuda().eval()

    def feed(bgr):                                            # rarp_nick_fullres was trained at 1072x1340 -> /32 = 1088x1344
        img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (1344, 1088), interpolation=cv2.INTER_LINEAR)
        return ((img.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy()

    inter = union = 0
    for split, short, fs, ks in jobs():
        path = ARM / split / f"{short}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(path, "w+", np.uint8, (len(ks), QH, QW))
        for i in range(0, len(ks), 4):
            x = torch.from_numpy(np.stack([feed(fs.jpg(k)) for k in ks[i:i + 4]])).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                a = (net(x).argmax(1) == SEG_ARM).float()
            a = torch.nn.functional.interpolate(a[:, None], size=(QH, QW), mode="area")[:, 0] >= 0.3
            out[i:i + len(x)] = a.byte().cpu().numpy()
        out.flush()
        msg = f"{split} {short}: predicted arm {out.mean():.1%} of the frame"
        if split == "train":
            vi = vu = 0
            for i, k in enumerate(ks[::10]):
                g, p = quarter(fs.labels(k) == ARM_ID), out[i * 10] > 0
                vi, vu = vi + (g & p).sum(), vu + (g | p).sum()
            inter, union = inter + vi, union + vu
            msg += f", IoU with the drawn arm {vi / max(vu, 1):.2f}"
        print(msg, flush=True)
    print(f"predicted vs drawn arm on the training videos: IoU {inter / max(union, 1):.3f}", flush=True)


def feats(args):
    import torch
    f, C = backbone("dinov3_surg", "cuda")

    class DS(torch.utils.data.Dataset):
        def __init__(self, split, fs, ks, pred):
            self.split, self.fs, self.ks, self.pred = split, fs, ks, pred

        def __len__(self):
            return len(self.ks)

        def __getitem__(self, i):
            v = void_of(self.split, self.fs, self.ks[i], self.pred[i])
            img = self.fs.jpg(self.ks[i])
            img[cv2.resize(v.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0] = 0
            v32 = cv2.resize(v.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(prep(img)), torch.from_numpy(v32), torch.from_numpy(v)

    for split, short, fs, ks in jobs():
        pred = np.load(ARM / split / f"{short}.npy", mmap_mode="r")
        path = PURE / "feats" / "dinov3_surg_noarm" / split / f"{short}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.lib.format.open_memmap(path, "w+", np.float16, (len(ks), C, GH, GW))
        v32 = np.lib.format.open_memmap(ARM / split / f"{short}_void32.npy", "w+", np.float16, (len(ks), GH, GW))
        vq = np.lib.format.open_memmap(ARM / split / f"{short}_void.npy", "w+", np.uint8, (len(ks), QH, QW))
        k = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, v, q in torch.utils.data.DataLoader(DS(split, fs, ks, pred), batch_size=32, num_workers=16):
                arr[k:k + len(x)] = f(x.cuda(non_blocking=True)).float().cpu().numpy()
                v32[k:k + len(x)], vq[k:k + len(x)] = v.numpy(), q.numpy()
                k += len(x)
        for a in (arr, v32, vq):
            a.flush()
        print(f"{split} {short}: {k} frames, void {np.asarray(vq).mean():.1%}", flush=True)


def compare(args):
    import torch
    dev = "cuda"
    shorts, YT, ST = train_data()
    tests = test_frames()
    yt, st = torch.from_numpy(YT).to(dev), torch.from_numpy(ST).to(dev)
    occl = {}
    for s in TEST:                                             # is the consensus tip itself under the predicted arm?
        vq = np.load(ARM / "test" / f"{s}_void.npy", mmap_mode="r")
        xy = np.clip((tests[s][1] / 4).astype(int), 0, [QW - 1, QH - 1])
        occl[s] = np.array([vq[i, y, x] > 0 for i, (x, y) in enumerate(xy)])
    print("test frames with the consensus tip under the predicted arm: "
          + " ".join(f"{s} {o.mean():.0%}" for s, o in occl.items()), flush=True)

    configs = [("baseline (arm visible)", "dinov3_surg", False),
               ("arm blacked out", "dinov3_surg_noarm", False),
               ("arm blacked + void", "dinov3_surg_noarm", True)]
    rows = []
    for name, fb, void in configs:
        X = np.concatenate([np.load(PURE / "feats" / fb / "train" / f"{s}.npy") for s in shorts])
        XT = {s: np.asarray(np.load(PURE / "feats" / fb / "test" / f"{s}.npy", mmap_mode="r")) for s in TEST}
        if void:
            X = np.concatenate([X, np.concatenate([np.load(ARM / "train" / f"{s}_void32.npy") for s in shorts])[:, None]], 1)
            XT = {s: np.concatenate([XT[s], np.load(ARM / "test" / f"{s}_void32.npy")[:, None]], 1) for s in TEST}
        X = torch.from_numpy(X).to(dev)
        mu, sd = norm_stats(X)
        for seed in range(args.seeds):
            head = fit(X, yt, st, mu, sd, args.method, seed, args.steps, args.bs, dev, void=void)
            err = {s: np.linalg.norm(infer_with(head, mu, sd, XT[s], dev)[0][:, 0] - tests[s][1], axis=1) for s in TEST}
            E, O = np.concatenate(list(err.values())), np.concatenate([occl[s] for s in TEST])
            rows.append(dict(config=name, seed=seed, macro=float(np.mean([e.mean() for e in err.values()])),
                             median=float(np.median(E)), within50=float((E <= 50).mean()),
                             tip_hidden=float(E[O].mean()) if O.any() else None, tip_visible=float(E[~O].mean()),
                             per_video={s: float(e.mean()) for s, e in err.items()}))
            print(f"{name:<24} seed {seed}: macro {rows[-1]['macro']:6.1f}  " + " ".join(f"{s} {v:.0f}" for s, v in rows[-1]["per_video"].items()),
                  flush=True)
            del head
        del X
    out = ROOT / "outputs" / "arch_tip_pure"
    (out / f"noarm_{args.method}.json").write_text(json.dumps(dict(method=args.method, dilate_cells=ARM_DILATE, rows=rows,
                                                                    tip_hidden_share={s: float(o.mean()) for s, o in occl.items()}), indent=1))
    n_hidden = sum(int(o.sum()) for o in occl.values())
    print(f"\n=== robot arm as void: surgical DINOv3, target {args.method}, {args.seeds} seeds; tip error, 7 test videos ===")
    print(f"{'setup':<25}{'macro':>14}{'median':>8}{'<=50':>6}{'tip hidden':>12}{'visible':>9}  " + " ".join(f"{s:>8}" for s in TEST))
    for name, _, _ in configs:
        rs = [r for r in rows if r["config"] == name]
        mac = [r["macro"] for r in rs]
        hid = [r["tip_hidden"] for r in rs if r["tip_hidden"] is not None]
        print(f"{name:<25}{np.mean(mac):7.1f} +-{np.std(mac):4.1f}{np.mean([r['median'] for r in rs]):8.1f}"
              f"{np.mean([r['within50'] for r in rs]):6.0%}{(np.mean(hid) if hid else float('nan')):12.1f}"
              f"{np.mean([r['tip_visible'] for r in rs]):9.1f}  " + " ".join(f"{np.mean([r['per_video'][s] for r in rs]):8.1f}" for s in TEST))
    print(f"('tip hidden' = the {n_hidden} test frames whose consensus tip lies under the predicted arm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["arm", "feats", "compare", "all"])
    ap.add_argument("--method", default="arc7")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    for step in (("arm", "feats", "compare") if a.cmd == "all" else (a.cmd,)):
        {"arm": arm, "feats": feats, "compare": compare}[step](a)


if __name__ == "__main__":
    q = np.zeros((QH, QW), np.uint8)
    q[100, 100] = 1
    assert void_of("test", None, None, q).sum() == (2 * ARM_DILATE + 1) ** 2
    main()
