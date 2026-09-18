"""Arch tip from the "Pure Arch" set: train on the new videos (arch + anatomy masks), test on the 7 three-annotator videos.

TRAIN: every video in E:/Geknipte Videos batch 2/20archesfortesting/Pure Arch except the test videos (46867a8e is in
there too), K=500 frames per video with both an arch and a mask, evenly spaced -> every video weighs the same.
Masks are the labelling tool's palette (transfer_atlas_mod/gui/cutie/utils/palette.py): 1 urethra, 2 prostate,
3 dorsal venous plexus, 4 catheter, 5 non-anatomical, plus red = 6 (not in the palette; unnamed).
TEST: the 7 videos Nick + Veerle + Aron all annotated; per video, the half of its 3-annotator frames with the best
agreement (agree_px <= that video's median), scored against the 3-way consensus tip; macro = mean over the 7 videos.

Methods, each on 3 frozen backbones (surgenet_rarp / surgenetxl CAFormer-S18 stage 3 + upsampled 4 = 832 ch, surgical
DINOv3 ViT-L/16 last layer = 1024 ch; all on the 32x40 grid of a 512x640 feed):
  tip       vote head (softmax over cells of centre + offset -> tip and both arch ends; may leave the frame)
  tip+seg   same head, plus an anatomy head trained on the masks (7-way soft CE on the grid) as extra supervision
  seg->tip  anatomy head only; tip = top of the largest predicted urethra region + the median train offset
  prior     train-mean tip        human: each annotator vs the mean of the other two

    python scripts/arch_tip_pure.py pack --src "E:/Geknipte Videos batch 2/20archesfortesting/Pure Arch" --out <dir>
    sbatch jobs/arch_tip_pure_feats.sh      # GPU: backbone features for train + test frames
    sbatch jobs/arch_tip_pure_run.sh        # GPU: all methods x backbones x seeds -> outputs/arch_tip_pure
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_tip_data import A, FrameStore, append  # noqa: E402
from arch_tip_prep import oriented  # noqa: E402
from compare_arch_multi import apex_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PURE = ROOT.parent / "data" / "processed" / "arch_tip_pure"
TEST = ("46867a8e", "4db28d2e", "749c8234", "acd22d98", "cada5bef", "RARP_062", "RARP_063")
COLORS = {(255, 255, 0): 1, (255, 0, 255): 2, (0, 0, 255): 3, (0, 255, 0): 4, (128, 128, 128): 5, (255, 0, 0): 6}
NCLS = 7
GH, GW = 32, 40
CELL = 1340 / GW
BACKBONES = ("surgenet_rarp", "surgenetxl", "dinov3_surg")
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def even(frames, k):
    frames = sorted(frames)
    return [frames[i] for i in sorted(set(np.linspace(0, len(frames) - 1, min(k, len(frames))).round().astype(int)))]


def short_of(name):
    return name[:8]


# ------------------------------------------------------------------------------------------------ pack (local)
def colour_ids(bgr):
    """Palette colour mask -> uint8 class ids; any other colour -> 0 (returned as a fraction for reporting)."""
    key = bgr[..., 2].astype(np.int32) << 16 | bgr[..., 1].astype(np.int32) << 8 | bgr[..., 0].astype(np.int32)
    ids = np.zeros(key.shape, np.uint8)
    known = np.zeros(key.shape, bool)
    for (r, g, b), i in COLORS.items():
        sel = key == (r << 16 | g << 8 | b)
        ids[sel], known = i, known | sel
    return ids, float((~known & (key != 0)).mean())


def pack(src, out, k):
    out.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in Path(src).iterdir() if p.is_dir()):
        short = short_of(d.name)
        if short in TEST:
            print(f"{short}: test video, skipped")
            continue
        arch = {int(f): e for f, e in json.loads((d / "arches.json").read_text())["frames"].items()}
        both = sorted(set(arch) & {int(f[:7]) for f in os.listdir(d / "masks")})
        fr, index, labels, unknown = even(both, k), [], {}, []
        (out / short).mkdir(exist_ok=True)
        with open(out / short / "frames.bin", "wb") as fh:
            for f in fr:
                ids, u = colour_ids(cv2.imread(str(d / "masks" / f"{f:07d}.png")))
                unknown.append(u)
                append(fh, index, f, (d / "images" / f"{f:07d}.jpg").read_bytes(), cv2.imencode(".png", ids)[1].tobytes())
                left, right, _, _ = oriented(arch[f], np.zeros(2))
                labels[f] = dict(tip=apex_of(arch[f])[0].tolist(), left=left.tolist(), right=right.tolist())
        np.save(out / short / "frames.npy", np.array(index, np.int64))
        (out / short / "labels.json").write_text(json.dumps(dict(video=d.name, labels=labels)))
        print(f"{short}: {len(fr)} of {len(both)} arch+mask frames, off-palette px {np.mean(unknown):.4%}", flush=True)


# ------------------------------------------------------------------------------------------------ test frames
def test_frames():
    """{short: (frames, consensus tips, per-annotator tips)}: per video, the better-agreeing half of its 3-annotator frames."""
    rows = json.loads((A / "labels_all.json").read_text())["rows"]
    out = {}
    for short in TEST:
        rs = sorted((r for r in rows if r["short"] == short and r["n_annot"] == 3), key=lambda r: r["frame"])
        med = np.median([r["agree_px"] for r in rs])
        rs = [r for r in rs if r["agree_px"] <= med]
        out[short] = ([r["frame"] for r in rs], np.array([r["tip"] for r in rs], np.float32), [r["tips"] for r in rs])
    return out


# ------------------------------------------------------------------------------------------------ features (GPU)
def backbone(name, dev):
    import torch
    import torch.nn.functional as F
    if name in ("surgenet_rarp", "surgenetxl"):
        sys.path.insert(0, str(ROOT / "third_party" / "surgenet"))
        from finetune_segmentation import load_encoder
        from metaformer import MetaFormerFPN
        ckpt = {"surgenet_rarp": ROOT.parent / "backbones" / "RARP_checkpoint_epoch0050_teacher.pth",
                "surgenetxl": ROOT.parent / "backbones" / "SurgeNetXL" / "SurgeNetXL_checkpoint_epoch0050_teacher.pth"}[name]
        m = MetaFormerFPN(num_classes=1, pretrained="SurgeNet", pretrained_weights=None)
        load_encoder(m, str(ckpt))
        net = m.metaformer.to(dev).eval()

        def f(x):
            s = net.forward_features(x)[1]
            return torch.cat([s[2], F.interpolate(s[3], size=(GH, GW), mode="bilinear", align_corners=False)], 1)
        return f, 832
    sys.path.insert(0, os.path.expanduser("~/pylibs/dinov3"))
    from dinov3.models.vision_transformer import DinoVisionTransformer
    sd = torch.load(ROOT.parent / "backbones" / "SurgeNetXL" / "DINOv3_ViTl16_size336_SurgeNetXL.pth",
                    map_location="cpu", weights_only=False)
    sd = {k.replace(".gamma_1", ".ls1.gamma").replace(".gamma_2", ".ls2.gamma"): v for k, v in sd.items()}
    m = DinoVisionTransformer(img_size=336, patch_size=16, embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4.0,
                              layerscale_init=1e-5, n_storage_tokens=0, mask_k_bias=False, pos_embed_rope_base=100,
                              pos_embed_rope_normalize_coords="separate", pos_embed_rope_dtype="fp32")
    m.load_state_dict(sd, strict=True)
    m = m.to(dev).eval()
    return (lambda x: m.get_intermediate_layers(x, n=[23], reshape=True, norm=True)[0]), 1024


def feats(args):
    import torch

    class DS(torch.utils.data.Dataset):
        def __init__(self, fs, ks):
            self.fs, self.ks = fs, ks

        def __len__(self):
            return len(self.ks)

        def __getitem__(self, i):
            return torch.from_numpy(prep(self.fs.jpg(self.ks[i])))

    jobs = [("train", s, FrameStore(s, root=PURE), None) for s in sorted(p.name for p in PURE.iterdir() if (p / "frames.npy").exists())]
    for short, (fr, _, _) in test_frames().items():
        fs = FrameStore(short)
        jobs.append(("test", short, fs, [fs.pos[f] for f in fr]))
    for name in args.backbones:
        f, C = backbone(name, "cuda")
        for split, short, fs, ks in jobs:
            ks = list(range(len(fs))) if ks is None else ks
            path = PURE / "feats" / name / split / f"{short}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            arr = np.lib.format.open_memmap(path, "w+", np.float16, (len(ks), C, GH, GW))
            k = 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for x in torch.utils.data.DataLoader(DS(fs, ks), batch_size=32, num_workers=16):
                    arr[k:k + len(x)] = f(x.cuda(non_blocking=True)).float().cpu().numpy()
                    k += len(x)
            arr.flush()
            print(f"{name} {split} {short}: {k} frames", flush=True)


# ------------------------------------------------------------------------------------------------ model (GPU)
def make_head(cin, width=256):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.red = nn.Conv2d(cin, width, 1)

            def block(i, o, d):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d), nn.GroupNorm(8, o), nn.ReLU(inplace=True))
            self.body = nn.Sequential(block(width + 2, width, 1), block(width, width, 2), block(width, width, 4),
                                      block(width, width, 1))
            self.vote, self.seg = nn.Conv2d(width, 9, 1), nn.Conv2d(width, NCLS, 1)
            ys, xs = torch.meshgrid(torch.arange(GH), torch.arange(GW), indexing="ij")
            self.register_buffer("cx", ((xs + 0.5) * CELL).float().flatten())
            self.register_buffer("cy", ((ys + 0.5) * CELL).float().flatten())
            self.register_buffer("coords", torch.stack([xs / (GW - 1), ys / (GH - 1)]).float()[None] * 2 - 1)

        def forward(self, x):
            f = F.dropout2d(self.red(x), 0.1, self.training)
            h = self.body(torch.cat([f, self.coords.expand(len(x), -1, -1, -1)], 1))
            o = self.vote(h).flatten(2)
            w = o[:, :3].softmax(-1)
            pts = torch.stack([(w * (self.cx + o[:, 3:6] * CELL)).sum(-1), (w * (self.cy + o[:, 6:9] * CELL)).sum(-1)], -1)
            return pts, self.seg(h)
    return Head()


def seg_targets(short, n):
    """Soft 7-class targets on the 32x40 grid (area-downsampled one-hot of the packed masks), cached per video."""
    path = PURE / short / "seg32.npy"
    if not path.exists():
        fs = FrameStore(short, root=PURE)
        t = np.stack([np.stack([cv2.resize((fs.labels(k) == c).astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
                                for c in range(NCLS)]) for k in range(len(fs))]).astype(np.float16)
        np.save(path, t)
    t = np.load(path)
    assert len(t) == n
    return t


def urethra_tip(seg_logits):
    """Top of the largest predicted urethra region, in 1340x1072 px; None if no urethra."""
    import torch
    import torch.nn.functional as F
    p = F.interpolate(seg_logits[None].float(), size=(268, 335), mode="bilinear", align_corners=False)[0].softmax(0)[1]
    p = p.cpu().numpy()
    m = p > 0.5
    if m.sum() < 10:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    m = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(m)
    band = ys <= ys.min() + 2
    return np.array([xs[band].mean() * 4 + 2, ys.min() * 4 + 2], np.float32)


def train_data():
    """Training videos, (N,3,2) tip/left/right targets and (N,7,32,40) soft anatomy targets, video by video."""
    shorts = sorted(p.name for p in PURE.iterdir() if (p / "frames.npy").exists())
    Y = {s: json.loads((PURE / s / "labels.json").read_text())["labels"] for s in shorts}
    fr = {s: FrameStore(s, root=PURE).frames for s in shorts}
    YT = np.concatenate([np.array([[Y[s][str(f)][k] for k in ("tip", "left", "right")] for f in fr[s]], np.float32)
                         for s in shorts])
    return shorts, YT, np.concatenate([seg_targets(s, len(fr[s])) for s in shorts])


def norm_stats(X):
    """Per-channel mean / std of an (N, C, H, W) float16 GPU tensor, in chunks."""
    m = sq = 0
    for c in X.split(256):
        c = c.float()
        m, sq = m + c.sum((0, 2, 3)), sq + (c * c).sum((0, 2, 3))
    n = X.shape[0] * GH * GW
    mu = (m / n).view(1, -1, 1, 1)
    return mu, (sq / n - mu.flatten() ** 2).clamp_min(1e-6).sqrt().view(1, -1, 1, 1)


def fit(X, yt, st, mu, sd, method, seed, steps, bs, dev):
    """tip: vote loss only; tip+seg: vote + anatomy loss; seg->tip: anatomy loss only."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    head = make_head(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    wv, ws = (1.0, 0.0) if method == "tip" else (1.0, 1.0) if method == "tip+seg" else (0.0, 1.0)
    pw = torch.tensor([1.0, 0.25, 0.25], device=dev)
    head.train()
    for _ in range(steps):
        b = torch.randint(0, len(X), (bs,), device=dev)
        pts, seg = head((X[b].float() - mu) / sd)
        lv = (F.smooth_l1_loss(pts / 100, yt[b] / 100, beta=0.2, reduction="none").sum(-1) * pw).sum(-1).mean()
        ls = -(st[b].float() * F.log_softmax(seg, 1)).sum(1).mean()
        loss = wv * lv + ws * ls
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return head.eval()


def infer_with(head, mu, sd, arr, dev, bs=256):
    """(N,3,2) tip/left/right px and (N,7,32,40) anatomy logits; arr = numpy/memmap or GPU tensor of features."""
    import torch
    P, S = [], []
    with torch.no_grad():
        for i in range(0, len(arr), bs):
            x = (arr[i:i + bs] if torch.is_tensor(arr) else torch.from_numpy(np.asarray(arr[i:i + bs]))).to(dev).float()
            p, s = head((x - mu) / sd)
            P.append(p.cpu().numpy())
            S.append(s)
    return np.concatenate(P), torch.cat(S)


def prep(bgr):
    """GUI-blacked 1340x1072 BGR crop -> normalised 3x512x640 float32 (every backbone's feed)."""
    img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (640, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return ((img - MEAN) / STD).transpose(2, 0, 1).copy()


def predict(args):
    """Every frame of one test video: train the method on the Pure Arch videos with each seed, predict, save per seed
    <out>/pred/<method>_<backbone>/s<seed>/<short>.npz (frames, tip, left, right, test_frames) for arch_tip_render."""
    import torch
    dev, bb, short = "cuda", args.backbones[0], args.video
    shorts, YT, ST = train_data()
    X = torch.from_numpy(np.concatenate([np.load(PURE / "feats" / bb / "train" / f"{s}.npy") for s in shorts])).to(dev)
    mu, sd = norm_stats(X)
    yt, st = torch.from_numpy(YT).to(dev), torch.from_numpy(ST).to(dev)
    f, _ = backbone(bb, dev)
    fs = FrameStore(short)
    XA = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(fs), 32):
            x = torch.from_numpy(np.stack([prep(fs.jpg(k)) for k in range(i, min(i + 32, len(fs)))])).to(dev)
            XA.append(f(x).half())
    XA = torch.cat(XA)
    tf, tips, _ = test_frames()[short]
    tk = [fs.pos[t] for t in tf]
    for seed in range(args.seeds):
        head = fit(X, yt, st, mu, sd, args.method, seed, args.steps, args.bs, dev)
        P, _ = infer_with(head, mu, sd, XA, dev)
        out = ROOT / "outputs" / "arch_tip_pure" / "pred" / f"{args.method}_{bb}" / f"s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{short}.npz", frames=fs.frames, tip=P[:, 0], left=P[:, 1], right=P[:, 2], test_frames=np.array(tf))
        print(f"{args.method} {bb} seed {seed}: {len(fs)} frames, test error {np.linalg.norm(P[tk, 0] - tips, axis=1).mean():.1f} px",
              flush=True)


def run(args):
    import torch
    dev = "cuda"
    train_shorts, YT, ST = train_data()
    tests = test_frames()
    prior = YT[:, 0].mean(0)
    rows = []

    def score(name, bb, seed, preds):
        per = {s: float(np.linalg.norm(preds[s] - tests[s][1], axis=1).mean()) for s in TEST}
        pooled = np.concatenate([np.linalg.norm(preds[s] - tests[s][1], axis=1) for s in TEST])
        rows.append(dict(method=name, backbone=bb, seed=seed, macro=float(np.mean(list(per.values()))),
                         median=float(np.median(pooled)), within50=float((pooled <= 50).mean()), per_video=per))
        print(f"{name:<9} {bb:<14} seed {seed}: macro {rows[-1]['macro']:6.1f}  median {rows[-1]['median']:6.1f}  "
              + " ".join(f"{s} {v:.0f}" for s, v in per.items()), flush=True)

    score("prior", "-", 0, {s: np.repeat(prior[None], len(tests[s][1]), 0) for s in TEST})
    hum = {s: np.mean([[np.linalg.norm(np.array(t[n]) - np.mean([t[m] for m in t if m != n], 0)) for n in t]
                       for t in tests[s][2]], 1) for s in TEST}
    rows.append(dict(method="human", backbone="-", seed=0, macro=float(np.mean([h.mean() for h in hum.values()])),
                     median=float(np.median(np.concatenate(list(hum.values())))),
                     within50=float((np.concatenate(list(hum.values())) <= 50).mean()),
                     per_video={s: float(h.mean()) for s, h in hum.items()}))

    for bb in args.backbones:
        X = torch.from_numpy(np.concatenate([np.load(PURE / "feats" / bb / "train" / f"{s}.npy") for s in train_shorts])).to(dev)
        XT = {s: np.load(PURE / "feats" / bb / "test" / f"{s}.npy", mmap_mode="r") for s in TEST}
        mu, sd = norm_stats(X)
        yt, st = torch.from_numpy(YT).to(dev), torch.from_numpy(ST).to(dev)
        for method, seed in itertools.product(("tip", "tip+seg", "seg->tip"), range(args.seeds)):
            head = fit(X, yt, st, mu, sd, method, seed, args.steps, args.bs, dev)

            def infer(arr):
                P, S = infer_with(head, mu, sd, arr, dev)
                return P[:, 0], S

            if method == "seg->tip":
                _, S = infer(X)                                   # train predictions -> median offset to the arch tip
                tr = [(urethra_tip(s), y) for s, y in zip(S, YT[:, 0])]
                off = np.median([y - t for t, y in tr if t is not None], 0)
                preds = {}
                for s in TEST:
                    _, S = infer(XT[s])
                    tips = [urethra_tip(z) for z in S]
                    preds[s] = np.array([t + off if t is not None else prior for t in tips], np.float32)
                print(f"   seg->tip {bb}: train urethra found {np.mean([t is not None for t, _ in tr]):.0%}, "
                      f"offset {off.round(0)}", flush=True)
            else:
                preds = {s: infer(XT[s])[0] for s in TEST}
            score(method, bb, seed, preds)
        del X

    out = ROOT / "outputs" / "arch_tip_pure"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(dict(train=train_shorts, test={s: len(tests[s][0]) for s in TEST},
                                                      rows=rows), indent=1))
    print(f"\n=== test: 7 three-annotator videos, better-agreeing half per video ({sum(len(t[0]) for t in tests.values())} "
          f"frames); train: {len(train_shorts)} videos x {len(YT) // len(train_shorts)} frames ===")
    print(f"{'method':<10}{'backbone':<15}{'macro':>14}{'median':>8}{'<=50':>6}  " + " ".join(f"{s:>8}" for s in TEST))
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], r["backbone"]), []).append(r)
    for (method, bb), rs in sorted(groups.items(), key=lambda kv: np.mean([r["macro"] for r in kv[1]])):
        mac = [r["macro"] for r in rs]
        print(f"{method:<10}{bb:<15}{np.mean(mac):7.1f} +-{np.std(mac):4.1f}{np.mean([r['median'] for r in rs]):8.1f}"
              f"{np.mean([r['within50'] for r in rs]):6.0%}  "
              + " ".join(f"{np.mean([r['per_video'][s] for r in rs]):8.1f}" for s in TEST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "feats", "run", "predict"])
    ap.add_argument("--video", default="cada5bef", help="predict: test video short id")
    ap.add_argument("--method", default="tip", choices=["tip", "tip+seg"], help="predict: method")
    ap.add_argument("--src", default="E:/Geknipte Videos batch 2/20archesfortesting/Pure Arch")
    ap.add_argument("--out", default=str(PURE))
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.src, Path(args.out), args.k)
    elif args.cmd == "feats":
        feats(args)
    elif args.cmd == "predict":
        predict(args)
    else:
        run(args)


if __name__ == "__main__":
    ids, u = colour_ids(np.array([[[0, 255, 255], [255, 0, 255], [128, 128, 128], [0, 0, 255], [7, 7, 7]]], np.uint8))
    assert ids.tolist() == [[1, 2, 5, 6, 0]] and abs(u - 0.2) < 1e-9, (ids, u)       # BGR in: yellow magenta grey red other
    main()
