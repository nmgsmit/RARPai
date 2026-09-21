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
  tip+mid   tip + the centre of the arch chord (weights 1, 0.5)
  arc5/arc7 5 / 7 points along the drawn arch (tip, then outwards to both ends); loss weight 1 at the tip,
            falling linearly to 0.25 at the ends. Only the tip (point 0) is ever scored.
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
PURE = Path(os.environ.get("ARCH_TIP_PURE", ROOT.parent / "data" / "processed" / "arch_tip_pure"))   # training set root
OUT = ROOT / "outputs" / PURE.name          # results per training set (arch_tip_pure = the 14-video set)
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
        a = d / "arches.json"
        arch = {int(f): e for f, e in json.loads(a.read_text())["frames"].items()} if a.exists() else {}
        both = sorted(set(arch) & {int(f[:7]) for f in os.listdir(d / "masks")}) if (d / "masks").is_dir() else []
        if not both:
            print(f"{short}: no frame with both an arch and a mask, skipped")
            continue
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
            if path.exists():                             # e.g. the shared test features
                print(f"{name} {split} {short}: exists, skipped", flush=True)
                continue
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
def make_head(cin, npts=3, width=256, void=False):
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
            self.vote, self.seg = nn.Conv2d(width, 3 * npts, 1), nn.Conv2d(width, NCLS, 1)
            ys, xs = torch.meshgrid(torch.arange(GH), torch.arange(GW), indexing="ij")
            self.register_buffer("cx", ((xs + 0.5) * CELL).float().flatten())
            self.register_buffer("cy", ((ys + 0.5) * CELL).float().flatten())
            self.register_buffer("coords", torch.stack([xs / (GW - 1), ys / (GH - 1)]).float()[None] * 2 - 1)
            self.void = void

        def forward(self, x, v=None):
            f = F.dropout2d(self.red(x), 0.1, self.training)
            h = self.body(torch.cat([f, self.coords.expand(len(x), -1, -1, -1)], 1))
            o = self.vote(h).flatten(2)
            heat = o[:, :npts]                           # per point: heat, dx, dy (point 0 = tip)
            if v is not None:                            # void cells (robot arm) may not vote; offsets still reach under them
                heat = heat.masked_fill((v.flatten(1) > 0.5)[:, None, :], -1e4)
            w = heat.softmax(-1)
            pts = torch.stack([(w * (self.cx + o[:, npts:2 * npts] * CELL)).sum(-1),
                               (w * (self.cy + o[:, 2 * npts:] * CELL)).sum(-1)], -1)
            return pts, self.seg(h)
    return Head()


def seg_targets(short, n, root=None):
    """Soft 7-class targets on the 32x40 grid (area-downsampled one-hot of the packed masks), cached per video."""
    root = root or PURE
    path = root / short / "seg32.npy"
    if not path.exists():
        fs = FrameStore(short, root=root)
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


# Training targets. Point 0 is always the tip (the only thing scored); weights fall with distance from the tip.
POINTS = {"tip1": None, "tip": None, "tip+mid": None, "arc5": [0, -0.5, 0.5, -1, 1], "arc7": [0, -1 / 3, 1 / 3, -2 / 3, 2 / 3, -1, 1]}


def arc_points(tip, left, right, us, power=2.0):
    """Points on the annotation tool's arch (apex = tip, ends = left/right) at chord positions us in [-1, 1]."""
    mid, ch = (left + right) / 2, right - left
    d = np.linalg.norm(ch, axis=-1, keepdims=True) / 2
    t = ch / (2 * d + 1e-9)
    nrm = np.stack([t[..., 1], -t[..., 0]], -1)
    h = ((tip - mid) * nrm).sum(-1, keepdims=True)
    return np.stack([mid + u * d * t + (1 - abs(u) ** power) * h * nrm for u in us], -2)


def targets(yt, method):
    """(N,3,2) tip/left/right tensor -> (N,P,2) target points and P weights for this method."""
    import torch
    if method == "tip1":                              # the tip alone: no arch point enters training
        return yt[:, :1], [1.0]
    if method in ("tip", "tip+seg", "seg->tip"):
        return yt, [1.0, 0.25, 0.25]
    if method == "tip+mid":
        return torch.stack([yt[:, 0], (yt[:, 1] + yt[:, 2]) / 2], 1), [1.0, 0.5]
    us = POINTS[method]
    Y = yt.cpu().numpy()
    pts = arc_points(Y[:, 0], Y[:, 1], Y[:, 2], us)
    return torch.from_numpy(pts.astype(np.float32)).to(yt.device), [1 - 0.75 * abs(u) for u in us]


def train_data(root=None):
    """Training videos, (N,3,2) tip/left/right targets and (N,7,32,40) soft anatomy targets, video by video."""
    root = root or PURE
    shorts = sorted(p.name for p in root.iterdir() if (p / "frames.npy").exists())
    Y = {s: json.loads((root / s / "labels.json").read_text())["labels"] for s in shorts}
    fr = {s: FrameStore(s, root=root).frames for s in shorts}
    YT = np.concatenate([np.array([[Y[s][str(f)][k] for k in ("tip", "left", "right")] for f in fr[s]], np.float32)
                         for s in shorts])
    return shorts, YT, np.concatenate([seg_targets(s, len(fr[s]), root) for s in shorts])


def norm_stats(X):
    """Per-channel mean / std of an (N, C, H, W) float16 GPU tensor, in chunks."""
    m = sq = 0
    for c in X.split(256):
        c = c.float()
        m, sq = m + c.sum((0, 2, 3)), sq + (c * c).sum((0, 2, 3))
    n = X.shape[0] * GH * GW
    mu = (m / n).view(1, -1, 1, 1)
    return mu, (sq / n - mu.flatten() ** 2).clamp_min(1e-6).sqrt().view(1, -1, 1, 1)


def fit(X, yt, st, mu, sd, method, seed, steps, bs, dev, void=False):
    """yt = (N,3,2) tip/left/right; targets() turns it into the method's points + weights. tip+seg adds the anatomy
    loss, seg->tip uses only the anatomy loss."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    pts_t, wts = targets(yt, method)
    head = make_head(X.shape[1], pts_t.shape[1], void=void).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    wv = 0.0 if method == "seg->tip" else 1.0                  # point loss for every point method
    ws = 1.0 if method in ("tip+seg", "seg->tip") else 0.0     # anatomy loss only where asked
    pw = torch.tensor(wts, device=dev, dtype=torch.float32)
    head.train()
    for _ in range(steps):
        b = torch.randint(0, len(X), (bs,), device=dev)
        xb = X[b].float()
        pts, seg = head((xb - mu) / sd, xb[:, -1] if void else None)   # void = last input channel, raw
        lv = (F.smooth_l1_loss(pts / 100, pts_t[b] / 100, beta=0.2, reduction="none").sum(-1) * pw).sum(-1).mean()
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
            p, s = head((x - mu) / sd, x[:, -1] if getattr(head, "void", False) else None)
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
        out = OUT / "pred" / f"{args.method}_{bb}" / f"s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        ends = P[:, -2:] if P.shape[1] > 2 else np.stack([P[:, 0], P[:, 0]], 1)   # outermost points = arch ends
        np.savez(out / f"{short}.npz", frames=fs.frames, tip=P[:, 0], left=ends[:, 0], right=ends[:, 1], test_frames=np.array(tf))
        print(f"{args.method} {bb} seed {seed}: {len(fs)} frames, test error {np.linalg.norm(P[tk, 0] - tips, axis=1).mean():.1f} px",
              flush=True)


# ------------------------------------------------------------------------------------------------ ablation (GPU)
def depth3(z):
    """(268,335) mm -> (3,32,40): log-depth minus its median + normalised log-depth x/y gradients (as round 2)."""
    L = cv2.GaussianBlur(np.log(np.clip(np.asarray(z, np.float32), 1, None)), (0, 0), 2)
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0) / 8, cv2.Sobel(L, cv2.CV_32F, 0, 1) / 8
    m = np.median(np.hypot(gx, gy)) + 1e-6
    ch = [L - np.median(L), np.clip(gx / m, -20, 20), np.clip(gy / m, -20, 20)]   # clip: near-flat frames blow up 1/m
    return np.stack([cv2.resize(c, (GW, GH), interpolation=cv2.INTER_AREA) for c in ch])


def ablate(args):
    """Extra INPUT channels on top of the frozen backbone features, same targets / seeds / test set as `run`:
       feats | +depth (UniDepth, 3 ch) | +anatomy (7-class probabilities, predicted) | +depth+anatomy.
    Test videos have no masks, so anatomy maps are always PREDICTED: training frames get maps from a model trained on
    the other half of the training videos (2-fold by video), test frames from a model trained on all 14."""
    import torch
    dev, bb = "cuda", args.backbones[0]
    shorts, YT, ST = train_data()
    n_per = [len(FrameStore(s, root=PURE)) for s in shorts]
    vid = np.repeat(np.arange(len(shorts)), n_per)
    tests = test_frames()
    X = torch.from_numpy(np.concatenate([np.load(PURE / "feats" / bb / "train" / f"{s}.npy") for s in shorts])).to(dev)
    XT = {s: np.asarray(np.load(PURE / "feats" / bb / "test" / f"{s}.npy", mmap_mode="r")) for s in TEST}
    mu, sd = norm_stats(X)
    yt, st = torch.from_numpy(YT).to(dev), torch.from_numpy(ST).to(dev)

    dep = np.concatenate([np.stack([depth3(z) for z in np.load(PURE / s / "depth.npy", mmap_mode="r")]) for s in shorts])
    depT = {}
    for s in TEST:
        fs, zs = FrameStore(s), np.load(A / s / "depth.npy", mmap_mode="r")
        depT[s] = np.stack([depth3(zs[fs.pos[f]]) for f in tests[s][0]])
    print("depth channels ready", dep.shape, flush=True)

    ana = np.zeros((len(X), NCLS, GH, GW), np.float16)
    for fold in (0, 1):
        held = np.isin(vid, np.arange(fold, len(shorts), 2))
        tr, te = torch.from_numpy(np.nonzero(~held)[0]).to(dev), np.nonzero(held)[0]
        head = fit(X[tr], yt[tr], st[tr], mu, sd, "seg->tip", 0, args.steps, args.bs, dev)
        ana[te] = infer_with(head, mu, sd, X[torch.from_numpy(te).to(dev)], dev)[1].softmax(1).half().cpu().numpy()
        del head
    head = fit(X, yt, st, mu, sd, "seg->tip", 0, args.steps, args.bs, dev)
    anaT = {s: infer_with(head, mu, sd, XT[s], dev)[1].softmax(1).half().cpu().numpy() for s in TEST}
    agree = float((ana.argmax(1) == ST.argmax(1)).mean())
    print(f"anatomy maps ready: cross-fitted train maps match the mask argmax in {agree:.1%} of grid cells", flush=True)

    configs = {"feats": (), "+depth": ("depth",), "+anatomy": ("anatomy",), "+depth+anatomy": ("depth", "anatomy")}
    rows = []
    for name, parts in configs.items():
        ext = [{"depth": dep, "anatomy": ana}[p].astype(np.float16) for p in parts]
        Xc = torch.cat([X] + [torch.from_numpy(e).to(dev) for e in ext], 1) if ext else X
        XTc = {s: np.concatenate([XT[s]] + [{"depth": depT, "anatomy": anaT}[p][s].astype(np.float16) for p in parts], 1)
               for s in TEST}
        muc, sdc = norm_stats(Xc)
        for seed in range(args.seeds):
            head = fit(Xc, yt, st, muc, sdc, args.method, seed, args.steps, args.bs, dev)
            per = {s: float(np.linalg.norm(infer_with(head, muc, sdc, XTc[s], dev)[0][:, 0] - tests[s][1], axis=1).mean())
                   for s in TEST}
            pooled = np.concatenate([np.linalg.norm(infer_with(head, muc, sdc, XTc[s], dev)[0][:, 0] - tests[s][1], axis=1)
                                     for s in TEST])
            rows.append(dict(config=name, seed=seed, macro=float(np.mean(list(per.values()))), median=float(np.median(pooled)),
                             within50=float((pooled <= 50).mean()), per_video=per))
            print(f"{name:<16} seed {seed}: macro {rows[-1]['macro']:6.1f}  " + " ".join(f"{s} {v:.0f}" for s, v in per.items()),
                  flush=True)
        if ext:
            del Xc
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ablation_{args.method}_{bb}.json").write_text(json.dumps(dict(method=args.method, backbone=bb, rows=rows), indent=1))
    print(f"\n=== input ablation: {bb}, target {args.method}, {args.seeds} seeds; tip error vs consensus, 7 test videos ===")
    print(f"{'inputs':<17}{'macro':>14}{'median':>8}{'<=50':>6}  " + " ".join(f"{s:>8}" for s in TEST))
    for name in configs:
        rs = [r for r in rows if r["config"] == name]
        mac = [r["macro"] for r in rs]
        print(f"{name:<17}{np.mean(mac):7.1f} +-{np.std(mac):4.1f}{np.mean([r['median'] for r in rs]):8.1f}"
              f"{np.mean([r['within50'] for r in rs]):6.0%}  " + " ".join(f"{np.mean([r['per_video'][s] for r in rs]):8.1f}" for s in TEST))


# ------------------------------------------------------------------------------------------------ consistency (GPU)
def jitter(frames, tips):
    """Frame-to-frame tip movement (px) between CONSECUTIVE frame numbers only."""
    frames, tips = np.asarray(frames), np.asarray(tips, np.float32)
    ok = np.diff(frames) == 1
    return np.linalg.norm(np.diff(tips, axis=0), axis=1)[ok]


def consistency(args):
    """Every frame of the 7 test videos, frozen surgical DINOv3 + arc7 trained on each training set (3 seeds): tip jitter
    between consecutive frames vs the annotators' own jitter, and on the scored frames the error split into a constant
    per-video offset and the scatter around it. Per-frame predictions -> outputs/<training set>/pred_all/<short>.npz."""
    import torch
    dev, bb = "cuda", "dinov3_surg"
    base = ROOT.parent / "data" / "processed"
    sets = {"14 videos": base / "arch_tip_pure", "36 videos": base / "arch_tip_pure40"}
    allf = base / "arch_tip_pure" / "feats" / bb / "allframes"
    tests = test_frames()
    rows = json.loads((A / "labels_all.json").read_text())["rows"]
    human = {}
    for s in TEST:                                           # annotators' consensus tip on consecutive 3-annotator frames
        rs = sorted((r for r in rows if r["short"] == s and r["n_annot"] == 3), key=lambda r: r["frame"])
        human[s] = jitter([r["frame"] for r in rs], [r["tip"] for r in rs])

    need = [s for s in TEST if not (allf / f"{s}.npy").exists()]
    if need:
        f, C = backbone(bb, dev)
        for s in need:
            fs = FrameStore(s)
            allf.mkdir(parents=True, exist_ok=True)
            arr = np.lib.format.open_memmap(allf / f"{s}.npy", "w+", np.float16, (len(fs), C, GH, GW))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, len(fs), 32):
                    x = torch.from_numpy(np.stack([prep(fs.jpg(k)) for k in range(i, min(i + 32, len(fs)))])).to(dev)
                    arr[i:i + len(x)] = f(x).float().cpu().numpy()
            arr.flush()
            print(f"all-frame features {s}: {len(fs)} frames", flush=True)

    report = {}
    for name, root in sets.items():
        shorts, YT, ST = train_data(root)
        X = torch.from_numpy(np.concatenate([np.load(root / "feats" / bb / "train" / f"{t}.npy") for t in shorts])).to(dev)
        mu, sd = norm_stats(X)
        yt, st = torch.from_numpy(YT).to(dev), torch.from_numpy(ST).to(dev)
        heads = [fit(X, yt, st, mu, sd, args.method, seed, args.steps, args.bs, dev) for seed in range(args.seeds)]
        del X
        out = ROOT / "outputs" / root.name / "pred_all"
        out.mkdir(parents=True, exist_ok=True)
        report[name] = {}
        for s in TEST:
            fs = FrameStore(s)
            feats_all = np.load(allf / f"{s}.npy", mmap_mode="r")
            P = np.stack([infer_with(h, mu, sd, feats_all, dev)[0][:, 0] for h in heads])     # (seeds, frames, 2)
            ens = P.mean(0)
            np.savez(out / f"{s}.npz", frames=fs.frames, tip_seeds=P, tip=ens)
            j_ens, j_seed = jitter(fs.frames, ens), np.concatenate([jitter(fs.frames, q) for q in P])
            tf, gt, _ = tests[s]
            d = ens[[fs.pos[fr] for fr in tf]] - gt
            off = d.mean(0)
            report[name][s] = dict(jitter_median=float(np.median(j_ens)), jitter_mean=float(j_ens.mean()),
                                   jitter_p95=float(np.percentile(j_ens, 95)), jitter_seed_median=float(np.median(j_seed)),
                                   error=float(np.linalg.norm(d, axis=1).mean()), offset=float(np.linalg.norm(off)),
                                   offset_xy=off.round(1).tolist(), scatter=float(np.linalg.norm(d - off, axis=1).mean()),
                                   human_jitter_median=float(np.median(human[s])))
            r = report[name][s]
            print(f"{name} {s}: jitter median {r['jitter_median']:.1f} px/frame, error {r['error']:.1f} "
                  f"= offset {r['offset']:.1f} + scatter {r['scatter']:.1f}", flush=True)
        del heads
    (ROOT / "outputs" / "arch_tip_pure40" / f"consistency_{args.method}.json").write_text(json.dumps(report, indent=1))

    print()
    print("=== frame-to-frame tip movement (px per frame, consecutive frames, all frames of each test video) ===")
    print(f"{'video':<10}{'annotators':>11}" + "".join(f"{n + ' median':>18}{'mean':>7}{'p95':>7}" for n in sets))
    for s in TEST:
        print(f"{s:<10}{report['14 videos'][s]['human_jitter_median']:11.1f}" + "".join(
            f"{report[n][s]['jitter_median']:18.1f}{report[n][s]['jitter_mean']:7.1f}{report[n][s]['jitter_p95']:7.1f}"
            for n in sets))
    for n in sets:
        print(f"{n}: median over videos {np.median([report[n][s]['jitter_median'] for s in TEST]):.1f} px/frame "
              f"(single seed {np.median([report[n][s]['jitter_seed_median'] for s in TEST]):.1f}; 3-seed average above)")
    print()
    print("=== error on the scored frames = constant per-video offset + scatter around it (3-seed average, px) ===")
    print(f"{'video':<10}" + "".join(f"{n + ' error':>17}{'offset':>8}{'scatter':>9}" for n in sets))
    for s in TEST:
        print(f"{s:<10}" + "".join(f"{report[n][s]['error']:17.1f}{report[n][s]['offset']:8.1f}{report[n][s]['scatter']:9.1f}"
                                   for n in sets))


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
        for method, seed in itertools.product(args.methods, range(args.seeds)):
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

    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{'_'.join(args.methods)}_{'_'.join(args.backbones)}.json").write_text(json.dumps(dict(train=train_shorts, test={s: len(tests[s][0]) for s in TEST},
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
    ap.add_argument("cmd", choices=["pack", "feats", "run", "predict", "ablate", "consistency"])
    ap.add_argument("--video", default="cada5bef", help="predict: test video short id")
    ap.add_argument("--method", default="tip", choices=list(POINTS) + ["tip+seg"], help="predict: method")
    ap.add_argument("--methods", nargs="+", default=["tip", "tip+mid", "arc5", "arc7"],
                    choices=list(POINTS) + ["tip+seg", "seg->tip"], help="run: methods to compare")
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
    elif args.cmd == "ablate":
        ablate(args)
    elif args.cmd == "consistency":
        consistency(args)
    else:
        run(args)


if __name__ == "__main__":
    ids, u = colour_ids(np.array([[[0, 255, 255], [255, 0, 255], [128, 128, 128], [0, 0, 255], [7, 7, 7]]], np.uint8))
    assert ids.tolist() == [[1, 2, 5, 6, 0]] and abs(u - 0.2) < 1e-9, (ids, u)       # BGR in: yellow magenta grey red other
    _y = np.array([[670.0, 250.0]]), np.array([[270.0, 550.0]]), np.array([[1070.0, 550.0]])
    _p = arc_points(*_y, POINTS["arc7"])[0]
    assert np.allclose(_p[0], _y[0][0]) and np.allclose(_p[-2], _y[1][0]) and np.allclose(_p[-1], _y[2][0]), _p
    assert np.allclose(arc_points(*_y, [0.5])[0, 0], [870, 250 + 0.25 * 300]), arc_points(*_y, [0.5])
    main()
