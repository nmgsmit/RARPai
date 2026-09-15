"""Arch tip head on frozen SurgeNet-RARP CAFormer-S18 features (+ UniDepth depth), over the 20 patient splits.

Per frame on the 32x40 grid of the 512x640 feed (arch_tip_features.py): stage-3 (320 ch) and stage-4 (512 ch,
16x20, upsampled) features, the tool mask (catheter | non-anatomical, rarp_nick_fullres) and, for variant 'rgbd',
log-depth minus its median + normalised log-depth gradients. Output: tip, left end, right end, each a softmax
vote over cells of (cell centre + learned offset), so a point may lie OUTSIDE the frame (on 46867a8e the
annotators put the tip ~200 px above it).

Train = every annotated frame (>= 1 annotator, every --train-every) of the 7 non-test patients, weighted by
annotator count and agreement; loss on tip (1) and the arch ends (0.25). Fixed recipe, nothing tuned on test.
Predictions for EVERY frame of the test videos -> <out>/<variant>/s<k>/<short>.npz for arch_tip_cv.py.

    python scripts/arch_tip_head.py --selfcheck
    sbatch jobs/arch_tip_head.sh
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_tip_data import A, FrameStore, cv_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GH, GW = 32, 40
CELL = 1340 / GW                                   # 33.5 px, == 1072 / GH
PW = torch.tensor([1.0, 0.25, 0.25])               # loss weight: tip, left, right


class Head(nn.Module):
    def __init__(self, extra, width=256):
        super().__init__()
        self.red4 = nn.Conv2d(512, 128, 1)

        def block(i, o, d):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d), nn.GroupNorm(8, o), nn.ReLU(inplace=True))

        self.body = nn.Sequential(block(320 + 128 + extra + 2, width, 1), block(width, width, 2),
                                  block(width, width, 4), block(width, width, 1))
        self.out = nn.Conv2d(width, 9, 1)               # 3 heat + 3 dx + 3 dy
        ys, xs = torch.meshgrid(torch.arange(GH), torch.arange(GW), indexing="ij")
        self.register_buffer("cx", ((xs + 0.5) * CELL).float().flatten())
        self.register_buffer("cy", ((ys + 0.5) * CELL).float().flatten())
        self.register_buffer("coords", torch.stack([xs / (GW - 1), ys / (GH - 1)]).float()[None] * 2 - 1)

    def forward(self, s3, s4, extra):
        f = torch.cat([s3, F.interpolate(self.red4(s4), size=(GH, GW), mode="bilinear", align_corners=False)], 1)
        f = F.dropout2d(f, 0.1, self.training)
        o = self.out(self.body(torch.cat([f, extra, self.coords.expand(len(s3), -1, -1, -1)], 1))).flatten(2)
        w = o[:, :3].softmax(-1)
        px = (w * (self.cx + o[:, 3:6] * CELL)).sum(-1)
        py = (w * (self.cy + o[:, 6:9] * CELL)).sum(-1)
        return torch.stack([px, py], -1)                # (B, 3, 2) px: tip, left, right


class Store:
    """One video's grid inputs, indexed by frame number."""

    def __init__(self, short, depth):
        d = A / short
        self.frames = FrameStore(short).frames
        self.idx = {int(f): k for k, f in enumerate(self.frames)}
        self.s3 = np.load(d / "feats_s3.npy", mmap_mode="r")
        self.s4 = np.load(d / "feats_s4.npy", mmap_mode="r")
        self.t32 = cached(d / "tools32.npy", lambda: tools32(d))
        self.d32 = cached(d / "depth32.npy", lambda: depth32(d)) if depth else None


def cached(path, make):
    if not path.exists():
        np.save(path, make())
    return np.load(path)


def tools32(d):
    t = np.load(d / "tools.npy", mmap_mode="r")
    return np.stack([cv2.resize(np.asarray(m, np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
                     for m in t])[:, None].astype(np.float16)


def depth32(d):
    zs = np.load(d / "depth.npy", mmap_mode="r")

    def one(k):
        z = np.asarray(zs[k], np.float32)
        L = cv2.GaussianBlur(np.log(np.clip(z, 1, None)), (0, 0), 2)
        gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0) / 8, cv2.Sobel(L, cv2.CV_32F, 0, 1) / 8
        m = np.median(np.hypot(gx, gy)) + 1e-6
        return np.stack([cv2.resize(c, (GW, GH), interpolation=cv2.INTER_AREA)
                         for c in (L - np.median(L), gx / m, gy / m)]).astype(np.float16)
    with ThreadPoolExecutor(16) as ex:
        return np.stack(list(ex.map(one, range(len(zs)))))


def extra_of(st, ks, variant):
    return np.concatenate([st.t32[ks], st.d32[ks]], 1) if variant == "rgbd" else st.t32[ks]


def gather(stores, rows, variant):
    N, ne = len(rows), 4 if variant == "rgbd" else 1
    s3 = np.empty((N, 320, GH, GW), np.float16)
    s4 = np.empty((N, 512, GH // 2, GW // 2), np.float16)
    ex = np.empty((N, ne, GH, GW), np.float16)
    by = {}
    for j, r in enumerate(rows):
        by.setdefault(r["short"], []).append((stores[r["short"]].idx[r["frame"]], j))
    for short, pairs in by.items():
        st = stores[short]
        ks, js = (np.array(x) for x in zip(*sorted(pairs)))
        s3[js], s4[js], ex[js] = st.s3[ks], st.s4[ks], extra_of(st, ks, variant)
    return s3, s4, ex


def stats(t):
    """Per-channel mean/std of a (N, C, H, W) float16 GPU tensor, in chunks."""
    m = sq = 0
    for c in t.split(256):
        c = c.float()
        m, sq = m + c.sum((0, 2, 3)), sq + (c * c).sum((0, 2, 3))
    n = t.shape[0] * t.shape[2] * t.shape[3]
    m = m / n
    return m.view(1, -1, 1, 1), (sq / n - m * m).clamp_min(1e-6).sqrt().view(1, -1, 1, 1)


def train(s3, s4, ex, Y, Wt, a, dev, seed=0):
    torch.manual_seed(seed)
    s3, s4, ex = (torch.from_numpy(x).to(dev) for x in (s3, s4, ex))
    Y, Wt, pw = torch.from_numpy(Y).to(dev), torch.from_numpy(Wt).to(dev), PW.to(dev)
    norm = (*stats(s3), *stats(s4))
    model = Head(ex.shape[1], a.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * ((len(Y) + a.bs - 1) // a.bs))
    for _ in range(a.epochs):
        model.train()
        for b in torch.randperm(len(Y), device=dev).split(a.bs):
            p = model((s3[b].float() - norm[0]) / norm[1], (s4[b].float() - norm[2]) / norm[3], ex[b].float())
            l = F.smooth_l1_loss(p / 100, Y[b] / 100, beta=0.2, reduction="none").sum(-1)
            loss = ((l * pw).sum(-1) * Wt[b]).sum() / Wt[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval(), norm


@torch.no_grad()
def predict(model, norm, st, variant, dev, bs=256):
    out = []
    for k0 in range(0, len(st.frames), bs):
        ks = np.arange(k0, min(k0 + bs, len(st.frames)))
        s3, s4, ex = (torch.from_numpy(np.asarray(x)).to(dev).float()
                      for x in (st.s3[ks], st.s4[ks], extra_of(st, ks, variant)))
        out.append(model((s3 - norm[0]) / norm[1], (s4 - norm[2]) / norm[3], ex).cpu().numpy())
    return np.concatenate(out)


def weight(r):
    return r["n_annot"] / 3 * (np.exp(-r["agree_px"] / 50) if r["agree_px"] is not None else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["rgb", "rgbd"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--train-every", type=int, default=2)
    ap.add_argument("--splits", type=int, nargs="*", help="subset of split indices (debug)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "arch_tip_cv" / "head"))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    dev = "cuda"
    rows = json.loads((A / "labels_all.json").read_text())["rows"]
    splits = cv_splits(rows)
    short_of = {r["patient"]: r["short"] for r in rows}
    stores = {s: Store(s, "rgbd" in a.variants) for s in sorted(set(short_of.values()))}
    out, metrics = Path(a.out), []
    for variant in a.variants:
        for k, test in enumerate(splits):
            if a.splits and k not in a.splits:
                continue
            tr = [r for r in rows if r["patient"] not in test and r["frame"] % a.train_every == 0]
            Y = np.array([[r["tip"], r["arch"]["left"], r["arch"]["right"]] for r in tr], np.float32)
            model, norm = train(*gather(stores, tr, variant), Y, np.array([weight(r) for r in tr], np.float32), a, dev, k)
            ev = {}
            for p in sorted(test):
                st = stores[short_of[p]]
                P = predict(model, norm, st, variant, dev)
                (out / variant / f"s{k:02d}").mkdir(parents=True, exist_ok=True)
                np.savez(out / variant / f"s{k:02d}" / f"{short_of[p]}.npz",
                         frames=st.frames, tip=P[:, 0], left=P[:, 1], right=P[:, 2])
                e = [np.linalg.norm(P[st.idx[r["frame"]], 0] - r["tip"]) for r in rows
                     if r["patient"] == p and r["best_half"]]
                ev[p] = float(np.mean(e))
            metrics.append(dict(variant=variant, split=k, test=sorted(test), n_train=len(tr), patient_mean=ev,
                                macro=float(np.mean(list(ev.values())))))
            print(f"{variant} s{k:02d} train {len(tr)}  test macro {metrics[-1]['macro']:6.1f} px  "
                  + "  ".join(f"{p[:8]} {v:5.1f}" for p, v in ev.items()), flush=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))


def selfcheck():
    """Tip position encoded as a bump (where) + a value (how far past that cell): the vote head must recover it,
    including tips up to 200 px above the frame."""
    rng = np.random.default_rng(0)
    N = 384
    s3 = rng.normal(0, 1, (N, 320, GH, GW)).astype(np.float16)
    s4 = rng.normal(0, 1, (N, 512, GH // 2, GW // 2)).astype(np.float16)
    Y = np.zeros((N, 3, 2), np.float32)
    for j in range(N):
        tx, ty = rng.uniform(200, 1140), rng.uniform(-200, 800)
        cx, cy = int(tx // CELL), int(np.clip(ty, 0, 1071) // CELL)
        s3[j, 0, cy, cx] = 8
        s3[j, 1, cy, cx] = (ty - (cy + 0.5) * CELL) / CELL
        Y[j] = [[tx, ty], [tx - 300, ty + 300], [tx + 300, ty + 300]]
    a = SimpleNamespace(epochs=60, bs=32, lr=3e-3, width=64)
    ex = np.zeros((N, 1, GH, GW), np.float16)
    model, norm = train(s3, s4, ex, Y, np.ones(N, np.float32), a, "cpu")
    with torch.no_grad():
        P = model((torch.from_numpy(s3).float() - norm[0]) / norm[1], (torch.from_numpy(s4).float() - norm[2]) / norm[3],
                  torch.from_numpy(ex).float()).numpy()
    err = np.linalg.norm(P[:, 0] - Y[:, 0], axis=1)
    off = Y[:, 0, 1] < 0
    print(f"selfcheck: tip error mean {err.mean():.1f} px, off-frame tips {err[off].mean():.1f} px (n={off.sum()})")
    assert err.mean() < 40 and err[off].mean() < 60, "vote head failed to localise planted tips"


if __name__ == "__main__":
    main()
