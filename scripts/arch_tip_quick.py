"""Quick test: does more (more diverse) training data help the arch tip head?

Fixed held-out TEST patients (round 1): 46867a8e, RARP_062, cada5bef. Every TRAINING patient contributes the same
K frames (evenly spaced over its annotated frames; K = the smallest patient), so no video dominates. Conditions:
  old7          the 7 round-2 training patients
  all           old7 + the new workspace patients (transfer_atlas_mod/workspace, Nick only, GUI already black)
  n=3,5,7       random patient subsets of `all` (learning curve), several draws x seeds
Same model as round 2: head_rgb (frozen SurgeNet-RARP CAFormer stage 3/4, SurgeNet variant), no tool/depth/class
channels (not available for the new frames yet). Scored like round 2: tip error vs consensus, per test patient
mean, macro over the 3; on best-half frames and on all annotated frames. prior = train-mean tip of that run.

    sbatch jobs/arch_tip_quick.sh
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "surgenet"))
from arch_tip_data import A, FrameStore  # noqa: E402
from arch_tip_features import MEAN, STD  # noqa: E402
from arch_tip_head import GH, GW, train  # noqa: E402
from compare_arch_multi import apex_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT.parent / "data" / "processed" / "arch_tip_new"
TEST = ("46867a8e", "RARP_062", "cada5bef")


def even(frames, k):
    """k frames spread evenly over a sorted list (all of them if there are fewer)."""
    frames = sorted(frames)
    return [frames[i] for i in sorted(set(np.linspace(0, len(frames) - 1, min(k, len(frames))).round().astype(int)))]


@torch.no_grad()
def encode(imgs, dev):
    from finetune_segmentation import load_encoder
    from metaformer import MetaFormerFPN
    m = MetaFormerFPN(num_classes=1, pretrained="SurgeNet", pretrained_weights=None)
    load_encoder(m, str(ROOT.parent / "backbones" / "RARP_checkpoint_epoch0050_teacher.pth"))
    net = m.metaformer.to(dev).eval()
    s3, s4 = [], []
    for i in range(0, len(imgs), 32):
        x = np.stack([((cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (640, 512), interpolation=cv2.INTER_AREA)
                        .astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for im in imgs[i:i + 32]])
        f = net.forward_features(torch.from_numpy(x).to(dev))[1]
        s3.append(f[2].half().cpu().numpy())
        s4.append(f[3].half().cpu().numpy())
    return np.concatenate(s3), np.concatenate(s4)


def new_annotations():
    """{short: {frame: arch entry}} for workspace patients with at least one annotated frame."""
    out = {}
    for d in sorted(p for p in NEW.iterdir() if (p / "arches.json").exists()):
        ann = {int(f): e for f, e in json.loads((d / "arches.json").read_text())["frames"].items()}
        if ann:
            out[d.name[:8]] = (d, ann)
    return out


def load_patients(rows, k, dev):
    """{patient: dict(new, frames, tips, s3, s4)} for every training candidate, K frames each."""
    out = {}
    for short in sorted({r["short"] for r in rows} - set(TEST)):
        rs = {r["frame"]: r for r in rows if r["short"] == short}
        fs, fr = FrameStore(short), even(list(rs), k)
        ks = [fs.pos[f] for f in fr]
        out[short] = dict(new=False, frames=fr, tips=np.array([rs[f]["tip"] for f in fr], np.float32),
                          s3=np.load(A / short / "feats_s3.npy", mmap_mode="r")[ks],
                          s4=np.load(A / short / "feats_s4.npy", mmap_mode="r")[ks])
    for short, (d, ann) in new_annotations().items():
        fr = even(list(ann), k)
        s3, s4 = encode([cv2.imread(str(d / "images" / f"{f:07d}.jpg")) for f in fr], dev)
        out[short] = dict(new=True, frames=fr, tips=np.array([apex_of(ann[f])[0] for f in fr], np.float32),
                          s3=s3, s4=s4)
    return out


@torch.no_grad()
def predict(model, norm, s3, s4, dev, bs=256):
    out = []
    for i in range(0, len(s3), bs):
        a = torch.from_numpy(np.asarray(s3[i:i + bs])).to(dev).float()
        b = torch.from_numpy(np.asarray(s4[i:i + bs])).to(dev).float()
        out.append(model((a - norm[0]) / norm[1], (b - norm[2]) / norm[3],
                         torch.zeros(len(a), 0, GH, GW, device=dev))[:, 0].cpu().numpy())
    return np.concatenate(out)


def summary(results, name):
    rs = [r for r in results if r["cond"] == name]
    f = lambda m: f"{np.mean([r[m] for r in rs]):6.1f} +- {np.std([r[m] for r in rs]):4.1f}"  # noqa: E731
    per = " ".join(f"{s} {np.mean([r[s]['head_best'] for r in rs]):.0f}" for s in TEST)
    return (f"{name:<6} (n={rs[0]['n']:2d}, runs {len(rs):2d})  head best-half {f('head_best')}  all {f('head_all')}"
            f"  | prior best-half {f('prior_best')}  all {f('prior_all')}  | head best-half per patient: {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0, help="frames per training patient (0 = the smallest patient)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "arch_tip_quick"))
    a = ap.parse_args()
    dev = "cuda"

    rows = json.loads((A / "labels_all.json").read_text())["rows"]
    counts = {s: len({r["frame"] for r in rows if r["short"] == s}) for s in {r["short"] for r in rows} - set(TEST)}
    counts.update({s: len(ann) for s, (_, ann) in new_annotations().items()})
    k = a.k or min(counts.values())
    print("annotated frames per training patient:", dict(sorted(counts.items())), "-> K =", k, flush=True)
    P = load_patients(rows, k, dev)
    old, allp = sorted(p for p in P if not P[p]["new"]), sorted(P)

    test = {}
    for short in TEST:
        rs = sorted((r for r in rows if r["short"] == short), key=lambda r: r["frame"])
        fs = FrameStore(short)
        ks = [fs.pos[r["frame"]] for r in rs]
        test[short] = dict(tips=np.array([r["tip"] for r in rs], np.float32), best=np.array([r["best_half"] for r in rs]),
                           s3=np.load(A / short / "feats_s3.npy", mmap_mode="r")[ks],
                           s4=np.load(A / short / "feats_s4.npy", mmap_mode="r")[ks])

    conds = [("old7", tuple(old)), ("all", tuple(allp))]
    rng = random.Random(0)
    for n in (3, 5, 7):
        if n < len(allp):
            conds += [(f"n={n}", tuple(sorted(rng.sample(allp, n)))) for _ in range(a.draws)]
    results = []
    for (name, pats), seed in itertools.product(conds, range(a.seeds)):
        tips = np.concatenate([P[p]["tips"] for p in pats])
        Y = np.stack([tips, tips, tips], 1).astype(np.float32)     # quick test is tip-only: ends = tip
        spe = -(-len(Y) // 64)
        args = SimpleNamespace(epochs=max(1, -(-a.steps // spe)), bs=64, lr=1e-3, width=256)
        model, norm = train(np.concatenate([P[p]["s3"] for p in pats]), np.concatenate([P[p]["s4"] for p in pats]),
                            np.zeros((len(Y), 0, GH, GW), np.float16), Y, np.ones(len(Y), np.float32), args, dev, seed)
        mu = tips.mean(0)
        r = dict(cond=name, patients=list(pats), n=len(pats), seed=seed, n_new=sum(P[p]["new"] for p in pats))
        for short, t in test.items():
            e = np.linalg.norm(predict(model, norm, t["s3"], t["s4"], dev) - t["tips"], axis=1)
            ep = np.linalg.norm(mu - t["tips"], axis=1)
            r[short] = dict(head_best=float(e[t["best"]].mean()), head_all=float(e.mean()),
                            prior_best=float(ep[t["best"]].mean()), prior_all=float(ep.mean()))
        for m in ("head_best", "head_all", "prior_best", "prior_all"):
            r[m] = float(np.mean([r[s][m] for s in TEST]))
        results.append(r)
        print(f"{name:<6} n={len(pats):2d} new={r['n_new']} seed {seed}: head best {r['head_best']:6.1f} all "
              f"{r['head_all']:6.1f} | prior best {r['prior_best']:6.1f} all {r['prior_all']:6.1f}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(dict(k=k, counts=counts, results=results), indent=1))
    print("\n=== macro tip error over the 3 test patients (mean +- sd over draws x seeds) ===")
    for name in dict.fromkeys(c for c, _ in conds):
        print(summary(results, name))


if __name__ == "__main__":
    assert even(list(range(100)), 5) == [0, 25, 50, 74, 99] and even([3, 9], 5) == [3, 9]
    main()
