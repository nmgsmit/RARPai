"""Fine-tune the last blocks of the surgical DINOv3 for the arch tip (Pure Arch train set, 7 three-annotator test videos).

Same data, targets and test as scripts/arch_tip_pure.py (14 videos x 500 frames; test = per video the better-agreeing
half of its 3-annotator frames; only the tip is scored).

  cache  frozen DINOv3 ViT-L/16 (SurgeNetXL) through its first CUT=20 of 24 blocks, once, for the training and test frames
         -> <PURE>/tokens/{train,test}/<short>.npy  (N, 1 + 32*40, 1024) float16  (cls + patch tokens of a 512x640 feed).
         Self-check: replaying blocks 20..23 + the final norm on the stored tokens must reproduce
         get_intermediate_layers(n=[23], norm=True) on the same batch.
  ft     for n in --train-blocks (0 = frozen reference, 2, 4): blocks 20..23 with the last n trainable (lr --lr-blocks) +
         final norm + the arch head (arc7 targets, lr 1e-3); 3 seeds. Reports each run, the 3-seed ensemble (mean tip),
         and whether seed disagreement flags large errors.

    sbatch jobs/arch_tip_ft_cache.sh
    sbatch jobs/arch_tip_ft.sh
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_tip_data import FrameStore  # noqa: E402
from arch_tip_pure import GH, GW, OUT, PURE, ROOT, TEST, make_head, prep, targets, test_frames, train_data  # noqa: E402

CUT = 20
TOK = PURE / "tokens"


def load_dinov3(dev):
    import torch
    sys.path.insert(0, os.path.expanduser("~/pylibs/dinov3"))
    from dinov3.models.vision_transformer import DinoVisionTransformer
    sd = torch.load(ROOT.parent / "backbones" / "SurgeNetXL" / "DINOv3_ViTl16_size336_SurgeNetXL.pth",
                    map_location="cpu", weights_only=False)
    sd = {k.replace(".gamma_1", ".ls1.gamma").replace(".gamma_2", ".ls2.gamma"): v for k, v in sd.items()}
    m = DinoVisionTransformer(img_size=336, patch_size=16, embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4.0,
                              layerscale_init=1e-5, n_storage_tokens=0, mask_k_bias=False, pos_embed_rope_base=100,
                              pos_embed_rope_normalize_coords="separate", pos_embed_rope_dtype="fp32")
    m.load_state_dict(sd, strict=True)
    return m.to(dev).eval()


def tail_forward(blocks, norm, rope, tok):
    """Stored tokens after block CUT-1 -> normalised patch tokens as a (B, 1024, 32, 40) grid."""
    x = tok
    for blk in blocks:
        x = blk(x, rope)
    x = norm(x)[:, 1:]
    return x.transpose(1, 2).reshape(len(x), -1, GH, GW)


def cache(args):
    import torch
    dev = "cuda"
    m = load_dinov3(dev)
    jobs = [("train", s, FrameStore(s, root=PURE), None) for s in sorted(p.name for p in PURE.iterdir() if (p / "frames.npy").exists())]
    for short, (fr, _, _) in test_frames().items():
        fs = FrameStore(short)
        jobs.append(("test", short, fs, [fs.pos[f] for f in fr]))
    checked = False
    for split, short, fs, ks in jobs:
        ks = list(range(len(fs))) if ks is None else ks
        path = TOK / split / f"{short}.npy"
        if path.exists():                                     # e.g. the shared test tokens
            print(f"{split} {short}: exists, skipped", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.lib.format.open_memmap(path, "w+", np.float16, (len(ks), 1 + GH * GW, 1024))
        for i in range(0, len(ks), 16):
            x = torch.from_numpy(np.stack([prep(fs.jpg(k)) for k in ks[i:i + 16]])).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t, (H, W) = m.prepare_tokens_with_masks(x)
                assert (H, W) == (GH, GW), (H, W)
                rope = m.rope_embed(H=H, W=W)
                for blk in m.blocks[:CUT]:
                    t = blk(t, rope)
                t16 = t.half()
                if not checked:                               # the split must reproduce the normal forward
                    ref = m.get_intermediate_layers(x, n=[23], reshape=True, norm=True)[0].float()
                    rep = tail_forward(m.blocks[CUT:], m.norm, rope, t16).float()
                    rel = ((rep - ref).norm() / ref.norm()).item()
                    print(f"split self-check: relative difference replay vs normal forward = {rel:.4f}", flush=True)
                    assert rel < 0.05, rel
                    checked = True
            arr[i:i + len(x)] = t16.cpu().numpy()
        arr.flush()
        print(f"{split} {short}: {len(ks)} frames cached", flush=True)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ft(args):
    import torch
    import torch.nn.functional as F
    dev = "cuda"
    shorts, YT, _ = train_data()
    tests = test_frames()
    T = torch.from_numpy(np.concatenate([np.load(TOK / "train" / f"{s}.npy") for s in shorts])).to(dev)
    TT = {s: np.load(TOK / "test" / f"{s}.npy", mmap_mode="r") for s in TEST}
    dino = load_dinov3(dev)
    yt = torch.from_numpy(YT).to(dev)
    pts_t, wts = targets(yt, args.method)
    pw = torch.tensor(wts, device=dev, dtype=torch.float32)
    rope = dino.rope_embed(H=GH, W=GW)

    # head input normalisation: per-channel stats of the FROZEN tail output, fixed for every run
    m = sq = n_cells = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(T), 10 * 64):
            f = tail_forward(dino.blocks[CUT:], dino.norm, rope, T[i:i + 64]).float()
            m, sq, n_cells = m + f.sum((0, 2, 3)), sq + (f * f).sum((0, 2, 3)), n_cells + f.shape[0] * GH * GW
    mu = (m / n_cells).view(1, -1, 1, 1)
    sd = (sq / n_cells - mu.flatten() ** 2).clamp_min(1e-6).sqrt().view(1, -1, 1, 1)

    preds, rows = {}, []
    for n_train in args.train_blocks:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            blocks = torch.nn.ModuleList(copy.deepcopy(dino.blocks[CUT:])).to(dev)
            norm = copy.deepcopy(dino.norm).to(dev)
            for p in list(blocks.parameters()) + list(norm.parameters()):
                p.requires_grad_(False)
            tuned = []
            if n_train:
                for blk in blocks[len(blocks) - n_train:]:
                    tuned += list(blk.parameters())
                tuned += list(norm.parameters())
                for p in tuned:
                    p.requires_grad_(True)
            head = make_head(1024, pts_t.shape[1]).to(dev)
            groups = [dict(params=list(head.parameters()), lr=1e-3, weight_decay=1e-2)]
            if tuned:
                groups.append(dict(params=tuned, lr=args.lr_blocks, weight_decay=0.05))
            opt = torch.optim.AdamW(groups)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=args.steps)
            head.train()
            blocks.train()
            for _ in range(args.steps):
                b = torch.randint(0, len(T), (args.bs,), device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = tail_forward(blocks, norm, rope, T[b])
                    pts, _ = head((f.float() - mu) / sd)
                lv = (F.smooth_l1_loss(pts.float() / 100, pts_t[b] / 100, beta=0.2, reduction="none").sum(-1) * pw).sum(-1).mean()
                opt.zero_grad()
                lv.backward()
                opt.step()
                sched.step()
            head.eval()
            blocks.eval()
            out = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in TEST:
                    P = []
                    for i in range(0, len(TT[s]), 64):
                        f = tail_forward(blocks, norm, rope, torch.from_numpy(np.asarray(TT[s][i:i + 64])).to(dev))
                        P.append(head((f.float() - mu) / sd)[0][:, 0].float().cpu().numpy())
                    out[s] = np.concatenate(P)
            preds[(n_train, seed)] = out
            per = {s: float(np.linalg.norm(out[s] - tests[s][1], axis=1).mean()) for s in TEST}
            pooled = np.concatenate([np.linalg.norm(out[s] - tests[s][1], axis=1) for s in TEST])
            rows.append(dict(train_blocks=n_train, seed=seed, macro=float(np.mean(list(per.values()))),
                             median=float(np.median(pooled)), within50=float((pooled <= 50).mean()), per_video=per))
            print(f"last {n_train} blocks trained, seed {seed}: macro {rows[-1]['macro']:6.1f}  median {rows[-1]['median']:6.1f}  "
                  + " ".join(f"{s} {v:.0f}" for s, v in per.items()), flush=True)
            del blocks, norm, head, opt

    ens = {}
    for n_train in args.train_blocks:
        tips = {s: np.mean([preds[(n_train, sd_)][s] for sd_ in range(args.seeds)], 0) for s in TEST}
        spread = {s: np.mean([np.linalg.norm(preds[(n_train, sd_)][s] - tips[s], axis=1) for sd_ in range(args.seeds)], 0)
                  for s in TEST}
        err = {s: np.linalg.norm(tips[s] - tests[s][1], axis=1) for s in TEST}
        E, S = np.concatenate(list(err.values())), np.concatenate(list(spread.values()))
        lo = S <= np.median(S)
        ens[n_train] = dict(macro=float(np.mean([e.mean() for e in err.values()])), median=float(np.median(E)),
                            within50=float((E <= 50).mean()), per_video={s: float(e.mean()) for s, e in err.items()},
                            spread_vs_error_spearman=spearman(S, E), error_low_spread=float(E[lo].mean()),
                            error_high_spread=float(E[~lo].mean()))

    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ft_{args.method}.json").write_text(json.dumps(dict(method=args.method, cut=CUT, lr_blocks=args.lr_blocks,
                                                                     steps=args.steps, rows=rows, ensemble=ens), indent=1))
    print(f"\n=== surgical DINOv3, target {args.method}: fine-tuning the last blocks (tokens cached after block {CUT}) ===")
    print(f"{'trained blocks':<16}{'macro (seeds)':>16}{'median':>8}{'<=50':>6} | {'ensemble':>9}{'median':>8}{'<=50':>6}  "
          + " ".join(f"{s:>8}" for s in TEST))
    for n_train in args.train_blocks:
        rs = [r for r in rows if r["train_blocks"] == n_train]
        mac, e = [r["macro"] for r in rs], ens[n_train]
        print(f"{n_train:<16}{np.mean(mac):9.1f} +-{np.std(mac):4.1f}{np.mean([r['median'] for r in rs]):8.1f}"
              f"{np.mean([r['within50'] for r in rs]):6.0%} | {e['macro']:9.1f}{e['median']:8.1f}{e['within50']:6.0%}  "
              + " ".join(f"{e['per_video'][s]:8.1f}" for s in TEST))
    print("\nseed disagreement as a failure flag (ensemble):")
    for n_train in args.train_blocks:
        e = ens[n_train]
        print(f"  last {n_train} trained: Spearman(spread, error) {e['spread_vs_error_spearman']:+.2f}; mean error on the "
              f"half with low spread {e['error_low_spread']:.1f} px vs high spread {e['error_high_spread']:.1f} px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "ft"])
    ap.add_argument("--method", default="arc7")
    ap.add_argument("--train-blocks", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--lr-blocks", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cache(args) if args.cmd == "cache" else ft(args)


if __name__ == "__main__":
    assert abs(spearman(np.arange(10.0), np.arange(10.0) ** 3) - 1) < 1e-9
    main()
