"""Visualize the worst-residual tip disagreement between Nick and Veerle's
arch annotations, on the actual video frame.

Left panel : both arches in their own raw coordinates (shows the per-video
             coordinate-frame offset -- NOT real disagreement).
Right panel: Veerle's arch shifted by that video's estimated constant offset,
             so it lines up with Nick's frame -- this is the real, residual
             tip-to-tip disagreement after removing the offset.

Reads outputs/arch_tip_comparison_residual.json (written by
compare_arch_annotators.py) to find the worst frame and the per-video offset,
re-reads both raw arches.json for the full arch (left/right/height/power),
and draws the parabola with the same formula as
transfer_atlas_mod/gui/retzius_arch.py (v(u) = height*(1-|u|^power)).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data" / "InterAnnotator_Arch_Comparison"
NICK_DIR = DATA / "JSONfileNick"
VEERLE_DIR = DATA / "JSONfilesVeerle"
VEERLE_PREFIX = "InterAnnotator_Arch_Comparison__"
VEERLE_SUFFIX = "_arches.json"
FRAMES_ROOT = ROOT.parent / "transfer_atlas_mod" / "workspace" / "InterAnnotator_Arch_Comparison"


def arch_points(frame, n=200):
    left = np.array(frame["left"], dtype=np.float64)
    right = np.array(frame["right"], dtype=np.float64)
    mid = (left + right) / 2
    chord = right - left
    d = float(np.linalg.norm(chord)) / 2
    t = chord / (2 * d) if d > 1e-9 else np.array([1.0, 0.0])
    n_vec = np.array([t[1], -t[0]])
    power = frame.get("power", 2.0)
    us = np.linspace(-1.0, 1.0, n)
    shape = 1 - np.abs(us) ** power
    pts = mid[None, :] + (us[:, None] * d) * t[None, :] + (shape[:, None] * frame["height"]) * n_vec[None, :]
    apex = mid + frame["height"] * n_vec
    return pts, apex, left, right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", choices=["worst", "median", "video-frame"], default="worst")
    ap.add_argument("--video")
    ap.add_argument("--frame", type=int)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    resid = json.loads((ROOT / "outputs" / "arch_tip_comparison_residual.json").read_text())
    if args.pick == "worst":
        worst = max(resid, key=lambda r: r["resid_px"])
    elif args.pick == "median":
        vals = sorted(r["resid_px"] for r in resid)
        med = vals[len(vals) // 2]
        worst = min(resid, key=lambda r: abs(r["resid_px"] - med))
    else:
        worst = next(r for r in resid if r["video"] == args.video and r["frame"] == args.frame)
    video, frame_idx = worst["video"], worst["frame"]
    print(f"[{args.pick}] {video} frame {frame_idx}: {worst['resid_px']:.1f}px "
          f"({worst['resid_pct_chord']:.1f}% of chord)")

    nick_frames = json.loads((NICK_DIR / video / "arches.json").read_text())["frames"]
    veerle_frames = json.loads(
        (VEERLE_DIR / f"{VEERLE_PREFIX}{video}{VEERLE_SUFFIX}").read_text())["frames"]
    nf = nick_frames[str(frame_idx)]
    vf = veerle_frames[str(frame_idx)]

    n_pts, n_apex, n_left, n_right = arch_points(nf)
    v_pts, v_apex, v_left, v_right = arch_points(vf)

    # per-video mean offset, computed the same way compare_arch_annotators.py did
    all_rows = json.loads((ROOT / "outputs" / "arch_tip_comparison.json").read_text())
    vr = [r for r in all_rows if r["video"] == video]
    dx = np.mean([r["veerle_tip"][0] - r["nick_tip"][0] for r in vr])
    dy = np.mean([r["veerle_tip"][1] - r["nick_tip"][1] for r in vr])
    offset = np.array([dx, dy])
    print(f"per-video offset (veerle - nick): dx={dx:.1f} dy={dy:.1f}")

    v_pts_corr = v_pts - offset
    v_apex_corr = v_apex - offset

    img_path = FRAMES_ROOT / video / "images" / f"{frame_idx:07d}.jpg"
    img = np.asarray(Image.open(img_path)) if img_path.exists() else None
    if img is None:
        print(f"[warn] frame image not found at {img_path}, plotting on blank canvas")

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, title, v_pts_use, v_apex_use in (
        (axes[0], "Raw coordinates (per-video offset NOT removed)", v_pts, v_apex),
        (axes[1], "Offset-corrected (Veerle shifted onto Nick's frame)", v_pts_corr, v_apex_corr),
    ):
        if img is not None:
            h, w = img.shape[:2]
            ax.imshow(img, extent=(0, w, h, 0))
        ax.plot(n_pts[:, 0], n_pts[:, 1], color="cyan", lw=3, label="Nick's arch")
        ax.plot(v_pts_use[:, 0], v_pts_use[:, 1], color="magenta", lw=3, label="Veerle's arch")
        ax.scatter(*n_apex, color="cyan", s=140, edgecolor="black", zorder=5, marker="*", label="Nick's tip")
        ax.scatter(*v_apex_use, color="magenta", s=140, edgecolor="black", zorder=5, marker="*", label="Veerle's tip")
        ax.plot([n_apex[0], v_apex_use[0]], [n_apex[1], v_apex_use[1]], "w--", lw=1.5, alpha=0.9)
        dist = float(np.linalg.norm(np.array(n_apex) - np.array(v_apex_use)))
        mid = (np.array(n_apex) + np.array(v_apex_use)) / 2
        ax.annotate(f"{dist:.0f}px", mid, color="yellow", fontsize=14, fontweight="bold",
                    ha="center", va="bottom")

        # autoscale to fit image + both arches + a margin, keep y inverted (image coords)
        all_x = np.concatenate([n_pts[:, 0], v_pts_use[:, 0], [0, img.shape[1] if img is not None else 0]])
        all_y = np.concatenate([n_pts[:, 1], v_pts_use[:, 1], [0, img.shape[0] if img is not None else 0]])
        pad_x = 0.06 * (all_x.max() - all_x.min())
        pad_y = 0.06 * (all_y.max() - all_y.min())
        ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
        ax.set_ylim(all_y.max() + pad_y, all_y.min() - pad_y)

        ax.set_title(title, fontsize=12)
        ax.legend(loc="upper right")
        ax.set_facecolor("black")

    fig.suptitle(f"[{args.pick}] {video}  frame {frame_idx}\n"
                 f"raw distance={worst['dist_px']:.0f}px  |  "
                 f"residual (real) distance={worst['resid_px']:.0f}px "
                 f"({worst['resid_pct_chord']:.1f}% of chord)", fontsize=13)
    fig.tight_layout()
    out_path = Path(args.out) if args.out else ROOT / "outputs" / f"{args.pick}_arch_tip_diff.png"
    fig.savefig(out_path, dpi=130, facecolor="white")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
