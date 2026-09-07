"""Diagnostic panel for one video's SUL: frame + urethra mask + mirror axis, and the
metric depth map with the two sampled endpoints. Use it when a depth SUL looks wrong.

python scripts/sul_depth_debug.py 107a5601 --root ..\transfer_atlas_mod\workspace\SUL_img3x_5x4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sul_measure as S
from gui_depth_measure import (DEFAULT_CKPT, DEFAULT_K_NORM, DEFAULT_MAX_DEPTH,
                               DEFAULT_MIN_DEPTH, DEFAULT_SHAPE, DepthBackend,
                               backproject, colorize, sample_depth, segment_length)


def label(img, lines, org=(16, 44), color=(255, 255, 255)):
    for i, t in enumerate(lines):
        y = org[1] + i * 40
        cv2.putText(img, t, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(img, t, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="video id or any unique prefix of it")
    ap.add_argument("--root", default="../transfer_atlas_mod/workspace/SUL_img3x_5x4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    groups = S.load_groups(root)
    hits = [(c, f) for c, f in groups.items() if args.video in c and S.URETHRA in f]
    if len(hits) != 1:
        raise SystemExit(f"[abort] {args.video} matched {len(hits)} clips with a urethra mask")
    clip, frames = hits[0]
    stem = frames[S.URETHRA]

    u = np.array(Image.open(root / "masks" / f"{stem}.png")) == S.URETHRA
    ax = S.mirror_axis(u)
    p0, p1 = ax["p0"], ax["p1"]

    pil = Image.open(root / "images" / f"{stem}.png").convert("RGB")
    W, H = pil.size
    depth = DepthBackend(DEFAULT_CKPT, DEFAULT_SHAPE, DEFAULT_MIN_DEPTH,
                         DEFAULT_MAX_DEPTH).load().predict(pil)
    a, b = (p0[0] / W, p0[1] / H), (p1[0] / W, p1[1] / H)
    z0, z1 = sample_depth(depth, *a), sample_depth(depth, *b)
    mm3d = segment_length(a, b, z0, z1, DEFAULT_K_NORM)
    A, B = backproject(*a, z0, DEFAULT_K_NORM), backproject(*b, z1, DEFAULT_K_NORM)
    inplane = float(np.linalg.norm((B - A)[:2]))

    # scale from the catheter arch, same as sul_measure
    scale = None
    arches = S.load_arches(root)
    c_stem = frames.get(S.CATHETER)
    if c_stem in arches:
        left, right = arches[c_stem]
        scale = S.CATHETER_MM / float(np.linalg.norm(left - right))

    left_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cont, _ = cv2.findContours(u.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left_img, cont, -1, (0, 0, 255), 3)
    q0, q1 = (tuple(np.round(p).astype(int)) for p in (p0, p1))
    cv2.line(left_img, q0, q1, (0, 255, 0), 3)
    for q, c in ((q0, (0, 255, 255)), (q1, (255, 0, 255))):
        cv2.circle(left_img, q, 12, c, -1)
    label(left_img, [f"span {ax['span']} px  tilt {ax['tilt']:+.1f} deg",
                     f"2D SUL {ax['span'] * scale:.1f} mm" if scale else "no arch scale",
                     f"symmetry {ax['score']:.2f}  elong {ax['elongation']:.1f}"])

    right_img = cv2.cvtColor(colorize(depth), cv2.COLOR_RGB2BGR)
    right_img = cv2.resize(right_img, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.line(right_img, q0, q1, (0, 255, 0), 3)
    for q, c in ((q0, (0, 255, 255)), (q1, (255, 0, 255))):
        cv2.circle(right_img, q, 12, c, -1)
    label(right_img, [f"depth map {depth.shape[1]}x{depth.shape[0]}  "
                      f"{depth.min():.0f}-{depth.max():.0f} mm",
                      f"z0 {z0:.1f} mm (yellow)   z1 {z1:.1f} mm (magenta)",
                      f"dz {z1 - z0:+.1f} mm   in-plane {inplane:.1f} mm",
                      f"depth SUL {mm3d:.1f} mm"])

    panel = np.hstack([left_img, right_img])
    out = Path(args.out) if args.out else root / "visualization" / f"{S.video_id(clip)}__depth_debug.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), panel)
    print(f"[done] {out}")
    print(f"  span {ax['span']} px, 2D {ax['span'] * scale:.2f} mm" if scale else "")
    print(f"  z0 {z0:.1f}  z1 {z1:.1f}  dz {z1 - z0:+.1f} mm")
    print(f"  in-plane 3D {inplane:.2f} mm + depth term -> {mm3d:.2f} mm")


if __name__ == "__main__":
    main()
