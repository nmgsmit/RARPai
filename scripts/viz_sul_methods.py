"""Pictures for choosing between the SUL methods, from eval_sul_methods.py's rows.

  <clip>_methods.mp4   one panel per method (A-D), one row per depth source, Nick's masks. Each
                       panel: the frame, the mask outline, Nick's two clicks, and where THAT method
                       puts the start and the end. The strip underneath traces every method's SUL
                       over the clip against Nick's.
  sul_errors.png       every annotated frame as a dot: SUL minus Nick, per method, in each of the
                       four mask x depth combinations. Accuracy (where the dots sit) and stability
                       (how far they spread) in one look.

Colours are the data-viz reference palette, slots 1-4 = A, B, C, D, validated for both surfaces
(light figure, dark video). Marker shape carries the clip, so identity never rests on colour alone.

    python scripts/viz_sul_methods.py --eval outputs/sul_eval/eval_rows.csv
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urethra_cylinder as cur                                    # noqa: E402

METHODS = ["A mask ends", "B first cylinder", "C current cylinder", "D mask along tube"]
SHORT = {"A mask ends": "A  mask ends", "B first cylinder": "B  first version",
         "C current cylinder": "C  current", "D mask along tube": "D  mask ends along tube"}
DEPTHS = [("stereo", "your masks + stereo depth"),
          ("mono_scaled", "your masks + monocular depth (rescaled per frame)")]
CLIPS = ["control", "short", "long"]
RUN_OF = {"control": "seg3_t30", "short": "noocc_seg3_14s", "long": "noocc_5e27_16s"}
# reference palette, dark steps (video, surface #1a1a19) and light steps (figure, #fcfcfb)
BGR = {"A mask ends": (229, 135, 57), "B first cylinder": (38, 89, 217),
       "C current cylinder": (112, 158, 25), "D mask along tube": (0, 133, 201)}
HEX = {"A mask ends": "#2a78d6", "B first cylinder": "#eb6834",
       "C current cylinder": "#1baf7a", "D mask along tube": "#eda100"}
INK, MUTED, GRID, SURFACE = (255, 255, 255), (129, 135, 137), (42, 44, 44), (25, 26, 26)
PW, PH = 469, 375                                                  # panel = 0.35 of 1340x1072


def text(img, s, org, scale=0.46, col=INK, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def dot(img, q, col, r=5):
    """Filled marker with a 2px surface ring, so it stays legible over the tissue."""
    if not np.all(np.isfinite(q)):
        return
    q = (int(round(q[0])), int(round(q[1])))
    cv2.circle(img, q, r + 2, SURFACE, -1, cv2.LINE_AA)
    cv2.circle(img, q, r, col, -1, cv2.LINE_AA)


def panel(img, mask, clicks, row, sc):
    """One method on one frame: mask outline, Nick's clicks, the method's start and end."""
    out = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_AREA)
    cnts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(out, [(c * sc).astype(np.int32) for c in cnts], -1, MUTED, 1, cv2.LINE_AA)
    for q in clicks:                                               # Nick's two ruler points
        x, y = int(round(q[0] * sc)), int(round(q[1] * sc))
        for dx, dy in ((-7, -7, ), (-7, 7)):
            cv2.line(out, (x + dx, y + dy), (x - dx, y - dy), SURFACE, 4, cv2.LINE_AA)
        cv2.drawMarker(out, (x, y), INK, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
    col = BGR[row["method"]]
    s = np.array([row["s_u"], row["s_v"]], float) * sc
    e = np.array([row["e_u"], row["e_v"]], float) * sc
    if np.all(np.isfinite(np.r_[s, e])):
        cv2.line(out, tuple(s.astype(int)), tuple(e.astype(int)), col, 2, cv2.LINE_AA)
    dot(out, s, col)
    dot(out, e, col)
    band = np.full((22, PW, 3), SURFACE, np.uint8)
    if np.isfinite(row["sul"]):
        text(band, "SUL %.1f mm    you %.1f    %+.1f" % (row["sul"], row["ref"], row["sul"] - row["ref"]),
             (8, 16), 0.44)
    else:
        text(band, "no measurement", (8, 16), 0.44, MUTED)
    if not row["counted"]:
        text(band, "not counted", (PW - 104, 16), 0.44, MUTED)
    return np.vstack([out, band])


def strip(series, ref, i_now, w, h, title):
    """SUL over the clip: one 2px line per method, Nick's in ink, cursor on the current frame."""
    img = np.full((h, w, 3), SURFACE, np.uint8)
    L, R, T, B = 54, 12, 30, 22
    vals = [v for s in series.values() for v in s if np.isfinite(v)] + [v for v in ref if np.isfinite(v)]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    pad = max(1.0, 0.08 * (hi - lo))
    lo, hi = lo - pad, hi + pad
    n = max(len(ref), 2)
    X = lambda i: int(L + i / (n - 1) * (w - L - R))
    Y = lambda v: int(T + (hi - v) / (hi - lo) * (h - T - B))
    for tick in np.linspace(lo, hi, 4):                            # hairline solid grid, recessive
        cv2.line(img, (L, Y(tick)), (w - R, Y(tick)), GRID, 1)
        text(img, "%.0f" % tick, (6, Y(tick) + 4), 0.4, MUTED)
    cv2.line(img, (X(i_now), T), (X(i_now), h - B), MUTED, 1)
    def draw(vals_, col, th):
        pts = [(X(i), Y(v)) for i, v in enumerate(vals_) if np.isfinite(v)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, col, th, cv2.LINE_AA)
    draw(ref, INK, 2)
    for m in METHODS:
        draw(series[m], BGR[m], 2)
    text(img, title, (L, 18), 0.46, MUTED)
    text(img, "SUL mm", (6, 18), 0.4, MUTED)
    return img


def legend(w):
    img = np.full((30, w, 3), SURFACE, np.uint8)
    x = 14
    for m in METHODS:
        cv2.circle(img, (x, 15), 5, BGR[m], -1, cv2.LINE_AA)
        text(img, SHORT[m], (x + 12, 20), 0.46)
        x += 20 + 9 * len(SHORT[m])
    cv2.circle(img, (x, 15), 5, INK, -1, cv2.LINE_AA)
    text(img, "you (ruler points)", (x + 12, 20), 0.46)
    return img


def video(clip, rows, root, out, fps):
    run = RUN_OF[clip]
    frames = sorted({r["frame"] for r in rows})
    by = {(r["frame"], r["depth"], r["method"]): r for r in rows}
    with open(os.path.join(root, run, "hand_points.csv")) as fh:
        pts = {r["frame"]: (np.array([float(r["ax"]), float(r["ay"])]),
                            np.array([float(r["bx"]), float(r["by"])])) for r in csv.DictReader(fh)}
    ser = {d: {m: [by.get((f, d, m), {}).get("sul", np.nan) for f in frames] for m in METHODS}
           for d, _ in DEPTHS}
    ref = [by.get((frames[i], "stereo", "A mask ends"), {}).get("ref", np.nan) for i in range(len(frames))]
    vw, sc = None, PW / 1340.0
    for i, f in enumerate(frames):
        img = cv2.imread(os.path.join(root, run, "images", f + ".png"))
        mask = cur.load_mask(os.path.join(root, run, "hand_masks", f + ".png"), img.shape[:2]) == cur.URETHRA
        blocks = []
        for depth, label in DEPTHS:
            head = np.full((24, PW * 4, 3), SURFACE, np.uint8)
            text(head, label, (10, 17), 0.5)
            cols = []
            for m in METHODS:
                r = by.get((f, depth, m))
                cell = panel(img, mask, pts[f], r, sc) if r else np.full((PH + 22, PW, 3), SURFACE, np.uint8)
                tag = np.full((22, PW, 3), SURFACE, np.uint8)
                cv2.circle(tag, (12, 11), 5, BGR[m], -1, cv2.LINE_AA)
                text(tag, SHORT[m], (24, 16), 0.46)
                cols.append(np.vstack([tag, cell]))
            blocks += [head, np.hstack(cols)]
        W = PW * 4
        strips = np.hstack([strip(ser[d], ref, i, W // 2, 210, lab.split(" + ")[1]) for d, lab in DEPTHS])
        frame = np.vstack(blocks + [legend(W), strips])
        if vw is None:
            vw = cv2.VideoWriter(os.path.join(out, clip + "_methods.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    vw.release()
    print("  wrote %s_methods.mp4  (%d frames, %s)" % (clip, len(frames), frame.shape[1::-1]))


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, PRIM, SEC, MUT, GRD, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    shape = {"control": "o", "short": "s", "long": "^"}
    conds = [("hand", "stereo", "your masks + stereo"), ("hand", "mono_scaled", "your masks + monocular"),
             ("model", "stereo", "model masks + stereo"), ("model", "mono_scaled", "model masks + monocular")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.4), sharey=True, facecolor=SURF)
    lo, hi = -15, 20
    rng = np.random.default_rng(0)
    for ax, (mask, depth, title) in zip(axes, conds):
        ax.set_facecolor(SURF)
        ax.axhline(0, color=BASE, lw=1, zorder=1)
        ax.yaxis.grid(True, color=GRD, lw=1)
        ax.set_axisbelow(True)
        for x, m in enumerate(METHODS):
            sel = [r for r in rows if r["mask"] == mask and r["depth"] == depth and r["method"] == m
                   and r["counted"] and np.isfinite(r["sul"])]
            err = np.array([r["sul"] - r["ref"] for r in sel])
            for clip in CLIPS:
                e = np.array([r["sul"] - r["ref"] for r in sel if r["clip"] == clip])
                if not e.size:
                    continue
                ax.scatter(x + rng.uniform(-0.22, 0.22, e.size), np.clip(e, lo, hi), s=26,
                           marker=shape[clip], c=HEX[m], edgecolors=SURF, linewidths=1, zorder=3)
            if err.size:
                ax.plot([x - 0.34, x + 0.34], [np.median(err)] * 2, color=PRIM, lw=2.5, zorder=4)
                ax.text(x, hi - 1.2, "%.1f" % np.median(np.abs(err)), ha="center", va="top",
                        color=PRIM, fontsize=10, fontweight="semibold")
                out = int((err > hi).sum() + (err < lo).sum())
                if out:
                    ax.text(x, lo + 0.6, "%d off-scale" % out, ha="center", color=MUT, fontsize=8)
        ax.set_xticks(range(4), [m.split()[0] for m in METHODS], color=PRIM)
        for x, m in enumerate(METHODS):
            ax.plot(x, lo - 2.0, "o", color=HEX[m], ms=7, clip_on=False)
        ax.set_title(title, color=SEC, fontsize=11, pad=16)
        ax.set_ylim(lo, hi)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASE)
        ax.tick_params(colors=MUT, length=0)
    axes[0].set_ylabel("SUL minus your measurement (mm)", color=SEC)
    axes[0].text(-0.45, hi - 1.2, "median |error|", color=SEC, fontsize=9, va="top")
    handles = [plt.Line2D([], [], marker=shape[c], color=MUT, ls="", mec=SURF, label=c) for c in CLIPS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, labelcolor=SEC,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("SUL error per frame, by method -- dot = one annotated frame, bar = median, "
                 "shape = clip", color=PRIM, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)
    print("  wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="outputs/sul_eval/eval_rows.csv")
    ap.add_argument("--root", default="outputs/temporal_stereo")
    ap.add_argument("--out", default="outputs/sul_eval")
    ap.add_argument("--fps", type=float, default=4.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    num = lambda v: float(v) if v not in ("", "nan", None) else np.nan
    rows = []
    with open(a.eval) as fh:
        for r in csv.DictReader(fh):
            r.update(sul=num(r["sul"]), ref=num(r["ref"]), counted=r["counted"] == "True",
                     s_u=num(r.get("s_u")), s_v=num(r.get("s_v")),
                     e_u=num(r.get("e_u")), e_v=num(r.get("e_v")))
            rows.append(r)
    figure(rows, os.path.join(a.out, "sul_errors.png"))
    byclip = defaultdict(list)
    for r in rows:
        if r["mask"] == "hand" and r["depth"] in dict(DEPTHS):
            byclip[r["clip"]].append(r)
    for clip in CLIPS:
        video(clip, byclip[clip], a.root, a.out, a.fps)


if __name__ == "__main__":
    main()
