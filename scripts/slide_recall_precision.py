"""
Explainer slide (16:9 PNG): what recall and precision look like as MASKS.

Six overlap cases, drawn to scale -- the radii actually produce the quoted ratios,
so the picture cannot disagree with the caption. For two circles the overlap has a
closed form (`lens_area`), and `solve_offset` inverts it numerically, which is how
the "half off target" and "this model" panels get their separation instead of being
eyeballed. --self-test checks the drawn geometry against the labels.

    python scripts/slide_recall_precision.py --out outputs/slide_recall_precision.png
"""
from __future__ import annotations
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Circle, Patch         # noqa: E402

GT_C = "#0FB5CE"        # ground-truth outline (cyan, as on the overlay slides)
PR_C = "#FFC400"        # prediction fill (yellow)


def lens_area(r1, r2, d):
    """Area shared by two circles of radii r1, r2 whose centres are d apart."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    tri = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return r1 * r1 * a1 + r2 * r2 * a2 - tri


def solve_offset(r1, r2, target):
    """Centre distance giving `target` overlap area. Overlap falls monotonically
    with d, so bisection is exact enough and needs no derivative."""
    lo, hi = 0.0, r1 + r2
    for _ in range(80):
        mid = (lo + hi) / 2
        if lens_area(r1, r2, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def case_geometry(recall, precision, r_gt=1.0):
    """Radii and offset realising a (recall, precision) pair, GT radius fixed.

    overlap = recall * A_gt  and  A_pred = overlap / precision, so the predicted
    radius follows from the ratio and the offset from solve_offset.
    """
    a_gt = math.pi * r_gt ** 2
    overlap = recall * a_gt
    if precision == 0 or recall == 0:
        return r_gt, r_gt, (r_gt * 2.15)              # disjoint, but only just
    a_pred = overlap / precision
    r_pred = math.sqrt(a_pred / math.pi)
    return r_gt, r_pred, solve_offset(r_gt, r_pred, overlap)


CASES = [
    ("perfect overlap",   1.00, 1.00),
    ("paints too little", 0.50, 1.00),
    ("paints too much",   1.00, 0.50),
    ("half off target",   0.50, 0.50),
    ("completely missed", 0.00, 0.00),
    ("this model",        0.79, 0.89),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/slide_recall_precision.png")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default="What recall and precision mean for a mask")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        # the drawn circles must reproduce the labels, or the figure lies
        for name, rc, pr in CASES:
            r1, r2, d = case_geometry(rc, pr)
            ov = lens_area(r1, r2, d)
            got_r = ov / (math.pi * r1 ** 2)
            got_p = ov / (math.pi * r2 ** 2) if r2 else 0.0
            assert abs(got_r - rc) < 0.01, (name, got_r, rc)
            assert abs(got_p - pr) < 0.01, (name, got_p, pr)
            print(f"  {name:18s} drawn recall={got_r:.3f} precision={got_p:.3f}  ok")
        assert lens_area(1, 1, 0) == math.pi                       # identical
        assert lens_area(1, 1, 5) == 0                             # disjoint
        assert abs(lens_area(2, 1, 0) - math.pi) < 1e-9            # contained
        print("[self-test] geometry matches every label")
        return

    # widest case is the disjoint one (-r1 .. d+r2); size the shared window to it
    WIN = max(max(r1, d + r2) - min(-r1, d - r2)
              for r1, r2, d in (case_geometry(rc, pr) for _, rc, pr in CASES)) * 1.06
    fig, axes = plt.subplots(2, 3, figsize=(13.333, 7.5), dpi=args.dpi)
    fig.patch.set_facecolor("white")

    for ax, (name, rc, pr) in zip(axes.ravel(), CASES):
        r1, r2, d = case_geometry(rc, pr)
        # prediction first, outline on top, so the boundary is never hidden
        ax.add_patch(Circle((d, 0), r2, facecolor=PR_C, edgecolor="none", alpha=0.9))
        ax.add_patch(Circle((0, 0), r1, facecolor="none", edgecolor=GT_C, linewidth=3.5))
        # ONE window size for every panel, so the ground-truth circle renders at
        # the same size throughout -- panels with per-panel autoscale look like the
        # anatomy changed size, which is exactly the comparison this figure makes.
        lo, hi = min(-r1, d - r2), max(r1, d + r2)
        cx = (lo + hi) / 2
        ax.set_xlim(cx - WIN / 2, cx + WIN / 2)
        ax.set_ylim(-WIN / 2, WIN / 2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=19, pad=6)
        ax.text(0.5, -0.10, f"recall {rc:.2f}   precision {pr:.2f}",
                transform=ax.transAxes, ha="center", fontsize=15, color="#444444")

    fig.suptitle(args.title, fontsize=26, fontweight="bold", y=0.975)
    fig.legend(handles=[Patch(fc=PR_C, ec="none", label="Model prediction"),
                        Patch(fc="none", ec=GT_C, lw=3, label="Ground truth (outline)")],
               loc="lower center", ncol=2, frameon=False, fontsize=15,
               bbox_to_anchor=(0.5, 0.012))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.17,
                        wspace=0.05, hspace=0.40)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
