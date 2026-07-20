#!/usr/bin/env python
"""Cut clips around da Vinci "move cue" bars, with the GUI masked out.

A cue is a striped bar with a small digit marker at each end. Matching the digit
templates alone is not enough: the same digits sit permanently in the bottom HUD
and also occur in popup-dialog lettering (an "I" scores as a "1"). Two
constraints make detection reliable:

  WHERE  a cue only ever occupies CUE_ROW (a horizontal bar) or an EDGE_BAND
         inset from the *content* edge. Matching is restricted to those bands,
         which both speeds up the scan and removes most false positives.
  WHAT   a marker pair counts only if at least --min-bars stripe templates
         (bar_*.png) lie on the line between them.

A bar whose second digit scores under threshold has no pair to validate, and used
to mask NOTHING. So a lone digit with --min-bars stripes beside it infers its bar
instead: the busiest side gives the direction, and the bar is drawn at its fixed
nominal length, wrapping the screen corner if the edge runs out. See cue_paths.

Thresholds were chosen with scripts/sweep_cue_thresholds.py.

    python scripts/cut_cue_clips.py --videos "data/raw/*.mp4" --out data/clips --mask-gui
"""
import argparse
import csv
import glob
import os
import sys
import time
from multiprocessing.pool import ThreadPool   # cv2 frees the GIL; threads also
                                              # dodge Windows process-spawn errors

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gui_mask import (  # noqa: E402
    BOTTOM_BAR_H, CONNECT_TEMPLATES, PAIR_MAX_DIST, REF_H, _content_box, gui_mask,
    load_templates,
)

GUI_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "templates")
BAR_PREFIX = "bar_"    # data/templates/Move_Que: bar_*.png are stripe segments,
                       # marker_*.png are the digit end-caps

# --- where a cue can appear (measured on this console) -----------------------
# A horizontal bar sits on CUE_ROW, counted from the frame top. The other three
# sides sit EDGE_BAND px in from the CONTENT edge, so the pillarbox offset is
# added automatically and a differently matted clip still lines up.
CUE_ROW = (985, 1005)
EDGE_BAND = (20, 40)
BAND_PAD = 24          # a match at row y spans y..y+th, so slices need slack

# --- detection tuning --------------------------------------------------------
FLOOR_MARK, FLOOR_BAR = 0.45, 0.50   # collection floors, below any usable threshold
CORRIDOR = 22          # px a bar segment may sit off the marker-to-marker line
MAX_MARK, MAX_BAR = 60, 240          # candidates kept per frame, BY SCORE
MAX_PAIR_MARK = 8      # markers entering the pair search (a bar has two ends)
NMS_RADIUS = 8

# --- masking -----------------------------------------------------------------
MARKER_SPAN = 20       # px: largest marker template dimension
CUE_MASK_PAD = 14      # px grown around the bar; the templates crop tight

# --- bar geometry (measured on this console) ---------------------------------
# A cue bar is one digit block at each end with stripe blocks between: 14 blocks
# straight, so ~340 x 22 px overall (340 - 2*17 digits = 306 px over 14 blocks =
# 22 px/block, which is exactly the stripe template width). An L bar wraps a
# screen corner and holds 13 blocks instead of 14 -- the corner eats one.
#
# The bar always rides the perimeter RING of the content box (EDGE_BAND inset on
# every side; note CUE_ROW is just the bottom leg of that ring). So a bar with
# less than BAR_LEN of edge left in front of it does not stop -- it turns the
# corner and continues along the next edge. That is what walk_ring does.
BAR_LEN = 340
BAR_THICK = 22
STRIPE_MAX_DIST = 350  # a stripe further than this from a digit is not its bar
# Inference deliberately uses the SAME high --bar-thresh as the pair path.
# Measured on 43bf7ef9 f200/f205, where a real bar is present: at 0.95 the bar
# yields 9-10 distinct blocks and the rest of the frame yields ZERO matches, and
# at 0.70 it yields 11 blocks against 142 elsewhere. The stripe templates
# separate cleanly at the top of their range, so lowering the threshold to "see
# more of the bar" only buys tissue -- bright yellow anatomy with dark veins
# mimics a hazard stripe, which is what produced the f175 phantom bar.
STRIPE_GAP_MAX = 50    # largest gap allowed inside one stripe run: 2 blocks (44 px)
                       # plus slack, so ONE unmatched block bridges but tissue does not
# Matches closer together than this are the SAME block: _peaks runs NMS per
# template, so the four stripe templates all fire on one physical block and it
# would otherwise count four times -- which is how three tissue matches used to
# add up to a run. Counting is per distinct block, just under the 22 px pitch.
STRIPE_MIN_SEP = 20
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ----------------------------------------------------------------- geometry
def content_box(gray):
    """Content rectangle (pillarbox removed). Static per video, so hoist it."""
    H = gray.shape[0]
    return _content_box(gray, max(H - int(round(BOTTOM_BAR_H * H / REF_H)), 1))


def cue_bands(box):
    """(x_offset, y_offset, slice) for each band a cue marker can occupy."""
    left, top, right, bottom = box
    lo, hi = EDGE_BAND
    r0, r1 = max(CUE_ROW[0], top), min(CUE_ROW[1], bottom)
    bands = []
    if r1 > r0:                                                    # bottom cue row
        bands.append((left, r0, (slice(r0, min(r1 + BAND_PAD, bottom)), slice(left, right))))
    bands.append((left, top + lo,                                  # top edge
                  (slice(top + lo, min(top + hi + BAND_PAD, bottom)), slice(left, right))))
    bands.append((left + lo, top,                                  # left edge
                  (slice(top, bottom), slice(left + lo, min(left + hi + BAND_PAD, right)))))
    bands.append((max(right - hi - BAND_PAD, left), top,           # right edge
                  (slice(top, bottom), slice(max(right - hi - BAND_PAD, left), right - lo))))
    return bands


def in_cue_band(x, y, box):
    """True if a match at (x, y) lies where a cue marker can actually be."""
    left, top, right, bottom = box
    lo, hi = EDGE_BAND
    if not (left <= x <= right and top <= y <= bottom):
        return False       # pillarbox: no GUI is ever drawn out there
    if CUE_ROW[0] <= y <= CUE_ROW[1]:
        return True
    return (lo <= y - top <= hi) or (lo <= x - left <= hi) or (lo <= right - x <= hi)


# ---------------------------------------------------------------- detection
def _peaks(res, thr, tw, th, cap):
    """Local maxima of a matchTemplate result -> [(x, y, score)].

    Vectorised NMS: a repeating stripe yields hundreds of raw matches, which the
    O(n^2) python dedupe in gui_mask cannot handle.
    """
    mx = cv2.dilate(res, np.ones((th, tw), np.uint8))
    ys, xs = np.where((res >= thr) & (res >= mx - 1e-6))
    if len(xs) == 0:
        return []
    sc = res[ys, xs]
    return [(int(xs[i]), int(ys[i]), float(sc[i])) for i in np.argsort(-sc)[:cap]]


def _nms(pts, r=NMS_RADIUS):
    """Best-first spatial dedupe."""
    kept = []
    for x, y, sc in sorted(pts, key=lambda q: -q[2]):
        if not any(abs(x - kx) < r and abs(y - ky) < r for kx, ky, _ in kept):
            kept.append((x, y, sc))
    return kept


def collect_frame(gray, markers, bars, box):
    """Marker and bar candidates above the collection floors, full-frame coords."""
    mk, br = [], []
    for x0, y0, sl in cue_bands(box):
        band = gray[sl]
        for group, floor, cap, sink in ((markers, FLOOR_MARK, MAX_MARK, mk),
                                        (bars, FLOOR_BAR, MAX_BAR, br)):
            for tpl in group.values():
                if tpl.shape[0] > band.shape[0] or tpl.shape[1] > band.shape[1]:
                    continue
                th, tw = tpl.shape[:2]
                res = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED)
                # bars ride the same bands as the markers
                sink += [(x0 + x, y0 + y, sc) for x, y, sc in _peaks(res, floor, tw, th, cap)
                         if in_cue_band(x0 + x, y0 + y, box)]
    # cap BY SCORE, never band order, or the first band fills the quota with
    # noise before the real bar in a later band is reached
    mk.sort(key=lambda q: -q[2])
    br.sort(key=lambda q: -q[2])
    return (np.asarray(mk[:MAX_MARK], np.float32).reshape(-1, 3),
            np.asarray(br[:MAX_BAR], np.float32).reshape(-1, 3))


def bars_between(p0, p1, bx, by):
    """Count bar segments on the line joining two markers (bx, by are arrays)."""
    (x0, y0), (x1, y1) = p0, p1
    if len(bx) == 0:
        return 0
    # counted per distinct BLOCK: the stripe templates overlap, so one block can
    # produce several matches and inflate the count
    if abs(y0 - y1) <= CORRIDOR:                       # horizontal bar
        ym, lo, hi = (y0 + y1) / 2, min(x0, x1), max(x0, x1)
        return _distinct(bx[(np.abs(by - ym) <= CORRIDOR) & (bx > lo) & (bx < hi)])
    if abs(x0 - x1) <= CORRIDOR:                       # vertical bar
        xm, lo, hi = (x0 + x1) / 2, min(y0, y1), max(y0, y1)
        return _distinct(by[(np.abs(bx - xm) <= CORRIDOR) & (by > lo) & (by < hi)])
    # ponytail: L-shaped cues (wrapping a screen corner) score 0 and are rejected.
    # None occur in the four reference videos; add the two-leg case if one shows up.
    return 0


def cue_span(mk, br, m_thr, b_thr, min_bars):
    """The validated cue as a marker pair (p0, p1), or None if this is not a cue.

    Returns the WIDEST validated pair: the bar runs the full span, and picking a
    shorter sub-pair would leave its tail unmasked.
    """
    if len(mk) == 0:
        return None
    m = _nms([tuple(q) for q in mk[mk[:, 2] >= m_thr]])[:MAX_PAIR_MARK]
    if len(m) < 2:
        return None
    b = br[br[:, 2] >= b_thr] if len(br) else br
    bx = b[:, 0] if len(b) else np.empty(0, np.float32)
    by = b[:, 1] if len(b) else np.empty(0, np.float32)
    best, best_len = None, 0
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            (x0, y0, _), (x1, y1, _) = m[i], m[j]
            span = abs(x0 - x1) + abs(y0 - y1)
            if span > PAIR_MAX_DIST or span <= best_len:
                continue
            if bars_between((x0, y0), (x1, y1), bx, by) >= min_bars:
                best, best_len = ((x0, y0), (x1, y1)), span
    return best


def _distinct(vals, sep=STRIPE_MIN_SEP):
    """Count positions at least `sep` apart, i.e. distinct blocks not raw matches."""
    n, last = 0, None
    for v in np.sort(vals):
        if last is None or v - last >= sep:
            n += 1
            last = v
    return n


def _run_len(along):
    """Blocks in the stripe run starting AT the digit, walking outwards.

    Counting every stripe within STRIPE_MAX_DIST instead lets three unrelated
    tissue matches 250 px away validate a bar that is not there -- observed on
    43bf7ef9 f155, which masked a 340 px strip of anatomy. A real bar is a
    CONTIGUOUS chain of blocks butted against the digit, so the run has to start
    next to the digit, may never skip more than a block, and only counts a block
    once however many templates matched it.
    """
    n = 0
    reach = STRIPE_GAP_MAX          # how far the chain currently extends
    last = None
    for a in np.sort(along):
        if a > reach:
            break                   # gap too wide: the chain ended before this stripe
        if last is not None and a - last < STRIPE_MIN_SEP:
            continue                # same block, another template
        n += 1
        last = a
        reach = a + STRIPE_GAP_MAX
    return n


def stripe_dir(c, bx, by):
    """(direction, count) of the stripe run beside a digit: the busiest side wins.

    A digit with stripes on BOTH sides is ambiguous -- the bar can only go one
    way, so the side holding the longer run is the bar and the other side is
    noise (or the neighbouring leg of an L). Stripes beyond STRIPE_MAX_DIST
    belong to some other bar and never count.
    """
    best, best_n = None, 0
    for d in DIRS:
        along = (bx - c[0]) * d[0] if d[0] else (by - c[1]) * d[1]
        perp = np.abs(by - c[1]) if d[0] else np.abs(bx - c[0])
        keep = (perp <= CORRIDOR) & (along > 0) & (along <= STRIPE_MAX_DIST)
        n = _run_len(along[keep])
        if n > best_n:
            best, best_n = d, n
    return best, best_n


def _inset(p, d, box):
    """How far p sits in from the ring edge it is travelling along."""
    left, top, right, bottom = box
    return min(p[1] - top, bottom - p[1]) if d[0] else min(p[0] - left, right - p[0])


def _limit(p, d, box, ins):
    """Distance from p to the ring corner ahead, travelling in d."""
    left, top, right, bottom = box
    if d == (1, 0):
        return (right - ins) - p[0]
    if d == (-1, 0):
        return p[0] - (left + ins)
    if d == (0, 1):
        return (bottom - ins) - p[1]
    return p[1] - (top + ins)


def _turn(p, d, box):
    """At a ring corner, the perpendicular leg that stays INSIDE the box.

    Travelling right along the bottom leg and hitting the right edge, only 'up'
    continues along the ring; 'down' leaves the content entirely. So pick the
    perpendicular pointing away from whichever edge we are closest to.
    """
    left, top, right, bottom = box
    if d[0]:
        return (0, -1) if (bottom - p[1]) < (p[1] - top) else (0, 1)
    return (-1, 0) if (right - p[0]) < (p[0] - left) else (1, 0)


def walk_ring(p, d, length, box, max_legs=3):
    """Walk `length` px from p along the ring, turning corners -> polyline points."""
    pts = [p]
    for _ in range(max_legs):
        ins = _inset(p, d, box)
        step = max(min(length, _limit(p, d, box, ins)), 0)
        p = (p[0] + d[0] * step, p[1] + d[1] * step)
        pts.append(p)
        length -= step
        if length <= 1:
            break
        d = _turn(p, d, box)
    return pts


def _centre(x, y):
    """Match coords are template top-left; the bar runs through the middle."""
    return (x + MARKER_SPAN / 2, y + MARKER_SPAN / 2)


def cue_paths(mk, br, box, m_thr, b_thr, min_bars):
    """Every cue bar in the frame as a polyline (2 points straight, 3+ for an L).

    Strict first: a validated marker PAIR pins both ends exactly, so no length has
    to be assumed. Only when that fails does a LONE digit infer its bar from the
    stripe run beside it -- previously such a frame masked nothing at all, because
    a bar whose second digit scored under threshold has no pair to validate. The
    inferred bar is drawn at its full nominal BAR_LEN: the length is fixed by the
    console, so stopping at the last matched stripe would leave the tail showing.
    """
    span = cue_span(mk, br, m_thr, b_thr, min_bars)
    if span is not None:
        return [[_centre(*span[0]), _centre(*span[1])]]
    if len(mk) == 0 or len(br) == 0:
        return []
    b = br[br[:, 2] >= b_thr]
    if len(b) == 0:
        return []
    bx, by = b[:, 0], b[:, 1]
    paths = []
    for q in _nms([tuple(v) for v in mk[mk[:, 2] >= m_thr]])[:MAX_PAIR_MARK]:
        d, n = stripe_dir((q[0], q[1]), bx, by)
        if n >= min_bars:
            paths.append(walk_ring(_centre(q[0], q[1]), d, BAR_LEN, box))
    return paths


def frame_has_cue(mk, br, box, m_thr, b_thr, min_bars):
    """Whether this frame shows a real cue. Used by sweep_cue_thresholds."""
    return bool(cue_paths(mk, br, box, m_thr, b_thr, min_bars))


# ------------------------------------------------------------------ masking
def ring_mask(shape, box):
    """Every band a cue can occupy -- the only pixels a cue mask may ever cover."""
    m = np.zeros(shape[:2], bool)
    for _, _, sl in cue_bands(box):
        m[sl] = True
    return m


def cue_mask(shape, paths, box=None, pad=CUE_MASK_PAD):
    """Bool mask covering every cue bar polyline, drawn at the bar's thickness.

    Clipped to the cue bands when `box` is given. The bar only ever rides those
    bands, so anything the thick line spills outside them is tissue that no cue
    could have covered -- a mask wider than the search region can only destroy
    real pixels, never GUI.
    """
    m = np.zeros(shape[:2], np.uint8)
    for pts in paths:
        for a, b in zip(pts, pts[1:]):
            cv2.line(m, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255,
                     BAR_THICK + 2 * pad)
    m = m.astype(bool)
    return m & ring_mask(shape, box) if box is not None else m


def band_mask(shape, mk, br, box, m_thr, b_thr):
    """Whole search band blacked out as soon as ONE match lands in it.

    cue_span only masks between two VALIDATED markers, so a bar whose second
    marker scored under threshold keeps its tail visible -- a false negative that
    leaks GUI pixels into the clip. A cue never occupies anything but these bands,
    so blacking the entire band on a single hit trades masked area (which the
    valid mask simply drops) for never missing part of a bar.
    """
    m = np.zeros(shape[:2], bool)
    hits = ([(q[0], q[1]) for q in mk if q[2] >= m_thr] +
            [(q[0], q[1]) for q in br if q[2] >= b_thr])
    if not hits:
        return m
    for _, _, sl in cue_bands(box):
        rows, cols = sl
        if any(cols.start <= x < cols.stop and rows.start <= y < rows.stop for x, y in hits):
            m[sl] = True
    return m


def black_gui(frame, panels, markers, bars, box, m_thr, b_thr, min_bars, dilate,
              bands=False):
    """Black out every GUI element in place: fixed HUD, popup panels, cue bar.

    Black rather than inpaint because finetune_depth's `valid` mask is
    `mean(RGB) > vignette_thresh` (0.04), so zeroed pixels drop out of the
    photometric, smoothness and anchor losses automatically.

    Popups and the cue bar are handled separately because they fail differently.
    Popups land OUTSIDE the cue bands by design (a dialog is not a cue) and are
    caught by the wide panel templates. The bar uses cue_span, NOT gui_mask's
    connector: that connector joins whichever pair matched, and on voorbeeld_A
    f149 it spanned x402-954 while the bar really ran to x1254, leaving the tail
    visible in the clip.

    The mask is dilated because the dataset resizes BILINEAR *before* computing
    its valid mask, so an undilated black/tissue edge interpolates to greys above
    0.04 and leaves a rim of GUI-contaminated pixels marked valid.
    """
    mk, br = collect_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), markers, bars, box)
    m = (gui_mask(frame, panels) |
         cue_mask(frame.shape, cue_paths(mk, br, box, m_thr, b_thr, min_bars), box))
    if bands:
        m = m | band_mask(frame.shape, mk, br, box, m_thr, b_thr)
    m = m.astype(np.uint8)
    if dilate > 0:
        m = cv2.dilate(m, np.ones((2 * dilate + 1,) * 2, np.uint8))
    frame[m.astype(bool)] = 0
    return frame


# -------------------------------------------------------------- clip spans
def runs(hits, pad_before, pad_after, n, min_len, gap):
    """Consecutive True runs -> padded (start, end) inclusive, merged across gaps.

    Padding ADDS context around the cue; it never trims the cue itself.
    """
    spans, start = [], None
    for i, h in enumerate(hits):
        if h and start is None:
            start = i
        elif not h and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(hits) - 1))

    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] - 1 <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    padded = [(max(0, s - pad_before), min(n - 1, e + pad_after)) for s, e in merged]
    # min_len is the length of the CLIP that gets written, padding included
    return [(s, e) for s, e in padded if e - s + 1 >= min_len]


# ------------------------------------------------------------------ workers
# Each worker opens its own VideoCapture over a contiguous frame range, so decode
# is parallel too. ThreadPool shares _W directly, so no config is pickled.
_W = {}


def _scan_chunk(job):
    """(start, count) -> [(frame_idx, is_cue)] for the strided frames in range."""
    start, count = job
    cap = cv2.VideoCapture(_W["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for i in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        if i % _W["stride"] == 0:   # decode is unavoidable; matching is what we skip
            mk, br = collect_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                   _W["markers"], _W["bars"], _W["box"])
            out.append((start + i, int(frame_has_cue(mk, br, _W["box"], _W["m_thr"],
                                                     _W["b_thr"], _W["min_bars"]))))
    cap.release()
    return out


def _cut_clip(job):
    """Write one clip (masked mp4, plus frames if asked). Clips are independent."""
    n, s, e = job
    cap = cv2.VideoCapture(_W["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    dst = os.path.join(_W["outdir"], f"{_W['stem']}_clip{n:03d}_f{s:06d}-{e:06d}.mp4")
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), _W["fps"], _W["wh"])
    imgdir = os.path.join(_W["framedir"], f"clip_{n:03d}", "images") if _W["framedir"] else None
    if imgdir:
        os.makedirs(imgdir, exist_ok=True)
    for j in range(e - s + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if _W["panels"] is not None:
            frame = black_gui(frame, _W["panels"], _W["markers"], _W["bars"], _W["box"],
                              _W["m_thr"], _W["b_thr"], _W["min_bars"], _W["dilate"],
                              _W["bands"])
        vw.write(frame)
        if imgdir:
            cv2.imwrite(os.path.join(imgdir, f"frame_{j:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
    vw.release()
    cap.release()
    return os.path.basename(dst), e - s + 1


def _dump_negatives(flags, every):
    """Write every Nth non-cue frame (masked) to nocue/ so misses are inspectable."""
    dst = os.path.join(_W["outdir"], "nocue")
    os.makedirs(dst, exist_ok=True)
    cap = cv2.VideoCapture(_W["path"])
    n = 0
    for i, cued in enumerate(flags):
        ok, frame = cap.read()
        if not ok:
            break
        if cued or i % every:
            continue
        if _W["panels"] is not None:
            frame = black_gui(frame, _W["panels"], _W["markers"], _W["bars"], _W["box"],
                              _W["m_thr"], _W["b_thr"], _W["min_bars"], _W["dilate"],
                              _W["bands"])
        cv2.imwrite(os.path.join(dst, f"frame_{i:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    return n


def _run_pool(fn, jobs, nw):
    if nw > 1 and len(jobs) > 1:
        with ThreadPool(min(nw, len(jobs))) as pool:
            return pool.map(fn, jobs)
    return [fn(j) for j in jobs]


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+", required=True, help="video files or globs")
    p.add_argument("--templates", default="data/templates/Move_Que")
    p.add_argument("--out", default="data/clips")
    # Defaults from scripts/sweep_cue_thresholds.py: the only point whose every
    # neighbour also yields 28 clips with 0 bad on the four reference videos.
    p.add_argument("--marker-thresh", type=float, default=0.55,
                   help="digit-marker threshold (low is safe; the bars do the rejecting)")
    p.add_argument("--bar-thresh", type=float, default=0.90, help="bar-segment threshold")
    p.add_argument("--min-bars", type=int, default=4,
                   help="stripe segments required to call it a cue: between the two markers "
                        "of a validated pair, or beside a lone digit")
    p.add_argument("--pad-before", type=int, default=0,
                   help="context frames ADDED before the cue (0 = start on the cue)")
    p.add_argument("--pad-after", type=int, default=0, help="context frames ADDED after")
    p.add_argument("--min-len", type=int, default=15,
                   help="skip clips shorter than this many frames (padding included)")
    p.add_argument("--gap", type=int, default=3, help="merge runs this many frames apart")
    p.add_argument("--mask-gui", action="store_true", help="black out the GUI in clips")
    p.add_argument("--band-mask", action="store_true",
                   help="black the WHOLE search band on a single match; no partial bars, more "
                        "masked area (needs --mask-gui)")
    p.add_argument("--dilate", type=int, default=3,
                   help="px to grow the GUI mask; covers the bilinear-resize rim")
    p.add_argument("--frames", action="store_true",
                   help="also write frame_*.jpg in finetune_depth's */*/clip_*/images layout")
    p.add_argument("--neg-every", type=int, default=0,
                   help="also dump every Nth NON-cue frame (masked) to nocue/ for inspection")
    p.add_argument("--scan-only", action="store_true", help="write scores.csv, cut nothing")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="worker threads")
    p.add_argument("--stride", type=int, default=1,
                   help="match every Nth frame; 3 is safe here and ~2x faster")
    args = p.parse_args()

    allt = {os.path.splitext(os.path.basename(f))[0]: cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            for f in sorted(glob.glob(os.path.join(args.templates, "*.png")))}
    markers = {k: v for k, v in allt.items() if not k.startswith(BAR_PREFIX)}
    bars = {k: v for k, v in allt.items() if k.startswith(BAR_PREFIX)}
    if not markers or not bars:
        raise SystemExit(f"need marker AND bar templates in {args.templates}")
    # panels = every GUI template that is NOT a cue marker; these mask the popups
    panels = ({k: v for k, v in load_templates(GUI_TEMPLATE_DIR).items()
               if k not in CONNECT_TEMPLATES} if args.mask_gui else None)
    print(f"markers: {sorted(markers)}")
    print(f"bars   : {sorted(bars)}")
    print(f"marker>={args.marker_thresh} bar>={args.bar_thresh} min_bars={args.min_bars}")

    nw = max(1, args.workers)
    cv2.setNumThreads(1)      # the pool provides the parallelism

    for path in [q for v in args.videos for q in (glob.glob(v) or [v])]:
        stem = os.path.splitext(os.path.basename(path))[0]
        outdir = os.path.join(args.out, stem)   # created only if this video has a cue

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"skip (cannot open): {path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame0 = cap.read()
        cap.release()
        if not ok:
            print(f"skip (empty): {path}")
            continue

        _W.update(path=path, markers=markers, bars=bars, stride=args.stride,
                  # the pillarbox is static per video -> measure once, not per frame
                  box=content_box(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)),
                  m_thr=args.marker_thresh, b_thr=args.bar_thresh, min_bars=args.min_bars,
                  panels=panels, dilate=args.dilate, bands=args.band_mask,
                  fps=fps, wh=(w, h), outdir=outdir,
                  stem=stem,
                  framedir=os.path.join(args.out, stem, "cues") if args.frames else None)

        t0 = time.time()
        chunk = max(1, -(-n_frames // nw))
        jobs = [(s, min(chunk, n_frames - s)) for s in range(0, n_frames, chunk)]
        scanned = sorted(x for part in _run_pool(_scan_chunk, jobs, nw) for x in part)
        print(f"{stem}: scanned {len(scanned)}/{n_frames} frames in {time.time() - t0:.0f}s "
              f"({nw} workers, stride {args.stride})")

        # strided samples -> per-frame flags: a cued sample covers its whole stride
        flags = [False] * n_frames
        for idx, cued in scanned:
            for j in range(idx, min(idx + args.stride, n_frames)):
                flags[j] = bool(cued)

        spans = runs(flags, args.pad_before, args.pad_after, len(flags), args.min_len, args.gap)
        print(f"{stem}: {len(flags)} frames, {sum(flags)} cued, {len(spans)} clips")
        if not spans:
            print(f"  no cue -> nothing written for {stem}")
            continue
        os.makedirs(outdir, exist_ok=True)

        with open(os.path.join(outdir, "scores.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "cue"])
            wr.writerows([[j, int(c)] for j, c in enumerate(flags)])

        if args.neg_every > 0:
            print(f"  nocue frames: {_dump_negatives(flags, args.neg_every)}")
        if args.scan_only:
            continue

        t0 = time.time()
        for name, nf in _run_pool(_cut_clip, [(n, s, e) for n, (s, e) in enumerate(spans)], nw):
            print(f"  {name}  ({nf} frames)")
        print(f"  cut in {time.time() - t0:.0f}s")


def _self_test():
    # asymmetric padding: starts on the cue, extends past the end
    assert runs([0, 0, 1, 1, 0, 0, 0], 0, 2, 7, 1, 0) == [(2, 5)]
    # padding never trims the cue: the run stays fully inside the span
    s, e = runs([0, 1, 1, 0], 0, 2, 4, 1, 0)[0]
    assert s <= 1 and e >= 2
    # clamping at both ends
    assert runs([1, 1, 0, 0], 3, 3, 4, 1, 0) == [(0, 3)]
    # gap merging: two runs 2 apart merge at gap=3, stay split at gap=1
    f = [1, 1, 0, 0, 1, 1]
    assert runs(f, 0, 0, 6, 1, 3) == [(0, 5)]
    assert runs(f, 0, 0, 6, 1, 1) == [(0, 1), (4, 5)]
    # min_len measures the WRITTEN clip, so padding counts toward it:
    # a 3-frame run + 2 trailing pad = 5 frames, kept at min_len 5, dropped at 6
    f5 = [0, 1, 1, 1, 0, 0, 0]
    assert runs(f5, 0, 2, 7, 5, 0) == [(1, 5)]
    assert runs(f5, 0, 2, 7, 6, 0) == []
    # a cue band never extends into the pillarbox
    box = (286, 3, 1631, 1023)
    assert in_cue_band(300, 995, box) and not in_cue_band(100, 995, box)
    # cue_mask spans both markers plus padding
    m = cue_mask((1080, 1920), [[(936, 987), (1254, 987)]])
    assert m[987, 936] and m[987, 1254] and m[987, 1100] and not m[987, 500]
    # clipped to the bands: the bar row survives, the spill above it does not
    mc = cue_mask((1080, 1920), [[(936, 987), (1254, 987)]], box)
    assert mc[987, 1100] and not mc[960, 1100], "mask spilled outside the cue band"

    # --- single-digit inference ---------------------------------------------
    # busiest side wins: 6 stripes right, 2 left -> the bar runs right
    bx = np.array([1000, 1022, 1044, 1066, 1088, 1110, 950, 928], np.float32)
    by = np.full(8, 990, np.float32)
    assert stripe_dir((980, 990), bx, by) == ((1, 0), 6)
    # a stripe past STRIPE_MAX_DIST belongs to some other bar
    assert stripe_dir((980, 990), np.array([980 + STRIPE_MAX_DIST + 10], np.float32),
                      np.array([990], np.float32))[1] == 0
    # the run must butt against the digit: 3 stripes 250 px away are tissue noise,
    # not a bar (the 43bf7ef9 f155 false positive)
    far = np.array([1230, 1252, 1274], np.float32)
    assert stripe_dir((980, 990), far, np.full(3, 990, np.float32))[1] == 0
    # ... and one missed block inside a run still bridges
    assert _run_len(np.array([22, 66, 88], np.float32)) == 3
    assert _run_len(np.array([22, 100, 122], np.float32)) == 1
    # four templates firing on ONE block is one block, not a run of four
    assert _run_len(np.array([20, 22, 24, 26], np.float32)) == 1
    assert _distinct(np.array([20, 22, 24, 26], np.float32)) == 1
    assert _distinct(np.array([20, 42, 64, 86], np.float32)) == 4
    # perpendicular stripes do not count toward a horizontal run
    assert stripe_dir((980, 990), np.array([980], np.float32),
                      np.array([1010], np.float32))[0] == (0, 1)

    # straight walk: room to spare, so one leg of exactly BAR_LEN
    pts = walk_ring((600, 990), (1, 0), BAR_LEN, box)
    assert len(pts) == 2 and pts[1] == (600 + BAR_LEN, 990)
    # near the right edge the walk turns the corner and heads up, total = BAR_LEN
    pts = walk_ring((1400, 990), (1, 0), BAR_LEN, box)
    path = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(pts, pts[1:]))
    assert len(pts) >= 3 and abs(path - BAR_LEN) < 2, (pts, path)
    assert pts[1][1] > pts[2][1], "corner turned down/outside instead of up into the box"

    # one digit + enough stripes infers a bar; too few infers nothing
    mk1 = np.array([[980, 990, .9]], np.float32)
    br6 = np.stack([bx[:6], by[:6], np.full(6, .95, np.float32)], 1)
    assert len(cue_paths(mk1, br6, box, .55, .9, 3)) == 1
    assert cue_paths(mk1, br6[:2], box, .55, .9, 3) == []
    # cue_span picks the WIDEST validated pair, not the first (span < PAIR_MAX_DIST)
    mk = np.array([[100, 990, .9], [250, 990, .9], [400, 990, .9]], np.float32)
    br = np.array([[x, 990, .95] for x in range(110, 400, 20)], np.float32)
    assert cue_span(mk, br, .55, .9, 5) == ((100, 990), (400, 990))
    # ... and rejects a pair with no bar between it
    assert cue_span(mk, np.empty((0, 3), np.float32), .55, .9, 5) is None
    # a validated PAIR still wins over inference: exact span, not an assumed 340 px
    assert cue_paths(mk, br, box, .55, .9, 5) == [[_centre(100, 990), _centre(400, 990)]]
    print("self-test ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
