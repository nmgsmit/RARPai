"""
da Vinci GUI masking via fixed geometry + template matching.

Method C from the GUI detection notebook — fast, reliable for this console.

Usage:
    from gui_mask import gui_mask

    mask = gui_mask(frame)  # bool mask, 255 = GUI
    # or with templates:
    mask = gui_mask(frame, templates=[tpl1, tpl2], search_band=165)
"""

from pathlib import Path

import cv2
import numpy as np


# Hardcoded geometry for 1920x1080 da Vinci Xi footage
REF_W, REF_H = 1920, 1080
BOTTOM_BAR_H = 56         # bottom bar height, full width
TAB_H, TAB_X, TAB_W = 13, 961, 332   # tab sitting above the bottom bar


def fixed_gui_mask(shape, scale=True):
    """Hard-coded da Vinci GUI geometry: bottom bar + tab.

    Args:
        shape: frame shape (H, W, ...) or (H, W)
        scale: if True, rescale boxes proportionally for non-1920x1080 frames

    Returns:
        bool mask, True = GUI overlay
    """
    H, W = shape[:2]
    sy, sx = (H / REF_H, W / REF_W) if scale else (1.0, 1.0)
    bar = int(round(BOTTOM_BAR_H * sy))
    tab_h = int(round(TAB_H * sy))
    tab_x0 = int(round(TAB_X * sx))
    tab_x1 = int(round((TAB_X + TAB_W) * sx))

    m = np.zeros((H, W), dtype=bool)
    m[max(H - bar, 0):H, :] = True                    # bottom bar, full width
    y1 = max(H - bar, 0)
    y0 = max(y1 - tab_h, 0)
    m[y0:y1, min(tab_x0, W):min(tab_x1, W)] = True    # tab above the bar
    return m


CONNECT_TEMPLATES = ("marker_1", "marker_2", "marker_4")   # digit end-caps, joined in pairs
                                                           # by a thin line
CONNECT_WIDTH = 15                       # px thickness of the drawn connector
CONNECT_PAD = 3                          # px extra on each side — 15 alone leaves the bar edges showing
CONNECT_ALIGN_TOL = 20                   # px: closer than this on an axis = straight bar, else L
# Measured on RARP_voorbeeld_A: real paired cue markers score 0.705-0.828, isolated
# false positives 0.703-0.727 -- the bands overlap, so no threshold separates the
# two on score alone. These are tuned for recall, on the assumption the caller
# rejects the leftovers some other way.
# NOTE: cut_cue_clips no longer relies on these. It validates a marker pair by
# counting bar segments between the markers (see cue_span there), which does
# separate cues from popup lettering. These values still drive mask_video.py.
THRESH_OVERRIDE = {"marker_4": 0.60, "marker_1": 0.65, "marker_2": 0.75}
# The connector markers only ever sit on a few lines inset from the *content* edge (measured:
# 24 px from the left, 39 and 79 px from the bottom). Anything off those lines is a false match.
EDGE_INSETS = (24, 39, 79)
EDGE_TOL = 6
# A marker is only real if it has a partner within one bar length. The bar templates
# (the 334 px-wide popup panels) set the scale; an L pair wraps a corner, so measure
# along the path.
PAIR_MAX_DIST = 334 + 20
MARKER_PAD = 5           # px grown around each connector-marker box; the template crop sits tight


def _dedupe(matches, tw, th):
    """Keep one match per cluster (matchTemplate fires on every near-pixel)."""
    kept = []
    for x, y in matches:
        if not any(abs(x - kx) < tw and abs(y - ky) < th for kx, ky in kept):
            kept.append((x, y))
    return kept


def _paired(hits):
    """Drop hits with no partner within a bar length (path distance, so L pairs count)."""
    return [a for a in hits
            if any(b is not a and abs(a[0] - b[0]) + abs(a[1] - b[1]) <= PAIR_MAX_DIST
                   for b in hits)]


def _content_box(gray, bar_top):
    """(left, top, right, bottom) of the actual image, ignoring the black side bars.

    Insets are measured from here, not from the frame, so a differently pillarboxed
    clip still lines up. The bottom is the GUI bar, which occludes the content anyway.
    """
    lit = gray[:bar_top] > 20
    cols, rows = np.where(lit.any(0))[0], np.where(lit.any(1))[0]
    if not len(cols) or not len(rows):
        return 0, 0, gray.shape[1] - 1, bar_top - 1
    return int(cols[0]), int(rows[0]), int(cols[-1]), bar_top - 1


def _on_gui_line(x, y, box):
    """True if (x, y) sits on one of the lines the GUI markers live on."""
    left, top, right, bottom = box
    for d in (x - left, right - x, y - top, bottom - y):
        if any(abs(d - i) <= EDGE_TOL for i in EDGE_INSETS):
            return True
    return False


def _elbow(p0, p1, shape):
    """Of the two possible L corners, the one bending *away* from frame centre.

    The GUI connector wraps around the nearest screen corner, so the elbow is
    always the candidate furthest from the middle — this picks the right bend in
    all four corners, unlike a fixed horizontal-then-vertical rule.
    """
    H, W = shape[:2]
    cx, cy = W / 2, H / 2
    dist = lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2
    return max([(p1[0], p0[1]), (p0[0], p1[1])], key=dist)


def _connect(m, p0, p1):
    """Draw a bar (or L, if the points are off-axis) between two centres."""
    line = np.zeros(m.shape, dtype=np.uint8)
    w = CONNECT_WIDTH + 2 * CONNECT_PAD
    if abs(p0[0] - p1[0]) < CONNECT_ALIGN_TOL or abs(p0[1] - p1[1]) < CONNECT_ALIGN_TOL:
        cv2.line(line, p0, p1, 255, w)
    else:
        corner = _elbow(p0, p1, m.shape)
        cv2.line(line, p0, corner, 255, w)
        cv2.line(line, corner, p1, 255, w)
    m |= line.astype(bool)


def gui_mask(bgr, templates=None, thresh=0.70, search_band=None, scale=True, roi_bottom=None):
    """Detect GUI: fixed geometry + template matching in a band above the overlay.

    Args:
        bgr: input frame (BGR, uint8)
        templates: list of grayscale template images (uint8), or a {name: image}
            dict as returned by load_templates. Names in CONNECT_TEMPLATES that
            match twice get a bar/L drawn between the two hits.
        thresh: matchTemplate score threshold (0.0-1.0) for TM_CCOEFF_NORMED,
            overridden per template by THRESH_OVERRIDE
        search_band: pixels above the bottom bar to search for templates (None = whole frame)
        scale: if True, rescale fixed boxes for non-1920x1080 frames
        roi_bottom: height (pixels) of bottom ROI to crop before template matching (None = full frame)

    Returns:
        bool mask of shape (H, W), True = GUI overlay
    """
    m = fixed_gui_mask(bgr.shape, scale=scale)
    fixed = m.copy()   # hits already inside the hard-coded boxes don't count as pair ends

    if templates:
        H, W = bgr.shape[:2]

        # Crop to bottom region if specified (faster template matching)
        if roi_bottom:
            crop_top = max(0, H - roi_bottom)
            gray = cv2.cvtColor(bgr[crop_top:, :], cv2.COLOR_BGR2GRAY)
            offset_y = crop_top
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            offset_y = 0

        # Search band above the bottom bar, or the whole (possibly ROI-cropped) frame
        bar_h = int(round(BOTTOM_BAR_H * (H / REF_H if scale else 1.0)))
        bar_top = H - bar_h
        if search_band is None:
            band_top, band = offset_y, gray
        else:
            band_top = max(max(0, bar_top - search_band), offset_y)  # Respect ROI crop
            band = gray[band_top - offset_y:bar_top - offset_y, :]

        items = templates.items() if isinstance(templates, dict) else enumerate(templates)
        for name, tpl in items:
            if tpl is None or tpl.shape[0] > band.shape[0] or tpl.shape[1] > band.shape[1]:
                continue
            res = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED)
            th, tw = tpl.shape[:2]
            t = THRESH_OVERRIDE.get(name, thresh)
            hits = _dedupe([(int(x), int(y)) for y, x in zip(*np.where(res >= t))], tw, th)
            if name in CONNECT_TEMPLATES:
                bx = _content_box(gray, bar_top - offset_y)
                box = (bx[0], bx[1] + offset_y, bx[2], bx[3] + offset_y)   # back to full-frame y
                # on an inset line, not already inside the fixed boxes, and part of a pair
                hits = [(x, y) for x, y in hits if _on_gui_line(x, band_top + y, box)]
                hits = [(x, y) for x, y in hits
                        if not fixed[min(band_top + y + th // 2, H - 1), min(x + tw // 2, W - 1)]]
                hits = _paired(hits)
                # A bar has exactly two ends. Extra hits used to skip _connect below
                # (it needs len == 2), leaving the bar itself unmasked -- so rank by
                # match score and keep the two most confident.
                hits = sorted(hits, key=lambda h: -res[h[1], h[0]])[:2]
            pad = MARKER_PAD if name in CONNECT_TEMPLATES else 0
            for x, y in hits:
                # Translate match coords back to full-frame space
                m[max(band_top + y - pad, 0):band_top + y + th + pad,
                  max(x - pad, 0):x + tw + pad] = True
            if name in CONNECT_TEMPLATES and len(hits) == 2:
                _connect(m, *[(x + tw // 2, band_top + y + th // 2) for x, y in hits])

    return m


def crop_template(frame, x, y, w, h, save_as=None):
    """Cut a grayscale template out of a frame.

    Args:
        frame: input frame (BGR)
        x, y, w, h: template bounding box (top-left and size)
        save_as: path to save the template as grayscale PNG, or None

    Returns:
        grayscale template (uint8)
    """
    tpl = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
    if save_as:
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_as), tpl)
    return tpl


def load_templates(template_dir):
    """Load all .png files from a directory as grayscale templates.

    Args:
        template_dir: Path or str to a directory of template images

    Returns:
        {stem: grayscale uint8 image}, e.g. {"marker_1": ...}. Names matter:
        see CONNECT_TEMPLATES.
    """
    if not template_dir:
        return {}
    path = Path(template_dir)
    if not path.exists():
        return {}
    return {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            for p in sorted(path.glob("*.png"))}


if __name__ == "__main__":
    # 400x600 frame -> bar_top 380, content bottom 379; markers must sit on an inset line
    tpl = np.zeros((10, 10), np.uint8)   # needs variance: TM_CCOEFF_NORMED is undefined on a flat patch
    tpl[2:8, 2:8] = 255
    row = 379 - 39   # bottom line
    frame = np.zeros((400, 600, 3), np.uint8)
    frame[row + 2:row + 8, 102:108] = 255
    frame[row + 2:row + 8, 402:408] = 255
    m = gui_mask(frame, {"marker_2": tpl}, search_band=300)
    assert m[row + 5, 250], "straight bar missing between aligned pair"

    # a match off every inset line is a false positive and must be dropped
    frame = np.zeros((400, 600, 3), np.uint8)
    frame[row + 2:row + 8, 102:108] = 255
    frame[202:208, 302:308] = 255        # mid-frame, on no line
    m = gui_mask(frame, {"marker_2": tpl}, search_band=300)
    assert not m[205, 305], "off-line false positive was not filtered"

    # on the line, but too far apart to be one bar -> both dropped
    frame = np.zeros((400, 600, 3), np.uint8)
    frame[row + 2:row + 8, 22:28] = 255
    frame[row + 2:row + 8, 562:568] = 255
    m = gui_mask(frame, {"marker_2": tpl}, search_band=300)
    assert not m[row + 5, 25] and not m[row + 5, 300], "unpaired hits were not dropped"

    # off-axis pair -> L bending away from centre, in each of the four corners
    shape = (400, 600)
    for name, (a, b), corner in [
        ("top-left",     ((100, 100), (200, 160)), (100, 160)),
        ("top-right",    ((500, 100), (400, 160)), (500, 160)),
        ("bottom-left",  ((100, 300), (200, 240)), (100, 240)),
        ("bottom-right", ((500, 300), (400, 240)), (500, 240)),
    ]:
        assert _elbow(a, b, shape) == corner, f"{name}: elbow bends the wrong way"

    up = 379 - 79    # second line, 40 px above the first
    frame = np.zeros((400, 600, 3), np.uint8)
    frame[up + 2:up + 8, 102:108] = 255
    frame[row + 2:row + 8, 402:408] = 255
    m = gui_mask(frame, {"marker_2": tpl}, search_band=340)
    assert m[row - 10, 105] and m[row + 5, 250], "L connector missing between off-axis pair"
    assert not m[up + 5, 405], "elbow bent toward frame centre instead of away"
    print("ok")
