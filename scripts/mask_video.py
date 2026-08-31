"""Black out the da Vinci GUI in a video.

    python scripts/mask_video.py in.mp4 out.mp4 [workers] [--crop]

With --crop the frame is also cut down to the content box: pillarbox black edges
(left/right/top) and the bottom GUI bar all removed, instead of left as black. The
box is measured once from the first frame and held fixed for the rest of the video
-- pillarboxing doesn't change mid-clip, and a VideoWriter needs one size throughout.

Template matching is the whole cost and each frame is independent, so frames are
masked in a process pool; decode and write stay sequential.
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from gui_mask import BOTTOM_BAR_H, REF_H, _content_box, gui_mask, load_templates

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "data" / "templates"
BATCH = 256          # frames held in memory at once (~1.5 GB at 1080p)

_templates = None


def _init():
    global _templates
    _templates = load_templates(TEMPLATE_DIR)
    cv2.setNumThreads(1)   # the pool provides the parallelism


def _mask_one(frame):
    frame[gui_mask(frame, _templates)] = 0
    return frame


def main(src, dst, workers=1, crop=False):
    global _templates
    _templates = load_templates(TEMPLATE_DIR)   # used directly when workers == 1
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    box = (0, 0, w - 1, h - 1)
    if crop:
        ok, frame0 = cap.read()
        if not ok:
            raise SystemExit(f"[abort] {src}: no frames to size the crop from")
        bar_h = int(round(BOTTOM_BAR_H * h / REF_H))
        box = _content_box(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY), h - bar_h)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind past the frame we just peeked
    left, top, right, bottom = box
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                           (right - left + 1, bottom - top + 1))

    def batches():
        buf = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buf.append(frame)
            if len(buf) == BATCH:
                yield buf
                buf = []
        if buf:
            yield buf

    t0 = time.perf_counter()
    n = 0
    pool = Pool(workers, initializer=_init) if workers > 1 else None
    for buf in batches():
        masked = pool.map(_mask_one, buf) if pool else [_mask_one(f) for f in buf]
        for f in masked:
            out.write(f[top:bottom + 1, left:right + 1] if crop else f)
        n += len(buf)
        print(f"{n} frames, {(time.perf_counter() - t0) / n:.3f} s/frame", flush=True)
    if pool:
        pool.close()
        pool.join()
    cap.release()
    out.release()

    dt = time.perf_counter() - t0
    print(f"{n} frames in {dt:.1f} s ({dt / max(n, 1):.3f} s/frame, {workers} workers) -> {dst}")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        # 40 px black pillarbox on all sides + a 6 px bottom bar (bar_h at REF scale
        # would be BOTTOM_BAR_H, so shrink REF_H to make a small frame's bar visible).
        frame = np.zeros((120, 120, 3), np.uint8)
        frame[40:100, 40:100] = 200
        bar_h = int(round(BOTTOM_BAR_H * 120 / REF_H))
        box = _content_box(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 120 - bar_h)
        assert box == (40, 40, 99, 120 - bar_h - 1), f"content box wrong: {box}"
        print("ok")
        raise SystemExit

    args = [a for a in sys.argv[1:] if a != "--crop"]
    main(args[0], args[1], int(args[2]) if len(args) > 2 else 1, "--crop" in sys.argv)
