"""Black out the da Vinci GUI in a video.

    python scripts/mask_video.py in.mp4 out.mp4 [workers]

Template matching is the whole cost and each frame is independent, so frames are
masked in a process pool; decode and write stay sequential.
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

from gui_mask import gui_mask, load_templates

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


def main(src, dst, workers=1):
    global _templates
    _templates = load_templates(TEMPLATE_DIR)   # used directly when workers == 1
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
            out.write(f)
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
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
