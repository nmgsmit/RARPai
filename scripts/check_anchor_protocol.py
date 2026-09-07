"""
QC a new annotation dump against the anchor protocol, BEFORE any GPU time is spent on it.

The thing that decides whether annotations are usable is not how many there are, it is the DEPTH
SPREAD they cover: calibration solves Z' = a*Z + b, and anchors all sitting at one distance leave
the two terms confounded. Measured on the existing data (scripts/anchor_spread_curve.py): a 5 mm
band gives 23.7% length error (p90 36%), 15 mm gives 15.7%, 30 mm gives 12.7% -- the floor.

Nothing here needs a model. The distance to a known-size object follows from its size in pixels:

    z = fx * mm / L_px          fx in pixels for the annotated frame size

so a dump can be graded the moment it is annotated.

    python scripts/check_anchor_protocol.py ../data/processed/<new_dump>
    python scripts/check_anchor_protocol.py <dir> --class-id 3      # grade the arm on its own
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = {1: "Ruler", 2: "Catheter tip", 3: "Robot arm"}
# Optical-flow propagation copies one drawn object across a clip: it multiplies the COUNT while
# adding almost no depth spread, so the two are reported apart. In the ruler dumps 1246 of 1472
# ruler objects are "tracked", and every one of the 416 arm objects is "measured" -- there is not
# a single hand-drawn arm anchor, which is why --anchor-sources manual silently drops the arm.
PROPAGATED = {"tracked", "hold"}
MIN_SPREAD_MM = 30.0        # anchor_spread_curve: 30 mm reaches the 12.7% floor, 15 mm = 15.7%
MIN_OBJECTS = 20            # same table: 20 -> 16.7%, 50 -> 13.4%, fewer than 10 is unusable


def objects(video_dir, fx):
    """(class_id, z_true, n_px, source) for every annotated segment under one video."""
    out = []
    for p in sorted(Path(video_dir).glob("*/scale_objects.json")):
        d = json.loads(p.read_text())
        sw = d["frame_size"][0]
        for objs in d["frames"].values():
            for o in objs:
                pts = np.asarray(o.get("points") or [], np.float32)
                if len(pts) < 2 or not o.get("mm"):
                    continue
                px = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                if px < 1:
                    continue
                out.append((int(o["class_id"]), fx * sw * float(o["mm"]) / px, px,
                            o.get("source") or "manual"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dump dir holding <video>/<clip>/scale_objects.json")
    ap.add_argument("--fx", type=float, default=0.82, help="normalised focal (DEFAULT_K_NORM[0])")
    ap.add_argument("--class-id", type=int, default=None, help="grade only this class")
    ap.add_argument("--min-spread", type=float, default=MIN_SPREAD_MM)
    ap.add_argument("--min-objects", type=int, default=MIN_OBJECTS)
    a = ap.parse_args()

    vids = sorted(p for p in Path(a.root).rglob("*") if p.is_dir()
                  and any(p.glob("*/scale_objects.json")))
    if not vids:
        raise SystemExit(f"no <video>/<clip>/scale_objects.json under {a.root}")
    print(f"{'video':<26} {'class':<13} {'n':>4} {'drawn':>6} "
          f"{'z p5-p95 (mm)':>16} {'spread':>7}  verdict")
    fails = 0
    for v in vids:
        rows = objects(v, a.fx)
        classes = sorted({c for c, *_ in rows}) if a.class_id is None else [a.class_id]
        for c in classes:
            z = np.array([r[1] for r in rows if r[0] == c])
            man = sum(1 for r in rows if r[0] == c and r[3] not in PROPAGATED)
            if not len(z):
                continue
            lo, hi = np.percentile(z, [5, 95])
            spread, ok = hi - lo, []
            if spread < a.min_spread:
                ok.append(f"SPREAD {spread:.0f}<{a.min_spread:.0f}mm")
            if man < a.min_objects:                # propagated copies are not independent
                ok.append(f"DRAWN {man}<{a.min_objects}")
            fails += bool(ok)
            print(f"{v.name[:24]:<26} {CLASS_NAMES.get(c, c):<13} {len(z):>4} {man:>6} "
                  f"{lo:>7.0f}-{hi:<8.0f} {spread:>6.0f}  "
                  + ("FAIL: " + ", ".join(ok) if ok else "ok"))
    print(f"\n{fails} video/class group(s) below protocol "
          f"(need >={a.min_objects} objects spanning >={a.min_spread:.0f} mm of depth).")
    print("A group that fails SPREAD cannot be calibrated -- re-annotate that object at a "
          "DIFFERENT working distance; more annotations at the same distance will not help.")


def _selfcheck(tmp):
    """Two clips: one spanning 40mm of depth, one stuck at a single distance."""
    root = Path(tmp) / "vidA"
    for name, pxs in (("clip_0", [40, 50, 70, 120]), ("clip_1", [80, 80, 80, 80])):
        (root / name).mkdir(parents=True)
        (root / name / "scale_objects.json").write_text(json.dumps(
            {"frame_size": [1000, 800], "frames": {
                str(i): [{"class_id": 3, "mm": 8.0, "points": [[0, 0], [px, 0]]}]
                for i, px in enumerate(pxs)}}))
    z = np.array([r[1] for r in objects(root, 0.82)])
    assert len(z) == 8, len(z)
    near = 0.82 * 1000 * 8.0 / 120                    # biggest px = nearest object
    assert abs(z.min() - near) < 1e-3, (z.min(), near)
    assert np.ptp(z) > 60, np.ptp(z)                  # the varied clip supplies the spread
    print("[selfcheck] anchor protocol check OK")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            _selfcheck(t)
    else:
        main()
