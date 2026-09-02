"""Smoke test for the GUI's inference path: build EndoDAC exactly as the trainer does, save a
RANDOM-weight checkpoint, load it back through DepthBackend and predict.

Weights are random, so the depth values are meaningless -- what this proves is the plumbing:
the vit_base input_size patch grid at 392x490, the state_dict key contract, disp_to_depth with
the ruler run's 20/200 band, and that a measurement comes out finite. Needs torch; run it once
on a new machine before trusting the GUI there.

    python scripts/tests/smoke_depth_backend.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import gui_depth_measure as G                      # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    tmp = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/depth_backend_smoke")
    tmp.mkdir(parents=True, exist_ok=True)
    ckpt = tmp / "random_best.pth"

    sys.path.insert(0, str(REPO / "third_party" / "endodac"))
    G._shim_dotenv()
    from finetune_depth import make_endodac        # noqa: E402

    shape = G.DEFAULT_SHAPE
    model = make_endodac(shape)
    sd = model.state_dict()
    check(len(sd) == 389, f"state_dict has the documented 389 keys (got {len(sd)})")
    torch.save(sd, ckpt)

    be = G.DepthBackend(ckpt, shape, G.DEFAULT_MIN_DEPTH, G.DEFAULT_MAX_DEPTH, device="cpu")
    be.load(log=lambda m: print("        " + m))

    img = Image.fromarray((np.random.default_rng(0).random((1080, 1344, 3)) * 255).astype(np.uint8))
    depth = be.predict(img)
    # The DPT head returns disp at its own resolution (448x560 for a 392x490 feed), NOT the feed
    # size. Harmless here precisely because every measurement is done in normalised coordinates --
    # but it must keep the feed's aspect, or normalised sampling would shear.
    check(abs(depth.shape[1] / depth.shape[0] - shape[1] / shape[0]) < 1e-6,
          f"depth map keeps the feed aspect (got {depth.shape} for feed {shape})")
    check(np.isfinite(depth).all(), "depth map is finite everywhere")
    check(G.DEFAULT_MIN_DEPTH - 1e-3 <= depth.min() and depth.max() <= G.DEFAULT_MAX_DEPTH + 1e-3,
          f"depth lies inside the 20-200 mm band ({depth.min():.1f}-{depth.max():.1f})")

    sess = G.Session()
    m = sess.measure(depth, (0.3, 0.5), (0.7, 0.5))
    check(np.isfinite(m.mm) and m.mm > 0, f"a measurement comes out finite ({m.mm:.2f} mm)")

    # a second image must give a different map -- guards against a frozen/cached prediction
    img2 = Image.fromarray((np.random.default_rng(1).random((1080, 1344, 3)) * 255).astype(np.uint8))
    check(not np.allclose(depth, be.predict(img2)), "a different image gives a different depth map")

    print("FAILURES:", FAILS if FAILS else "none")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
