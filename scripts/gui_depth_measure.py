r"""
Point-to-point METRIC MEASUREMENT GUI on top of a trained EndoDAC depth model.

Pick an image -> the model predicts a depth map in MILLIMETRES -> click two points ->
the app back-projects both pixels to 3D with the camera intrinsics and reports the
Euclidean distance between them.

Why this is metric at all: outputs/depth_ruler_range_sw05 was trained with the ruler
scale loss (--scale-w 0.5, --min-depth 20 --max-depth 200), the only term that pins the
otherwise-arbitrary global scale. Held-out test scale ratio was 0.974, i.e. ~2.6% short on
average -- so a length here is "mm, +-a few percent", not a caliper. Use Calibrate (below)
against a known object in the same shot when you need better than that.

    python scripts/gui_depth_measure.py                       # defaults, see below
    python scripts/gui_depth_measure.py --ckpt path\to\best.pth --data-dir path\to\images
    python scripts/gui_depth_measure.py --self-test           # geometry + cache, no torch needed
    python scripts/gui_depth_measure.py --no-model            # UI only (synthetic depth)

Every depth map is computed once and kept, in `outputs/depth_cache/<run>_<ckpt>_<fingerprint>/`
as float32 mm under the npz key `depth`. The checkpoint dropdown lists every `outputs/*/*.pth`
and switching to one switches to ITS cache folder, so its maps are picked straight back up --
the folder is keyed by the checkpoint's CONTENTS, so re-copying the same best.pth keeps the
cache while a genuinely different one can never serve a stale map. "Precompute depth for folder"
fills the whole file list ahead of time; cached frames are marked with a dot.

Geometry (identical convention to finetune_depth.scale_loss, which is what made the model
metric): with NORMALISED intrinsics (fx, fy, cx, cy) and a pixel (u, v) in an image of size
(W, H), X = (u/W - cx) * z / fx, Y = (v/H - cy) * z / fy, Z = z. Normalised means the maths
is resolution-free: sampling the depth map at its own resolution and clicking at display
resolution give the same answer.

Framing matters more than anything else here. The model was trained on the 5:4 ruler dumps
(pillarbox bars already removed), and the assumed SCARED intrinsics describe THAT view. Feed
it a raw 16:9 console frame with black bars and both the depth and the mm are wrong. The
crop panel (auto-detects the bars on load) exists to put the image back into training
framing -- check the aspect readout says ~1.25 before trusting a number.

ponytail: Tkinter (stdlib -> nothing to install on Windows beyond torch/PIL/numpy), single
file, model loaded lazily so --self-test and --no-model run on a laptop with no checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Bump on every user-visible change. Printed at start-up and shown in the title bar, so
# "I don't see the new buttons" is one glance instead of a guess about which copy is running.
BUILD = 4   # 1: measure  2: depth cache + checkpoint picker  3: scrollable panel, build stamp
            # 4: per-stage timing in the status line + faster (uncompressed) cache writes

# ------------------------------------------------------------------ defaults (best run)
DEFAULT_CKPT = REPO / "outputs" / "depth_ruler_range_sw05" / "best.pth"
DEFAULT_DATA = REPO.parent / "data"
DEFAULT_CACHE = REPO / "outputs" / "depth_cache"
DEFAULT_SHAPE = (392, 490)        # what the run was trained/eval'd at
DEFAULT_MIN_DEPTH = 20.0          # NOT the 0.1/150 default: the ruler run's endoscopic band
DEFAULT_MAX_DEPTH = 200.0
DEFAULT_K_NORM = (0.82, 1.02, 0.5, 0.5)   # EndoDAC / SCARED assumed K, normalised
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ------------------------------------------------------------------------- geometry
def backproject(u_n, v_n, z, k_norm):
    """Normalised pixel (u/W, v/H) + depth -> 3D point in the camera frame, same units as z."""
    fx, fy, cx, cy = k_norm
    return np.array([(u_n - cx) * z / fx, (v_n - cy) * z / fy, z], dtype=np.float64)


def segment_length(p0_n, p1_n, z0, z1, k_norm):
    """Euclidean 3D length (mm) of the segment between two clicked pixels."""
    a = backproject(p0_n[0], p0_n[1], z0, k_norm)
    b = backproject(p1_n[0], p1_n[1], z1, k_norm)
    return float(np.linalg.norm(b - a))


def sample_depth(depth, u_n, v_n, radius=2):
    """Median depth over a (2r+1)^2 patch around the normalised point. Median, not the raw
    pixel: on thin structures (ruler edge, instrument shaft) a single sample can sit on the
    depth discontinuity and be off by centimetres."""
    h, w = depth.shape
    u = int(round(np.clip(u_n, 0, 1) * (w - 1)))
    v = int(round(np.clip(v_n, 0, 1) * (h - 1)))
    r = max(0, int(radius))
    patch = depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    return float(np.median(patch)) if patch.size else float(depth[v, u])


def auto_bars(img_np, thresh=18.0, max_frac=0.35):
    """Detect pillarbox/letterbox: leading+trailing rows/cols whose mean intensity is near
    black. Returns (side, top, bottom) as fractions. Symmetric side crop (the console frame
    is), independent top/bottom (a banner is not)."""
    g = img_np.mean(2) if img_np.ndim == 3 else img_np
    h, w = g.shape
    col, row = g.mean(0), g.mean(1)
    l = int(np.argmax(col > thresh)) if (col > thresh).any() else 0
    r = int(np.argmax(col[::-1] > thresh)) if (col > thresh).any() else 0
    t = int(np.argmax(row > thresh)) if (row > thresh).any() else 0
    b = int(np.argmax(row[::-1] > thresh)) if (row > thresh).any() else 0
    side = min(max(l, r) / w, max_frac)
    return (side, min(t / h, max_frac), min(b / h, max_frac))


def crop_box(size, side, top, bottom):
    """(W,H) + fractions -> PIL box (l, t, r, b), always non-degenerate."""
    w, h = size
    l, r = int(round(w * side)), int(round(w * (1 - side)))
    t, b = int(round(h * top)), int(round(h * (1 - bottom)))
    if r - l < 8:
        l, r = 0, w
    if b - t < 8:
        t, b = 0, h
    return (l, t, r, b)


def colorize(disp, lo=None, hi=None):
    """disp HxW -> magma uint8 HxWx3 (near = bright), 5-95 pct stretch like the training panels."""
    import matplotlib
    lo = np.percentile(disp, 2) if lo is None else lo
    hi = np.percentile(disp, 98) if hi is None else hi
    out = np.clip((disp - lo) / (hi - lo + 1e-8), 0, 1)
    return (matplotlib.colormaps["magma"](out)[:, :, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------- depth backend
def _shim_dotenv():
    """finetune_depth does `from dotenv import load_dotenv` for wandb creds this GUI never uses.
    Rather than make measuring depend on python-dotenv being installed on the laptop, stand in a
    no-op module when it is genuinely missing."""
    try:
        import dotenv  # noqa: F401
    except ImportError:
        import types as _t
        m = _t.ModuleType("dotenv")
        m.load_dotenv = lambda *a, **k: False
        sys.modules["dotenv"] = m


class DepthBackend:
    """Lazy EndoDAC wrapper. Nothing here is imported until the first predict()."""

    def __init__(self, ckpt, image_shape, min_depth, max_depth, device=None):
        self.ckpt, self.image_shape = Path(ckpt), tuple(image_shape)
        self.min_depth, self.max_depth = min_depth, max_depth
        self.device, self.model, self._fns = device, None, None

    def load(self, log=print):
        import torch
        sys.path.insert(0, str(REPO / "scripts"))
        sys.path.insert(0, str(REPO / "third_party" / "endodac"))
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        _shim_dotenv()
        # Reuse the trainer's builders rather than copying them: make_endodac carries the exact
        # GUI arg set + the vit_base(input_size=) monkeypatch, and a drift between the two would
        # silently reshape the patch grid and ruin the depth.
        from finetune_depth import make_endodac, round14, disp_to_depth   # noqa: E402

        h, w = round14(self.image_shape[0]), round14(self.image_shape[1])
        self.image_shape = (h, w)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log(f"building endodac{(h, w)} on {self.device} ...")
        model = make_endodac((h, w)).to(self.device).eval()
        sd = torch.load(self.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        mdict = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in mdict and v.shape == mdict[k].shape}
        model.load_state_dict(keep, strict=False)
        log(f"loaded {len(keep)}/{len(mdict)} tensors from {self.ckpt.name}")
        if len(keep) < 0.9 * len(mdict):
            log("WARNING: most weights did not match -- wrong checkpoint or --image-shape?")
        self.model, self._fns = model, (torch, disp_to_depth)
        return self

    def predict(self, pil_img):
        """PIL RGB (already cropped to training framing) -> depth map (H,W) float32, mm.

        The DPT head returns disp at its own resolution (448x560 for a 392x490 feed), not the
        feed size; it is kept as-is and sampled in normalised coordinates, which is exact
        because the aspect matches."""
        if self.model is None:
            self.load()
        torch, disp_to_depth = self._fns
        from torchvision import transforms
        h, w = self.image_shape
        x = transforms.ToTensor()(pil_img.convert("RGB").resize((w, h)))
        with torch.no_grad():
            disp = self.model(x.unsqueeze(0).to(self.device))[("disp", 0)]
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        return depth[0, 0].float().cpu().numpy()


class SyntheticBackend:
    """--no-model: a smooth depth ramp so the UI (and the maths) can be exercised without a
    checkpoint. Numbers it produces are meaningless by construction."""

    def __init__(self, image_shape=DEFAULT_SHAPE, **_):
        self.image_shape = tuple(image_shape)

    def predict(self, pil_img):
        h, w = self.image_shape
        yy = np.linspace(0, 1, h)[:, None] * np.ones((1, w))       # depth increases downwards
        xx = np.linspace(-1, 1, w)[None, :] * np.ones((h, 1))      # + a mild bowl, so a length
        return (60.0 + 40.0 * yy + 10.0 * xx ** 2).astype(np.float32)   # depends on both points


# ----------------------------------------------------------------------- depth cache
def file_fingerprint(path, digest_size=6, chunk=1 << 22):
    """Short blake2b of a file's CONTENTS. Content, not mtime/size: re-downloading the same
    best.pth must land in the cache directory that already holds its depth maps, while an
    actually-different checkpoint must never share one."""
    h = hashlib.blake2b(digest_size=digest_size)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class DepthCache:
    """One directory per checkpoint; one .npz per (image, crop), named `<stem>_<img>_<crop>.npz`.

    Depth is a deterministic function of checkpoint, image, crop, feed shape and depth band, so a
    hit is exact and can never be stale: a different checkpoint lands in a different directory,
    anything else changes the filename. Nothing is invalidated, so the folder doubles as a
    permanent depth dump other scripts can read (`np.load(f)["depth"]` -> mm, float32).

    The name splits into an image part and a crop part on purpose: "is this frame cached at all"
    is then a dict lookup, so the file list can be marked without opening 2000 images to work out
    what their auto-crop would be.
    """

    VERSION = 1

    def __init__(self, root, ckpt, image_shape, min_depth, max_depth, log=print):
        self.image_shape, self.min_depth, self.max_depth = tuple(image_shape), min_depth, max_depth
        ckpt = Path(ckpt)
        fp = file_fingerprint(ckpt) if ckpt.is_file() else "nockpt"
        stem = f"{ckpt.parent.name}_{ckpt.stem}" if ckpt.is_file() else "synthetic"
        self.name = f"{stem}_{fp}"
        self.dir = Path(root) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "meta.json"
        if not meta.exists():
            meta.write_text(json.dumps(
                {"version": self.VERSION, "checkpoint": str(ckpt), "fingerprint": fp,
                 "image_shape": list(self.image_shape), "min_depth": min_depth,
                 "max_depth": max_depth,
                 "note": "depth maps in mm, float32, under key 'depth'"}, indent=2))
        self.index = {}                          # img_key -> {crop_key, ...}
        for f in self.dir.glob("*.npz"):
            parts = f.stem.rsplit("_", 2)
            if len(parts) == 3:
                self.index.setdefault(parts[1], set()).add(parts[2])
        log(f"cache {self.name}: {self.count()} maps over {len(self.index)} frames")

    def count(self):
        return sum(len(v) for v in self.index.values())

    def img_key(self, img_path):
        """Identity of the source frame. Size+mtime are in it, so re-exporting a frame under the
        same name recomputes instead of silently serving the old depth."""
        img_path = Path(img_path)
        try:
            st = img_path.stat()
            stamp = f"{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            stamp = "?"
        raw = f"{img_path.resolve()}|{stamp}|{self.image_shape}|{self.min_depth}:{self.max_depth}|{self.VERSION}"
        return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()

    def crop_key(self, fracs):
        return hashlib.blake2b(",".join(f"{f:.6f}" for f in fracs).encode(),
                               digest_size=4).hexdigest()

    def path_for(self, img_path, ik, ck):
        return self.dir / f"{Path(img_path).stem}_{ik}_{ck}.npz"

    def has(self, img_path, fracs):
        return self.crop_key(fracs) in self.index.get(self.img_key(img_path), ())

    def has_any(self, img_path):
        """Any crop of this frame cached -- what the file list marks. Accurate in practice because
        the auto-crop is a deterministic function of the image."""
        return bool(self.index.get(self.img_key(img_path)))

    def get(self, img_path, fracs):
        ik, ck = self.img_key(img_path), self.crop_key(fracs)
        if ck not in self.index.get(ik, ()):
            return None
        try:
            with np.load(self.path_for(img_path, ik, ck)) as z:
                return z["depth"]
        except Exception:                        # noqa: BLE001 - a truncated file is just a miss
            self.index.get(ik, set()).discard(ck)
            return None

    def put(self, img_path, fracs, depth):
        ik, ck = self.img_key(img_path), self.crop_key(fracs)
        # Plain (uncompressed) npz: benchmarked at <1ms/write vs ~50ms compressed for a
        # 448x560 float32 map, for ~11% more disk -- irrelevant next to seconds of CPU
        # inference, and it adds up over a big precompute run for nothing in return.
        np.savez(self.path_for(img_path, ik, ck), depth=depth.astype(np.float32),
                meta=json.dumps({"source": str(Path(img_path).resolve()),
                                 "crop_fracs": list(fracs),
                                 "image_shape": list(self.image_shape),
                                 "min_depth": self.min_depth,
                                 "max_depth": self.max_depth,
                                 "version": self.VERSION}))
        self.index.setdefault(ik, set()).add(ck)


class DepthProvider:
    """backend + cache + a small in-memory LRU. `depth_for` is the only entry point the GUI uses.

    Three tiers because they cost three different things: RAM (instant, for prev/next flipping),
    the cache directory (~10 ms, survives restarts), the model (seconds on CPU)."""

    def __init__(self, backend, cache=None, mem=8):
        self.backend, self.cache, self.mem_max = backend, cache, mem
        self.mem = {}                            # key -> depth, insertion-ordered = LRU
        self.computed = 0                        # model runs this session (the smoke test reads it)

    def _remember(self, key, depth):
        self.mem[key] = depth
        while len(self.mem) > self.mem_max:
            self.mem.pop(next(iter(self.mem)))
        return depth

    def cached(self, img_path, fracs=None):
        """Is a map for this frame available without running the model? With `fracs`, that exact
        crop; without, any crop (what the file list marks)."""
        if self.cache is None:
            return False
        return self.cache.has(img_path, fracs) if fracs is not None \
            else self.cache.has_any(img_path)

    def depth_for(self, img_path, pil_crop, fracs, on_compute=None):
        """Returns (depth, tier, elapsed_seconds). Timing is reported, not guessed at: when
        something is slow, the status line already says whether that was the model or disk."""
        t0 = time.perf_counter()
        if self.cache is None:
            self.computed += 1
            return self.backend.predict(pil_crop), "model", time.perf_counter() - t0
        key = (self.cache.img_key(img_path), self.cache.crop_key(fracs))
        if key in self.mem:
            self.mem[key] = self.mem.pop(key)    # touch: most recently used goes last
            return self.mem[key], "memory", time.perf_counter() - t0
        hit = self.cache.get(img_path, fracs)
        if hit is not None:
            return self._remember(key, hit), "cache", time.perf_counter() - t0
        if on_compute:
            on_compute()
        self.computed += 1
        depth = self.backend.predict(pil_crop)
        t_model = time.perf_counter() - t0
        self.cache.put(img_path, fracs, depth)
        return self._remember(key, depth), "model", time.perf_counter() - t0


# ------------------------------------------------------------------------ app state
@dataclass
class Measurement:
    p0: tuple           # normalised (u, v) in the CROPPED image
    p1: tuple
    z0: float
    z1: float
    mm: float           # already scale-corrected
    label: str = ""


@dataclass
class Session:
    k_norm: tuple = DEFAULT_K_NORM
    scale: float = 1.0                      # user scale correction on top of the model
    radius: int = 2
    items: list = field(default_factory=list)

    def measure(self, depth, p0, p1):
        z0 = sample_depth(depth, *p0, self.radius)
        z1 = sample_depth(depth, *p1, self.radius)
        mm = segment_length(p0, p1, z0, z1, self.k_norm) * self.scale
        return Measurement(p0, p1, z0 * self.scale, z1 * self.scale, mm)

    def recompute(self, depth):
        """Re-run every measurement after a scale / radius / intrinsics change."""
        self.items = [self.measure(depth, m.p0, m.p1) for m in self.items]


# ------------------------------------------------------------------------- self-test
def self_test():
    """Fronto-parallel plane: a horizontal segment has an exactly known metric length.
    Mirrors finetune_depth._selfcheck_scale_loss, so a green run here means this GUI measures
    the same way the scale loss that trained the model did."""
    k = DEFAULT_K_NORM
    fx = k[0]
    h, w, z = 384, 480, 50.0
    depth = np.full((h, w), z, np.float32)
    u0, u1 = 120.0, 360.0
    p0, p1 = (u0 / w, 0.5), (u1 / w, 0.5)
    expect = (u1 - u0) / w * z / fx
    got = segment_length(p0, p1, z, z, k)
    assert abs(got - expect) < 1e-6, (got, expect)

    # vertical segment uses fy
    v0, v1 = 100.0, 300.0
    got_v = segment_length((0.5, v0 / h), (0.5, v1 / h), z, z, k)
    assert abs(got_v - (v1 - v0) / h * z / k[1]) < 1e-6, got_v

    # pure depth difference: length is |z1-z0| when both points sit at the principal point
    assert abs(segment_length((0.5, 0.5), (0.5, 0.5), 40.0, 70.0, k) - 30.0) < 1e-9

    # resolution invariance: same normalised points on a 4x depth map -> same mm
    d4 = np.full((h * 4, w * 4), z, np.float32)
    assert abs(sample_depth(d4, *p0) - sample_depth(depth, *p0)) < 1e-6

    # patch median ignores a single outlier pixel (thin-structure robustness)
    d = np.full((64, 64), 50.0, np.float32)
    d[32, 32] = 500.0
    assert abs(sample_depth(d, 0.5, 0.5, radius=2) - 50.0) < 1e-6

    # scale correction is a pure multiplier; calibration inverts it exactly
    s = Session(scale=1.0)
    m = s.measure(depth, p0, p1)
    s.scale = 25.0 / m.mm
    assert abs(s.measure(depth, p0, p1).mm - 25.0) < 1e-9

    # ---- cache: a hit must be exact, and a miss must be a miss
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="depthcache_selftest_"))
    try:
        ck1, ck2 = tmp / "a.pth", tmp / "b.pth"
        ck1.write_bytes(b"weights-A"); ck2.write_bytes(b"weights-B")
        img = tmp / "frame_0001.png"
        img.write_bytes(b"not really a png, only its stat() matters here")
        c1 = DepthCache(tmp / "cache", ck1, (392, 490), 20, 200, log=lambda m: None)
        d = np.full((8, 10), 42.0, np.float32)
        assert not c1.has(img, (0, 0, 0)) and not c1.has_any(img)
        c1.put(img, (0, 0, 0), d)
        assert c1.has(img, (0, 0, 0)) and c1.has_any(img) and c1.count() == 1
        assert np.array_equal(c1.get(img, (0, 0, 0)), d)              # round-trips exactly
        assert c1.get(img, (0.15, 0, 0)) is None                      # other crop -> miss
        c1.put(img, (0.15, 0, 0), d * 2)
        assert c1.count() == 2 and len(c1.index) == 1                 # 2 crops, 1 frame
        assert np.array_equal(c1.get(img, (0, 0, 0)), d)              # crops do not collide

        c1b = DepthCache(tmp / "cache", ck1, (392, 490), 20, 200, log=lambda m: None)
        assert c1b.dir == c1.dir and c1b.count() == 2                 # index rebuilt from disk
        assert np.array_equal(c1b.get(img, (0, 0, 0)), d)

        c2 = DepthCache(tmp / "cache", ck2, (392, 490), 20, 200, log=lambda m: None)
        assert c2.dir != c1.dir and c2.count() == 0                   # other checkpoint, own dir
        c3 = DepthCache(tmp / "cache", ck1, (392, 448), 20, 200, log=lambda m: None)
        assert c3.dir == c1.dir and not c3.has(img, (0, 0, 0))        # other feed shape -> miss

        ck1.write_bytes(b"weights-A")                                 # same CONTENT, new mtime
        assert DepthCache(tmp / "cache", ck1, (392, 490), 20, 200,
                          log=lambda m: None).dir == c1.dir           # fingerprint, not mtime

        # ---- provider: model runs at most once per (image, crop)
        class CountingBackend:
            image_shape = (392, 490)
            calls = 0

            def predict(self, pil):
                CountingBackend.calls += 1
                return np.full((8, 10), 7.0, np.float32)

        prov = DepthProvider(CountingBackend(), DepthCache(tmp / "cache2", ck1, (392, 490), 20, 200,
                                                           log=lambda m: None))
        a, src_a, _ = prov.depth_for(img, None, (0, 0, 0))
        b_, src_b, _ = prov.depth_for(img, None, (0, 0, 0))
        assert (src_a, src_b) == ("model", "memory") and CountingBackend.calls == 1
        assert np.array_equal(a, b_)
        prov.mem.clear()
        _, src_c, _ = prov.depth_for(img, None, (0, 0, 0))
        assert src_c == "cache" and CountingBackend.calls == 1        # disk hit, no model run
        assert prov.cached(img) and prov.cached(img, (0, 0, 0))
        assert not prov.cached(img, (0.2, 0, 0))
        prov.mem_max = 2                                              # LRU evicts, never grows
        for k in range(5):
            prov._remember(f"k{k}", d)
        assert list(prov.mem) == ["k3", "k4"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # bar detection recovers a known pillarbox, and crop_box is pixel-exact
    img = np.zeros((108, 192, 3), np.uint8)
    img[6:100, 29:163] = 200
    side, top, bot = auto_bars(img)
    assert abs(side - 29 / 192) < 0.02 and abs(top - 6 / 108) < 0.02 and abs(bot - 8 / 108) < 0.02
    assert crop_box((192, 108), 29 / 192, 6 / 108, 8 / 108) == (29, 6, 163, 100)
    assert crop_box((192, 108), 0.49, 0.49, 0.49) == (0, 0, 192, 108)   # degenerate -> full frame
    print("self-test OK")


# ------------------------------------------------------------------------------- GUI
def run_gui(args, on_ready=None):
    """Build and run the window. `on_ready(ctx)` is a test hook: it gets the live widgets and
    closures 200 ms after start-up so a headless smoke test can drive the UI (see
    scripts/tests/smoke_gui_depth_measure.py); production callers never pass it."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk

    sess = Session(k_norm=tuple(args.intrinsics), scale=args.scale, radius=args.patch_radius)

    root = tk.Tk()
    root.title(f"RARPai - depth ruler (build {BUILD})")
    root.geometry("1500x900")

    state = dict(path=None, full=None, crop=None, depth=None, disp_img=None, tkimg=None,
                 zoom=1.0, base=1.0, pending=None, files=[], idx=-1,
                 provider=None, abort=False)

    # ---------------------------------------------------------------- left: controls
    # The control column SCROLLS. Tk clips a too-tall frame silently -- on a laptop screen that
    # quietly hides whichever controls do not fit, which looks exactly like the app being out of
    # date. A canvas + inner frame means new sections can never disappear.
    left_outer = ttk.Frame(root)
    left_outer.pack(side="left", fill="y")
    left_canvas = tk.Canvas(left_outer, width=252, highlightthickness=0, takefocus=0)
    left_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
    left_canvas.configure(yscrollcommand=left_scroll.set)
    left_scroll.pack(side="right", fill="y")
    left_canvas.pack(side="left", fill="y", expand=True)
    left = ttk.Frame(left_canvas, padding=8)
    left_win = left_canvas.create_window((0, 0), window=left, anchor="nw")
    left.bind("<Configure>",
              lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
    left_canvas.bind("<Configure>",
                     lambda e: left_canvas.itemconfigure(left_win, width=e.width))
    for w in (left_canvas, left):
        w.bind("<MouseWheel>", lambda e: left_canvas.yview_scroll(
            -1 if e.delta > 0 else 1, "units"))

    canvas = tk.Canvas(root, bg="#1e1e1e", highlightthickness=0, cursor="tcross")
    canvas.pack(side="right", fill="both", expand=True)

    def sect(text):
        ttk.Label(left, text=text, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 2))

    def num_row(parent, label, init, width=7):
        row = ttk.Frame(parent); row.pack(fill="x")
        ttk.Label(row, text=label, width=11).pack(side="left")
        var = tk.StringVar(value=str(init))
        ttk.Entry(row, textvariable=var, width=width).pack(side="left")
        return var

    sect("Depth model")
    ckpt_var = tk.StringVar(value="")
    ckpt_box = ttk.Combobox(left, textvariable=ckpt_var, state="readonly",
                            values=[], width=28)
    ckpt_box.pack(fill="x")
    ttk.Button(left, text="Browse for checkpoint...",
               command=lambda: browse_ckpt()).pack(fill="x", pady=2)
    cache_lbl = ttk.Label(left, text="cache -", foreground="#666", wraplength=230,
                          justify="left")
    cache_lbl.pack(anchor="w")
    ttk.Button(left, text="Precompute depth for folder",
               command=lambda: precompute_all()).pack(fill="x", pady=2)

    sect("Image")
    ttk.Button(left, text="Open folder...", command=lambda: pick_folder()).pack(fill="x")
    ttk.Button(left, text="Open image...", command=lambda: pick_file()).pack(fill="x", pady=2)
    filebox = tk.Listbox(left, height=10, exportselection=False)
    filebox.pack(fill="x")
    nav = ttk.Frame(left); nav.pack(fill="x", pady=2)
    ttk.Button(nav, text="< Prev", command=lambda: step(-1)).pack(side="left", expand=True, fill="x")
    ttk.Button(nav, text="Next >", command=lambda: step(1)).pack(side="left", expand=True, fill="x")

    sect("Crop to training framing")
    v_side = num_row(left, "side frac", 0.0)
    v_top = num_row(left, "top frac", 0.0)
    v_bot = num_row(left, "bottom frac", 0.0)
    aspect_lbl = ttk.Label(left, text="aspect -", foreground="#666")
    aspect_lbl.pack(anchor="w")
    cr = ttk.Frame(left); cr.pack(fill="x", pady=2)
    ttk.Button(cr, text="Auto bars", command=lambda: set_auto_crop()).pack(side="left", expand=True, fill="x")
    ttk.Button(cr, text="Apply", command=lambda: reload_image(keep_marks=False)).pack(side="left", expand=True, fill="x")

    sect("View")
    v_alpha = tk.DoubleVar(value=0.45)
    ttk.Scale(left, from_=0.0, to=1.0, variable=v_alpha,
              command=lambda _=None: redraw()).pack(fill="x")
    ttk.Label(left, text="depth overlay opacity", foreground="#666").pack(anchor="w")
    zr = ttk.Frame(left); zr.pack(fill="x", pady=2)
    for txt, f in (("-", 1 / 1.25), ("Fit", 0.0), ("+", 1.25)):
        ttk.Button(zr, text=txt, width=5, command=lambda f=f: zoom(f)).pack(side="left", expand=True, fill="x")

    sect("Measurement")
    v_scale = num_row(left, "scale x", f"{args.scale:g}")
    v_rad = num_row(left, "patch px", args.patch_radius)
    v_fx = num_row(left, "fx", args.intrinsics[0])
    v_fy = num_row(left, "fy", args.intrinsics[1])
    ttk.Button(left, text="Apply params", command=lambda: apply_params()).pack(fill="x", pady=2)

    listbox = tk.Listbox(left, height=9, font=("Consolas", 9), exportselection=False)
    listbox.pack(fill="x", pady=(6, 2))
    br = ttk.Frame(left); br.pack(fill="x")
    ttk.Button(br, text="Undo", command=lambda: undo()).pack(side="left", expand=True, fill="x")
    ttk.Button(br, text="Clear", command=lambda: clear()).pack(side="left", expand=True, fill="x")
    ttk.Button(left, text="Calibrate from selected...",
               command=lambda: calibrate()).pack(fill="x", pady=2)
    er = ttk.Frame(left); er.pack(fill="x")
    ttk.Button(er, text="Export CSV", command=lambda: export_csv()).pack(side="left", expand=True, fill="x")
    ttk.Button(er, text="Save PNG", command=lambda: save_png()).pack(side="left", expand=True, fill="x")

    status = ttk.Label(left, text="ready", foreground="#0a0", wraplength=230, justify="left")
    status.pack(anchor="w", pady=(10, 0))
    readout = ttk.Label(left, text="", font=("Consolas", 9), foreground="#444")
    readout.pack(anchor="w")

    def log(msg, color="#0a0"):
        status.configure(text=str(msg), foreground=color)
        root.update_idletasks()

    # ------------------------------------------------------------------ checkpoint / cache
    def discover_ckpts():
        """Every *.pth under outputs/, best.pth first -- the runs this repo produces."""
        found = sorted(Path(REPO / "outputs").glob("*/*.pth")) if (REPO / "outputs").exists() else []
        found.sort(key=lambda q: (q.name != "best.pth", str(q)))
        return found

    def ckpt_label(q):
        return f"{Path(q).parent.name}/{Path(q).name}"

    def refresh_ckpt_box(current=None):
        opts = {ckpt_label(q): q for q in discover_ckpts()}
        if current is not None:
            opts.setdefault(ckpt_label(current), Path(current))
        state["ckpt_opts"] = opts
        ckpt_box.configure(values=list(opts))
        if current is not None:
            ckpt_var.set(ckpt_label(current))

    def update_cache_label():
        prov = state["provider"]
        if prov is None or prov.cache is None:
            cache_lbl.configure(text="cache off")
            return
        c = prov.cache
        cache_lbl.configure(text=f"cache: {c.count()} maps / {len(c.index)} frames\n{c.dir}")

    def set_checkpoint(path, synthetic=False):
        """Load a checkpoint and switch to ITS cache directory. Depth already computed for that
        checkpoint is picked straight back up; nothing is recomputed on a switch back."""
        path = Path(path)
        if synthetic:
            be = SyntheticBackend(image_shape=args.image_shape)
        else:
            if not path.is_file():
                log(f"checkpoint not found:\n{path}", "#c00")
                return False
            be = DepthBackend(path, args.image_shape, args.min_depth, args.max_depth, args.device)
            try:
                log(f"loading {ckpt_label(path)} (first time takes ~30 s on CPU) ...", "#06c")
                t0 = time.perf_counter()
                be.load(log=lambda m: log(m, "#06c"))
                log(f"loaded {ckpt_label(path)} in {time.perf_counter() - t0:.1f}s", "#06c")
            except Exception as exc:                          # noqa: BLE001 - surface to the UI
                log(f"model load failed: {exc}", "#c00")
                messagebox.showerror("Checkpoint", str(exc))
                return False
        cache = None if args.no_cache else DepthCache(
            args.cache_dir, path, be.image_shape, args.min_depth, args.max_depth,
            log=lambda m: log(m, "#06c"))
        state["provider"] = DepthProvider(be, cache)
        refresh_ckpt_box(path)
        update_cache_label()
        mark_files()
        if state["full"] is not None:
            reload_image(keep_marks=False)
        else:
            log(f"model ready ({'synthetic' if synthetic else be.device})")
        return True

    def on_ckpt_pick(_=None):
        q = state.get("ckpt_opts", {}).get(ckpt_var.get())
        if q and state["provider"] is not None and getattr(state["provider"].backend, "ckpt", None) == Path(q):
            return
        if q:
            set_checkpoint(q)

    def browse_ckpt():
        f = filedialog.askopenfilename(title="Depth checkpoint",
                                       initialdir=str(REPO / "outputs"),
                                       filetypes=[("PyTorch checkpoint", "*.pth *.pt")])
        if f:
            set_checkpoint(f)

    def precompute_all():
        """Walk the whole file list and fill the cache, skipping what is already there. Uses the
        same auto-crop `load` would, so every map it writes is one a later click will hit."""
        prov = state["provider"]
        if prov is None or not state["files"] or state.get("busy"):
            return                               # `busy`: root.update() below re-enters callbacks
        todo = [q for q in state["files"] if not prov.cached(q)]
        if not todo:
            log(f"all {len(state['files'])} frames already cached")
            return
        state["abort"], state["busy"] = False, True
        done = 0
        for i, q in enumerate(todo, 1):
            if state["abort"]:
                break
            try:
                full = Image.open(q).convert("RGB")
                fr = auto_bars(np.asarray(full))
                prov.depth_for(q, full.crop(crop_box(full.size, *fr)), fr)
                done += 1
            except Exception as exc:                          # noqa: BLE001 - keep going
                log(f"skipped {Path(q).name}: {exc}", "#c60")
            log(f"precomputing {i}/{len(todo)} ...", "#06c")
            root.update()                        # keep the window alive; Esc sets abort
        state["busy"] = False
        update_cache_label(); mark_files()
        log(f"cached {done}/{len(todo)} new maps" + (" (stopped)" if state["abort"] else ""))

    # ------------------------------------------------------------------ image loading
    def set_files(paths, select=0):
        state["files"] = list(paths)
        mark_files()
        if state["files"]:
            select_index(select)

    def mark_files():
        """Redraw the file list, prefixing frames whose depth is already on disk with a dot."""
        prov, sel = state["provider"], state["idx"]
        filebox.delete(0, "end")
        for q in state["files"]:
            hit = prov is not None and prov.cached(q)
            filebox.insert("end", f"{'* ' if hit else '  '}{Path(q).name}")
        if 0 <= sel < len(state["files"]):
            filebox.selection_clear(0, "end")
            filebox.selection_set(sel)

    def select_index(i):
        if not state["files"]:
            return
        state["idx"] = max(0, min(i, len(state["files"]) - 1))
        filebox.selection_clear(0, "end")
        filebox.selection_set(state["idx"])
        filebox.see(state["idx"])
        load(state["files"][state["idx"]])

    def step(d):
        select_index(state["idx"] + d)

    def pick_folder():
        d = filedialog.askdirectory(title="Folder with images",
                                    initialdir=str(args.data_dir if Path(args.data_dir).exists() else REPO))
        if not d:
            return
        files = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not files:
            messagebox.showwarning("Nothing found", f"No images under {d}")
            return
        log(f"{len(files)} images")
        set_files(files)

    def pick_file():
        f = filedialog.askopenfilename(title="Image",
                                       initialdir=str(args.data_dir if Path(args.data_dir).exists() else REPO),
                                       filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if f:
            set_files(sorted(p for p in Path(f).parent.iterdir() if p.suffix.lower() in IMG_EXTS),
                      select=0)
            hit = [i for i, p in enumerate(state["files"]) if str(p) == f]
            if hit:
                select_index(hit[0])

    def set_auto_crop():
        if state["full"] is None:
            return
        s, t, b = auto_bars(np.asarray(state["full"]))
        for var, val in ((v_side, s), (v_top, t), (v_bot, b)):
            var.set(f"{val:.4f}")
        reload_image(keep_marks=False)

    def fracs():
        def f(var):
            try:
                return max(0.0, min(0.45, float(var.get())))
            except ValueError:
                return 0.0
        return f(v_side), f(v_top), f(v_bot)

    def load(path):
        state["path"] = Path(path)
        state["full"] = Image.open(path).convert("RGB")
        s, t, b = auto_bars(np.asarray(state["full"]))
        for var, val in ((v_side, s), (v_top, t), (v_bot, b)):
            var.set(f"{val:.4f}")
        reload_image(keep_marks=False, fit=True)

    def reload_image(keep_marks=True, fit=False):
        if state["full"] is None:
            return
        if not keep_marks:
            sess.items.clear(); state["pending"] = None; refresh_list()
        state["crop"] = state["full"].crop(crop_box(state["full"].size, *fracs()))
        w, h = state["crop"].size
        aspect_lbl.configure(text=f"aspect {w / h:.3f} (training 5:4 = 1.250)  {w}x{h}",
                             foreground="#666" if abs(w / h - 1.25) < 0.08 else "#c60")
        prov = state["provider"]
        if prov is None:
            log("no checkpoint loaded", "#c00")
            return
        try:
            state["depth"], src, dt = prov.depth_for(
                state["path"], state["crop"], fracs(),
                on_compute=lambda: log(f"predicting depth for {state['path'].name} ...", "#06c"))
        except Exception as exc:                              # noqa: BLE001 - surface to the UI
            state["depth"] = None
            log(f"depth failed: {exc}", "#c00")
            messagebox.showerror("Depth model", str(exc))
            return
        d = state["depth"]
        state["disp_img"] = Image.fromarray(colorize(1.0 / np.clip(d, 1e-6, None))).resize(state["crop"].size)
        log(f"{state['path'].name} [{src} {dt * 1000:.0f} ms]  depth {d.min():.0f}-{d.max():.0f} mm "
            f"(median {np.median(d):.0f})")
        if src == "model":
            update_cache_label(); mark_files()
        if fit:
            zoom(0.0)
        else:
            redraw()

    # ------------------------------------------------------------------------ drawing
    def fit_scale():
        if state["crop"] is None:
            return 1.0
        cw = max(canvas.winfo_width(), 50); ch = max(canvas.winfo_height(), 50)
        w, h = state["crop"].size
        return min(cw / w, ch / h)

    def zoom(factor):
        state["zoom"] = fit_scale() if factor == 0.0 else state["zoom"] * factor
        redraw()

    def redraw(_=None):
        if state["crop"] is None:
            return
        if state["zoom"] <= 0:
            state["zoom"] = fit_scale()
        w, h = state["crop"].size
        z = state["zoom"]
        base = state["crop"]
        if state["disp_img"] is not None and v_alpha.get() > 0.01:
            base = Image.blend(base, state["disp_img"], float(v_alpha.get()))
        img = base.resize((max(1, int(w * z)), max(1, int(h * z))))
        state["tkimg"] = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=state["tkimg"], tags="img")
        canvas.configure(scrollregion=(0, 0, img.width, img.height))
        for i, m in enumerate(sess.items):
            draw_measure(m, i + 1)
        if state["pending"] is not None:
            x, y = to_canvas(state["pending"])
            canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="#0ff", width=2)

    def to_canvas(p_n):
        w, h = state["crop"].size
        return p_n[0] * w * state["zoom"], p_n[1] * h * state["zoom"]

    def to_norm(cx, cy):
        w, h = state["crop"].size
        return (min(max(cx / (w * state["zoom"]), 0.0), 1.0),
                min(max(cy / (h * state["zoom"]), 0.0), 1.0))

    def draw_measure(m, n):
        x0, y0 = to_canvas(m.p0); x1, y1 = to_canvas(m.p1)
        canvas.create_line(x0, y0, x1, y1, fill="#00ff88", width=2)
        for x, y in ((x0, y0), (x1, y1)):
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#00ff88",
                               fill="#003322", width=2)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        txt = f"{n}: {m.mm:.1f} mm"
        canvas.create_text(mx + 1, my - 11, text=txt, fill="#000",
                           font=("Segoe UI", 10, "bold"))
        canvas.create_text(mx, my - 12, text=txt, fill="#00ff88",
                           font=("Segoe UI", 10, "bold"))

    # ------------------------------------------------------------------------- events
    def on_click(ev):
        if state["depth"] is None:
            return
        p = to_norm(canvas.canvasx(ev.x), canvas.canvasy(ev.y))
        if state["pending"] is None:
            state["pending"] = p
        else:
            sess.items.append(sess.measure(state["depth"], state["pending"], p))
            state["pending"] = None
            refresh_list()
        redraw()

    def on_motion(ev):
        if state["depth"] is None:
            return
        u, v = to_norm(canvas.canvasx(ev.x), canvas.canvasy(ev.y))
        z = sample_depth(state["depth"], u, v, sess.radius) * sess.scale
        w, h = state["crop"].size
        readout.configure(text=f"({u * w:6.0f},{v * h:6.0f})  z={z:7.1f} mm")

    def on_wheel(ev):
        state["zoom"] *= 1.25 if ev.delta > 0 else 1 / 1.25
        redraw()

    canvas.bind("<Button-1>", on_click)
    canvas.bind("<Motion>", on_motion)
    canvas.bind("<MouseWheel>", on_wheel)                       # Windows / macOS
    canvas.bind("<Button-4>", lambda e: (state.update(zoom=state["zoom"] * 1.25), redraw()))
    canvas.bind("<Button-5>", lambda e: (state.update(zoom=state["zoom"] / 1.25), redraw()))
    canvas.bind("<ButtonPress-3>", lambda e: canvas.scan_mark(e.x, e.y))
    canvas.bind("<B3-Motion>", lambda e: canvas.scan_dragto(e.x, e.y, gain=1))
    canvas.bind("<Configure>", lambda e: redraw())
    filebox.bind("<<ListboxSelect>>",
                 lambda e: (filebox.curselection() and select_index(filebox.curselection()[0])))
    ckpt_box.bind("<<ComboboxSelected>>", on_ckpt_pick)
    root.bind("<Escape>", lambda e: (state.update(pending=None, abort=True), redraw()))
    root.bind("<Left>", lambda e: step(-1))
    root.bind("<Right>", lambda e: step(1))
    root.bind("<Control-z>", lambda e: undo())

    # ---------------------------------------------------------------------- list / io
    def refresh_list():
        listbox.delete(0, "end")
        for i, m in enumerate(sess.items, 1):
            listbox.insert("end", f"{i:2d}  {m.mm:7.1f} mm   z {m.z0:5.0f}/{m.z1:5.0f}")
        if sess.items:
            listbox.selection_clear(0, "end")
            listbox.selection_set("end")

    def undo():
        if state["pending"] is not None:
            state["pending"] = None
        elif sess.items:
            sess.items.pop()
        refresh_list(); redraw()

    def clear():
        sess.items.clear(); state["pending"] = None
        refresh_list(); redraw()

    def apply_params():
        try:
            sess.scale = float(v_scale.get())
            sess.radius = max(0, int(float(v_rad.get())))
            sess.k_norm = (float(v_fx.get()), float(v_fy.get()), sess.k_norm[2], sess.k_norm[3])
        except ValueError:
            messagebox.showerror("Parameters", "scale / patch / fx / fy must be numbers")
            return
        if state["depth"] is not None:
            sess.recompute(state["depth"])
        refresh_list(); redraw()
        log(f"scale x{sess.scale:g}, patch {sess.radius}px, fx {sess.k_norm[0]:g} fy {sess.k_norm[1]:g}")

    def calibrate():
        """Type the TRUE length of a measurement you took on a known object -> solves the
        scale correction and re-scales every measurement. This is the honest way to get
        below the model's ~2-3% residual scale error on a given scene."""
        from tkinter.simpledialog import askfloat
        sel = listbox.curselection()
        if not sel:
            messagebox.showinfo("Calibrate", "Select a measurement first (one of known length).")
            return
        m = sess.items[sel[0]]
        true_mm = askfloat("Calibrate", f"Measured {m.mm:.2f} mm.\nTrue length in mm:",
                           minvalue=0.01)
        if not true_mm:
            return
        raw = m.mm / sess.scale                          # undo the current correction
        sess.scale = true_mm / raw
        v_scale.set(f"{sess.scale:.5g}")
        sess.recompute(state["depth"])
        refresh_list(); redraw()
        log(f"calibrated: scale x{sess.scale:.4f}")

    def export_csv():
        if not sess.items:
            return
        f = filedialog.asksaveasfilename(defaultextension=".csv",
                                         initialfile=f"{state['path'].stem}_measurements.csv")
        if not f:
            return
        w, h = state["crop"].size
        with open(f, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["image", "crop_w", "crop_h", "x0", "y0", "x1", "y1",
                         "z0_mm", "z1_mm", "length_mm", "scale", "fx", "fy", "patch_px"])
            for m in sess.items:
                wr.writerow([state["path"].name, w, h,
                             f"{m.p0[0] * w:.1f}", f"{m.p0[1] * h:.1f}",
                             f"{m.p1[0] * w:.1f}", f"{m.p1[1] * h:.1f}",
                             f"{m.z0:.2f}", f"{m.z1:.2f}", f"{m.mm:.2f}",
                             f"{sess.scale:.5f}", sess.k_norm[0], sess.k_norm[1], sess.radius])
        log(f"wrote {Path(f).name}")

    def save_png():
        if state["crop"] is None:
            return
        from PIL import ImageDraw
        f = filedialog.asksaveasfilename(defaultextension=".png",
                                         initialfile=f"{state['path'].stem}_measured.png")
        if not f:
            return
        w, h = state["crop"].size
        img = state["crop"].copy()
        if state["disp_img"] is not None and v_alpha.get() > 0.01:
            img = Image.blend(img, state["disp_img"], float(v_alpha.get()))
        dr = ImageDraw.Draw(img)
        for i, m in enumerate(sess.items, 1):
            x0, y0, x1, y1 = m.p0[0] * w, m.p0[1] * h, m.p1[0] * w, m.p1[1] * h
            dr.line((x0, y0, x1, y1), fill=(0, 255, 136), width=3)
            for x, y in ((x0, y0), (x1, y1)):
                dr.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(0, 255, 136), width=3)
            dr.text(((x0 + x1) / 2, (y0 + y1) / 2 - 14), f"{i}: {m.mm:.1f} mm", fill=(0, 255, 136))
        img.save(f)
        log(f"wrote {Path(f).name}")

    # ---------------------------------------------------------------------- kick off
    refresh_ckpt_box()
    if args.no_model:
        set_checkpoint(args.ckpt, synthetic=True)
        log("--no-model: synthetic depth, numbers are meaningless", "#c60")
    else:
        set_checkpoint(args.ckpt)

    if args.image:
        p = Path(args.image)
        set_files(sorted(q for q in p.parent.iterdir() if q.suffix.lower() in IMG_EXTS))
        hit = [i for i, q in enumerate(state["files"]) if q == p]
        select_index(hit[0] if hit else 0)
    elif Path(args.data_dir).exists():
        files = sorted(q for q in Path(args.data_dir).rglob("*") if q.suffix.lower() in IMG_EXTS)[:2000]
        if files:
            set_files(files)
            log(f"{len(files)} images under {args.data_dir}")

    if on_ready is not None:
        ctx = dict(root=root, canvas=canvas, state=state, sess=sess, load=load,
                   reload_image=reload_image, redraw=redraw, on_click=on_click,
                   on_motion=on_motion, undo=undo, clear=clear, apply_params=apply_params,
                   refresh_list=refresh_list, listbox=listbox, to_canvas=to_canvas,
                   vars=dict(side=v_side, top=v_top, bot=v_bot, alpha=v_alpha,
                             scale=v_scale, rad=v_rad, fx=v_fx, fy=v_fy),
                   set_auto_crop=set_auto_crop, provider=lambda: state["provider"],
                   set_checkpoint=set_checkpoint, precompute_all=precompute_all,
                   mark_files=mark_files, filebox=filebox, cache_lbl=cache_lbl,
                   ckpt_box=ckpt_box, left=left, left_canvas=left_canvas,
                   export_csv=export_csv, save_png=save_png, calibrate=calibrate,
                   select_index=select_index, step=step, zoom=zoom)
        root.after(200, lambda: on_ready(ctx))

    root.mainloop()


def banner():
    """Which build, from which file. The answer to "why don't I see the new buttons"."""
    f = Path(__file__).resolve()
    return f"gui_depth_measure build {BUILD}  |  {f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="trained depth state_dict")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA), help="folder to browse for images")
    ap.add_argument("--image", default=None, help="open this image on start")
    ap.add_argument("--image-shape", type=int, nargs=2, default=list(DEFAULT_SHAPE),
                    metavar=("H", "W"), help="model feed size (must match the run)")
    ap.add_argument("--min-depth", type=float, default=DEFAULT_MIN_DEPTH)
    ap.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH)
    ap.add_argument("--intrinsics", type=float, nargs=4, default=list(DEFAULT_K_NORM),
                    metavar=("fx", "fy", "cx", "cy"), help="NORMALISED K; default = SCARED assumed")
    ap.add_argument("--scale", type=float, default=1.0, help="extra multiplier on predicted mm")
    ap.add_argument("--patch-radius", type=int, default=2, help="median window for depth sampling")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                    help="where depth maps are stored, one subfolder per checkpoint")
    ap.add_argument("--no-cache", action="store_true", help="always recompute, write nothing")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--no-model", action="store_true", help="UI only, synthetic depth")
    ap.add_argument("--self-test", action="store_true", help="geometry checks, no torch needed")
    ap.add_argument("--version", action="store_true",
                    help="print which build and which file is about to run, then exit")
    args = ap.parse_args()
    if args.version:
        print(banner())
        return
    if args.self_test:
        self_test()
        return
    print(banner(), flush=True)
    run_gui(args)


if __name__ == "__main__":
    main()
