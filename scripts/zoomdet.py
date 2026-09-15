"""Zoom label (1x/2x/4x) per clip from the bottom HUD, validated on 77 hand labels."""
import csv, glob, os, re
from multiprocessing.pool import ThreadPool
import cv2, numpy as np

cv2.setNumThreads(1)
H = os.path.expanduser("~")
GUI = f"{H}/data/UMCdissectionvid"
Y0, Y1, X0, X1 = 1034, 1066, 1212, 1262          # "1x 0°" label, 1920x1080
ZOOM = {"4d8eca93": "2x", **{k: "4x" for k in ["RARP_069", "RARP_077", "RARP_083", "RARP_086",
                                                "RARP_089", "349725a5", "7d96d613", "d7419222"]}}

def feature(path, start, n, step=4):
    """Mean high-passed label crop over the clip: the label is fixed, tissue behind it moves."""
    c = cv2.VideoCapture(path)
    if start:
        c.set(cv2.CAP_PROP_POS_FRAMES, start)
    acc = []
    for i in range(n):
        ok, f = c.read()
        if not ok:
            break
        if i % step == 0 and f.shape[:2] == (1080, 1920):
            g = cv2.cvtColor(f[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            acc.append(g - cv2.GaussianBlur(g, (0, 0), 3))
    c.release()
    if not acc:
        return None
    m = np.mean(acc, 0)
    return (m - m.mean()) / (m.std() + 1e-6)

def ncc(a, t, s=2):
    """Best normalised correlation of template t over +-s px shifts inside a."""
    best = -1.0
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            b = np.roll(np.roll(a, dy, 0), dx, 1)[s:-s, s:-s]
            u = t[s:-s, s:-s]
            best = max(best, float(((b - b.mean()) * (u - u.mean())).mean() / (b.std() * u.std() + 1e-6)))
    return best

def classify(f, T):
    sc = {k: ncc(f, v) for k, v in T.items()}
    k = max(sc, key=sc.get)
    return k, sc

# ---- 1) hand-labelled reference set (77 sharpest clips)
rows = [r for r in csv.DictReader(open(f"{H}/data/processed/depthclips_sharpest/sharpness_rank.csv")) if r["selected"]]
def ref_job(r):
    m = re.search(r"_f(\d+)-(\d+)\.mp4$", r["clip"])
    s, e = int(m.group(1)), int(m.group(2))
    return r["video"][:8], ZOOM.get(r["video"][:8], "1x"), feature(f"{GUI}/{r['video']}.mp4", s, e - s + 1)
with ThreadPool(32) as p:
    ref = [x for x in p.map(ref_job, rows) if x[2] is not None]
print(f"reference clips with a feature: {len(ref)}", flush=True)

def templates(exclude=None):
    T = {}
    for z in ("1x", "2x", "4x"):
        fs = [f for k, lab, f in ref if lab == z and k != exclude]
        if fs:
            m = np.mean(fs, 0); T[z] = (m - m.mean()) / (m.std() + 1e-6)
    return T

# leave-one-out: each clip is scored against templates built WITHOUT it
wrong, margins = [], []
for k, lab, f in ref:
    pred, sc = classify(f, templates(exclude=k))
    s1 = sc.get("1x", -1); snot = max(v for z, v in sc.items() if z != "1x") if len(sc) > 1 else -1
    margins.append((lab, round(s1 - snot, 3), k))
    ok = (pred == "1x") == (lab == "1x")               # what the filter needs: 1x vs zoomed
    if not ok or pred != lab:
        wrong.append((k, lab, pred, {z: round(v, 3) for z, v in sc.items()}))
print("LOO mistakes (label, pred, scores):", wrong if wrong else "none", flush=True)
m1 = [m for lab, m, _ in margins if lab == "1x"]; mz = [m for lab, m, _ in margins if lab != "1x"]
print(f"1x-minus-zoomed score: 1x clips min {min(m1):.3f} median {np.median(m1):.3f} | zoomed clips max {max(mz):.3f} "
      f"({[(k, m) for lab, m, k in margins if lab != '1x']})", flush=True)

T = templates()
os.makedirs(f"{H}/zoomcrop/zoom_templates", exist_ok=True)
for z, t in T.items():
    cv2.imwrite(f"{H}/zoomcrop/zoom_templates/{z}.png", cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    np.save(f"{H}/zoomcrop/zoom_templates/{z}.npy", t)

# ---- 2) manual clips (GUI visible in the clip itself)
man = sorted(glob.glob(f"{H}/data/depthclips_manual_raw/*.mp4"))
with open(f"{H}/zoomcrop/zoom_manual.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "zoom", "s1x", "s2x", "s4x"])
    for mp in man:
        f = feature(mp, 0, 10**6, step=2)
        z, sc = classify(f, T) if f is not None else ("?", {})
        w.writerow([os.path.basename(mp), z] + [round(sc.get(k, float("nan")), 3) for k in ("1x", "2x", "4x")])
        print(f"manual {os.path.basename(mp)[:8]} -> {z} {({k: round(v, 3) for k, v in sc.items()})}", flush=True)

# ---- 3) all staging clips > 1.5 MB (source = with-GUI original, frames from the clip name)
big = [p for p in sorted(glob.glob(f"{H}/data/depth_clips_staging/*/*.mp4")) if os.path.getsize(p) > 1.5 * 2**20]
def stg_job(p):
    v = os.path.basename(os.path.dirname(p)); m = re.search(r"_f(\d+)-(\d+)\.mp4$", p)
    src = f"{GUI}/{v}.mp4"
    if not m or not os.path.exists(src):
        return p, "missing", {}
    s, e = int(m.group(1)), int(m.group(2))
    f = feature(src, s, e - s + 1)
    return (p, *classify(f, T)) if f is not None else (p, "?", {})
with ThreadPool(32) as pool:
    res = pool.map(stg_job, big)
with open(f"{H}/zoomcrop/zoom_staging15.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["video", "clip", "zoom", "s1x", "s2x", "s4x"])
    for p, z, sc in res:
        w.writerow([os.path.basename(os.path.dirname(p)), os.path.basename(p), z] +
                   [round(sc.get(k, float("nan")), 3) for k in ("1x", "2x", "4x")])
from collections import Counter
print("staging >1.5MB:", len(res), "clips", dict(Counter(z for _, z, _ in res)), flush=True)
s1 = sorted(sc["1x"] - max(sc["2x"], sc["4x"]) for _, z, sc in res if sc)
print("staging 1x-minus-zoomed margin: p1/p5/median", np.percentile(s1, [1, 5, 50]).round(3), flush=True)
