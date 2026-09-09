"""Regenerate every figure in STEREO_REPORT.md into docs/stereo/.

Thesis figures have to be reproducible, so they are generated from the data and the stored
calibration rather than screenshotted. Needs outputs/stereo_calib/calib.json (produced by
scripts/calibrate_stereo_charuco.py).

    python scripts/stereo_report_figs.py
"""
import argparse
import glob
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_stereo_charuco import (BANNER_ROW, BOARDS, OUT_H, OUT_W, collect,
                                      eye_to_mono, make_board)
from rectify_mono_clips import load_maps, mono_crop

DATA = "../data"
FG, BG, ACC = "#e8e8e8", "#141414", "#4cc2ff"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": "#555",
    "xtick.color": FG, "ytick.color": FG, "grid.color": "#2e2e2e",
    "font.size": 10, "axes.titlesize": 11, "axes.grid": True, "grid.alpha": 0.5,
})


def label(img, text, h=42):
    out = cv2.copyMakeBorder(img, h, 8, 8, 8, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(out, text, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1,
                cv2.LINE_AA)
    return out


def tag(img, text, org, col, scale=0.72):
    """Text on a filled plate -- plain text over bright tissue is unreadable."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + 8), (18, 18, 18), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)


def center(rows):
    """Pad rows to a common width so they can be vstacked."""
    w = max(r.shape[1] for r in rows)
    return [cv2.copyMakeBorder(r, 0, 0, (w - r.shape[1]) // 2, w - r.shape[1] - (w - r.shape[1]) // 2,
                               cv2.BORDER_CONSTANT, value=(20, 20, 20)) for r in rows]


def grab(path, index):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read frame %d of %s" % (index, path))
    return fr


def stereo_clips():
    return sorted(glob.glob(os.path.join(DATA, "3D_ProxyGT", "*.mp4")))


def calib_clip():
    return sorted(glob.glob(os.path.join(DATA, "ARUCO_calibration", "*.mp4")))[0]


# ------------------------------------------------------------------ fig 1
def fig_geometry(out):
    """Where the two eyes sit in the SBS frame, and where they land in the training frame."""
    fr = grab(stereo_clips()[2], 1500)
    top = fr.copy()
    for (x0, name, col) in ((164, "LEFT EYE  x 164-799", (90, 220, 120)),
                            (1124, "RIGHT EYE  x 1124-1759  (= +960)", (60, 170, 255))):
        cv2.rectangle(top, (x0, 32), (x0 + 636, 1047), col, 4)
        tag(top, name, (x0 + 10, 24), col)
        cv2.line(top, (x0, 997), (x0 + 636, 997), (255, 90, 90), 2)
    tag(top, "636 x 1016 per eye, squeezed 2x horizontally", (164, 1074), (235, 235, 235), 0.7)
    tag(top, "GUI banner (excluded)", (1124, 1074), (255, 90, 90), 0.7)
    top = label(cv2.resize(top, (1600, 900)), "SOURCE  1920x1080 half-width anamorphic side-by-side")

    eyes = [label(cv2.resize(eye_to_mono(fr, r), (795, 636)), t) for r, t in
            ((False, "LEFT eye -> 1340x1072 (the mono training frame)"),
             (True, "RIGHT eye -> 1340x1072, same transform +960 in x"))]
    cv2.imwrite(out, np.vstack(center([top, np.hstack(eyes)])))
    print("  " + out)


# ------------------------------------------------------------------ fig 2
def fig_anamorphic(out):
    """The GUI badges are circles. That is what pins the squeeze at exactly 2x."""
    # The console draws the instrument badges 1..4 as circles. No fit, no overlay -- the two
    # crops are the whole argument; the rigorous version is fx/fy = 1.0000 in the calibration.
    fr = grab(stereo_clips()[2], 1500)
    band = fr[32 + 958:32 + 1016, 164:164 + 250]
    K = 5
    iso = cv2.resize(band, None, fx=K, fy=K, interpolation=cv2.INTER_NEAREST)
    uns = cv2.resize(band, (band.shape[1] * 2 * K, band.shape[0] * K),
                     interpolation=cv2.INTER_NEAREST)
    a = label(iso, "as stored, isotropic %dx  -  the badge rings around 1 and 2 are ELLIPSES" % K)
    b = label(uns, "after 2x horizontal un-squeeze  -  the same rings are CIRCLES")
    cv2.imwrite(out, np.vstack(center([a, b])))
    print("  " + out)


# ------------------------------------------------------------------ fig 3
def fig_charuco(out):
    board = make_board(BOARDS["A"])
    det = cv2.aruco.CharucoDetector(board)
    cap = cv2.VideoCapture(calib_clip())
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best = None
    for i in range(0, n, 7):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        got = {}
        for side, r in (("L", False), ("R", True)):
            img = eye_to_mono(fr, r)[:BANNER_ROW]
            cc, ci, mc, mi = det.detectBoard(img)
            if ci is None:
                break
            got[side] = (img, cc, ci, mc, mi)
        if len(got) == 2:
            k = min(len(got["L"][2]), len(got["R"][2]))
            if best is None or k > best[0]:
                best = (k, got)
    cap.release()
    k, got = best
    panels = []
    for side in ("L", "R"):
        img, cc, ci, mc, mi = got[side]
        v = img.copy()
        cv2.aruco.drawDetectedMarkers(v, mc, mi)
        cv2.aruco.drawDetectedCornersCharuco(v, cc.reshape(-1, 1, 2),
                                             ci.reshape(-1, 1).astype(np.int32), (0, 255, 120))
        panels.append(label(cv2.resize(v, (795, 605)),
                            "%s eye  -  %d ChArUco corners" % ("LEFT" if side == "L" else "RIGHT",
                                                               len(ci))))
    cv2.imwrite(out, np.hstack(panels))
    print("  %s  (best frame: %d common corners)" % (out, k))


# ------------------------------------------------------------------ fig 4
def fig_distortion(out, calib):
    KL, DL = np.array(calib["KL"]), np.array(calib["DL"])
    gx, gy = np.meshgrid(np.arange(0, OUT_W, 4, np.float32), np.arange(0, OUT_H, 4, np.float32))
    g = np.stack([gx, gy], -1).reshape(-1, 1, 2)
    d = np.linalg.norm(cv2.undistortPoints(g, KL, DL, P=KL).reshape(-1, 2) - g.reshape(-1, 2),
                       axis=1)
    r = np.linalg.norm(g.reshape(-1, 2) - [KL[0, 2], KL[1, 2]], axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.linspace(0, r.max(), 45)
    idx = np.digitize(r, bins) - 1
    mean = np.array([d[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins))])
    mx = np.array([d[idx == i].max() if (idx == i).any() else np.nan for i in range(len(bins))])
    rmax = calib.get("max_corner_radius")
    if rmax:
        ax[0].axvspan(rmax, r.max(), color="#ff4444", alpha=0.13)
        ax[0].axvline(rmax, color="#ff4444", lw=1.6, ls="--")
        ax[0].text(rmax + 12, ax[0].get_ylim()[1] * 0.05,
                   "EXTRAPOLATED\nno board data\nbeyond r=%.0f" % rmax,
                   color="#ff8080", fontsize=9, va="bottom")
    ax[0].plot(bins, mx, color="#ff7a45", lw=1.4, label="max")
    ax[0].plot(bins, mean, color=ACC, lw=2.2, label="mean")
    ax[0].axhline(1, color="#888", ls=":", lw=1)
    ax[0].set(xlabel="radius from optical axis (px)",
              ylabel="displacement from pinhole (px)",
              title="Lens distortion vs radius (frame corner at r=906)")
    ax[0].legend(facecolor=BG, edgecolor="#555", labelcolor=FG, loc="upper left")
    im = ax[1].imshow(d.reshape(gy.shape), extent=[0, OUT_W, OUT_H, 0], cmap="inferno")
    cs = ax[1].contour(gx, gy, d.reshape(gy.shape), levels=[1, 2, 5, 10, 20],
                       colors="w", linewidths=0.8)
    ax[1].clabel(cs, fmt="%g px", fontsize=8)
    if rmax:
        th = np.linspace(0, 2 * np.pi, 200)
        ax[1].plot(KL[0, 2] + rmax * np.cos(th), KL[1, 2] + rmax * np.sin(th),
                   color="#ff4444", lw=2, ls="--")
        ax[1].text(20, 40, "inside dashed circle = measured\noutside = extrapolated (%.0f%% of frame)"
                   % (100 * calib.get("extrapolated_frac", 0.26)), color="#ff8080", fontsize=9)
    ax[1].plot(KL[0, 2], KL[1, 2], "x", color="#4cff9a", ms=11, mew=2.5)
    ax[1].plot(OUT_W / 2, OUT_H / 2, "+", color="#888", ms=11, mew=2)
    ax[1].set(title="Distortion field (x = optical axis, + = frame centre)",
              xlabel="px", ylabel="px")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], label="px")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  " + out)


# ------------------------------------------------------------------ fig 5
def fig_rectification(out, calib):
    from rectify_mono_clips import preview
    m1, m2, K, D, _, _ = load_maps(calib, "left")
    src = sorted(glob.glob(os.path.join(DATA, "depthclips_ruler_NoGUI", "NOgui", "*", "clip_*.mp4")))[0]
    preview(mono_crop(grab(src, 8)), m1, m2, K, D, out)


# ------------------------------------------------------------------ fig 6
def fig_validation(out, calib, stride):
    KL, DL = np.array(calib["KL"]), np.array(calib["DL"])
    KR, DR = np.array(calib["KR"]), np.array(calib["DR"])
    R1, R2 = np.array(calib["R1"]), np.array(calib["R2"])
    P1, P2 = np.array(calib["P1"]), np.array(calib["P2"])
    f_b = -P2[0, 3]

    views = []
    for b in calib.get("board", "A"):                   # "A", "B" or "AB"
        views += collect(calib_clip(), make_board(BOARDS[b]), stride, 10, 40.0)
    zp, zd = [], []
    for v in views:
        ok, rv, tv = cv2.solvePnP(v["obj"], v["L"], KL, DL, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        qL = cv2.undistortPoints(v["L"].reshape(-1, 1, 2), KL, DL, R=R1, P=P1).reshape(-1, 2)
        qR = cv2.undistortPoints(v["R"].reshape(-1, 1, 2), KR, DR, R=R2, P=P2).reshape(-1, 2)
        zp.append((cv2.Rodrigues(rv)[0] @ v["obj"].T + tv)[2])
        zd.append(f_b / (qL[:, 0] - qR[:, 0]))
    zp, zd = np.concatenate(zp), np.concatenate(zd)
    e = zd - zp

    # surgical-clip depth distributions via SIFT correspondences
    sift, bf = cv2.SIFT_create(3000), cv2.BFMatcher()
    dists = {}
    for f in stereo_clips():
        cap = cv2.VideoCapture(f)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        acc = []
        for i in np.linspace(60, n - 60, 8).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            L, R = (eye_to_mono(fr, r)[:BANNER_ROW] for r in (False, True))
            k1, d1 = sift.detectAndCompute(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), None)
            k2, d2 = sift.detectAndCompute(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), None)
            pl, pr = [], []
            for a, b in bf.knnMatch(d1, d2, 2):
                if a.distance < 0.7 * b.distance:
                    pl.append(k1[a.queryIdx].pt)
                    pr.append(k2[a.trainIdx].pt)
            if len(pl) < 20:
                continue
            ql = cv2.undistortPoints(np.float32(pl).reshape(-1, 1, 2), KL, DL, R=R1, P=P1).reshape(-1, 2)
            qr = cv2.undistortPoints(np.float32(pr).reshape(-1, 1, 2), KR, DR, R=R2, P=P2).reshape(-1, 2)
            m = (np.abs(ql[:, 1] - qr[:, 1]) < 2) & ((ql[:, 0] - qr[:, 0]) > 5)
            acc.append(f_b / (ql[m, 0] - qr[m, 0]))
        cap.release()
        name = os.path.basename(f)
        key = name[:8] + ("-" + name.split("-")[-1][:4] if "seg" in name else "")
        dists[key] = np.concatenate(acc)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    lim = [zp.min() * 0.95, zp.max() * 1.05]
    ax[0].plot(lim, lim, color="#ff7a45", lw=1.4, ls="--", label="1:1")
    ax[0].scatter(zp, zd, s=3, alpha=0.15, color=ACC, edgecolors="none")
    ax[0].set(xlim=lim, ylim=lim, xlabel="depth from board PnP pose (mm)",
              ylabel="depth from stereo disparity (mm)",
              title="Validation: %d corners, MAE %.2f mm" % (len(zp), np.abs(e).mean()))
    ax[0].legend(facecolor=BG, edgecolor="#555", labelcolor=FG)
    ax[1].hist(e, bins=70, color=ACC)
    ax[1].axvline(0, color="#ff7a45", lw=1.4, ls="--")
    ax[1].set(xlabel="disparity depth - PnP depth (mm)", ylabel="corners",
              title="Residual: bias %+.2f mm, p95 |e| %.2f mm"
                    % (np.median(e), np.percentile(np.abs(e), 95)))
    for name, z in dists.items():
        z = z[(z > 5) & (z < 250)]
        ax[2].hist(z, bins=80, histtype="step", lw=1.8, label="%s  med %.0f mm"
                   % (name, np.median(z)), density=True)
    ax[2].set(xlabel="implied depth (mm)", ylabel="density",
              title="Transfer to the surgical clips")
    ax[2].legend(facecolor=BG, edgecolor="#555", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  %s  (MAE %.3f mm, %d corners)" % (out, np.abs(e).mean(), len(zp)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calib/stereo_calib.json")
    ap.add_argument("--out", default="docs/stereo")
    ap.add_argument("--stride", type=int, default=8, help="frame stride for the validation figure")
    ap.add_argument("--only", help="regenerate one figure by number, e.g. 4")
    a = ap.parse_args()
    with open(a.calib) as fh:
        calib = json.load(fh)
    os.makedirs(a.out, exist_ok=True)
    p = lambda n: os.path.join(a.out, n)
    jobs = {
        "1": lambda: fig_geometry(p("fig1_sbs_geometry.png")),
        "2": lambda: fig_anamorphic(p("fig2_anamorphic_proof.png")),
        "3": lambda: fig_charuco(p("fig3_charuco_detection.png")),
        "4": lambda: fig_distortion(p("fig4_distortion.png"), calib),
        "5": lambda: fig_rectification(p("fig5_rectification.png"), calib),
        "6": lambda: fig_validation(p("fig6_validation.png"), calib, a.stride),
    }
    for k in ([a.only] if a.only else sorted(jobs)):
        print("fig %s" % k)
        jobs[k]()


if __name__ == "__main__":
    main()
