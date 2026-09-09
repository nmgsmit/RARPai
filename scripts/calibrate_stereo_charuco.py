"""Stereo-calibrate the da Vinci 3D (side-by-side) feed from the ChArUco calibration clip.

The 3D feed is half-width anamorphic SBS: each eye sits in its own 960px half, squeezed
2x horizontally. Both eyes are mapped into the SAME 1340x1072 frame the mono depth
pipeline uses (source_crop.json: x=289 y=4 w=1340 h=1072 of the 1920x1080 console output),
so the K that comes out is directly the K the training code needs -- no conversion.

The eye->mono affine below was measured by cross-correlating the da Vinci GUI banner (a
fixed-size overlay, i.e. a free ruler) between the SBS halves and the raw mono videos:
scale 2.125, left edge x=286, identical on every surgical clip AND on this calibration
clip. An error in it is absorbed into fx/fy at calibration time and is harmless, PROVIDED
the same transform is applied to the surgical clips -- which it is, via eye_to_mono().

    python scripts/calibrate_stereo_charuco.py --video ../data/ARUCO_calibration/<clip>.mp4
"""
import argparse
import json
import os

import cv2
import numpy as np

# --- SBS eye -> mono 1340x1072 geometry (see module docstring) -----------------------
OUT_W, OUT_H = 1340, 1072
SRC_X, SRC_Y, SRC_W, SRC_H = 165.41, 35.76, 630.59, 1008.47   # left eye, in the 1920x1080 frame
EYE_DX = 960.0                                                # right eye offset
BANNER_ROW = 1020            # GUI banner starts here in the 1340x1072 output

# --- boards (OTHERS/charuco_endoscope_A4.pdf) ---------------------------------------
BOARDS = {
    "A": dict(squares=(9, 7), square=8.0, marker=6.0, ids=(0, 31)),    # main,  72x56 mm
    "B": dict(squares=(7, 5), square=6.0, marker=4.5, ids=(31, 48)),   # close, 42x30 mm
}


def eye_to_mono(frame, right):
    """One sub-pixel crop+resize: SBS eye -> the mono 1340x1072 frame."""
    sx, sy = OUT_W / SRC_W, OUT_H / SRC_H
    x0 = SRC_X + (EYE_DX if right else 0.0)
    m = np.float32([[sx, 0, -sx * x0], [0, sy, -sy * SRC_Y]])
    return cv2.warpAffine(frame, m, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)


def make_board(spec):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    b = cv2.aruco.CharucoBoard(spec["squares"], spec["square"], spec["marker"], d,
                               np.arange(*spec["ids"]))
    b.setLegacyPattern(False)      # boards were generated with the OpenCV >= 4.6 layout
    return b


def collect(video, board, stride, min_corners, blur_pct):
    """Views where BOTH eyes see >= min_corners of the same ChArUco corners."""
    det = cv2.aruco.CharucoDetector(board)
    all_obj = board.getChessboardCorners()          # (N,3) float32, millimetres
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    views = []
    for i in range(0, n, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        eyes = {}
        for side, right in (("L", False), ("R", True)):
            img = eye_to_mono(fr, right)[:BANNER_ROW]        # banner carries no board
            corners, ids, _, _ = det.detectBoard(img)
            if ids is None or len(ids) < min_corners:
                break
            eyes[side] = (corners.reshape(-1, 2), ids.ravel(), img)
        if len(eyes) < 2:
            continue
        common = np.intersect1d(eyes["L"][1], eyes["R"][1])
        if len(common) < min_corners:
            continue
        pick = {s: np.searchsorted(eyes[s][1], common) for s in eyes}   # detectBoard sorts ids
        # Sharpness of the board's bounding box. Motion blur is the dominant error source
        # here; a blurred view poisons the fit more than the extra pose diversity buys.
        p = eyes["L"][0]
        x0, y0 = np.maximum(p.min(0).astype(int) - 8, 0)
        x1, y1 = p.max(0).astype(int) + 8
        patch = cv2.cvtColor(eyes["L"][2][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        views.append(dict(
            frame=i, sharp=float(cv2.Laplacian(patch, cv2.CV_64F).var()),
            obj=all_obj[common].astype(np.float32),
            L=eyes["L"][0][pick["L"]].astype(np.float32),
            R=eyes["R"][0][pick["R"]].astype(np.float32)))
    cap.release()
    if not views:
        return []
    thr = np.percentile([v["sharp"] for v in views], blur_pct)
    return [v for v in views if v["sharp"] >= thr]


def spread(views, k):
    """Keep k views spread evenly in time -- the board is waved around, so time is a cheap
    stand-in for pose diversity. ponytail: if the fit is ill-conditioned, cluster on board
    pose instead."""
    if len(views) <= k:
        return views
    idx = np.linspace(0, len(views) - 1, k).round().astype(int)
    return [views[i] for i in idx]


def distortion_px(K, D, w=OUT_W, h=OUT_H):
    """How far the lens actually moves a pixel, by radius. The mono pipeline assumes a
    pinhole; this says where that assumption stops holding."""
    g = np.array([[x, y] for x in np.linspace(0, w - 1, 60) for y in np.linspace(0, h - 1, 48)],
                 np.float32).reshape(-1, 1, 2)
    d = np.linalg.norm(cv2.undistortPoints(g, K, D, P=K).reshape(-1, 2) - g.reshape(-1, 2), axis=1)
    r = np.linalg.norm(g.reshape(-1, 2) - [K[0, 2], K[1, 2]], axis=1)
    bands = [(lo, hi, float(d[(r >= lo) & (r < hi)].mean()), float(d[(r >= lo) & (r < hi)].max()))
             for lo, hi in ((0, 200), (200, 400), (400, 600), (600, 900)) if ((r >= lo) & (r < hi)).any()]
    return dict(median=float(np.median(d)), p95=float(np.percentile(d, 95)), max=float(d.max()),
                bands=bands)


def validate(views, cal):
    """Acceptance test: depth from disparity vs depth from the board's own PnP pose.

    These are independent -- PnP uses the left eye and the known 8mm board geometry, the
    disparity path uses both eyes and the baseline. If the rectification, the eye->mono
    affine or the board spec were wrong, they would not agree. Runs on the calibration
    views, so it costs nothing extra.
    """
    KL, DL = np.array(cal["KL"]), np.array(cal["DL"])
    KR, DR = np.array(cal["KR"]), np.array(cal["DR"])
    R1, R2 = np.array(cal["R1"]), np.array(cal["R2"])
    P1, P2 = np.array(cal["P1"]), np.array(cal["P2"])
    f_b = -P2[0, 3]
    zp, zd, dys = [], [], []
    for v in views:
        ok, rv, tv = cv2.solvePnP(v["obj"], v["L"], KL, DL, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        qL = cv2.undistortPoints(v["L"].reshape(-1, 1, 2), KL, DL, R=R1, P=P1).reshape(-1, 2)
        qR = cv2.undistortPoints(v["R"].reshape(-1, 1, 2), KR, DR, R=R2, P=P2).reshape(-1, 2)
        zp.append((cv2.Rodrigues(rv)[0] @ v["obj"].T + tv)[2])
        zd.append(f_b / (qL[:, 0] - qR[:, 0]))
        dys.append(qL[:, 1] - qR[:, 1])
    zp, zd, dy = np.concatenate(zp), np.concatenate(zd), np.concatenate(dys)
    e = zd - zp
    return dict(corners=int(len(zp)), mae_mm=float(np.abs(e).mean()),
                bias_mm=float(np.median(e)), p95_mm=float(np.percentile(np.abs(e), 95)),
                rel_mae=float(np.mean(np.abs(e) / zp)),
                epipolar_dy_med=float(np.median(np.abs(dy))),
                epipolar_dy_p95=float(np.percentile(np.abs(dy), 95)),
                z_range_mm=[float(zp.min()), float(zp.max())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--board", default="A", choices=list(BOARDS))
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--min-corners", type=int, default=10)
    ap.add_argument("--blur-pct", type=float, default=40.0, help="drop the blurriest N%%")
    ap.add_argument("--max-views", type=int, default=60)
    ap.add_argument("--out", default="outputs/stereo_calib")
    a = ap.parse_args()

    board = make_board(BOARDS[a.board])
    views = collect(a.video, board, a.stride, a.min_corners, a.blur_pct)
    print("usable stereo views after sharpness gate: %d" % len(views))
    views = spread(views, a.max_views)
    if len(views) < 8:
        raise SystemExit("only %d views -- not enough to calibrate" % len(views))
    obj = [v["obj"] for v in views]
    ptsL = [v["L"] for v in views]
    ptsR = [v["R"] for v in views]
    size = (OUT_W, OUT_H)
    print("calibrating on %d views, %d corner pairs" % (len(views), sum(len(o) for o in obj)))

    rmsL, KL, DL, *_ = cv2.calibrateCamera(obj, ptsL, size, None, None)
    rmsR, KR, DR, *_ = cv2.calibrateCamera(obj, ptsR, size, None, None)
    rms, KL, DL, KR, DR, R, T, *_ = cv2.stereoCalibrate(
        obj, ptsL, ptsR, KL, DL, KR, DR, size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(KL, DL, KR, DR, size, R, T,
                                              flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    baseline = float(np.linalg.norm(T))
    f_b = float(-P2[0, 3])                # P2[0,3] = -f*B, in the RECTIFIED frame -> mm*px
    # The console already ships a horizontally shifted ("converged") pair: this is the shift.
    conv = float(KL[0, 2] - KR[0, 2])
    deg = lambda M: float(np.degrees(np.linalg.norm(cv2.Rodrigues(M)[0])))

    rep = dict(
        video=os.path.basename(a.video), board=a.board, views=len(views),
        image_size=[OUT_W, OUT_H],
        eye_to_mono=dict(src_x=SRC_X, src_y=SRC_Y, src_w=SRC_W, src_h=SRC_H, eye_dx=EYE_DX),
        rms=dict(left=rmsL, right=rmsR, stereo=rms),
        KL=KL.tolist(), DL=DL.ravel().tolist(), KR=KR.tolist(), DR=DR.ravel().tolist(),
        R=R.tolist(), T=T.ravel().tolist(), Q=Q.tolist(),
        P1=P1.tolist(), P2=P2.tolist(), R1=R1.tolist(), R2=R2.tolist(),
        baseline_mm=baseline, f_times_B_rectified=f_b, console_convergence_px=conv,
        f_rectified=float(P1[0, 0]),
        stereo_rotation_deg=deg(R), rect_rot_left_deg=deg(R1), rect_rot_right_deg=deg(R2),
        distortion_left=distortion_px(KL, DL), distortion_right=distortion_px(KR, DR),
        # normalised the way the training code writes K: fx/W, fy/H, cx/W, cy/H
        K_norm_left=[KL[0, 0] / OUT_W, KL[1, 1] / OUT_H, KL[0, 2] / OUT_W, KL[1, 2] / OUT_H],
        K_norm_right=[KR[0, 0] / OUT_W, KR[1, 1] / OUT_H, KR[0, 2] / OUT_W, KR[1, 2] / OUT_H],
    )
    rep["validation"] = validate(views, rep)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "calib.json")
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=2)

    scared = (0.82, 1.02, 0.5, 0.5)       # SCARED / da Vinci Xi -- the borrowed base K
    kn = rep["K_norm_left"]
    v = rep["validation"]
    print("\nRMS reproj  left %.3f  right %.3f  stereo %.3f px" % (rmsL, rmsR, rms))
    print("fx %8.2f  fy %8.2f  cx %8.2f  cy %8.2f   (left, px)"
          % (KL[0, 0], KL[1, 1], KL[0, 2], KL[1, 2]))
    print("fx/fy %.4f   <- 1.0 confirms the 2.125 un-squeeze" % (KL[0, 0] / KL[1, 1]))
    print("baseline %.4f mm   stereo rotation %.3f deg" % (baseline, deg(R)))
    print("console convergence (cxL-cxR) %+.2f px" % conv)
    print("\ndistortion (px displacement from pinhole), left eye:")
    for lo, hi, mean, mx in rep["distortion_left"]["bands"]:
        print("   radius %3d-%3d px:  mean %5.2f  max %5.2f" % (lo, hi, mean, mx))
    print("\nnormalised K   fx      fy      cx      cy")
    print("  ours      %.4f  %.4f  %.4f  %.4f" % tuple(kn))
    print("  SCARED    %.4f  %.4f  %.4f  %.4f" % scared)
    print("  ratio     %.3f  %.3f  %.3f  %.3f" % tuple(k / s for k, s in zip(kn, scared)))
    print("\nrectify with R1/P1, R2/P2, then  Z_mm = %.1f / disparity_px" % f_b)
    print("VALIDATION (disparity depth vs the board's own PnP pose, %d corners, Z %.0f-%.0f mm)"
          % (v["corners"], *v["z_range_mm"]))
    print("  MAE %.3f mm   bias %+.3f mm   p95 %.3f mm   rel MAE %.2f%%"
          % (v["mae_mm"], v["bias_mm"], v["p95_mm"], 100 * v["rel_mae"]))
    print("  epipolar |dy| after rectification: median %.3f  p95 %.3f px"
          % (v["epipolar_dy_med"], v["epipolar_dy_p95"]))
    print("wrote %s" % path)
    if v["mae_mm"] > 3.0:
        raise SystemExit("VALIDATION FAILED: %.2f mm MAE -- calibration is not usable" % v["mae_mm"])


if __name__ == "__main__":
    main()
