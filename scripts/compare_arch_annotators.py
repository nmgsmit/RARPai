"""Compare Nick's vs Veerle's Retzius-arch annotations (arches.json), tip-focused.

Nick: data/InterAnnotator_Arch_Comparison/JSONfileNick/<video>.mp4/arches.json
      (per-video subfolder, tracked through the whole clip -> many frames)
Veerle: data/InterAnnotator_Arch_Comparison/JSONfilesVeerle/
        InterAnnotator_Arch_Comparison__<video>.mp4_arches.json
        (flat, one file per video)

Arch geometry (see transfer_atlas_mod/gui/retzius_arch.py):
    mid = (left+right)/2 ; chord = right-left ; d = |chord|/2
    t = chord/(2d) ; n = (t_y, -t_x)   (unit normal)
    apex (tip) = mid + height * n

The two annotators almost never clicked the SAME frame manually (TRACK fills
in everything between keyframes), so we compare on every frame index present
in BOTH files regardless of source ('manual' or 'tracked') -- that is what
actually overlaps, and it is what the arch overlay shows on screen at that
frame for each annotator either way. Frame source (manual/tracked) is kept
per side for transparency in the per-frame dump.

Deviation is reported in raw pixels, as a percentage of the arch chord length
(left-right span -- an annotation-intrinsic scale, averaged across the two
annotators' chords for that frame), and as a percentage of the frame's image
diagonal (from source_crop.json's output_size, a fixed scale per video).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data" / "InterAnnotator_Arch_Comparison"
NICK_DIR = DATA / "JSONfileNick"
VEERLE_DIR = DATA / "JSONfilesVeerle"
VEERLE_PREFIX = "InterAnnotator_Arch_Comparison__"
VEERLE_SUFFIX = "_arches.json"


def apex_of(frame):
    left = np.array(frame["left"], dtype=np.float64)
    right = np.array(frame["right"], dtype=np.float64)
    mid = (left + right) / 2
    chord = right - left
    d = float(np.linalg.norm(chord)) / 2
    t = chord / (2 * d) if d > 1e-9 else np.array([1.0, 0.0])
    n = np.array([t[1], -t[0]])
    apex = mid + frame["height"] * n
    return apex, 2 * d  # tip point, chord length


def load_json(path):
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text()), None
    except json.JSONDecodeError as e:
        return None, f"corrupt/truncated JSON ({e})"


def load_frames(path):
    data, err = load_json(path)
    if err:
        return None, err
    return data["frames"], None


def image_diag(video):
    data, err = load_json(NICK_DIR / video / "source_crop.json")
    if err or "output_size" not in data:
        return None
    w, h = data["output_size"]
    return float(np.hypot(w, h))


def main():
    videos = sorted(p.name for p in NICK_DIR.iterdir() if p.is_dir())
    rows = []
    for video in videos:
        nick_path = NICK_DIR / video / "arches.json"
        veerle_path = VEERLE_DIR / f"{VEERLE_PREFIX}{video}{VEERLE_SUFFIX}"
        nick_frames, n_err = load_frames(nick_path)
        veerle_frames, v_err = load_frames(veerle_path)
        if n_err or v_err:
            print(f"[skip] {video}: nick={n_err or 'ok'} | veerle={v_err or 'ok'}")
            continue

        nick_frames = {int(k): v for k, v in nick_frames.items()}
        veerle_frames = {int(k): v for k, v in veerle_frames.items()}
        shared = sorted(set(nick_frames) & set(veerle_frames))
        diag = image_diag(video)

        if not shared:
            print(f"[skip] {video}: no overlapping frame indices at all "
                  f"(nick {min(nick_frames)}-{max(nick_frames)}, "
                  f"veerle {min(veerle_frames)}-{max(veerle_frames)})")
            continue

        for fi in shared:
            nf, vf = nick_frames[fi], veerle_frames[fi]
            n_apex, n_chord = apex_of(nf)
            v_apex, v_chord = apex_of(vf)
            dist = float(np.linalg.norm(n_apex - v_apex))
            avg_chord = (n_chord + v_chord) / 2
            pct_chord = 100.0 * dist / avg_chord if avg_chord > 1e-9 else float("nan")
            pct_diag = 100.0 * dist / diag if diag else float("nan")
            rows.append(dict(video=video, frame=fi, dist_px=dist,
                              pct_of_chord=pct_chord, pct_of_diag=pct_diag,
                              avg_chord=avg_chord, img_diag=diag,
                              nick_source=nf.get("source"), veerle_source=vf.get("source"),
                              nick_tip=tuple(round(x, 1) for x in n_apex),
                              veerle_tip=tuple(round(x, 1) for x in v_apex)))
        d = np.array([r["dist_px"] for r in rows if r["video"] == video])
        print(f"[ok]   {video}: {len(shared)} shared frames, "
              f"tip dist mean={d.mean():.1f}px median={np.median(d):.1f}px max={d.max():.1f}px")

    if not rows:
        print("\nNo shared frames found across any video -- nothing to compare.")
        return

    dists = np.array([r["dist_px"] for r in rows])
    pct_c = np.array([r["pct_of_chord"] for r in rows])
    pct_d = np.array([r["pct_of_diag"] for r in rows if not np.isnan(r["pct_of_diag"])])

    print(f"\n=== overall: {len(rows)} compared frames across "
          f"{len(set(r['video'] for r in rows))} videos ===")
    print(f"tip distance (px):        mean={dists.mean():.1f}  median={np.median(dists):.1f}  "
          f"min={dists.min():.1f}  max={dists.max():.1f}  std={dists.std():.1f}")
    print(f"tip distance (% chord):   mean={pct_c.mean():.1f}%  median={np.median(pct_c):.1f}%  "
          f"min={pct_c.min():.1f}%  max={pct_c.max():.1f}%  std={pct_c.std():.1f}%")
    if len(pct_d):
        print(f"tip distance (% diag):    mean={pct_d.mean():.1f}%  median={np.median(pct_d):.1f}%  "
              f"min={pct_d.min():.1f}%  max={pct_d.max():.1f}%  std={pct_d.std():.1f}%")

    # per-video summary table
    print(f"\n{'video':<70} {'n':>5} {'mean_px':>8} {'median_px':>10} {'mean_%chord':>12}")
    for video in sorted(set(r["video"] for r in rows)):
        vr = [r for r in rows if r["video"] == video]
        d = np.array([r["dist_px"] for r in vr])
        pc = np.array([r["pct_of_chord"] for r in vr])
        print(f"{video:<70} {len(vr):>5} {d.mean():>8.1f} {np.median(d):>10.1f} {pc.mean():>11.1f}%")

    out_path = ROOT / "outputs" / "arch_tip_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nfull per-frame results ({len(rows)} rows) written to {out_path}")

    # --- per-video coordinate-frame offset -------------------------------
    # Veerle's files carry no source_crop.json, and the raw tip distance
    # above is dominated by a near-constant per-video translation (std is
    # 5-15x smaller than the mean within a video, but the mean itself varies
    # video to video) -- i.e. the two annotators' coordinate origins don't
    # line up (almost certainly a different crop/letterbox detection per
    # side), not real disagreement. Subtracting each video's own mean offset
    # isolates the actual tip-placement disagreement.
    print(f"\n=== per-video coordinate offset (systematic, not disagreement) ===")
    print(f"{'video':<70} {'offset_dx':>10} {'offset_dy':>10} {'offset_|.|':>11}")
    resid = []
    for video in sorted(set(r["video"] for r in rows)):
        vr = [r for r in rows if r["video"] == video]
        dx = np.array([r["veerle_tip"][0] - r["nick_tip"][0] for r in vr])
        dy = np.array([r["veerle_tip"][1] - r["nick_tip"][1] for r in vr])
        ox, oy = dx.mean(), dy.mean()
        print(f"{video:<70} {ox:>10.1f} {oy:>10.1f} {np.hypot(ox, oy):>11.1f}")
        rdist = np.hypot(dx - ox, dy - oy)
        for r, rd in zip(vr, rdist):
            resid.append(dict(r, resid_px=float(rd),
                               resid_pct_chord=100.0 * float(rd) / r["avg_chord"] if r["avg_chord"] > 1e-9 else float("nan"),
                               resid_pct_diag=100.0 * float(rd) / r["img_diag"] if r.get("img_diag") else float("nan")))

    rp = np.array([r["resid_px"] for r in resid])
    rpc = np.array([r["resid_pct_chord"] for r in resid])
    print(f"\n=== residual tip disagreement, after removing each video's own offset ===")
    print(f"residual distance (px):      mean={rp.mean():.1f}  median={np.median(rp):.1f}  "
          f"min={rp.min():.1f}  max={rp.max():.1f}  std={rp.std():.1f}")
    print(f"residual distance (% chord): mean={rpc.mean():.1f}%  median={np.median(rpc):.1f}%  "
          f"max={rpc.max():.1f}%")
    print(f"{'video':<70} {'n':>5} {'mean_px':>8} {'median_px':>10} {'max_px':>8}")
    for video in sorted(set(r["video"] for r in resid)):
        vr = [r["resid_px"] for r in resid if r["video"] == video]
        vr = np.array(vr)
        print(f"{video:<70} {len(vr):>5} {vr.mean():>8.1f} {np.median(vr):>10.1f} {vr.max():>8.1f}")

    resid_path = ROOT / "outputs" / "arch_tip_comparison_residual.json"
    resid_path.write_text(json.dumps(resid, indent=2))
    print(f"\nresidual per-frame results written to {resid_path}")


if __name__ == "__main__":
    main()
