"""Compare arch-tip annotations across THREE annotators (Nick, Veerle, Aron),
generalizing compare_arch_annotators.py to N annotators.

Each annotator's export lives under data/InterAnnotator_Arch_Comparison/ in
its own layout:
    JSONfileNick/<video>/arches.json               (+ source_crop.json)
    JSONfilesVeerle/InterAnnotator_..._<video>_arches.json   (flat file)
    JSONfileAron/InterAnnotator_..._<video>/arches.json      (subfolder)
find_arches() tries all three layouts so new annotators can use any of them.

Nick is the reference frame (his coordinates line up with the actual cropped
video frame, per source_crop.json). Every other annotator gets a per-video,
per-pair constant offset removed the same way as compare_arch_annotators.py
found necessary (see that script / the earlier writeup: raw coordinates
across annotators are shifted by an apparent crop/letterbox mismatch, not
real disagreement) -- offset = mean(other_tip - nick_tip) over that pair's
shared frames in that video, subtracted before comparing.

Two outputs:
  1. Pairwise stats (Nick-Veerle, Nick-Aron, Veerle-Aron) over every frame
     each pair shares, like the original 2-annotator analysis.
  2. A combined per-frame disagreement score over frames ALL THREE annotators
     share: mean pairwise tip distance among the 3 corrected tips. Used to
     rank "worst / average / best" samples for visualization.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data" / "InterAnnotator_Arch_Comparison"
PREFIX = "InterAnnotator_Arch_Comparison__"
SUFFIX = "_arches.json"

ANNOTATORS = {
    "Nick": DATA / "JSONfileNick",
    "Veerle": DATA / "JSONfilesVeerle",
    "Aron": DATA / "JSONfileAron",
}
REFERENCE = "Nick"


def find_arches(annotator_dir, video):
    for cand in (
        annotator_dir / video / "arches.json",
        annotator_dir / f"{PREFIX}{video}" / "arches.json",
        annotator_dir / f"{PREFIX}{video}{SUFFIX}",
    ):
        if cand.exists():
            return cand
    return None


def load_frames(path):
    try:
        return json.loads(path.read_text())["frames"], None
    except json.JSONDecodeError as e:
        return None, f"corrupt/truncated JSON ({e})"


def apex_of(frame):
    left = np.array(frame["left"], dtype=np.float64)
    right = np.array(frame["right"], dtype=np.float64)
    mid = (left + right) / 2
    chord = right - left
    d = float(np.linalg.norm(chord)) / 2
    t = chord / (2 * d) if d > 1e-9 else np.array([1.0, 0.0])
    n = np.array([t[1], -t[0]])
    apex = mid + frame["height"] * n
    return apex, 2 * d


def main():
    videos = sorted(p.name for p in ANNOTATORS[REFERENCE].iterdir() if p.is_dir())
    names = list(ANNOTATORS)
    others = [n for n in names if n != REFERENCE]

    # ---- load everything -------------------------------------------------
    frames_by_annot_video = {}   # (annot, video) -> {frame_idx: entry}
    status = {}
    for name, d in ANNOTATORS.items():
        for video in videos:
            path = find_arches(d, video)
            if path is None:
                status[(name, video)] = "missing"
                continue
            frames, err = load_frames(path)
            if err:
                status[(name, video)] = err
                continue
            frames_by_annot_video[(name, video)] = {int(k): v for k, v in frames.items()}
            status[(name, video)] = "ok"

    print("=== data availability ===")
    print(f"{'video':<70}" + "".join(f"{n:>10}" for n in names))
    for video in videos:
        row = "".join(f"{status.get((n, video), 'missing'):>10}"
                       if status.get((n, video)) in ("ok",) else f"{'FAIL':>10}"
                       for n in names)
        print(f"{video:<70}{row}")

    # ---- per-pair, per-video offset + pairwise per-frame distances -------
    pair_rows = []
    offsets = {}   # (name, video) -> np.array offset relative to REFERENCE
    for other in others:
        for video in videos:
            ref_frames = frames_by_annot_video.get((REFERENCE, video))
            oth_frames = frames_by_annot_video.get((other, video))
            if ref_frames is None or oth_frames is None:
                continue
            shared = sorted(set(ref_frames) & set(oth_frames))
            if not shared:
                continue
            ref_apex = {fi: apex_of(ref_frames[fi])[0] for fi in shared}
            oth_apex = {fi: apex_of(oth_frames[fi])[0] for fi in shared}
            dx = np.mean([oth_apex[fi][0] - ref_apex[fi][0] for fi in shared])
            dy = np.mean([oth_apex[fi][1] - ref_apex[fi][1] for fi in shared])
            offsets[(other, video)] = np.array([dx, dy])
            for fi in shared:
                n_apex, n_chord = apex_of(ref_frames[fi])
                o_apex_raw, o_chord = apex_of(oth_frames[fi])
                o_apex = o_apex_raw - offsets[(other, video)]
                dist = float(np.linalg.norm(n_apex - o_apex))
                avg_chord = (n_chord + o_chord) / 2
                pair_rows.append(dict(pair=f"{REFERENCE}-{other}", video=video, frame=fi,
                                       resid_px=dist,
                                       resid_pct_chord=100 * dist / avg_chord if avg_chord > 1e-9 else float("nan")))

    print(f"\n=== pairwise residual tip disagreement (offset removed) ===")
    for other in others:
        pr = [r for r in pair_rows if r["pair"] == f"{REFERENCE}-{other}"]
        if not pr:
            print(f"{REFERENCE}-{other}: no shared frames")
            continue
        d = np.array([r["resid_px"] for r in pr])
        pc = np.array([r["resid_pct_chord"] for r in pr])
        print(f"{REFERENCE}-{other:<10} n={len(pr):5d}  px: mean={d.mean():6.1f} median={np.median(d):6.1f} "
              f"max={d.max():6.1f}   %chord: mean={pc.mean():5.1f}% median={np.median(pc):5.1f}%")

    # direct Veerle-Aron too, if they share any video/frames (both offset from Nick already,
    # so compare their nick-corrected positions to each other)
    va_rows = []
    for video in videos:
        if (REFERENCE, video) not in frames_by_annot_video:
            continue
        v_frames = frames_by_annot_video.get(("Veerle", video))
        a_frames = frames_by_annot_video.get(("Aron", video))
        if v_frames is None or a_frames is None or ("Veerle", video) not in offsets or ("Aron", video) not in offsets:
            continue
        shared = sorted(set(v_frames) & set(a_frames))
        for fi in shared:
            v_apex = apex_of(v_frames[fi])[0] - offsets[("Veerle", video)]
            a_apex, a_chord = apex_of(a_frames[fi])
            a_apex = a_apex - offsets[("Aron", video)]
            v_chord = apex_of(v_frames[fi])[1]
            dist = float(np.linalg.norm(v_apex - a_apex))
            avg_chord = (v_chord + a_chord) / 2
            va_rows.append(dict(pair="Veerle-Aron", video=video, frame=fi, resid_px=dist,
                                 resid_pct_chord=100 * dist / avg_chord if avg_chord > 1e-9 else float("nan")))
    if va_rows:
        d = np.array([r["resid_px"] for r in va_rows])
        pc = np.array([r["resid_pct_chord"] for r in va_rows])
        print(f"{'Veerle-Aron':<17} n={len(va_rows):5d}  px: mean={d.mean():6.1f} median={np.median(d):6.1f} "
              f"max={d.max():6.1f}   %chord: mean={pc.mean():5.1f}% median={np.median(pc):5.1f}%")
    pair_rows += va_rows

    (ROOT / "outputs" / "arch_tip_comparison_multi_pairs.json").write_text(json.dumps(pair_rows, indent=2))

    # ---- triple-overlap combined disagreement score -----------------------
    triple_rows = []
    for video in videos:
        frame_sets = []
        for name in names:
            f = frames_by_annot_video.get((name, video))
            if f is None:
                frame_sets = []
                break
            frame_sets.append(set(f))
        if not frame_sets:
            continue
        shared = sorted(set.intersection(*frame_sets))
        for fi in shared:
            tips = {}
            for name in names:
                entry = frames_by_annot_video[(name, video)][fi]
                apex, chord = apex_of(entry)
                if name != REFERENCE:
                    apex = apex - offsets.get((name, video), np.zeros(2))
                tips[name] = (apex, chord)
            apexes = {n: t[0] for n, t in tips.items()}
            chords = [t[1] for t in tips.values()]
            pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
            dists = {f"{a}-{b}": float(np.linalg.norm(apexes[a] - apexes[b])) for a, b in pairs}
            mean_dist = float(np.mean(list(dists.values())))
            max_dist = float(np.max(list(dists.values())))
            avg_chord = float(np.mean(chords))
            triple_rows.append(dict(video=video, frame=fi, mean_pairwise_px=mean_dist,
                                     max_pairwise_px=max_dist,
                                     mean_pairwise_pct_chord=100 * mean_dist / avg_chord if avg_chord > 1e-9 else float("nan"),
                                     dists=dists,
                                     tips={n: [round(x, 1) for x in apexes[n]] for n in names}))

    if triple_rows:
        m = np.array([r["mean_pairwise_px"] for r in triple_rows])
        print(f"\n=== 3-way combined disagreement (frames all 3 share, n={len(triple_rows)}) ===")
        print(f"mean pairwise distance (px): mean={m.mean():.1f} median={np.median(m):.1f} "
              f"min={m.min():.1f} max={m.max():.1f}")
        by_video = {}
        for r in triple_rows:
            by_video.setdefault(r["video"], []).append(r["mean_pairwise_px"])
        print(f"\n{'video':<70} {'n':>5} {'mean_px':>8} {'median_px':>10}")
        for video, vals in sorted(by_video.items()):
            vals = np.array(vals)
            print(f"{video:<70} {len(vals):>5} {vals.mean():>8.1f} {np.median(vals):>10.1f}")

    (ROOT / "outputs" / "arch_tip_comparison_multi_triple.json").write_text(json.dumps(triple_rows, indent=2))
    print(f"\nwrote outputs/arch_tip_comparison_multi_pairs.json ({len(pair_rows)} rows)")
    print(f"wrote outputs/arch_tip_comparison_multi_triple.json ({len(triple_rows)} rows)")


if __name__ == "__main__":
    main()
