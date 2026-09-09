# da Vinci 3D stereo — geometry, calibration, and metric depth

Findings from decoding the 3D (side-by-side) console recordings in `../data/3D_ProxyGT` and
stereo-calibrating them from the ChArUco clip in `../data/ARUCO_calibration`. The goal is a
**metric proxy ground truth** for monocular depth training.

Every number below was measured on the actual files, not assumed. Where something is still an
assumption it is marked **ASSUMPTION**.

- Code: [scripts/calibrate_stereo_charuco.py](scripts/calibrate_stereo_charuco.py)
- Output: `outputs/stereo_calib/calib.json`
- Design log entry: [CLAUDE_NOTES.md](CLAUDE_NOTES.md), 2026-09-09

---

## 1. What the 3D files actually are

| property | value |
|---|---|
| Container | 1920×1080, 59.94 fps, H.264 |
| Layout | **Half-width anamorphic side-by-side** (not frame-sequential) |
| Left eye | x `164..799`, y `32..1047` → 636×1016 |
| Right eye | x `1124..1759` (exactly +960), same rows |
| Per-eye native | 1272×1016 after a **2× horizontal** stretch |
| GUI banner | rows `965..1015` inside the eye crop |

Two independent confirmations that the horizontal squeeze is exactly 2× and not something else:

1. **The GUI badges.** Blown up isotropically, the da Vinci instrument badges (`1`,`2`,`3`,`4`)
   are unmistakable ellipses with a 2:1 axis ratio; at 2× horizontal they are circles.
2. **The calibration itself** — see §3, `fx/fy = 1.0003`.

### The encoder's transform

Cross-correlating the GUI banner — a fixed-size overlay rendered at native console resolution,
i.e. a free ruler — between the SBS halves and the raw 2D videos gives:

```
scale 2.125,  left edge x = 286
```

identical on **all three surgical clips, both eyes, and the calibration clip** (NCC ≈ 0.72).
So the SBS encoder took the whole console frame, scaled it uniformly by 0.94, then squeezed it
2× horizontally into each 960-wide half. The eye→console-frame map is:

```
raw_x = 286 + 2.125 · eye_x
raw_y =   0 + 1.063 · eye_y
```

Sanity check: the banner starts at eye row 965; 965 × 1.063 = 1025.8, and the banner in the 2D
videos starts at row 1024.

---

## 2. Mapping onto the existing 1340×1072 pipeline

From `source_crop.json` in the processed depth clips, the mono crop is:

```
1920×1080  →  x=289, y=4, w=1340, h=1072
```

Measuring the raw 2D videos: anatomy occupies rows `0..1023`, the banner rows `1024..1079`,
content columns `~285..1631`. **So the 1340×1072 frame is 1020 rows of anatomy plus 52 rows of
GUI banner** (rows 1020–1071), blacked out in the NoGUI clips. Worth knowing — it is not 1072
rows of anatomy.

Composing the eye→console map with the mono crop gives one sub-pixel crop-and-resize per eye,
straight into the training frame:

| | |
|---|---|
| Left eye source | `x ∈ [165.41, 796.00), y ∈ [35.76, 1044.23)` in the 1920×1080 frame |
| Right eye source | same, `+960` in x |
| Resize to | **1340×1072** |
| Banner rows | 1020–1071, excluded |

Implemented as `eye_to_mono()` in `calibrate_stereo_charuco.py`. **Do not re-derive it.**

> **Why an error in the 2.125 would not matter.** Any error in the affine is absorbed into fx/fy
> during calibration. Only *consistency* between the calibration clip and the surgical clips
> matters — and the banner correlation is bit-identical across all four files. This makes the
> geometry chain robust to my own measurement error.

---

## 3. Stereo calibration

ChArUco boards from `OTHERS/charuco_endoscope_A4.pdf`:

| board | dictionary | ids | squares | square | marker |
|---|---|---|---|---|---|
| A (main) | `DICT_4X4_50` | 0–30 | 9×7 | 8.00 mm | 6.00 mm |
| B (close) | `DICT_4X4_50` | 31–47 | 7×5 | 6.00 mm | 4.50 mm |

Board A gives ≥6 corners in *both* eyes on 43% of frames, reaching the full 48. 60 sharpness-
selected views, 2463 corner pairs, RMS reprojection **0.50 px**.

### Results (left eye, in the 1340×1072 frame)

```
fx 1061.78   fy 1061.51   cx 608.65   cy 535.68     px
fx/fy = 1.0003
baseline           4.1155 mm
stereo rotation    0.205°
console convergence (cxL − cxR)   −96.73 px
cyL − cyR                          +0.01 px
rectified:   Z_mm = 5030.5 / disparity_px
```

`fx/fy = 1.0003` is the independent confirmation of §1. The 2.125 un-squeeze was derived from
the GUI banner; if it had been wrong, the calibrated pixels would not have come out square. It is
right to 0.03%.

`cyL − cyR = 0.01 px` and a stereo rotation of 0.205° confirm the console ships an
**already-rectified** pair. The −96.73 px is a deliberate horizontal convergence shift for
comfortable 3D viewing — which is why raw disparities come out both positive and negative with a
median near zero.

---

## 4. Comparison with the SCARED / da Vinci Xi standard

Normalised the way the training code writes K (fx/W, fy/H, cx/W, cy/H):

| | fx | fy | cx | cy |
|---|---|---|---|---|
| **Ours (measured)** | 0.7924 | 0.9902 | **0.4542** | 0.4997 |
| SCARED / da Vinci Xi | 0.8200 | 1.0200 | 0.5000 | 0.5000 |
| ratio | 0.966 | 0.971 | **0.908** | 0.999 |

**The focal length was a good stand-in; the principal point was not.**

- fx, fy agree to ~3%. Borrowing SCARED's focal was defensible.
- **cx is off by 61 px (9%).** The optical axis is not at frame centre. This biases ray
  directions by up to 3.3° across the frame, systematically rather than randomly, so it does not
  average out. Every prior UMC depth run reprojected through this error.
- cy is exact to 0.03%, which argues the horizontal offset is a real property of how the console
  crops rather than fit noise.

The ~3% focal error is also a ~3% depth-scale error, which is directly relevant to the metric
scale work — some of the observed scale drift may simply be this.

---

## 5. Lens distortion is NOT negligible

Pixel displacement from a pure pinhole model, left eye:

| radius from optical axis | mean | max |
|---|---|---|
| 0–200 px | 0.16 px | 0.46 px |
| 200–400 px | 0.80 px | 2.30 px |
| 400–600 px | 2.41 px | 6.70 px |
| **600–900 px** | **5.37 px** | **34.70 px** |

The first validation pass skipped undistortion and came out **9% biased in depth**.
Undistorting and rectifying dropped it to 0.8%. The centre of the image is effectively pinhole;
the periphery is not.

**Consequence for the mono pipeline:** the pinhole assumption baked into
`crop_adjust_intrinsics` and the reprojection loss breaks down beyond r ≈ 400 px. This is
independent of the stereo work and worth fixing on its own.

---

## 6. Validation

Depth from stereo disparity versus depth from the board's own PnP pose. These are independent
chains — PnP uses one eye plus the known 8 mm board geometry; disparity uses both eyes plus the
baseline. If the rectification, the eye→mono affine, or the board spec were wrong, they would
not agree.

```
2463 corners, Z 52–203 mm
MAE 1.004 mm    bias −0.289 mm    p95 2.651 mm    rel MAE 0.81%
epipolar |dy| after rectification: median 0.197 px, p95 0.620 px
```

| depth band | n | MAE | rel |
|---|---|---|---|
| 40–80 mm | 73 | 0.76 mm | 1.13% |
| 80–120 mm | 1362 | 0.79 mm | 0.75% |
| 120–250 mm | 1125 | 1.16 mm | 0.83% |

This check runs automatically inside `calibrate_stereo_charuco.py` and exits non-zero above
3 mm MAE.

### Transfer to the surgical clips

| clip | disparity p5 / med / p95 | implied **Z_mm** p5 / med / p95 |
|---|---|---|
| `18de9c5a…seg2` | 46 / 79 / 157 | 32 / 64 / 109 |
| `18de9c5a…seg3` | 59 / 102 / 159 | 32 / 49 / 86 |
| `5e27066c` | 47 / 113 / 174 | 29 / 45 / 107 |

All three land in the endoscopic working range, **including `5e27066c`, which is a different
session from the calibration clip**. Independently, the pre-flight estimate from the
hand-annotated ruler / catheter / robot-arm objects gave median 55 / 61 / 45 mm. Two unrelated
measurement chains agreeing on working distance is a strong end-to-end check.

All four clips (calibration and all three surgical) display **1× 0°** in the console banner —
same digital zoom, same scope angle. This is what licenses transferring the calibration.

---

## 7. Consequences for the proxy-GT plan

**The proxy GT can be metric.** `Z_mm = 5030.5 / disparity_px` on the rectified pair, validated
to 1 mm. This replaces the scale-and-shift-invariant loss that would otherwise be required, and
gives a dense alternative to the 1677 sparse hand-annotated segments currently carrying the
metric scale signal.

**Rectified disparity search range is ~30–200 px** — measured, not guessed.

**Precision ceiling.** Each eye is horizontally subsampled 2× by the SBS encoder, so disparity
precision in native pixels is roughly half that of a true full-resolution stereo pair. Adequate
for surfaces, poor on thin structures (suture, instrument tips).

**Sampling.** At 59.94 fps consecutive frames are near-duplicates; ~10 fps (every 6th frame) is
the useful rate.

---

## 8. Open assumptions and risks

**ASSUMPTION — which eye is the 2D feed?** The mono training videos are 2D console output. The
calibration gives two different principal points (left cx 608.65, right cx 705.38 — a 97 px
convergence shift). Undistorting the mono clips requires knowing which one applies, and the
mono videos come from 18 surgeries with **no overlap** with the two stereo surgeries, so it
cannot be resolved by correlation.

This is not a small risk. Using the wrong eye's parameters:

| radius | error if uncorrected | residual if wrong eye used |
|---|---|---|
| 0–200 px | 0.16 px | 0.17 px |
| 200–400 px | 0.79 px | 0.36 px |
| 400–600 px | 2.41 px | 2.46 px |
| **600–900 px** | **5.36 px** | **11.59 px** |

**At the periphery, picking the wrong eye is worse than doing nothing.** Ways to settle it,
cheapest first: ask the OR / da Vinci contact which eye the 2D channel carries; or record the
ChArUco board once through the 2D output and calibrate it directly (~1 minute of recording).

**ASSUMPTION — calibration transfers across sessions.** Calibration comes from one session
(`18de9c5a`). The `1× 0°` readout matches everywhere checked, and the baseline is fixed
hardware, so the focal is the main risk. The `5e27066c` transfer check (§6) is reassuring but
is not proof.

**ASSUMPTION — the board was printed at exactly 100%.** All metric scale traces back to the
8.00 mm square pitch. The PDF carries a "measure me: must be exactly 100.0 mm" bar. If the
print was scaled, every depth scales with it. Worth verifying with a steel ruler if the printed
board still exists.

**Rectification crops FOV.** `stereoRectify(alpha=0)` raises the effective focal from 1061 to
1222 px, i.e. it zooms in to discard invalid border — losing precisely the periphery that was
least trustworthy. `alpha` is a knob if that FOV matters.

**Annotation coordinates move.** `scale_objects.json` points are in unrectified 1340×1072
pixels. Any rectification or undistortion of the images must transform those points too
(`cv2.undistortPoints` with the same R/P), or the metric scale loss silently samples the wrong
places.

---

## 9. Method notes for writing up

- The SBS layout, the 2.125 scale, and the banner-as-ruler trick are all **measurements**, with
  the `fx/fy = 1.0003` result as the independent cross-check.
- The calibration validation (§6) is a genuine held-out-style check: PnP and disparity share no
  parameters except the board geometry.
- The 9%-bias-from-skipping-undistortion result (§5) is worth reporting — it quantifies an
  assumption that endoscopic depth papers commonly leave implicit.
