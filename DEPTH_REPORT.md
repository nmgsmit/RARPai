# RARP depth fine-tuning — report

Self-supervised monocular depth fine-tune of **EndoDAC** on our RARP videos, producing a
checkpoint that drops straight into the ATLAS-Interactive GUI.

## What & where
- Train/infer code: [scripts/finetune_depth.py](scripts/finetune_depth.py) · job: [jobs/finetune_depth.sh](jobs/finetune_depth.sh) (gpu_h100)
- Vendored EndoDAC code: `third_party/endodac/{models,utils}`
- Warm-start init (Snellius, out of git): `~/backbones/EndoDAC/{depth_model.pth, pose.pth, pose_encoder.pth}`
- Deliverable: `outputs/rarp_depth/best.pth` — plain depth_model `state_dict` (encoder.* + depth_head.*, **389 tensors**)

## Data
RARPAtlas is **monocular** 1080p YouTube clips (no stereo), `<split>/rarp/<video>/clip_*/images/frame_*.jpg`,
consecutive frames per clip. We build (t−s, t, t+s) triplets with real camera motion.

| split | videos | clips | frames | triplets (stride 1) |
|-------|-------:|------:|-------:|--------------------:|
| Train | 7 | 32 | 5300 | ~5.2k |
| Validation | 1 | 6 | 545 | 533 |
| Test | 2 | 13 | 2668 | ~2.6k |

**Console-GUI overlay masking.** The videos are heterogeneous console captures whose baked-in
overlays are NOT anatomy and corrupt depth at the borders: full-bleed (Hugo), black L/R pillarbox +
da Vinci Xi instrument banner (~70px), circular vignette, and CMR Versius corner logos/icons composited
over full-bleed anatomy (no black bars). A single trick handles all of them: the overlay is **static
across a clip** while anatomy moves, so a per-clip **temporal-variance mask** (valid = per-pixel std over
~16 frames > `--overlay-std-thresh`) isolates anatomy. Validated on CMR-corner and Xi-banner clips
(valid_frac ~0.5–0.78). Clips too static to trust (valid_frac<0.25) fall back to no mask. The mask gates
the photometric loss, a masked edge-aware smoothness (no depth bleed across the banner edge), and the
viz normalization. The mean-RGB dark mask still removes the black vignette/bars; `--bottom-crop-frac`
remains available. Disable with `--no-overlay-mask`.

> GUI note: the model output in overlay regions is unsupervised, so `gui/depth_estimator.py` should also
> mask those pixels for display — its dark-mask catches black bars/banner but not bright CMR corner widgets.

**Is the "overlay corrupts the border" claim correct? — measured.** Partly. On the **released (un-finetuned)
EndoDAC**, overlay regions read as **near**: overlay disparity 95th-pct (0.61–0.67) *exceeds* the anatomy
range (0.60–0.64) and 8–12% of overlay pixels sit above the anatomy 95th-pct, so they **do inflate the
colormap** → the claim holds for the warm-start model and motivates handling the overlay. On our
**RARP-fine-tuned** model the effect is **gone**: with the overlay excluded from the loss the model drives
those regions to **far** (overlay disp ≈0.001–0.016 ≪ anatomy 0.12–0.20; 0% above anatomy 95th-pct), so
full-frame and anatomy-only colormap normalization are nearly identical. Conclusion: the dominant
border-corruption was a **warm-start / colormap-normalization** effect that fine-tuning (with the overlay
removed from supervision, by either mask or crop) already resolves; auto-/occlusion-masking neutralises the
*static* overlay in the photometric term, and the residual benefit of masking is avoiding smoothness bleed
across the overlay edge and not wasting capacity on non-anatomy.

## Methods and training procedure (overview — for the IEEE write-up)

Relevant prior work to cite: EndoDAC (Cui et al., 2024), AF-SfMLearner (Shao et al., 2022),
Monodepth2 (Godard et al., 2019), Depth Anything / DPT head (Ranftl et al., 2021),
DINOv2 (Oquab et al., 2023), LoRA (Hu et al., 2021).

### Network architecture
**Depth network (the deliverable).** EndoDAC = a **DINOv2 ViT-B/14** encoder + a **DPT** dense head,
adapted for endoscopy with **DV-LoRA** low-rank adapters (rank r=4) on every transformer block's MLP
(fc1/fc2), **residual conv blocks** injected at encoder blocks {2,5,8,11}, CLS token included; the DPT
head fuses 4 feature levels (RefineNet) and emits sigmoid **disparity** at 4 scales. 389 parameter
tensors (`encoder.*` + `depth_head.*`).
**Trainable vs frozen.** Only the **LoRA adapters, injected residual blocks, and depth output convs**
(`lora_* / residual_* / conv_depth_*`, ≈14.7 M params) are trained; the DINOv2 backbone and DPT fusion
trunk stay frozen, preserving the foundation-model prior while adapting cheaply.
**Auxiliary self-supervision nets (training only, discarded after).** AF-SfMLearner stack — **Pose**
(ResNet-18 over a 2-frame stack → 6-DoF relative motion), **Position** (ResNet-18 + U-Net → dense optical
flow registration + forward–backward occlusion mask), **Transform / appearance-flow** (ResNet-18 + U-Net →
3-channel residual giving an illumination-corrected target, robust to specular/non-Lambertian changes).

### Initialization
All networks warm-started from EndoDAC's released weights (depth 389/389 keys, 0 missing; pose/position/
transform encoders+decoders likewise). Fixed normalized intrinsics K (fx=0.82, fy=1.02, cx=cy=0.5 of image
size; EndoDAC/SCARED assumption; replaceable with measured da Vinci values). Depth is up-to-scale.

### Losses
Photometric similarity `pe(a,b) = 0.85·(1−SSIM)/2 + 0.15·‖a−b‖₁`. View synthesis warps each source frame
by back-projecting depth with K⁻¹, applying relative pose T, projecting with K, and bilinear sampling.
- **Plain path (`--no-refine`):** min-over-sources photometric + Monodepth2 **auto-masking** (identity
  baseline) + edge-aware disparity smoothness.
- **Full AF-SfMLearner path (`--refine`, default), two alternating stages per batch:**
  - *Stage 0 — Position net only:* registration photometric (occlusion-masked) + flow smoothness (1e-3).
  - *Stage 1 — depth + Pose + Transform:* photometric (warped source vs **refined** target) +
    transform-constraint·|refined−registration| (0.01) + appearance-smoothness (0.01) + disparity
    smoothness (1e-3). Occlusion mask replaces auto-masking; smoothness is masked to valid pixels.

### Data & preprocessing
Monocular 1080p RARPAtlas; Train 7 videos / 32 clips / 5300 frames, Val 1, Test 2. Training samples are
within-clip **triplets (t−s, t, t+s)**; stride s ablated. **Resolution decoupling:** ViT internal res is a
multiple of 14 (default 392×490, recorded for the GUI); pose/registration/reprojection geometry runs at a
**÷32 feed res** (384×480) for U-Net skip alignment — the depth net resizes its input internally and
disparity is up-sampled back to feed res. **Console-overlay handling** (ablated): (i) per-clip
**temporal-variance mask** (static overlay vs moving anatomy, std < ~6/255 → overlay); (ii) **hard 10%
L/R/bottom crop** → anatomy-only. A vignette dark-mask always applies; identical color-jitter per triplet.

### Optimization
**fp32** (AMP caused pose-net NaN divergence), Adam lr=1e-4 (+ separate Adam lr=1e-4 for the Position
stage), **grad-norm clip 1.0**, StepLR (×0.1 at half), non-finite-batch guard, 20 epochs, batch 8.
1× NVIDIA H100, ≈46 min (no-refine) / ≈110 min (refine).

## Validation & logging (no depth GT, no stereo)
wandb project `rarp` / entity `nmgtue`. Logged per epoch: photometric/SSIM/smoothness, LR,
pose rotation+translation stats, and a **fixed 8-frame Validation panel** rendered as
`[rgb | magma-depth]` (plus a before-training panel and a final Test panel). Model selection =
lowest **Validation photometric** proxy. SCARED forgetting-check skipped (SCARED not staged on
Snellius). Stereo metric: N/A (data is monocular).

## Results (run 24270397 — final, overlay-masked, 20 epochs, fp32, 1×H100)
- wandb run: https://wandb.ai/ngmtue/rarp/runs/wm0jcm1b (name `endodac-rarp`)
- **Best Validation photometric: 0.0615** → `outputs/rarp_depth/best.pth`
- **Test photometric: 0.0181** (smoothness 0.0148)
- pose_trans 0.0010→0.0028 finite throughout; no NaN, no skipped batches.
- Before/after depth panels: `qual/panel` (epoch 0 warm-start vs later) + `qual/test_panel` in the run.
  Depth maps now mask the console-GUI borders (black bars / banner / corner widgets) instead of reading
  them as "near", so anatomy gets the full colour range.
- best.pth verified GUI-compatible: `--self-test --ckpt outputs/rarp_depth/best.pth` → **389/389 keys, 0 missing**.

Note: the photometric proxy is NOT comparable across the overlay change — it now averages over anatomy-only
pixels. The pre-overlay run (24269078) read 0.0605 val / 0.0173 test over the full frame incl. overlay.

### Two issues found & fixed along the way
1. **NaN divergence (run 24268262):** fp16 AMP sent `pose_trans=nan` in epoch 2, the auto-mask collapsed to
   the static-identity baseline and metrics froze. Fixed: train fp32 (EndoDAC's regime) + grad-norm clip 1.0
   + non-finite-batch guard.
2. **Console-GUI overlay corrupting borders:** da Vinci/CMR overlays leaked into the loss/output. Fixed with
   the per-clip temporal-static mask (see Method). The ATLAS GUI loader got the same mask as an online
   rolling-buffer version (`gui/depth_estimator.py`, `MASK_OVERLAY`), so display is clean on all console types.

## Ablations
Selection is by qualitative depth quality (panels) + pose conditioning, **not** the photometric proxy —
a larger inter-frame baseline mechanically raises the residual, so the loss is not comparable across strides.

**Frame stride (with AF-SfMLearner refinement, overlay temporal-mask on):**

| stride | best val_photo | test photo | test smooth | verdict |
|---:|---:|---:|---:|---|
| **1** | 0.0766 | 0.0262 | 0.0363 | **selected** — sharpest, most coherent depth |
| 2 | 0.1023 | 0.0366 | 0.0299 | close, slightly blurrier |
| 3 | 0.1192 | 0.0445 | 0.0271 | noticeably noisier (occlusion/appearance change, no conditioning gain) |

Consecutive surgical frames already carry enough motion (pose_trans ≈ equal across strides), so larger
strides only add occlusion/appearance change. Runs `depth_s{1,2,3}`.

**Overlay handling — temporal-variance mask vs hard 10% L/R/bottom crop** (stride 1, refinement on):

| overlay handling | val_photo | test photo | test smooth | depth quality |
|---|---:|---:|---:|---|
| temporal-variance mask (`depth_s1`) | 0.0766 | 0.0262 | 0.0363 | coherent, but speckle on static central tissue |
| **hard 10% L/R/B crop** (`depth_crop`) | 0.0760 | 0.0260 | **0.0251** | **cleaner — no speckle, smoother, anatomy-only** |

Photometric is tied; the crop gives lower smoothness and **visibly cleaner depth maps** (no mask speckle,
no unsupervised overlay regions) — so the **hard crop is preferred** for simplicity and output cleanliness.
Caveats: 10% L/R/bottom clears the da Vinci Xi banner/bars but leaves CMR Versius **top-corner** logos
(add a top crop for those) and trims some peripheral tissue on full-bleed frames. **If trained with a crop,
feed the GUI the same crop at inference.**

## Verification (run before declaring done)
- `python scripts/finetune_depth.py --self-test` → **389/389 keys, 0 missing** (fresh build *and*
  vs the released `depth_model.pth`).
- `--smoke` → full depth+pose+reprojection+automask+smoothness+backward path runs (loss≈0.41).
- Data layer: 533 Validation triplets, color (3,392,490) ∈ [0,1], valid-frac ≈ 0.61.

## GUI swap (one line)
Copy the checkpoint to the path the GUI's `CHECKPOINT` points at:
```
cp outputs/<run>/best.pth  <...>/backbones/EndoDAC/depth_model_rarp_finetune.pth
```
Architecture/keys are identical, so no GUI code changes are needed. `IMAGE_SHAPE` there stays independent
of the train res. If you trained with a crop, apply the matching crop in `gui/depth_estimator.py`.
