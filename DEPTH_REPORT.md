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

## Method
- **Train res:** `--image-shape 392 490` (multiples of 14). Decoupled from the GUI's
  `IMAGE_SHAPE` (pos-embed interpolation handles a different inference res). Recorded here per spec.
- **Loss:** single-scale Monodepth2 — 0.85·SSIM + 0.15·L1 reprojection over frames {−1,+1},
  auto-masking (identity-reprojection baseline), edge-aware disparity smoothness (1e-3). Reuses
  EndoDAC's vendored `utils/layers.py` primitives.
- **Trainable:** depth_model LoRA adapters + encoder residual blocks + depth conv heads
  (`lora_*`, `residual_*`, `conv_depth_*` — EndoDAC's own recipe) + a mono pose net
  (ResNet18 ×2-frame encoder + PoseDecoder), all warm-started from the released weights.
- **Intrinsics:** EndoDAC/SCARED assumed normalised K (0.82, 1.02, 0.5, 0.5). Override real
  da Vinci values with `--intrinsics fx fy cx cy` (normalised).
- Dropped EndoDAC's full AF-SfMLearner registration/transform refinement (ponytail: core
  photometric only; re-add if specular/non-rigid artifacts appear in the panels).

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

## Verification (run before declaring done)
- `python scripts/finetune_depth.py --self-test` → **389/389 keys, 0 missing** (fresh build *and*
  vs the released `depth_model.pth`).
- `--smoke` → full depth+pose+reprojection+automask+smoothness+backward path runs (loss≈0.41).
- Data layer: 533 Validation triplets, color (3,392,490) ∈ [0,1], valid-frac ≈ 0.61.

## GUI swap (one line)
Copy the checkpoint over ATLAS's init and the GUI picks it up:
```
cp outputs/rarp_depth/best.pth  <...>/ATLAS-Interactive/../backbones/EndoDAC/depth_model.pth
```
(or point `CHECKPOINT` in `gui/depth_estimator.py` at `best.pth`). Architecture/keys are identical,
so no GUI code changes are needed. `IMAGE_SHAPE` there stays independent of the train res.
