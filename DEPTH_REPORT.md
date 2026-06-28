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

Bottom UI-banner crop is exposed (`--bottom-crop-frac`, default 0); the circular vignette /
black border is handled by a per-frame valid-mask (mean-RGB threshold) that excludes those
pixels from the photometric loss — same idea as the GUI's viz normalization.

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

## Results (run 24269078, 20 epochs, ~46 min on 1×H100, fp32)
- wandb run: https://wandb.ai/ngmtue/rarp/runs/fb33zvkr (name `endodac-rarp`)
- **Best Validation photometric: 0.0605 @ epoch 12** → `outputs/rarp_depth/best.pth`
- **Test photometric: 0.0173** (smoothness 0.0160)
- Trajectory: train_photo 0.0199→0.0134 (monotonic), val_photo 0.0659→0.0605 (plateaus after the
  epoch-10 LR×0.1 step), pose_trans 0.0011→0.0028 (finite throughout — pose net learns real motion).
- Before vs after depth panels: `qual/panel` epoch 0 (warm-start) vs later epochs, and `qual/test_panel`
  in the wandb run above.
- best.pth verified GUI-compatible: `--self-test --ckpt outputs/rarp_depth/best.pth` → **389/389 keys, 0 missing**.

### Note on the first attempt (run 24268262)
The initial run used fp16 AMP and went `pose_trans=nan` in epoch 2 — the auto-mask then collapsed to the
static-identity baseline and metrics froze. Fixed by training in fp32 (EndoDAC's own regime) + grad-norm
clip 1.0 + a non-finite-batch guard. The numbers above are the fixed run.

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
