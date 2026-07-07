# Claude's design log

Append-only. Newest on top. Record design choices made and where things were put, so future
sessions don't re-derive them. Keep entries one or two lines.

## 2026-07-07 — DEPTH: SCARED staged + sweep flips the stride verdict + wired into training
- SCARED *test* release (ds8/9) uploaded to `../data/SCARED/test_dataset_{8,9}.zip` = 10 keyframes,
  each only `{Left,Right}_Image.png` + `rgb.mp4` + `endoscope_calibration.yaml`. **No structured-light
  GT** (withheld from the challenge test set). So `scripts/export_scared_stereo_gt.py` builds METRIC
  GT from the stereo pair: cv2 stereoRectify → SGBM → reprojectImageTo3D, Z=depth(mm). Output
  `../data/SCARED/stereo_gt/{frames/, gt_depths.npz}`. Depth medians 30–116 mm (sane), valid 69–84%.
  Baseline |T|≈4.35 mm, fx≈1024 px. It's calibrated-stereo pseudo-GT (Hamlyn-style), not struct-light.
- SWEEP (`jobs/eval_scared_sweep.sh`, all depth ckpts, same GT), abs_rel / a1 zero-shot N=10:
  depth_s3 0.230/0.668 · depth_s1 0.254/0.622 · depth_s2 0.254/0.653 · depth_crop 0.314/0.573 ·
  rarp_depth 0.349/0.453. **The stride ranking FLIPS vs the RARP photometric proxy**: proxy said
  s1 best / s3 worst; SCARED says s3 BEST (monotonic s1→s3 on abs_rel, a1, rmse_log). Confirms the
  report's warning that photometric ≠ comparable across strides (bigger baseline mechanically raises
  the residual). ⇒ don't select depth models by the photometric proxy across strides; s3 is best,
  worth an s4/s5. depth_crop's RARP "preferred" status does NOT hold on metric GT.
- WIRED: `finetune_depth.py` now runs `eval_scared.run_scared_eval` on the selected best.pth at end of
  training (default on, `--no-scared` off, `--scared-dir`), logs `scared/*` to the run, same crop as
  training. `eval_scared.py` refactored: `_eval_pairs` + `run_scared_eval` + `--side/--bottom-crop-frac`.

## 2026-07-06 — DEPTH: SCARED benchmark eval (absolute GT metrics)
- `scripts/eval_scared.py` + `jobs/eval_scared.sh`: evaluate a trained `best.pth` on SCARED, the
  structured-light GT benchmark used by AF-SfM/EndoDAC → numbers comparable to those papers. Reuses
  finetune_depth's `build_depth_model`/`_filter_load`/`round14`/`colorize`/`disp_to_depth` + vendored
  `compute_errors` (no reimpl). Self-supervised depth is scale-ambiguous → per-frame MEDIAN SCALING,
  then 7 Monodepth2 metrics (abs_rel..a3), clamp band 1e-3..150 mm. wandb: metrics to summary+log,
  scale-ratio median/std, `[rgb|pred|gt]` overlay images. `--smoke` = numpy metric self-check (no GPU).
- GT input: `--gt-npz` = (N,H,W) mm depth maps (key "data"), produced by AF-SfM `export_gt_depth.py`
  from raw SCARED point-cloud tiffs. Frames via `--rgb-dir` (sorted ↔ npz order; script prints
  first/last pairing to eyeball) or `--list` file. NOT reimplementing the exporter (needs raw layout).
- SCARED is license-gated (Intuitive EULA, EndoVis'19 sub-challenge) — cannot be wget'd. Nick must
  sign the agreement, download dataset_8/9 test keyframes, run the AF-SfM exporter, then drop
  `test_left/` + `gt_depths.npz` under `../data/SCARED/` on Snellius. Then `sbatch jobs/eval_scared.sh`.

## 2026-06-29 — DEPTH: AF-SfMLearner refinement re-added + frame-stride ablation
- `--refine` (default ON) restores EndoDAC's full self-supervision in `scripts/finetune_depth.py`:
  Position net (ResnetEncoder n_in=2 + PositionDecoder, dense optical-flow registration + occlusion
  mask via `get_occu_mask_backward`) and Transform net (ResnetEncoder + TransformDecoder, appearance
  flow → illumination-corrected "refined" target). Two-stage per batch like the released trainer:
  opt0 trains Position (lr 1e-4), opt trains depth-LoRA + pose + Transform (lr `--lr`). Losses:
  registration (stage0) + smoothness; photometric vs refined + transform-constraint(0.01) +
  transform-smoothness(0.01, `get_smooth_bright`) + masked disp-smoothness (stage1). Pose uses
  EndoDAC's [f,0] pairing (no invert). `--no-refine` = the plain Monodepth2 path. Overlay/vignette
  `valid` mask gates every term. Warm-start needs position*/transform*.pth in `~/backbones/EndoDAC`.
- Weights uploaded to `~/backbones/EndoDAC`: + position.pth, position_encoder.pth, transform.pth,
  transform_encoder.pth (intrinsics_head.pth NOT needed — we use fixed K, no learned intrinsics).
- FRAME-STRIDE ablation: triplet baseline = `--frame-stride`. Run s∈{1,2,3} with refine, separate
  `--out`/`--run-name`, compare Validation photometric + qual panels. Commands in COMMANDS.md.
- Data for finetuning = Train split = 7 videos / 32 clips / 5300 frames. Heterogeneous console types
  (Hugo, da Vinci Xi, CMR Versius) → diversity matters; more videos likely helps generalization more
  than more frames/video (consecutive frames redundant). A video-count ablation is the way to confirm.
- GUI: Nick renamed the loader CHECKPOINT to `depth_model_rarp_finetune.pth`, so the swap is now
  `cp outputs/<run>/best.pth ../backbones/EndoDAC/depth_model_rarp_finetune.pth`.

## 2026-06-28 — DEPTH: console-GUI overlay masking + NaN fix (run 24269078 → next run)
- First real run diverged: fp16 AMP sent pose_trans→nan in epoch 2, automask collapsed to the static
  baseline, metrics froze. FIX: train fp32 (no AMP, EndoDAC's regime) + grad-norm clip 1.0 (`--grad-clip`)
  + skip non-finite batches. Stable run 24269078: best val_photo 0.0605 @ep12, test 0.0173.
- RARPAtlas videos are HETEROGENEOUS console captures: full-bleed (Hugo), black L/R pillarbox + bottom
  instrument banner (da Vinci Xi: "PROGRASP FORCEPS"…, ~70px), circular vignette, static corner logos/icons
  (CMR Versius — NO black bars). Nick flagged these non-anatomical overlays corrupting the depth.
- FIX: per-clip **temporal static-overlay mask** in `RARPTriplets` — overlay is baked-in & identical across
  frames (low temporal std) while anatomy moves; valid = std>thresh. One mask/clip handles ALL overlay types
  incl. bright widgets with no black bars. Validated on CMR-corner + Xi-banner clips (valid_frac ~0.69–0.78).
  Flags: `--no-overlay-mask`, `--overlay-frames` (16), `--overlay-std-thresh` (6.0, on 0–255). Static clips
  (valid_frac<0.25) fall back to no mask. Mask now gates BOTH photometric and (new) `smooth_loss_masked`, and
  feeds the qual-panel viz normalization so depth maps ignore the banner/logos.
- NOTE for GUI: model output in overlay regions is unsupervised → the GUI (`depth_estimator.py`) should also
  mask those pixels for display; its dark-mask won't catch bright CMR corner widgets.

## 2026-06-28 — DEPTH workstream (EndoDAC fine-tune, separate from segmentation)
- NEW: `scripts/finetune_depth.py` + `jobs/finetune_depth.sh` (gpu_h100). Self-supervised monocular
  depth fine-tune of EndoDAC on RARPAtlas. **Not segmentation**; does NOT use the RARP DINO teacher.
- Vendored EndoDAC code at `third_party/endodac/{models,utils}` (copied from `../backbones/EndoDAC/EndoDAC`,
  pycache stripped, ~200K, no weights). Script does `sys.path.insert(0, third_party/endodac)` then
  `import models.endodac / models.encoders / models.decoders` + `utils.layers` (namespace pkg, mirrors GUI).
- Warm-start weights live on Snellius `~/backbones/EndoDAC/` (depth_model.pth 396MB, pose.pth, pose_encoder.pth),
  uploaded by scp; kept OUT of git. Init = released `depth_model.pth`.
- GUI contract (NON-NEGOTIABLE): deliverable `outputs/rarp_depth/best.pth` is a plain depth_model state_dict
  (encoder.* + depth_head.*, **389 tensor keys**) loading into `ATLAS-Interactive/gui/depth_estimator.py`
  which builds `endodac.endodac(backbone_size="base", r=4, lora_type="dvlora", image_shape=IMAGE_SHAPE,
  pretrained_path=None, residual_block_indexes=[2,5,8,11], include_cls_token=True)`. We save raw
  `depth_model.state_dict()` (no height/width extras). `--self-test` rebuilds with those args + asserts 389/389.
- DATA: RARPAtlas is **monocular** 1080p YouTube clips (NOT stereo) at `<split>/rarp/<video>/clip_*/images/frame_*.jpg`,
  consecutive frames per clip. Counts: Train 7vid/32clip/5300img, Val 1/6/545, Test 2/13/2668. Dataset builds
  (t-stride,t,t+stride) triplets within a clip. masks/machine_masks exist but unused for depth.
- LOSS: single-scale Monodepth2 — 0.85·SSIM+0.15·L1 reprojection, auto-masking, edge-aware disp smoothness;
  reuses EndoDAC `utils/layers.py` primitives. **Dropped** EndoDAC's full AF-SfMLearner registration/transform/
  optical-flow refinement (ponytail: core photometric only; re-add if specular/non-rigid artifacts appear).
- Trainable = depth_model {lora_*, residual_*, conv_depth_*} (EndoDAC's own recipe) + mono pose net
  (ResnetEncoder n_in=2 + PoseDecoder), warm-started from pose_encoder.pth/pose.pth.
- TRAIN RES decoupled from GUI: `--image-shape H W` (mult of 14, default 392 490), recorded for the report.
  GUI IMAGE_SHAPE interpolates pos-embeds independently. Reprojection runs at the same H,W (single scale).
- INTRINSICS: `--intrinsics fx fy cx cy` NORMALISED; default = EndoDAC/SCARED assumed K (0.82,1.02,0.5,0.5).
  No real da Vinci values provided yet — swap them in via the flag if Nick supplies them.
- VALIDATION/LOGGING: wandb project `rarp` entity `nmgtue`. Logs photo/SSIM/smoothness, LR, pose rot/trans
  stats, and a FIXED 8-frame Validation panel as [rgb | magma-depth] each epoch (+ before-training panel +
  Test panel). Model selection = lowest Validation photometric proxy. No depth GT; no stereo pseudo-GT; SCARED
  forgetting-check skipped (SCARED not staged on Snellius) — note in report.
- GUI swap one-liner: copy `outputs/rarp_depth/best.pth` -> ATLAS `../backbones/EndoDAC/depth_model.pth`
  (or point `CHECKPOINT` in `gui/depth_estimator.py` at it).

## 2026-06-24
- SSH WORKING: alias `snellius` (user nsmit2, key `id_ed25519_snellius`, passphrase-less). Dropped ControlMaster — flaky over Git Bash sockets and unneeded with no passphrase. I run `ssh snellius "cd ~/RARPai && ..."` directly. Repo on Snellius = `~/RARPai`; `../data`=`~/data`, `../backbones`=`~/backbones`. Login prints a harmless post-quantum warning — filter with `grep -v`. Local-machine config, not in repo.
- Created `CLAUDE.md` (ops, auto-loaded), `COMMANDS.md` (Nick's cheatsheet), this log.
- Confirmed conventions from the repo: `venv` (not conda); encoder = `../backbones/RARP_checkpoint_epoch0050_teacher.pth`; trained ckpts -> `outputs/<run>/best.pth`; data = `../data/RARPSurgenet/fold1`.
- GPU training partition is `gpu_h100 --gpus-per-node=1`; `genoa` is CPU-only (overlays/viz).
- RESOLVED: encoder lives in `../backbones` (Nick confirmed `../checkpoints` was a slip); trained ckpts -> `outputs/`.
- Jobs pass `"$@"` through to the python script, so hyperparams are overridable at submit time.
