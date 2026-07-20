# Claude's design log

Append-only. Newest on top. Record design choices made and where things were put, so future
sessions don't re-derive them. Keep entries one or two lines.

## 2026-07-18 — GUI template matching: `data/templates/` for overlay detection
- `gui_mask.py` loads GUI templates via `load_templates(template_dir)` → all `.png` files in a dir,
  sorted by filename. Put your cropped template images in `data/templates/` (e.g., instrument icons,
  GUI elements from `gui_lines.ipynb`). Usage: `mask = gui_mask(frame, templates=load_templates("data/templates"))`.
  Fixed geometry (hardcoded da Vinci Xi bottom bar + tab) runs first; template matching in a band
  above it catches transient widgets (thresholdable via `thresh`, `search_band`). `roi_bottom=N`
  crops to the bottom N px before matching (~1.2x faster).
- TRIED AND DROPPED: colour-based detection of the #F5C83A hazard bar (gold/black striped L that
  hugs the LEFT/RIGHT edge of the CONTENT region, not the frame — 1920x1080 frames are pillarboxed
  to ~1344 wide). Colour alone is useless: fatty tissue is the same gold (142k px/frame). Adding a
  content-edge restriction + a "contains black stripes" test did separate it (bars 0.10-0.15 dark
  frac vs tissue <=0.044 at dark_level 45), but a bbox-fill test proved crop-dependent and the whole
  thing stayed heuristic. Reverted to plain template matching per Nick. If revisited: recognise the
  "1"/"2" endpoint glyphs (fixed ~15x20 cream-on-dark labels, matched at 0.92-1.00, and always at
  the content edge x=308) and synthesise the L between them — far more robust than colour.
- CAUTION: the GUI_segmentation overlay run OVERWROTE the source frames in place (green on GUI), so
  those originals are gone. `SUL_cut` / `Stump_ruler` are still clean. Write overlays to a separate
  output dir in future.

## 2026-07-08 — DEPTH: diagnosed why finetuning DEGRADES (static scope) + motion filter & teacher anchor
- Nick's qualitative check: epoch 0 best on UMC too; trained models show anatomy structure
  crisply but WRONG depth (robot arm depth lost, prostate/urethra "come closer") = texture-copy +
  geometry collapse. Mechanism: UMC scope is mostly STATIC (pose_trans ~0.001) -> no parallax ->
  photometric loss carries no depth signal; automask then keeps only the MOVING-TOOL pixels in the
  loss (rigid warp can't explain them) -> gradient actively corrupts tool depth; edge-aware
  smoothness paints image edges into disp ("structures visible" = bad sign, not progress).
- Fixes (both flags in finetune_depth.py): `--motion-top-frac F` keeps top-F of TRAIN triplets by
  whole-frame motion (median |frame diff| on cropped grayscale thumbs — median robust to tools,
  high only for real camera motion; Monodepth2 static-frame filtering, adapted). `--anchor-w W`
  adds L1 log-disp anchor to a FROZEN deepcopy of the warm-start model (preserves pretrained
  geometry while photometric adapts appearance). Val loss includes the anchor term when on.
- Final best.pth re-score now logs under `scared_best/*` (was appending to `scared/*`, faking a
  late "recovery" at step epochs+1 — the "epoch 11/12 improves" artifact Nick spotted).
- Runs: `endodac-umc-botcrop-motion` (motion 0.5, lr 1e-4 = isolates data fix vs completed botcrop)
  and `endodac-umc-botcrop-anchor` (motion 0.5 + anchor 0.3). Both botcrop geometry, 8 ep, sf 0.1.

## 2026-07-08 — DEPTH: --drop-gui screens transient GUI/overlay frames out of training
- `--drop-gui` (+`--gui-z`, default 5): before training, one warm-start no-grad pass computes each
  sample's per-sample photometric error (`scan_gui_frames`). A GUI overlay isn't in the 3D scene →
  can't be reprojected from neighbours → large robust outlier. Flags samples > median+z·1.4826·MAD
  (`_flag_outliers`), writes `outputs/<run>/suspected_gui.csv` (path,score,thr), logs a top-8 montage
  + counts to wandb, and trains on a `Subset` excluding them. Triplet centers are interior to a clip
  so video cuts don't false-trigger. Gated (default off) → the 3 running crop runs are unaffected.
  Dataset now also returns `out["path"]`.
- FINAL crop/GUI experiment = **3 runs, all bars-off (side 0.15) + correct K + overlay-mask ON**
  (the standalone full-frame-with-bars baseline was dropped as redundant — bars off is the new
  normal): **A** `umc_s1_cropLR` = bars off only (BASELINE); **B** `umc_s1_cropanat` = bars off +
  anatomy crop (spatial GUI removal); **dropgui** `umc_s1_dropgui` = A + `--drop-gui` (frame-level
  GUI removal). B-vs-A isolates cropping, dropgui-vs-A isolates frame-dropping.
- 4th run `umc_s1_dropgui_botcrop` (`--side 0.1712 --bottom 0.0648 --drop-gui`): crops the
  ALWAYS-present bottom-70px GUI banner AND frame-drops novel transient GUIs (belt+suspenders).
  Bottom-70 makes the region wider than 4:5, so sides trim a touch more to restore exact 4:5
  (1263x1010) → isotropic resize to the 4:5 feed, no anatomy squash. Auto fed-K (0.873,1.091,0.5,
  0.535); cy shifts up from the bottom crop. (4:5 crop isn't strictly required — per-axis K also
  corrects an anisotropic resize — but keeping ~4:5 avoids feeding squashed anatomy.)

## 2026-07-08 — DEPTH: SCARED-metric checkpoint selection + top-crop + crop A/B experiment
- `finetune_depth.py` now selects `best.pth` by per-epoch SCARED `abs_rel` (real GT), not the
  `val_photo` proxy (falls back to val_photo when no SCARED GT). Photometric proxy rewards
  texture-copying and can improve while geometry degrades.
- Added `--top-crop-frac` (threaded through `eval_scared._crop`/`_eval_pairs`/`run_scared_eval`
  too, so train & SCARED eval crop identically). UMCdissection frames are 1920x1080, pillarboxed
  with ~288px BLACK bars L/R (top/bottom ~3px). Content ≈1344x1080 ≈ 4:5 already.
- Crop A/B experiment (both stride1, epochs10, sample-frac0.1, refine on) vs full-image baseline
  `umc_s1_scaredsel`: A = `--side-crop-frac 0.15` (black bars off → ~4:5). B = anatomy crop
  `--side 0.24 --top 0.037 --bottom 0.222` → 999x801 (~4:5, 40px top / 240px bottom kills GUI
  banner). Tests whether crop-to-anatomy beats full frame + temporal overlay mask.
- `crop_adjust_intrinsics()`: crops AUTO-adjust K (fx/=kept-width, fy/=kept-height, principal
  point re-referenced). A cy-only fix is incoherent — the same transform rescales fx/fy (crop
  narrows FOV). Only affects the training reprojection, not SCARED eval (median-scaled, K-free).
- **CORRECTION**: base K (0.82,1.02,0.5,0.5) is SCARED = a **4:5** calib (square px on 1280x1024),
  i.e. the UMC content WITHOUT the pillarbox bars — NOT the full 1920x1080. Added `--black-bar-frac`
  (UMC=0.15): anchors the base K to the 4:5 content inside the raw frame, so K is correct whether or
  not a run crops the bars. With bar_side=0.15 → Baseline (full 16:9) fed-K fx **0.574** (prior
  full-frame UMC runs incl umc-s1-short used 0.82 = a real bug), A (bars off) recovers **0.82**,
  B fx **1.10** fy **1.38** cy **0.625**. All 3 crop-experiment runs now pass `--black-bar-frac 0.15`.

## 2026-07-07 — DEPTH: switched to UMCdissectionimg + stride 1 (wide strides degenerate here)
- Depth training data switched RARPAtlas -> `../data/UMCdissectionvid/UMCdissectionimg` (real target
  domain, 71 videos/46k frames; Nick moved HD snapshots into the vid folder, HD->img). `--sample-frac`
  subsamples redundant triplet centers (5-10 fps) to keep runs tractable; `jobs/finetune_depth_umc.sh`.
- Credit-friendly SHORT stride screen (`--epochs 10 --sample-frac 0.1`, chained 1-GPU, `jobs/sweep_umc_stride.sh`):
  s1 SCARED abs_rel **0.228** (a1 .547, rmse 17.8) — already ≈ best RARPAtlas run. s3 **DEGENERATED**
  (pose_trans collapsed 0.0004->0.0002 instead of growing ~3x, 9 non-finite batches skipped, val_photo
  spuriously low = degenerate shortcut). Cancelled s3+s5. ⇒ the RARPAtlas "bigger stride better" trend
  does NOT transfer to UMC (5-10 fps -> wide strides span too much/erratic motion). **Use stride 1 on UMC**;
  exploiting bigger baselines would need a stability fix (lower lr / warm-up / stronger pose init), not just stride.
- SCARED metrics were going only to `wandb.run.summary` in finetune (overlay via `wandb.log`), so runs
  showed the image but no metric scalar — fixed to `wandb.log` the metrics too; backfilled umc-s1-short via API.
- REFINE ablation (stride 1, short, identical seed/data): EndoDAC+AF-SfMLearner (`umc-s1-short`) BEATS plain
  EndoDAC (`umc-s1-norefine`) on all 7 SCARED metrics — abs_rel 0.228 vs 0.283 (-19%), rmse 17.8 vs 23.1,
  a1 .547 vs .496. ⇒ keep `--refine` on UMC (earns its ~2x compute). Caveat: norefine has LOWER scale-ratio
  std (43 vs 107) — more globally-consistent scale — but per-frame median scaling hides that and refine still
  wins accuracy; relevant later for the fixed-scale metric (SUL) step, not for SCARED ranking.

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

## 2026-07-10 — Learned camera intrinsics (EndoDAC IntrinsicsHead)
- Until now K was FIXED for a whole run: built once in `RARPTriplets.__init__` from `--intrinsics`
  (after `crop_adjust_intrinsics`) and handed back unchanged with every batch. Never a parameter.
- EndoDAC's `IntrinsicsHead` was vendored (`third_party/endodac/models/decoders/intrinsics_decoder.py`)
  but never imported. It's now wired in and **on by default**; `--no-learn-intrinsics` restores the
  old fixed-K behaviour. K is predicted per frame-pair from the pose-encoder bottleneck.
- `build_khead()` bias-initialises the head so it *starts* at the crop-adjusted `k_norm`. The vendored
  convs are `bias=False` and the stock init lands at fx~1.19*W (vs da Vinci ~0.82) — miles from where
  the pretrained depth weights expect to be. We swap in biased convs, zero the weights, set the bias to
  the inverse of the head's own parameterisation (`f=(softplus(z)+0.5)*size`, `c=(z+0.5)*size`). Weights
  still get gradients, so K is free to move off the calibrated start.
- Consequence of that parameterisation: normalised fx,fy are FLOORED at 0.5. `build_khead` asserts.
  A very aggressive side-crop can push fx below that -> assert fires, use `--no-learn-intrinsics`.
- No warm-start for the head: a released checkpoint's K belongs to its camera+crop, and its bias-free
  convs would half-load over our init. Saved to `outputs/<run>/intrinsics_head.pth` alongside `best.pth`.
- Logged to wandb as `train/k_fx`,`k_fy`,`k_cx`,`k_cy` (normalised). WATCH THESE: focal length is only
  identifiable from camera ROTATION (the rotational flow field depends on K but not on depth). Pure
  translation leaves fx and the global depth scale degenerate. Near-static UMC clips -> expect drift.
- SCARED eval is unaffected (depth model only, median-scaled), so learned K shapes the training signal,
  not the metric. A wrong fx hides under median scaling; a wrong fx/fy ratio or principal point does not.

## 2026-07-18 — GUI mask: paired markers, connector bars, video masking
- `load_templates()` now returns `{stem: image}` instead of a list — the *names* matter.
  `gui_mask()` still accepts a plain list; only the dict form enables the connector logic.
- `temp04`/`temp5`/`temp7` are the small end-caps of the hazard-stripe bars. When exactly two of
  one name survive filtering, a `CONNECT_WIDTH+2*CONNECT_PAD` (=21 px) bar is drawn between them,
  or an **L** if they are off-axis by more than `CONNECT_ALIGN_TOL`. `_elbow()` puts the corner at
  whichever candidate is furthest from frame centre — the bar wraps the nearest screen corner, so a
  fixed horizontal-then-vertical rule bends the wrong way in half the corners.
- False-positive filtering, in order: cluster dedupe -> on an inset line -> not already inside the
  fixed boxes -> has a partner within one bar length (`PAIR_MAX_DIST`, path distance so Ls count).
  On the sample frame that takes `temp5` from 18 raw matches to the 2 real ones.
- `EDGE_INSETS = (24, 39, 79)` are measured from `_content_box()`, i.e. **content-relative**, not
  frame-relative — the footage is pillarboxed (~287 px of black on the left at 1920 wide) and
  frame-relative offsets would be wrong on any differently cropped clip. Same class of bug as the
  SCARED crop. The insets come from ONE annotated frame; widen them if a clip masks nothing.
- Per-template `THRESH_OVERRIDE` (`temp04` 0.6, `temp5` 0.7, `temp7` 0.8): the big box templates
  (`temp4`/`temp05`, 334 px wide) never score above ~0.45 and currently never fire — likely need
  re-cropping. `temp5` at 0.5 explodes to 31 tissue matches, hence the per-template values.
- `scripts/mask_video.py` + `jobs/mask_video.sh` (genoa, CPU): blacks out the GUI over a video,
  frames masked in a `multiprocessing.Pool`, decode/write sequential. 1587 frames @1080p =
  **58 s on 32 cores** vs ~19 min single-process. Note a single genoa core is SLOWER than a laptop
  core — the win is entirely the pool, so always pass `--cpus-per-task`.
- `PAIR_MAX_DIST`, `EDGE_INSETS` and `MARKER_PAD` are absolute pixels at 1920x1080; they do not
  scale with frame size the way `fixed_gui_mask()` does.

## Template rename (2026-07-20)

All templates now live under `data/templates/`, named for what they are instead of `tempNN`:

- `data/templates/*.png` — **popup panels only** (7): `popup_visualiseer_{dim,lit}`,
  `popup_voer_op_{dim,lit}`, `popup_beweeg_greep_{a,b}`, `popup_visualiseer_beweeg`.
  Dutch console text; `dim`/`lit` are the greyed and highlighted states of the same dialog.
- `data/templates/Move_Que/*.png` — the cue bar: `marker_{1,2,4}` (digit end-caps, the digit
  IS the arm number) and `bar_{gray,yellow}_{h,v}` (stripe segments).

Marker/bar split is now **by `bar_` prefix** (`BAR_PREFIX` in `cut_cue_clips.py`), not a hardcoded
name tuple, so adding a stripe template no longer needs a code edit. `CONNECT_TEMPLATES` and
`THRESH_OVERRIDE` in `gui_mask.py` were renamed to match; the old `temp04`/`temp5`/`temp7` in the
notes above are `marker_4`/`marker_1`/`marker_2`.

Because the cue templates moved into a subdirectory, `load_templates("data/templates")` (non-recursive)
now returns popups only — so `gui_mask`'s connector machinery (`_paired`/`_connect`) never fires for
`mask_video.py` anymore. It is effectively dead code pending a decision on that script.

## Cue detection: single-digit inference (2026-07-20)

A bar whose second digit scored under threshold used to mask **nothing** - `cue_span` needs a
validated PAIR. Now a lone digit infers its bar (`cue_paths` in `cut_cue_clips.py`):

- **Direction** = the side of the digit holding the longer stripe run. `stripe_dir` scores all four.
- **Run** = CONTIGUOUS chain butted against the digit (`_run_len`): starts within `STRIPE_GAP_MAX`
  (50 px = 2 blocks, so one unmatched block bridges), never skips more. Counting every stripe
  within 350 px instead let 3 tissue matches 250 px away validate a phantom bar.
- **Distinct blocks only** (`STRIPE_MIN_SEP` 20 px). `_peaks` runs NMS *per template*, so the four
  `bar_*` templates all fire on one physical block; that block counted four times. Applied to
  `bars_between` too - the pair path had the same inflation.
- **Length** = fixed nominal `BAR_LEN` 340 px, since the console always draws the full bar.
  `walk_ring` walks the content box's inset perimeter ring and turns the corner when the edge runs
  out, so **L bars now work** - previously `bars_between` scored them 0 and they were rejected
  outright. Verified on 43bf7ef9 f3885 (a real L, both legs masked).
- **Mask clipped to the bands** (`ring_mask`). A cue only ever rides the ring, so anything the thick
  line spills outside it is tissue no cue could have covered.

### Threshold: HIGH, not low - this was measured the wrong way round first

Instinct says lower `--bar-thresh` to "see more of the bar". Measured on 43bf7ef9 f200/f205, which
contain a real bar:

| thr | distinct blocks ON the bar | raw matches elsewhere |
|-----|---------------------------|----------------------|
| 0.95 | 9-10 | **0** |
| 0.90 | 10 | 2-5 |
| 0.70 | 11 | 109-142 |

The stripe templates separate cleanly at the TOP of their range. Dropping to 0.70 buys ~1 extra
block and ~140 tissue matches. **Bright yellow anatomy with dark red veins reads as a yellow/black
hazard stripe** - that is what produced the phantom bar on 43bf7ef9 f175, where zooming into the raw
pixels shows no bar at all. Inference therefore uses the same `--bar-thresh` as the pair path.

Caution for anyone eyeballing overlays: a red mask drawn over tissue looks exactly like a red mask
drawn over a bar. Verify against the UNMASKED crop, not the overlay - that mistake cost a round trip
here.

Defaults now `--min-bars 4`, `--bar-thresh 0.90`, `--pad-after 0`. Over 10 videos at stride 5:
769 pair hits, 41 inferred, 14326 clear. Every inferred hit spot-checked against raw pixels was a
real bar.
