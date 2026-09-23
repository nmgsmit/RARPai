# Claude's design log

Append-only. Newest on top. Record design choices made and where things were put, so future
sessions don't re-derive them. Keep entries one or two lines.

## 2026-09-23 — STANDARD SUL: cylinder depth-step START + arch model END (`scripts/sul_arch_cyl.py`)
- Nick fixed the method on 2026-09-23: START = `urethra_cylinder.analyse` t_start (axis="mask", depth step near the mask's
  proximal end, knee rule, outward only), END = arch model tip (outputs/arch_tip_pure40 arc7 dinov3_surg, 3-seed mean) matched
  in the IMAGE to the nearest point of the tube's top line, SUL = t_end - t_start. Depth = UniDepth V2, no K, frozen ruler calib
  scale mode (z_m / s_only, as arch_tip_unidepth) on the GUI-blacked 1340x1072 crop, K = da Vinci K_NORM. Catheter-top start
  (sul_slide_video) is NOT the standard; nor the roof end.
- Layout: a labelling-tool folder <root>/{images,masks} (1920x1080 console frames, palette masks) -> <root>/depth/<stem>.npz
  ('depth' mm float32 full res, 'gui'), <root>/visualization/<stem>.jpg, <root>/sul.csv. genoa job `jobs/sul_arch_cyl.sh`.
- GoodRulerTest SUL (job 27061981, 4 pre-cut frames, hand masks; Snellius ~/data/GoodRulerTest/SUL, local transfer_atlas_mod/
  workspace/GoodRulerTest/SUL): real / SUL mm 1132f8e5 16 / 15.8, 9e125883 27 / 35.0, c9d54c9b 16 / 20.2, fa38ea7e 18 / 19.8
  -> MAE 3.5, mean signed +3.5. The depth step moves the start <= 1 mm from the mask's start (mask start: MAE 3.2).
  Cylinder roof end instead of the arch: 20.7 / 34.5 / none / 28.4 (worse on all 3). 9e125883's +8 is not tilt (axis 11 deg
  out of plane). The same frozen calibration read the ruler lines of these cases 1.30x long (GoodRulerTest entry): /1.30 would
  give MAE 1.8, but that factor is fitted on these same patients -> in-sample, not a result.
- Rerun (job 27062730): the ruler (non-anatomical, only annotated in 1132f8e5) + 15 px now gets NO depth read (zeroed
  like the GUI; the networks still see it): 1132f8e5 unchanged at 15.8. c9d54c9b replaced by a pre-cut seg1 frame
  (00.00.25.506): 16.1 vs 16. -> MAE 2.5, mean signed +2.4 (9e125883 +8.0 is the only big miss). Picture (Nick) = frame + 50%
  depth overlay + cylinder outline + SUL line (green start -> red end) | the case's ruler frame; no masks, no gap profile;
  <root>/sul_overview.jpg stacks all cases.

## 2026-09-16 - Urethra cylinder from the MASK is the default now (urethra_cylinder.mask_tube, analyse(axis="mask"))
- Why: on monocular calibrated UniDepth the free fit failed: the urethra's depth is only ~1-2 mm deeper at its edges
  than its middle -> radius hit the 8 mm bound (unbounded 8-20); the refit pulled in tissue beside the urethra
  (RARP_062 22% inliers outside the mask); 9192f353 put the axis IN FRONT of the surface.
- Method: radius = half the 3D width (3-97 pct) of the prostate-side half of the mask. Direction = per mask row
  (5% trimmed each end) midpoint of left/right edge, instrument notches (NONANAT in the urethra hull) filled; RANSAC
  line (15 px) through the midpoints in the image + separate line of row median depth (middle third) -> 3D top line;
  axis one radius behind it. NONANAT grown 15 px outside the urethra first (blurred depth at a tool edge read as roof).
  Start/end march unchanged.
- Each piece fixed a visible failure on SUL_img3x hand masks + ureth_fn class 4: bottom-half-only rows followed the
  ragged prostate end (RARP_064 leaned 84 px); least squares on midpoints followed RARP_087's skewed top -> RANSAC;
  a depth-step instrument test flagged sloping tissue (RARP_091) -> dropped.
- `axis="fit"` = old free fit. Self-test runs both; only changed expectation: "mask past the start" -> mask mode
  finds the base 0.6 mm outside t_mask (SUL 17.96, true 18); fit mode keeps "mask (step inside)" (SUL nan).
- CALLERS CHANGE: arch_cylinder_compare, sul_depth_models, eval_sul_methods now get the mask tube, stereo included;
  their earlier numbers are free-fit -> pass axis="fit" to reproduce. Not re-run yet.

## 2026-09-16 - UniDepth K: the ruler calibration let UniDepth GUESS its focal per frame
- Every calibrated UniDepth path (metric_calib_proxy, arch_tip_unidepth, unidepth_overlays --calib, GUI npz)
  calls `infer(rgb)` without K, while z_true on the ruler set uses the da Vinci K_NORM. New
  `metric_calib_proxy --k`: UniDepth gets K_NORM on the ruler set, stereo P1 on the proxy eyes; every run
  now logs UniDepth's own focal (npz `focal`, p5/50/95 printed). Run it into a SEPARATE --out; switch
  the downstream no-K paths only if it wins (proxy abs_rel, LOO, tracking slope).

## 2026-09-14 - DEPTH: stereo proxy-GT convergence eval drives best.pth (+ frame/GUI convention)
- CONVENTION agreed with Nick, written into CLAUDE.md: side bars CROPPED once (1340x1072 content
  frame); GUI = black pixels + `<stem>_mask.png`, consumers read the mask, never infer from black.
- `make_stereo_proxy_gt.run_images`: a `<stem>_mask.png` beside an SBS still invalidates depth
  where the rectified left pixel OR its right-eye match (x - disp) is GUI, grown 4 px
  (`gui_rect_mask`, self-test). Measured on 3 frames: 0 depth px left under the GUI; without the
  mask 5-13k px survive there at median 46.6-54.3 mm = the screen plane. Also writes
  `<stem>_left.png` + `<stem>_left_mask.png` (the mono model's input).
- `eval_scared.run_proxy_gt_eval`: `abs_rel`.. are METRIC (unscaled, mm), `ms_*` median-scaled,
  `scale_ratio_median` (1 = metric). `_eval_pairs` now also returns unscaled errors (4-tuple).
- `finetune_depth --proxy-gt-dir` (default `../data/processed/proxy_gt_nogui_ffs`): logged every
  epoch as `proxy_gt/*` and it SELECTS best.pth (outranks metric_val / SCARED). Overlays at 1/4 res.
- Leakage checked: surgeries 18de9c5a / 5e27066c are not in depthclips_ruler_NoGUI -> clean test.
- `finetune_depth --smoke` fails in `_selfcheck_scale_loss` (1.28 vs 1.32), untouched by this
  change -- flagged as a separate task.
- HOLES in the FFS stills (Nick: "many black spots"), 9 frames attributed: 10-29% of the frame is
  hole, but outside-geom 1.3 + GUI 1.1-2.3 + specular 0.4-1.7 -- the rest (6-26% of the usable
  area) is the LR check: half-occlusion bands LEFT of every foreground object, the left border
  strip whose match falls off the right image (disparity ~100 px), and the textureless/glare
  white instrument shaft. Correct for an eval GT -- do NOT inpaint. Half the "spots" were not
  holes at all: magma's far end is near-black, same as the holes. `depth_jpg`/`preview` now
  TURBO + grey (90,90,90) holes, as temporal_stereo_clip already did.
- Temporal fusion (the earlier hole fix, +30% of holes) needs video: only 30/83 stills lie inside
  a 3D_ProxyGT clip (seg2 4, seg3 14, 5e27 12); 53 have no source clip.
- FIRST RUN (26683314, inplane config): EndoDAC warm-start proxy_gt abs_rel 0.218, scale 1.027
  (near metric!); fine-tuning worsens it to 0.35-0.40, proxy_scale ~0.83 (pred ~20% too FAR),
  agreeing with metric_scale 1.8 on the anchors. Through epoch 7, best.pth is still epoch 0.

## 2026-09-14 - DEPTH 3D: HUD/GUI stripped from the SBS proxy-GT stills (scripts/mask_sbs_gui.py)
- Each SBS eye is un-squeezed into the MONO 1920x1080 frame (eye->mono affine + MONO_CROP), masked
  there by the unchanged mono code (`cut_cue_clips.full_gui_mask`, split out of `black_gui`), mask
  warped back; L|R masks OR-ed and applied to both eyes (overlays composited identically, +960 px).
- Out: `../data/3D_ProxyGT/proxyGTimg_nogui/<stem>.png` (GUI black, lossless) + `<stem>_mask.png`
  (255 = GUI). 83 frames: HUD+tab = 5.1% each; 8 carry a cue bar (5.5-6.0%), all checked against the
  UNMASKED pixels, both eyes; no popups; no misses on contact sheets. Templates still score >=0.64 on
  the 2.125x-upsampled eye, so the mono thresholds carry over.
- BUG FIXED in `cue_paths` (mono too): a CHAINED cue `2 =yellow= 2 =grey= 1` is two bars whose whole
  span > PAIR_MAX_DIST, and `cue_span` returned only the widest pair -> the grey link stayed visible.
  Now repeats `cue_span` skipping pairs lying on an already-taken span. Self-test added.
- The mask is the deliverable, not the black: black is identical in both eyes -> a flat ~47 mm
  screen-plane that passes the LR check. make_stereo_proxy_gt.py does NOT read `_mask.png` yet.

## 2026-09-11 - SUL: head-to-head of every method (scripts/eval_sul_methods.py) -- use the MASK's ends along the tube

- 71 annotated frames, 4 methods x {Nick's masks, model masks} x {stereo, mono scaled per frame to the stereo,
  mono raw}; fixed reference = Nick's ruler points back-projected with the STEREO depth (3D chord).
  A mask ends (main-axis end pixels -> 3D chord), B first cylinder (git 50834de, run as is), C current
  cylinder (knee3 end, outward-only start), D C's axis with both ends from the mask (1st/99th pct).
- Median |SUL error|, Nick's masks: stereo A 2.8 / B 8.6 / C 2.9 / D 2.0 mm; mono-shape A 3.3 / B 8.0 / C 3.2 /
  D 2.7. Per end along the tube (stereo): D within 1.6 mm at every end; C long end +2.8; B control end -3.6,
  short 0/28 counted (hidden-start check rejects the prostate junction), long end +18.6.
- B's good first number (control +1.4 with model masks) was the model mask's early start cancelling a 3.6 mm
  early end. The 5e27 instability was the MODEL MASK, not the view angle (as guessed at the time).
- Mono: C's depth-driven end degrades (control -5.1, long +5.0 mm); mask ends barely move. Raw mono scale is
  off 1.06 / 1.59 / 1.74 x (control / short / long) -> a per-frame metric anchor is mandatory.
- Model masks: every method fails on the long clip (start 12-15 mm too far); the prostate end of the mask is
  the dominant error. RECOMMENDATION: D as the primary SUL, C's roof crossing only as a fallback / QC.
- Caveat: rules (knee3, outward-only) were chosen on these same frames, one annotator, 3 clips -- no held-out
  set; D involves no tuning. The chord reference carries surface depth (short: chord 22.5 vs along-tube 19.4).

## 2026-09-11 - SUL: Nick's hand masks + ruler points -- the mask was the long clip's problem, the END definition the rest

- Nick annotated the rectified frames (`OTHERS/annotate_urethra` -> `transfer_atlas_mod/workspace/<clip>/masks`)
  and marked start/end with the Ruler tool (`scale_objects.json`, keyed by index into the sorted clip images;
  mostly tracked, manual and tracked agree to <= 0.3 mm on the median). Converted to `<run>/hand_points.csv`;
  hand-mask runs write `urethra_cyl_hand/`.
- Method minus Nick, signed per-frame medians (start +: later/distal; end -: earlier):
  control seg3 30-35 s: model start -1.5 end -3.5 SUL -2.0 | hand start -1.6 end -5.8 SUL -4.5 (Nick 22.4 mm)
  short seg3 no-arm:    model start +3.4 end -2.1 SUL -4.8 | hand start +4.4 end -2.0 SUL -6.6 (Nick 22.5 mm)
  long 5e27 no-arm:     model start -17.1 SUL +14.9       | hand start +3.9 end +2.9 SUL -1.5 (Nick 8.5 mm)
- The long clip's instability was the MODEL MASK (27.2 -> 7.5 mm with Nick's masks, radius off the 8 mm cap),
  NOT the viewing angle guessed earlier.
- With hand masks each end is still 2-6 mm off, in both directions -- no single bias. On seg3 the end lands at the
  zero crossing of the slow pre-knee drift, while Nick marks the knee (the steep rise).
- CAVEAT on the metric: a ruler point takes the depth of whatever surface lies under it (prostate bulge, roof),
  which can sit ~9 mm in front of the tube; projected on the tilted axis that moves it several mm (short clip:
  start 15 px apart in the image, 4.4 mm apart along the axis). Compare in the image, along the top line, instead.
- The long clip's reference itself spreads +-2 mm frame to frame (manual and tracked alike).
- FOLLOW-UP: `--end-rule/--start-rule knee` (two-segment fit on the gap profile; falls back to the
  zero rule if the break is past the detection or bends the wrong way) and hand points matched in the
  IMAGE to the tube's top line (hand_measure). Runs with the zero rule go to *_end-zero_start-zero/.
- RESULT (d21910e): the knee fixes the prostate START (short clip +1.4 -> +0.4 mm) but NOT the end
  (control -7.3 -> -8.0, short -3.2 -> -3.6; long +3.0 either way). On seg3 the gap profile bends TWICE
  (flat -> slow rise -> steep): a one-break fit takes the first bend, Nick marks the second, where the steep
  rise starts (gap ~1.5-2.2 mm there). Long 5e27 has one sharp step and Nick is ~3 mm before it.
  Image matching works: Nick's clicks sit 4-19 px from the tube line.
- knee3 (f631b35; DEFAULT end rule since the next commit, start stays on the two-segment knee):
  three-segment fit, end = start of the steepest later segment. End error vs Nick: control -7.3 -> +0.5,
  short -3.2 -> +0.9 mm (p10-p90 within +-1.4); SUL vs Nick +1.8 / +0.5 / -0.5 mm. Long 5e27 unchanged
  (+2.8, a few frames up to +14): one sharp step, and Nick marks ~3 mm before it.
- OUTWARD ONLY start: the depth may move the start past the mask's edge, never into it. Long 5e27:
  depth start 4.2 mm inside the mask in 19/28 frames (start error +3.6 vs Nick; mask edge -0.5); short and
  control unaffected (depth start at or outside the edge). Cost: an overshooting mask cannot be pulled back.
- RESULT (60e66e7). Nick's masks: start error -1.4 / +0.4 / -0.6 mm (long was +3.6), end +0.5 / +0.9 / +2.8,
  SUL vs Nick +1.8 / +0.6 / +3.3 mm on control / short / long -- the long clip got WORSE overall because its
  late start used to cancel its late end. Model masks: +2.9 / +0.7 / +13.7; overshooting model masks are now
  pinned (10/10 control, 20/28 short starts are 'mask (step inside)'). Long end: the depth step sits ~3 mm past
  the top of Nick's mask and his end -- tissue lying flush on the tube is invisible to the depth.

## 2026-09-11 - SUL cylinder v2: the mask says WHERE, the depth finds the tube and the start border

- Nick's three suggestions (df62aa9, bc9c0de). (1) Mask as a rough ROI: after the first fit, every
  depth point within 40 px of the mask (instrument/catheter excluded) that lies on the surface is
  re-selected and refit 3x; radius held to 2.5-8 mm (end-on views ran to 8.5-11 mm unconstrained).
  (2) Start found like the end: a depth step at the mask's start -- prostate base in front, open cut
  end behind, or the catheter -- searched ONLY from 3 mm inside to 5 mm past the mask's start; no step
  but depth seen keeps the mask's start; unknown depth past it = hidden. (3) Prostate mask drawn
  (magenta); a start landing on it is tagged "base (prostate)".
- DEAD END, do not retry: walking the whole tube proximally from mid-tube stops at the first bump --
  the bipolar jaw on seg3 (99/100 frames "base", SUL 23.9 -> 12.6) or where a curved urethra leaves the
  straight cylinder (short no-arm clip: "open end", 14.0 vs 28.4 mm from the mask).
- Crossings are interpolated to gap = 0 (the threshold walk-back put each one ~zero_tol/slope past,
  and with both ends from the depth the offsets add). Self-test: cut end 17.9/18.0, prostate base 18.0/18.0.
- RESULT seg3 30-35 s: 100/100 frames, start = catheter every frame, SUL 20.4 mm IQR 20.0-20.9. It was
  23.9: the END moved -- on real tissue the zero crossing lands on the slow pre-knee drift, ~5 mm before
  the steep rise where the tissue visibly covers the tube. OPEN: a hinge/knee fit is the better end.
- Short no-arm seg3 clip (0.47 s, all 28 frames): the start lands on the prostate junction as a clean
  depth step, but SUL still spans 12-25 mm within 0.45 s of fast camera motion -- the mask-start SUL moves
  the same way, so that spread is the end/mask, not the start logic. 5e27 views stay unreliable (r pinned
  at the 8 mm bound, IQR up to 17-33 mm).
- The prostate class also paints the tissue around the cut stump after transection -- not a usable
  start signal there.
- `jobs/urethra_cylinder.sh`: SLOW=N slows only the encoded video.
- `--masks DIR`: hand masks replace the model (tool ids mapped by name, palette index read with
  PIL, DVP -> background, no keep-largest); frames without a mask are skipped; output in
  `urethra_cyl_hand/`. Frames to annotate: `OTHERS/annotate_urethra/<clip>/images/`.
- `--points CSV` (frame,ax,ay,bx,by, from the labelling tool's Ruler lines in scale_objects.json,
  keyed by index into the sorted clip images): 3D chord = the annotator's SUL; both points projected
  on the fitted axis give start/end errors (method - annotator, proximal point = start).

## 2026-09-11 - SUL: urethra END point from a stereo cylinder + where the roof covers it

- `scripts/urethra_cylinder.py` + `jobs/urethra_cylinder.sh`, run on a `temporal_stereo_clip.py
  --save-depth` output. `ureth_fn` + keep-largest on the rectified left eye (the same 1340x1072
  frame it was trained in), back-projected with P1, robust cylinder (soft_l1 on dist-to-axis - r,
  mask eroded 7 px). END = walk the tube's camera-facing top line distally: gap = Z_top - Z_obs is
  ~0 while the tube is visible and rises once the roof sits in front; first sustained gap > 1.5 mm,
  walked back to <= 0.5 mm. SUL = end - start along the axis.
- RESULT seg3 30-35 s (urethra side-on): 100/100 frames, **SUL 23.9 mm, IQR 23.6-24.2** under
  camera motion; r 6.1 vs silhouette 5.2 mm. 5e27 33-38 s (looking down the lumen): cylinder
  ill-posed (r 8.5 vs 5.2), 49/100 frames counted, SUL 17.2 mm IQR 16.4-19.2, and the frames whose
  mask reaches the lumen read 19-21 mm -- UNRELIABLE. The method needs the urethra side-on.
- **The unstable end was the START, not the roof.** On 5e27 SUL tracked mask size at rho 0.92: an
  instrument over the proximal urethra truncates the mask. Fix is class-independent -- the roof test
  run backwards (something in front 0.5-3 mm proximal of the start = hidden, frame not counted).
  On 5e27 it also fires on some clean cut ends (end-on stump); a catheter exception was tried,
  changed nothing on real data, and was removed.
- Radius ratio fit/silhouette separates the two windows (1.18 vs 1.62) but does NOT track per-frame
  error (rho 0.05), so it is reported, not used as a gate.
- Ceilings: straight cylinder (a curved urethra biases the far end); ~+-1.5 mm in where the roof
  "starts" (on seg3 the gap drifts -1..+0.5 mm before the knee); NO ruler GT for these surgeries
  (`sul_reference` has none) -- this is precision, not accuracy.
- `--self-test`: ray-cast tube under a roof + 0.15 mm noise -> r 3.95/4.0, axis 0.03 deg, SUL
  18.7/18.0; no roof -> nothing found; instrument over the start -> flagged (would read 14.6).

## 2026-09-11 - DEPTH 3D: TEMPORAL stereo - a video closes 30% of one pair's holes, but does NOT sharpen it

- `scripts/temporal_stereo_clip.py` + `jobs/temporal_stereo.sh`. Sample one window at 20 fps
  (59.94/3), run the existing per-frame matcher, then warp every neighbour's disparity into the
  current frame along DIS optical-flow chains and take the median of what survives three gates:
  forward-backward flow consistency, >= `--min-support` frames agreeing, tight MAD.
- RESULT on 5 s of the seg3 clip, FFS at scale 0.5, 100 frames: **84.1% -> 88.9%** of the usable
  area solved, i.e. **30% of the holes closed**, leave-one-out agreement **0.11 mm**. Biggest
  gain exactly where the single pair is worst: the 5 worst frames go 46.5->62.9, 47.6->58.6,
  53.6->62.4, 55.4->56.4, 58.1->66.9%. Mean gain 4.7 pts, max 16.4.
- **The win is COVERAGE, not precision.** Measured frame-to-frame depth jitter along the flow is
  only 0.13 mm, so there was almost no temporal noise to average away, and by default valid
  pixels keep their own-frame value (`--temporal-median` changes that; it is off). Anyone
  expecting a video to make the depth *sharper* should be told it makes it *more complete*.
- Ceiling is reported honestly: `geom` (rectified overlap minus GUI banner) is 98.7% of the
  frame and NO method reaches the rest, so every percentage is over `geom`.
- WHY EACH REMAINING HOLE SURVIVES, counted separately - this is what drove the tuning.
  4.2% of the frame `blind` (no frame in the window saw that point - the real floor), 4.7%
  `flow`-gated, 0.0% `thin`, 2.1% `disagree`. The first guess lumped these together as "nothing
  there" and hid the fact that more was gated by my own thresholds than was genuinely occluded.
- SWEEP on one cached window (holes closed / agreement): w4 fb1.5 sup2 18%/0.10mm; w4 fb3 sup2
  19%; w8 fb1.5 sup2 20%; w8 fb3 sup2 22%; **w8 fb3 sup1 30%/0.11mm (adopted)**; w8 fb3 sup1
  mad3 34%/0.12mm. Defaults pinned to the adopted row.
- `--min-support 1` needed justifying, and pooled agreement could not do it. STRATIFIED BY
  SUPPORT: **0.62 mm at support 1, 0.29 at 2, 0.11 at 3+**. So an uncorroborated fill is ~6x
  worse than a well-supported one and still inside the 0.9 mm calibration MAE (STEREO_REPORT
  §3), and it is only 1.3% of the frame. That is the trade, stated rather than assumed.
- `--align-median` (default ON) removes the median own-vs-warped disparity offset per pair
  before fusing, because depth-constant-along-the-flow is false under camera motion. Median
  offset 0.14 px but **p95 29.9 px** - a few pairs really do move a lot, which is why the
  alignment exists and why the MAD gate has to come after it.
- Agreement is LEAVE-ONE-OUT by construction (fusion never sees the frame it fills) but still an
  **optimistic bound**: it can only be evaluated where own stereo also succeeded, i.e. on the
  easy pixels. Do not quote it as the error of the filled pixels.
- The matcher is ~95% of runtime and depends on no fusion knob, so disparity is cached under
  `outputs/temporal_stereo/_disp_cache/<key>` (float16, 0.125 px step = 0.015 mm). A sweep is
  then a CPU job: the six configs above cost one GPU run plus CPU minutes.
- Drawing decisions that were wrong first: per-frame percentile colour stretch (flickers, hides
  real depth change - now one FIXED range for the clip); magma (the 23-115 mm span collapsed
  into one dark purple - now turbo); black holes (invisible against turbo's dark violet far end,
  where most holes are - now neutral grey 90,90,90, a colour no colormap here produces).
- `--save-depth` writes the fused result in the proxy-GT layout (uint16 mm*16), so this is a
  drop-in better GT generator, not only a demo.

## 2026-09-09 - DEPTH 3D: the surgical footage picks the distortion model, no re-record needed

- PROBLEM: the ChArUco clip never reaches past r~592 while the frame corner is at r=906, so 26%
  of the frame is EXTRAPOLATED and the board cannot choose between distortion models. Nick
  cannot re-capture the edges.
- FIX: the stereo pair is its own calibration target out there. After rectification a
  correspondence must have dy=0, and that holds at EVERY radius on the surgical clips. Four
  models on the same 120 board views, ~11.5k SIFT correspondences from `../data/3D_ProxyGT`:

  | model | board RMS | displ @ r=906 | med \|dy\| 0-300 | 300-500 | 500-700 | 700-950 |
  |---|---|---|---|---|---|---|
  | **k1 only** | 0.519 | **9.5px** | 0.56 | 0.76 | **1.12** | **1.82** |
  | k1,k2 | 0.519 | 19.6px | 0.55 | 0.73 | 1.48 | 4.35 |
  | k1,k2+tang | 0.516 | 24.9px | 0.58 | 0.80 | 1.63 | 4.20 |
  | k1,k2,k3 | 0.512 | 39.4px | 0.59 | 0.78 | 1.34 | 2.85 |

  All fit the BOARD identically (RMS 0.512-0.519) but extrapolate over a 4x range. The richer
  models fit noise inside r<600 and pay for it outside. k1 wins 2.4x at r=700-950 and TIES in
  the centre (0.56 vs 0.55) -- the bounded edge behaviour is free.
- `--dist-model {k1,k1k2,k1k2tang,full}` in `calibrate_stereo_charuco.py`, **default k1**.
  Re-pinned `calib/stereo_calib.json`: fx 1064.01 fy 1064.27 cx 621.35 cy 535.20, baseline
  4.1179mm, convergence -96.54px, **Z_mm = 4707.2 / disparity_px**, validation MAE 0.900mm
  (0.83% rel) -- unchanged from the richer models, so nothing was lost.
- Distortion now bounded+monotone: 0.04 / 0.39 / 1.58 / 4.13 px mean by radius band, corner max
  8.8px (was 18-39). Safe to apply.
- CAVEAT: dy probes the RADIAL model; depth comes from dx. Radial distortion couples both so
  this is strong evidence, not proof. Direct test would be whether the 8mm robot-arm shaft
  measures 8mm at large radius too. NOT RUN: the radial distribution of `scale_objects.json`
  annotations (Snellius kept timing out) -- if the arm points sit inside r=600 the edge question
  is moot for the metric scale loss.
- Nick's priority: metric accuracy in the CENTRE. => weight the proxy-GT depth loss by radius
  rather than hard-cropping the periphery.

## 2026-09-09 - DEPTH 3D: da Vinci SBS geometry solved + real stereo calibration

- SPECULAR MASK BUG (fixed): "bright + desaturated" (V>240,S<40) also describes a white da Vinci
  instrument shaft, so the first version deleted 0.8-3.8% of each frame that was INSTRUMENT --
  the nearest objects with the sharpest depth edges, i.e. the best supervision in the set. Now
  masks SMALL blobs only (`max_blob=2000`): measured over 1849 components, genuine highlights are
  median 10px / p95 128px while instruments run 5k-32k, so size separates them cleanly. Instrument
  masking now 0.00%. Pinned by `scripts/tests/smoke_specular_mask.py`.
- BOARD PRINT SCALE CONFIRMED 100% (Nick measured the printed board by hand, 2026-09-09) -> the
  metric chain has no unverified scale factor left.

- DATA `../data/3D_ProxyGT/*.mp4` (3 clips) + `../data/ARUCO_calibration/*seg1.mp4`, all
  1920x1080 @59.94, **half-width anamorphic side-by-side**: left eye x164-799, right x1124-1759
  (exactly +960), y32-1047, 636x1016 per eye, squeezed 2x horizontally. Boards are
  `OTHERS/charuco_endoscope_A4.pdf`: A = DICT_4X4_50 ids 0-30, 9x7, square 8.0mm marker 6.0mm;
  B = ids 31-47, 7x5, square 6.0mm marker 4.5mm.
- GEOMETRY, measured not guessed: cross-correlating the da Vinci GUI banner (fixed-size overlay
  = a free ruler) between the SBS halves and the raw mono videos gives **scale 2.125, left edge
  x=286**, IDENTICAL on all 3 surgical clips, both eyes, and the calibration clip. Combined with
  mono `source_crop.json` (x289 y4 w1340 h1072) this makes one sub-pixel affine per eye ->
  the mono 1340x1072 frame: src x 165.41(+960 for R), y 35.76, w 630.59, h 1008.47. In
  `scripts/calibrate_stereo_charuco.py:eye_to_mono` -- USE THAT, do not re-derive.
  An error in the affine is absorbed into fx/fy at calibration time, so only consistency between
  calibration and inference matters. Confirmed sound: calibrated **fx/fy = 1.0003**.
- The mono 1340x1072 frame is 1020 rows of anatomy + **52 rows of GUI banner** (rows 1020-1071),
  blacked out in the NoGUI clips. `BANNER_ROW=1020`.
- CALIBRATION `scripts/calibrate_stereo_charuco.py` -> `outputs/stereo_calib/calib.json`.
  60 views, RMS 0.50px. **fx 1061.78 fy 1061.51 cx 608.65 cy 535.68** px @1340x1072,
  **baseline 4.1155 mm**, stereo rotation 0.205 deg. Console ships a pre-converged pair:
  cxL-cxR = **-96.73 px**, cyL-cyR = 0.01 px (already vertically rectified).
  Rectified: **Z_mm = 5030.5 / disparity_px**.
- vs the borrowed SCARED K (0.82,1.02,0.5,0.5): ours normalised = **0.7924, 0.9902, 0.4542,
  0.4997**. Focal was good to ~3%, cy exact, but **cx is off by 61px (9%)** -- the optical axis
  is NOT at frame centre. Every prior UMC depth run reprojected through that error.
- **DISTORTION IS NOT NEGLIGIBLE**: 0.16px mean inside r=200, but 2.4px at r=400-600 and 5.4px
  (max 34.7) at r=600-900. Skipping undistortion cost 9% depth bias in validation; rectifying
  fixed it. => the pinhole assumption in the mono pipeline breaks at the periphery.
- VALIDATION (in-script, reruns free): disparity depth vs the board's own PnP pose, 2463 corners
  over Z 52-203mm -> **MAE 1.00mm, bias -0.29mm, 0.81% rel**, epipolar |dy| 0.20px median.
  Transfer to the surgical clips: implied Z median 45-64mm, p5-p95 ~29-110mm -- matches the
  independent ruler/catheter/arm pre-flight estimate (55/61/45mm). Holds on `5e27066c` too,
  which is a DIFFERENT session from the calibration clip.

## 2026-09-08 — SEG: urethra loop — LOSS SHAPING IS A DEAD LEVER, post-processing won

- INSTRUMENT FIRST: `scripts/eval_urethra.py`. Dice cannot see WHICH way a mask is wrong.
  Reports U->P, P->U, and `leak` (predicted urethra not on urethra = 1-precision), then splits
  leak by connected component into HALO (attached to a blob that found real urethra) vs
  **STRAY** (component with ZERO GT overlap). Those mean opposite things and precision conflates
  them. Imports the trainer's own `clip_pairs`/`SegDataset` -- a split that differs from the
  run's own is a silent lie.
- **The reported failure was not the real one.** U<->P confusion is ~0.3-1.9% each way in every
  model -- a non-issue. The constraint violation was STRAY blobs: ~10% of predicted urethra,
  **~1 per frame**, on tissue labelled nothing. Only ~4.7% of leak is boundary slop.
- SWEEP (6 arms, 512px, DVP excluded, 50 ep, `--select-on urethra`): control / urethra weight
  x3 / x6 / a=.35 b=.65 / a=.65 b=.35 / a=.8 b=.2. **All six within 0.014 dice (0.789-0.803) =
  noise on one seed.** The precision-leaning arms (which SHOULD have cut leak) produced the MOST
  stray (12.3%, 12.0%): shrinking masks fragments them without suppressing confident small blobs.
  Hypothesis falsified in both directions. Do not spend more GPU on alpha/beta or class weights.
- **`--keep-largest` (inference-time, no training) is the fix.** The urethra is ONE structure, so
  keep the biggest component and drop the rest to BACKGROUND (never to another class -- that would
  invent the U<->P confusion we are avoiding). On `ureth_fn`: dice 0.8032 -> **0.8386**, precision
  0.7916 -> 0.8898, stray 10.95% -> **0.39%**, blobs 1.15 -> 0.02/frame. Dice went UP.
- WINNER = `outputs/ureth_fn/best.pth` + `--keep-largest`. vs the first clip-data model
  (`rarp_nick_dice`, 0.7685) that is +0.070 dice. Most of the gain is NOT the loss: it is
  `--select-on urethra` (save the best-urethra epoch, not the best-mean-dice one) + dropping DVP.
- **CAVEAT: keep-largest is NOT in the checkpoint.** Anything loading `best.pth` directly (the
  ATLAS GUI included) gets the unfiltered ~10% stray. `_keep_largest` is duplicated in
  `overlay_dir.py` and `eval_urethra.py`; a third consumer should get a shared helper.
- Remaining error is now RECALL (0.7930): ~21% of GT urethra missed, and keep-largest costs a
  little of it (0.8153 -> 0.7930) when a true urethra splits in two. That is the next lever, not
  the loss.
- Single seed, one 8-clip test split. The arm ranking is not significant; the keep-largest effect
  (28x) is far outside that noise.

## 2026-09-08 — SEG: retrain on the new clip-layout annotations (a NEW label scheme)

- DATA: `../data/processed/Segmentation/Nick` — 56 clips, `<clip>/images/*.jpg` +
  `<clip>/masks/*.png`, 6659 images / 6635 masks / **6584 paired**. Replaces the old
  `RARPSurgenet` 378/97/60. NOTE `../data/RARPSurgenet/fold1` NO LONGER EXISTS (splits sit
  directly under `RARPSurgenet/`) — every older `jobs/finetune_*.sh` still points at fold1 and
  would fail on resubmit.
- **DIFFERENT LABEL SCHEME.** Legend is the labelling tool's own, `transfer_atlas_mod/gui/cutie/
  utils/palette.py::custom_names`: 1 urethra, 2 prostate, 3 dorsal venous plexus, 4 catheter,
  5 non-anatomical. Old scheme was 1 catheter, 2 prostate, 3 urethra, 4 apicalvesicle. Only
  urethra/prostate/catheter exist in both; DVP + non-anatomical are new, apicalvesicle is gone.
  `OLD2NEW` in `finetune_seg_tversky.py` maps them BY NAME — never by id.
- **The masks are mode "P".** The label id is the palette INDEX, so `.convert("L")` maps id 1 to
  226 (yellow's luminance) and the remap then drops every foreground pixel to background. That
  cost a run (26466088): loss 0.0000, `val_dice=1.0000`, every per-class dice 0.0000 — because
  `validate()` averages over PRESENT classes and only background was present. **An all-background
  dataset reports a PERFECT score, not a zero.** Old masks were mode "L" so the convert was a
  no-op, which is why this never bit before. There is now a startup `[labels]` histogram + assert
  that every kept class actually appears; it dies in 2 s instead of looking great for 12 h.
- SPLIT IS BY CLIP, not by frame (`clip_pairs`, seeded): consecutive frames of a clip are
  near-duplicates, so a frame split leaks test into train. Pairing is a stem INTERSECTION — the
  export has 24 more images than masks, so zip-by-sort-order misaligns after the first gap.
- TWO test numbers: `[test]` = held-out clips (the honest one); `[compare]` = the old 60-frame
  RARPSurgenet test set with its labels mapped into the new scheme, shared classes only, so it is
  comparable to `rarp_tversky_dice` (catheter=0.8567 urethra=0.8167). Verified NO leakage between
  them: old test videos are RARP_001/017/033/053, new clips are RARP_064..092 + 23 uuid-named,
  intersection empty, and max frame-correlation 0.76 (nothing >0.8).
- Run = `jobs/finetune_tversky_nick.sh` -> `outputs/rarp_nick_dice`. Every hyperparameter copied
  from `finetune_tversky_dice.sh` (512px, batch 8, lr 1e-4, 50 ep, a=b=0.5, --bg-in-loss) so the
  comparison isolates the DATA change. 12 h walltime, not 8: ~17x the steps per epoch.
- Baselines for the write-up, from the old 60-frame test set: `rarp_tversky_dice` (5-class, 512)
  dice=0.8135 cath=0.8567 ureth=0.8167 — the best all-class model, copied to
  `outputs/bestseg/best.pth`. Best of ANY old run was `rarp_tversky_dice_1024_2class`
  (cath+ureth only) dice=0.8392, which was never promoted to `bestseg/`.

## 2026-09-07 — DEPTH: PER-VIDEO ARM FINETUNING DOES NOT WORK — calibrate instead
- Base `outputs/adapt_base` (all classes, full volume, --scale-inplane --anchor-w 0.1, 4 ep,
  seed 66, best ep1): held-out in-plane scale .894 ruler / .894 cath / .830 arm, slope .349.
  Then per held-out video, --only-videos + --scale-classes 3, 3 epochs, arm anchors only
  (`jobs/adapt_per_video_arm.sh`, job 26448739; --epochs 0 pass = the un-adapted baseline).
- IN-PLANE abs_rel, base -> adapted:  ARM (supervised)  .180 -> **.043**  (n=113)
                                      RULER+CATH (held out) .163 -> **.174** (n=390, WORSE)
  Per video the held-out number goes .158->.181 (7ee04683), .177->.170 (ee3be53d),
  .161->.164 (31e2c520). track_slope falls in 2 of 3 (.32->.24, .40->.26).
- SO: adaptation fits the arm 4x better and the depth map not at all. train_scale collapses to
  .002 while c1/c2 stand still — the same "satisfy the anchor without moving the depth" failure
  as the equal-volume ablation, now inside a single video.
- NO-TRAINING CONTROL (`jobs/adapt_calib_control.sh`, fit_affine_scale --calib-classes 3, job
  26448740, 607 objects): per-video arm-only **scale** = 90.5% median error (!), per-video
  arm-only **affine** (a*Z+b) = **16.8% / 1.87mm**, global arm-only affine 15.4% / 1.47mm.
  All-class per-clip affine is 7.3% / 0.78mm — the ceiling if you ever have richer anchors.
- WHY, measured on the base model's own depths: the arm's depth spread is 6.0mm within a clip and
  17.2mm within a video (16% of its range) vs the ruler's 24.8 / 45.5mm (48%). One multiplier
  fitted on the arm is unidentifiable and explodes on extrapolation; adding the OFFSET rescues
  it. The arm pins the offset, not the scale.
- DECISION: do NOT ship per-video arm finetuning. Ship base weights + a per-video (or per-clip)
  AFFINE-in-depth calibration fitted on the arm. Costs a CPU second, needs no GPU at the bedside,
  and beats 3 GPU epochs. Open lever if 16.8% is not enough: get depth DIVERSITY into the
  deployment anchor (annotate the arm at several working distances), not more epochs.

## 2026-09-07 — DEPTH: WHICH ANCHOR at EQUAL volume — the ROBOT ARM IS THE WORST SUPERVISOR
- Redo of the 2026-09-02 --scale-classes A/B with the volume confound REMOVED. New flag
  `--anchor-balance 1 2 3` keeps only TRAIN videos carrying all three classes, then subsamples
  each to the same count: **135 objects x 3 classes over the SAME 5 videos**. `--seed 66` is the
  split (searched over 200) where all three are also well represented on the 5 held-out videos
  (test n 383/143/133). 3 epochs, `jobs/ablate_anchor_class.sh`, jobs 26447579/80/81, base config
  = the operating point (scale-w 0.5, --scale-inplane, --anchor-w 0.1, K frozen).
- READ IN-PLANE, per class (`c*_inplane_*`, added to eval_metric_scale). A class the run did NOT
  supervise is the held-out cross-object check. scale / abs_rel, own class in bold:
  supervised     c1 Ruler        c2 Catheter     c3 Arm         held-out abs_rel  slope  SCARED
  Ruler       **.868/.169**     .721/.279      .737/.263            **.271**      .193   .0663
  Catheter      .672/.331     **.813/.187**    .730/.270              .301        .212   .0626
  Arm           .667/.336       .626/.374    **.842/.159**            .355        .058   .0575
- **ARM-ONLY IS NOT ENOUGH.** It is the worst supervisor of the other objects (held-out abs_rel
  .355 vs the ruler's .271) AND the only arm whose depth stops tracking distance at all
  (track_slope .058, i.e. back to the constant-distance regime; ruler/catheter ~.20). It also has
  the LOWEST training scale loss (.0014 vs .0042) and the BEST SCARED (.0575) — it satisfies its
  own anchors while barely moving the depth map, which is exactly the failure signature.
- WHY, and it is not annotation quality: z_true = fx*mm/px over the balanced train anchors spans
  p5-p95 **38.6-73.7 mm for the arm (std 12.4, log-std .216)** against 37.9-124.7 for the ruler
  (std 25.6) and 45.4-121.9 for the catheter. The arm is seen at ONE working distance. A narrow
  z range cannot separate "right scale" from "one constant", so it teaches the constant.
- CONSOLING: every held-out scale now lands in .63-.74 instead of the 2026-09-02 collapse
  (held-out ruler scale .027). Volume WAS the story there; at equal volume the anchors do
  partially transfer. Also the arm is the EASIEST object to be scored on (lowest debiased error
  in every column, .094-.111) — a good measurement target, a bad teacher.
- SO WHAT for deployment (arm is the only in-frame anchor at inference): do not train on the arm
  alone. Train on the ruler (or ruler+catheter) and CALIBRATE on the arm — the 2026-09-02
  arm-only calibration entry already covers that path. If arm-only supervision is unavoidable,
  the fix is depth-range diversity in the annotations, not more of them.

## 2026-09-07 — DEPTH: --anchor-w A/B on top of --scale-inplane — 0.1 is the operating point
- 4-epoch arms, jobs 26440931/26440932, `outputs/depth_inplane_a0{1,3}/best.pth`. TEST (n=665):
  anchor-w   track_slope   scared abs_rel / a1   inplane_scale   inplane_abs_rel
  0 (a00)    0.628         0.134 / 0.798         1.060           0.167
  0.1 (a01)  **0.430**     **0.070** / 0.957     0.885           0.175
  0.3 (a03)  0.311         0.067 / 0.963         0.852           0.191
  sw05       0.094         0.054 / 0.983         ~0.74           --
- MONOTONIC TRADE, with diminishing returns: 0.1 -> 0.3 costs a THIRD of the tracking (.43 ->
  .31) and buys almost no geometry (.070 -> .067). Take 0.1. Note the val slope (0.27-0.32)
  UNDERSTATES the test slope (0.43) -- judge on test.
- WHY THE CAP: anchor_loss pulls log-disp toward a frozen warm-start copy whose LEVEL is the
  wrong one, so it holds geometry and vetoes the level at the same time. It should only vote on
  SHAPE. Next: subtract each image's own mean log-disp from both student and teacher before the
  L1, so the term is level-blind. Expect slope toward .63 at scared ~.067 if that is the whole
  story.
- Best candidate deliverable is now a01, NOT sw05, IF the SUL measurement is taken in-plane --
  a01 still has inplane_scale .885 (12% too near) and the polyline scale is 2.06, i.e. depth is
  still rough along a segment. Re-run scripts/eval_catheter_ckpt.py on it before switching.

## 2026-09-07 — DEPTH: --scale-inplane WORKS (distance tracking) but costs LOCAL GEOMETRY
- `endodac-ruler-inplane` (job 26440067, gpu_a100 -- gpu_h100 was drained): scale-w 0.5,
  --scale-inplane, --anchor-w 0, 12 ep. Best epoch 2 by inplane_abs_rel; ckpt kept as
  `outputs/depth_ruler_inplane/{best,ep2_snapshot}.pth`.
- TEST (5 held-out videos, n=665), sw05 -> inplane: track_slope .094 -> **.628**,
  track_stdratio .226 -> .758, track_corr .417 -> .827, inplane_scale ~.74 -> **1.06**.
  THE DEPTH MAP TRACKS DISTANCE FOR THE FIRST TIME. Confirmed independently on the 69
  hand-annotated catheter widths: corr(error, apparent size) -- the constant-distance
  signature -- collapses **+0.845 -> +0.306**.
- THE COST, two ways of seeing the same thing: SCARED abs_rel .054 -> .134, a1 .983 -> .798;
  and the test POLYLINE scale is 1.97 while in-plane is 1.06, i.e. depth is now ROUGH ALONG a
  segment. In-plane supervision constrains the segment's MEAN depth and nothing constrains its
  variation, so the tilt that used to be the shortcut is now unconstrained noise. Nick saw this
  as "corrupted from epoch 6" in the qual panels.
- CATHETER SET (external, hand-drawn, `../data/sul_reference`), measured 3D / measured in-plane:
  sw05 scale 1.151 / 1.136, debiased 13.1% / 13.3% | inplane 1.927 / 1.639, 21.9% / **15.0%**.
  Measuring in-plane removes most of the inplane model's penalty => the raw +4.9mm is endpoint
  roughness, NOT a distance error. Level 1.64 here vs 1.06 on the ruler test set is consistent
  with slope .63 (not 1.0): these snapshots sit at the NEAR end (z_true ~26-59mm), and an
  under-tracking model over-reads there. No domain gap needed to explain it.
- SO: not yet a better deliverable -- sw05 still wins at the habitual working distance. But the
  failure mode changed from "cannot measure distance" to "measures it at 0.63 gain, with rough
  local depth". NEXT: restore geometry regularisation now that the scale term only moves the
  MEAN depth -- A/B --anchor-w 0.1 vs 0.3 with --scale-inplane, 4 epochs (everything past ep 5
  is the degenerate regime: pose_trans jumps 7x at ep 6). Watch track_slope AND scared_abs_rel
  together; either alone is gameable.

## 2026-09-07 — DEPTH: the scale loss was GAMED — training changed segment TILT, not distance
- Same 671 test objects through warm-start and sw05 dumps (`rays` byte-identical, so annotations
  and K match): per-object mean depth ratio sw05/warm **1.005** (IQR .997-1.013, corr .992,
  median z 42.5 -> 42.5 mm). Twelve epochs did not move the predicted DISTANCE at all.
- The reported metric scale .801 -> .974 came from somewhere else entirely: in-plane length with
  z held flat is UNCHANGED (5.76 -> 5.77 mm), while depth variation ALONG the annotated segment
  |dz| went **2.55 -> 4.39 mm (+72%)**. Tilting a segment out of the image plane lengthens it in
  3D without moving it. That is the whole "metric scale" gain.
- TWO CODE CAUSES, both real: (1) `scale_loss` compares the 3D POLYLINE length to mm, so segment
  tilt is a free way to satisfy it — and `eval_metric_scale` calls the SAME function, so the
  metric rewards the shortcut it should catch. (2) `anchor_loss` (--anchor-w 0.3) is an L1 pull
  of log-disp toward a FROZEN warm-start copy over the WHOLE image = "do not change the depth";
  the only degree of freedom left for the scale term was local shape. Training itself was real
  (29.58M trainable, train_photo .141 -> .025) — it just went into appearance, not distance.
- FIX DIRECTION: make length in `scale_loss` IN-PLANE (z flat at the segment median) so the only
  way to satisfy it is the distance, and run with --anchor-w 0. Caveat: in-plane length is
  mm*cos(theta), so oblique objects push z too FAR — a ONE-SIDED loss (penalise only predicted >
  mm) turns each anchor into an upper bound on z and lets the most fronto-parallel ones set it.
  Judge by the z_pred-vs-z_true slope, never by metric_scale.

## 2026-09-07 — DEPTH: the model does NOT track distance — it predicts a CONSTANT ~43 mm
- Test: for every annotated object, z_true = z_pred * mm / L_pred (known size vs predicted length),
  then regress z_pred on z_true. Run on `outputs/affine_{sw05,warmstart}_test.npz` (671 objects,
  5 held-out videos) and on the 69 hand-annotated UMC catheter widths in `sul_reference`.
- RULER TEST SET, sw05: z_pred pct5/50/95 = 35 / 42.6 / 47.6 (std 3.9) while z_true = 28 / 43 / 87
  (std 17.3). slope **0.094**, corr +0.42, std-ratio 0.23. Per class corr .22/.57/.79.
  WARM-START IS IDENTICAL (slope 0.094, std-ratio 0.16) => scale training MOVED THE CONSTANT, it
  did not create distance sensitivity. Not a range-mapping artefact this time: 28-87 mm needs
  sigmoid 0.14-0.68 at --min/max-depth 20/200, nowhere near saturation.
- UMC SUL SNAPSHOTS (69 manual catheter widths, external check): same picture. near half z_true
  33 -> z_pred 43.1 | far half z_true 45 -> z_pred 44.9. corr(ratio, width_px) = **+0.84**, i.e.
  the over-read is entirely "object is closer than the model thinks". Slope is INVARIANT to fx,
  so a wrong focal cannot explain it (and fit_fx_catheter's 1.36x cannot fix it either).
- => `metric/test scale = 1.000` is a MEDIAN. It says the constant is well centred, nothing more.
  The ~20% per-object floor in the 2026-09-02 conclusion IS this compression, and it explains why
  per-clip calibration halves the error (it re-fits the constant per clip) while more scale
  supervision never does. Anchor depth spread is NOT the cause: training anchors span
  z 31-104 mm (n=2317, std 22.8).
- SO WHAT: a DPT/DepthAnything head is affine-invariant per image by construction; the absolute
  level has nowhere to live. Next lever is a per-frame scale (ZoeDepth-style: CLS token -> one
  positive scalar multiplying depth, trained by the existing scale loss), not more --scale-w.
  Metric to judge it: slope / std-ratio of z_pred vs z_true, NOT metric_scale.

## 2026-09-02 — DEPTH: CONCLUSION — training fixes SCALE; the ~20% per-object error is a floor
- CHECKPOINT A/B on the 5 held-out videos. Each run differs from the `range` pivot (scale-w 0.1,
  anchor-w 0.3, 12 ep, K frozen) by ONE flag. wandb metric_test/abs_rel, n=665:
  `range` mbbm1208 .211 (best ep 4) | `range-noanchor` iz883ljm (anchor 0) .200 (best ep **1**)
  | `range-sw05` afku6qek (sw 0.5) .205 (best ep 10). Test scale .919 / .933 / **.974**.
  mm view (`eval_scale_table.py`, n=671): warm-start 25.2% / 2.30mm -> sw05 **19.9% / 1.88mm**.
- The "+1 GLOBAL CONSTANT" trick is DEAD once the depth range is right: warm-start 25.2% raw ->
  25.6% after dividing by the fitted k. One constant RE-CENTRES, it does not shrink spread. It
  only looked impressive in the old 0.1/150 runs because it was silently undoing the range bug.
- SCALE and PER-OBJECT error trade off, and ONLY SCALE RESPONDS TO TRAINING. sw05 wins scale
  (.974) but not per-object (deb .205); noanchor wins per-object (deb .187) at best epoch 1, i.e.
  barely trained. Raising scale-w .1->.5 moves scale .919->.974 and extends useful training
  (best ep 4->10) but leaves the per-object residual alone: every config lands in 19-22%.
- `--anchor-w` (L1 log-disp to a FROZEN warm-start teacher) buys TRAINING LENGTH, not accuracy:
  with it a run stays productive to ep 4/10; without it selection picks ep 1 and all later epochs
  degrade. NAME COLLISION, will bite again: `--anchor-w` is the geometry regulariser and has
  NOTHING to do with the scale_objects "anchors", which are driven by `--scale-w`.
- SO WHAT for SUL: the weights need not carry the scale if it is fitted per-clip at inference
  (see the affine entry) — but fine-tuning is still NOT redundant, because it helps AFTER
  calibration too (per-clip affine-disp 9.3% sw05 vs 12.6% warm-start). The ~20% floor is
  per-object noise on thin objects plus the uncalibrated focal (DEFAULT_K_NORM, never measured
  on this scope) — neither of which more scale supervision can touch.
- OPEN: the affine grid was never run on `range-noanchor`. If its deb .187 survives per-clip
  calibration it could beat sw05's 9.3%, making "1 epoch + per-clip calibration" the whole
  recipe. Cost: one GPU dump + a CPU fit (~5 min).

## 2026-09-02 — DEPTH: --scale-classes A/B — dropping the Ruler makes metric scale WORSE
- `--scale-classes ID...` restricts scale supervision to given class_ids; the metric eval still
  scores every class, so an excluded one is a held-out cross-object check.
- A/B on the same split (4 ep, K frozen, scale-w 0.1): all-classes `jiui7izm` vs cath+arm-only
  `a1dlnj65`. Test scale 0.559 vs 0.062; test abs_rel 0.516 vs 0.938. Held-out Ruler scale 0.027
  in run 2 = the learned scale did NOT transfer to an unsupervised class at all.
- CAUSE is supervision VOLUME, not ruler quality: of 856 annotated TRAIN frames, 850 carry a
  ruler but only 283 carry a catheter/arm, so 2/3 of batches contribute zero scale gradient.
  The per-batch loss is w-weighted, so those frames are silently free. The ruler is not bad data
  — at warm-start all 3 classes agree exactly (scale 0.005/0.005/0.006).
- BUT run 2 is better per-object once the global offset is divided out: test abs_rel_deb catheter
  0.384 (vs 0.742) and arm 0.350 (vs 0.371); SCARED 0.195 vs 0.213. So cath+arm supervision is
  CLEANER but too sparse; the ruler is ABUNDANT but the net deforms it locally to satisfy the
  anchors. Neither run beats a single global constant on the frozen warm-start (deb 0.191).
  => next lever is `--anchor-w`, not the class mix.

## 2026-09-02 — DEPTH: ARM-ONLY calibration (the real deployment case) — use SCALE, not affine
- Deployment has only the Robot arm as an in-frame anchor. `--calib-classes 3` calibrates on the
  arm alone and scores only the other classes; `diagnose()` reports the depth spread that decides
  whether the shift b is identifiable at all.
- SPREAD (test split, sw05): the shift needs anchors at DIFFERENT depths. Measured:
  within-object (along the annotated segment) Ruler 5.6 / Catheter 1.9 / **Arm 3.2mm**;
  in-frame pooled 0.0 for every class (only ONE instance per class per frame);
  in-clip Arm 3.8mm, in-video Arm 5.8mm = **10% of the arm's 37mm working distance**.
- => The arm's 8mm annotation is the shaft DIAMETER measured ACROSS the shaft (3.2mm of depth
  spread), NOT a run along its length. The "long cylinder gives you many depths" idea is right
  physics but is NOT in the current annotations. Getting it needs the 8mm width annotated at
  SEVERAL STATIONS ALONG the shaft in the same frame.
- => The arm does move in depth between frames, but only 3.8-5.8mm. Sweeping a minimum-spread
  requirement shows extra spread does NOT rescue the affine fit: clip-level arm-only affine
  25.2% (>=3mm) -> 28.3% (>=6mm) -> 29.9% (>=10mm) while coverage collapses 44% -> 24% -> 11%.
  Every affine variant loses to scale-only. Degenerate fits show up as 1e8 mm in the MEAN column.
- ARM-ONLY RESULTS (sw05, median rel err / median mm / coverage):
  global scale 26.1% 2.22 (86%) | **video scale 21.5% 2.02 (56%)** <- best
  | clip scale 22.6% 1.99 (44%) | frame scale 23.6% 2.02 (22%)
  Per class from an arm-only calibration: Ruler ~20-23%, Catheter ~26-32%.
- COVERAGE IS THE BIGGER RISK: the arm is in only 26% of frames and 50% of CLIPS. Half the clips
  have no arm at all, so no calibration is computable there — a deployment failure, not an error.
- RECOMMENDATION: arm-only => scale-only fit, pooled over the video/clip (not per frame), budget
  ~21% median (~4.2mm on a 20mm urethra), and design for the 50% of clips with no anchor.

## 2026-09-02 — DEPTH: affine (scale+shift) calibration — LOCALITY beats the shift
- `scripts/fit_affine_scale.py`: runs a FROZEN model once, dumps per-object rays+depths to .npz,
  then fits a calibration in numpy (no scipy). Grid = {scale, affine-in-depth, affine-in-disp} x
  {global, per-video, per-clip, per-frame}. Every LOCAL fit is scored leave-one-object-out; the
  `cross-class` rows calibrate on the OTHER classes and predict this one (the deployment case).
- FIT TRICK: fixing r = b/a makes length proportional to a (depth) or 1/a (disp), so the best
  scale for any r is closed form (median of log(L/mm), L1-optimal => robust). 2-D fit -> 1-D
  search over r. `--selfcheck` plants a known calibration and asserts recovery.
- RESULTS, 5 held-out videos, 671 objects (median rel err / median abs mm):
  frozen warm-start        | trained range-sw05
  global scale   25.6% 2.01 | 20.4% 1.65
  global affine  24.8% 1.89 | 18.3% 1.56      <- shift alone buys only ~1-2 points
  per-clip scale 16.7% 1.19 | 14.1% 1.19
  per-clip affine(disp) **12.6% 1.02** | **9.3% 0.77**  <- best
  per-frame scale 26.0% (87% cov) | 22.4%     <- per-frame is WORSE, not better
  per-frame affine 43.8% (24% cov) | 44.1%    <- LOO leaves ~1 anchor; fit is noise
- CONCLUSIONS: (1) LOCALITY dominates the shift — global->per-clip roughly halves the error,
  adding the shift then takes off another 3-5 points. (2) affine in DISPARITY beats affine in
  depth at clip level (9.3 vs 10.8), consistent with EndoDAC's DepthAnything backbone being
  affine-invariant in inverse depth — SUL_10_week_plan Phase 3 writes Z = a*d + b, which is the
  weaker of the two. (3) per-clip is the sweet spot; per-frame has too few anchors to fit 2
  params (24% coverage) and even scale-only degrades. Aggregate over the clip, per the plan's
  temporal-aggregation step. (4) an under-determined affine fit extrapolates wildly — read the
  median mm column, not the mean, on the per-frame rows.
- SOBERING: CROSS-CLASS (calibrate on other object classes, predict this one) is ~22% at BOTH
  global and per-clip on sw05 — locality does NOT help when the calibration object differs from
  the measured one. The 9.3% best benefits from other instances of the SAME class in the clip.
  For SUL (calibrate on catheter, measure urethra) budget ~22%, i.e. ~4.4mm on a 20mm urethra.

## 2026-09-02 — DEPTH: --min-depth/--max-depth WAS THE BUG (metric scale solved)
- ROOT CAUSE of the first ruler run's failure: the defaults `--min-depth 0.1 --max-depth 150`.
  disp_to_depth maps sigmoid disp -> depth via min_disp=1/max_depth, max_disp=1/min_depth, so with
  those bounds the endoscopic range 35-120mm needs disp in [0.00017, 0.0022] = **0.2% of the
  network's output range**, jammed against zero. That single fact explains all three symptoms:
  warm-start "scale 0.005" (a mid-range disp ~0.36 decodes to 0.28mm), the scale loss's slow crawl
  (must drive disp to ~0.001 through a saturating region), and the class divergence (at disp~0.001
  a 0.001 step moves depth 22mm, so the catheter landed 8x nearer than the ruler).
- FIX = one flag: `--min-depth 20 --max-depth 200`. Warm-start metric scale jumps 0.005 -> **0.894**
  with NO training, abs_rel 0.995 -> 0.186. SCARED unchanged (0.054). Use these bounds for anything
  metric on mm data; the 0.1/150 defaults are a KITTI/SCARED-unit convention, not mm.
- RESULTS, 5 held-out videos, n=665, no median scaling (all 12 ep, K frozen):
  `range` (anchor .3, sw .1) scale .919 abs_rel .211 | `range-noanchor` (sw .1) .933 / .200
  | **`range-sw05` (anchor .3, sw .5) scale .974, abs_rel .205, SCARED .0557** <- best.
  Val scale 1.002. Checkpoints in outputs/depth_ruler_range{,_noanchor,_sw05}/best.pth.
- CROSS-ANCHOR CHECK NOW PASSES (SUL_10_week_plan Phase 3): sw05 per-class test scale
  Ruler 1.036 / Catheter 0.982 / Robot arm 0.892 — three independent known-size objects agree
  within ~14% on unseen surgeries. The catheter's 0.08 in the old-range runs was range compression,
  not a bad annotation.
- ANCHOR REVERSAL: with the OLD range `--anchor-w` FOUGHT `--scale-w` (anchor_loss is absolute L1
  on log-disp to the frozen teacher => it pins the global scale at the teacher's wrong value; train
  scale converged while val scale collapsed back to 0.036). With the corrected range the teacher is
  already ~metric, so anchor+scale are complementary: sw05 trains stably to ep10, noanchor peaks at
  ep1 then drifts (final val scale 0.797). Keep --anchor-w 0.3.
- REMAINING CEILING: abs_rel ~= abs_rel_deb (.205 vs .205) => the residual ~20% is NOT a global
  offset any calibration can remove, it is genuine per-object error. Scale is solved, accuracy is
  not. Uncalibrated f is worth ~3% of that, not 20%. Also watch sw05's test/photo (0.158 vs ~0.049
  elsewhere) while its SCARED depth score is fine -> the pose/refine nets took the hit, not depth.

## 2026-09-02 — DEPTH: metric scale loss from scale_objects.json (+ first result)
- DATA: `../data/processed/depthclips_ruler_NoGUI/NOgui/<video>/clip_NNN.mp4/` (note: clip dir is
  named `*.mp4`) holds `images/` + `masks/` + `scale_objects.json` + `source_crop.json`. 1677
  frames / 65 clips / 18 videos, 89% of frames annotated. Objects = straight segments with known
  mm: class 1 Ruler (annotator-typed 10-20mm), 2 Catheter tip (16Fr/3 = 5.333mm), 3 Robot arm
  (8mm), each with `a`,`b` + 5 collinear `points` in 1340x1072 px (already cropped from 1920x1080
  = the pillarbox removal, so K = the base 4:5 content K, all crop fracs 0).
- LOSS `scale_loss` (finetune_depth.py): sample depth at the annotated points, back-project with
  inv_K, compare polyline 3D length to the true mm as a Huber log-ratio, conf-weighted. Log space
  = dimensionless (one `--scale-w` for all classes) and multiplicative on the global scale.
  Do NOT use the json's `mm_per_px`: it is Z/f, depth-dependent, only valid at its own object.
- `split_by_video` splits by SURGERY not clip; `eval_metric_scale` reports per class
  `scale` (median pred/true length; 1.0 = metric), `abs_rel`, and `abs_rel_deb` (after dividing
  out the global median = how much is ONE constant vs real shape error). Metric outranks SCARED
  for checkpoint selection (SCARED is median-scaled => blind to scale).
- PRE-FLIGHT check that the geometry is sound: `Z = f*S/p` from the annotations alone gives
  median 55mm (ruler) / 61mm (catheter) / 45mm (arm) — all in endoscopic range, arm nearest.
- FIRST RESULT (`endodac-ruler-scale`, jiui7izm, 4 ep, K frozen, scale-w 0.1): warm-start EndoDAC
  is off by ONE constant — all 3 classes scale 0.005-0.006, debiased err 0.09-0.26, SCARED
  abs_rel 0.057. Training raises scale 0.005 -> 0.605 monotonically (loss still falling, not
  converged) BUT the classes DIVERGE (arm 0.855 / ruler 0.449 / catheter 0.365) and ruler
  debiased err grows 0.26 -> 0.92, SCARED 0.057 -> 0.213. The scale term pins depth only at
  annotated pixels, so the net satisfies it by locally deforming the map instead of rescaling it.
  => a single global constant on the FROZEN warm-start beats 4 epochs of this. Next: `--anchor-w`
  (frozen-teacher log-disp) to hold geometry while only the scale moves.

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


## 2026-09-02 - depth measurement GUI (`scripts/gui_depth_measure.py`)

Click two points on a frame, get the distance in mm. Local Windows app (Tkinter, so nothing to
install beyond torch/PIL/matplotlib/opencv/fvcore), CPU inference, no Snellius involvement.
Defaults are pinned to the run that made depth metric: `outputs/depth_ruler_range_sw05/best.pth`,
`--image-shape 392 490`, `--min-depth 20 --max-depth 200`, K = the SCARED assumed
(0.82, 1.02, 0.5, 0.5). Nick has no calibrated da Vinci intrinsics, so the assumed K stands.

**The measurement is the same maths as the scale loss that trained the model.** `segment_length`
back-projects both clicked pixels with normalised K -- X = (u/W - cx)z/fx, Y = (v/H - cy)z/fy --
and takes the norm, exactly `scale_loss`'s per-segment length with N=2. `--self-test` asserts that
against a fronto-parallel plane the same way `_selfcheck_scale_loss` does. Consequence worth
keeping: everything is in NORMALISED coordinates, so display resolution, crop size and the depth
map's own resolution are all decoupled.

**The DPT head does not return the feed resolution.** A 392x490 feed comes back as a 448x560 disp.
Aspect is preserved, so normalised sampling stays exact and the GUI keeps the native map instead of
resizing it (training interpolated to the round32 `hw`; either is fine, this one loses less).

**Framing is the failure mode to watch, not the model.** The assumed K describes the 5:4 ruler-dump
view. Hand the model a raw 16:9 console frame and both the depth and the mm are wrong, silently.
So the GUI auto-detects the pillarbox on load (`auto_bars`, near-black border rows/cols), exposes
the three crop fracs, and shows the resulting aspect in orange unless it is within 0.08 of 1.25.

**Scale honesty.** Held-out test scale ratio was 0.974, per-class 1.04/0.98/0.89 -- so a number
here is mm +-a few percent, and the per-object spread is wider than the global scale error. The
`scale x` box and *Calibrate from selected...* (type the true length of a measured known object ->
solves the multiplier, rescales everything) are how you do better on a specific scene. Depth is
sampled as a patch median (default r=2) because a single sample on a thin instrument edge can land
on the discontinuity and be off by centimetres.

**Tests.** `scripts/tests/smoke_gui_depth_measure.py` drives the real Tk window under Xvfb
(26 checks: auto-crop, click->measure, scale/patch/calibrate, undo/clear, CSV+PNG export) with the
dialogs stubbed, since a modal would deadlock the mainloop headless; `run_gui(args, on_ready=...)`
exists only as that hook. `scripts/tests/smoke_depth_backend.py` builds EndoDAC, saves a
random-weight checkpoint and runs it through `DepthBackend` -- proves the 389-key contract, the
patch grid at 392x490 and the 20/200 band without needing the trained weights.

`DepthBackend` imports `make_endodac`/`disp_to_depth` from `finetune_depth` rather than copying
them (the vit_base `input_size` monkeypatch is easy to get subtly wrong), and shims a no-op
`dotenv` if it is missing so measuring does not depend on the wandb creds path.

### Depth cache + checkpoint picker (same day)

Depth is a deterministic function of (checkpoint, image, crop, feed shape, depth band), so it is
computed at most once and written to `outputs/depth_cache/<run>_<ckpt>_<fingerprint>/<frame>_<img>_<crop>.npz`
(float32 mm under key `depth`, `meta.json` beside it). Three tiers, because they cost three
different things: an 8-entry in-memory LRU (instant, for flipping prev/next), the cache directory
(~10 ms, survives restarts), the model (seconds on CPU). `DepthProvider.depth_for` is the only
entry point the GUI uses and reports which tier answered.

**Keyed by checkpoint CONTENTS, not path or mtime.** `file_fingerprint` is a blake2b of the file,
so re-scp-ing the same `best.pth` lands back in the folder that already holds its maps, while a
genuinely different checkpoint can never serve a stale map - which is the whole reason the
directory is per-checkpoint rather than per-run-name. Choosing a checkpoint in the dropdown
(populated from `outputs/*/*.pth`) switches cache folders with it, so A/B-ing two runs on the same
frames costs one prediction each and nothing thereafter.

**The filename splits into an image part and a crop part on purpose.** "Is this frame cached at
all" is then a dict lookup, so the file list can mark 2000 frames without opening any of them to
work out what their auto-crop would be. Exactness is not lost: a click still looks up the exact
(image, crop) pair, and the mark is accurate in practice because `auto_bars` is deterministic.
Nothing is ever invalidated - the folder is a permanent dump other scripts can read.

`--self-test` now covers the cache without torch (round-trip, crop collisions, per-checkpoint
directories, fingerprint stability across mtime, LRU eviction, and a counting backend proving the
model runs at most once per image+crop); the Xvfb suite is up to 38 checks, including precompute,
the `*` marks, and returning to a checkpoint's folder with its maps intact.

### Build stamp + scrolling control column

Two bug reports on the same day ("I don't see the new buttons", "still slow") had one likely cause:
an old copy running. There was no way to tell by looking, so `BUILD` is now printed at start-up,
shown in the title bar and available as `--version` -- bump it on every user-visible change.

The control column also scrolls now (canvas + inner frame). Tk clips a too-tall frame silently, so
on a laptop screen whichever section did not fit simply vanished, which is indistinguishable from
the app being out of date. Sections can be added freely from here.

### Timing in the status line, and npz compression dropped (same day)

Follow-up report: "still slow" even on build 3 with the cache in place. Suspicion was the .npz
format. Benchmarked before touching anything (448x560 float32 map, the actual size the DPT head
returns): `savez_compressed` 50 ms/write, plain `savez` <1 ms/write, `np.load` read 6 ms either
way. All three are noise next to a multi-second CPU forward pass -- npz was never the bottleneck.
Dropped the compression anyway (no reason to pay 50ms x thousands of frames on a big precompute
run for ~11% smaller files), but the real fix is diagnostic, not speed: `DepthProvider.depth_for`
now returns elapsed time alongside the tier, and the status line shows both, e.g.
`frame_0142.jpg [cache 6 ms]` vs `[model 2400 ms]`. "Slow" no longer needs a guess about which
stage -- the line says model vs disk vs something else in the click/redraw path.

BUILD -> 4.

### 4-annotator arch comparison: Vivian marked a different structure (2026-09-11)

`scripts/compare_arch_4way.py` adds Vivian to Nick/Veerle/Aron. Vivian did not track the
comparison videos: her `JSONfileVivian/SUL_img3x/arches.json` is keyed by index into the sorted
234-image SUL_img3x still set (same as `transfer_atlas_mod/workspace/SUL_img3x/arches.json`).
Ten of those stills come from the comparison videos; `JSONfileVivian/SUL_img3x/frame_map.json`
(new, hand-written) maps key -> (video, frame), found by pixel-matching each still against the
mp4 (MAD ~3 vs ~5 runner-up). Her coords are full-frame; shifted into Nick's crop by
source_crop (289, 4). Veerle/Aron keep the per-video mean offset vs Nick from compare_arch_multi.

Result: Nick/Veerle/Aron agree to 35-65 px mean tip distance at these frames, Vivian is ~460 px
(~100% of chord) from all three. Overlay confirms the mapping is right but her arch sits on the
catheter tip / urethral opening (~150 px chord), not the Retzius arch (~900 px chord). Her numbers
measure a different target, not annotator disagreement. Her PNGs/CSVs were briefly deleted, then
restored on request from the identical workspace copies.

### `--smoke` tilt self-check: the 1.2x margin never held (2026-09-14)

`_selfcheck_scale_loss` asserted 3D ratio > 1.2 x in-plane ratio for the depth ramp added in
8153a28. For that fixed geometry (Z=50 -> 60 over u=30..50, fx=65.6) both are closed-form:
3D = hypot(dX, dZ)/mm = 1.2808, in-plane = 55/50 = 1.1000, so 1.164x -- it failed from the commit
that introduced it. Same numbers on Snellius torch 2.12+cu130 and local 2.14+cpu; neither
`scale_loss` nor the geometry changed since. Now asserts the exact 3D value (1e-3) and
3D > in-plane, like the in-plane line already did. No loss code touched.

### DEPTH: sharpest-clip-per-patient set vs ruler clips, judged on proxy-GT SHAPE (2026-09-14)

Question (Nick): does sharp, patient-diverse data make mono depth converge to the stereo proxy-GT
better than the ruler clips? Those clips have no known-size objects -> `--scale-w 0` -> RELATIVE
depth, so only `proxy_gt/ms_abs_rel` (per-frame median-scaled) is comparable, never `abs_rel`.
- `scripts/prep_sharpest_clips.py` (+ `jobs/prep_sharpest_clips.sh`, genoa): ranks every clip
  > 1 MB in `/home/nsmit2/data/depth_clips_staging` (1069 of 1121) by rank_sharpness `sharp`
  (var of Laplacian), keeps the top clip per PATIENT (uuid / RARP_NNN prefix; 77 patients), crops
  to 5:4 content, split 5 val / 5 test / rest train. Output `../data/processed/depthclips_sharpest`,
  ranking in its `sharpness_rank.csv`. The only earlier ranking
  (`data/depthclips_ruler_NoGUI/sharpness_rank.csv`) covers the 64 ruler clips, not staging.
- GUI masks: first version = pixels black in every frame (templates were missing on Snellius).
  Templates restored the same day -> `--masks-only` rewrote them with `cut_cue_clips.full_gui_mask`
  on the SOURCE frames (`/home/nsmit2/data/UMCdissectionvidNOgui/<video>.mp4`, frames S.. from the
  clip name). Staging frames are blacked so templates can't match there; the NOgui sources only
  lack the fixed HUD (5.1% black vs staging 5.9-6.3%), which full_gui_mask draws from geometry,
  while cue bars/popups are still visible. Logged `covers_black` = share of near-black jpg px the
  template mask covers. finetune_depth does not read masks today.
- Template-only pass (job 26690902): covers_black median 1.000 but min .668 -- on 5/77 clips the
  miss is ONE solid 334-474 px rectangle at rows ~900-1017 (a popup / cue panel the source-frame
  match misses but cut-time masking blacked). Final mask = template | black-in-every-frame
  (dilated 3). Tissue is never black in all frames of a clip, so this is cut-time GUI, not black.
- RESULT (job 26690383, wandb 7zfhhion, 3 ep, scale-w 0, anchor-w 0.3): proxy_gt ms_abs_rel
  .1660 / .1672 / .1660 / .1659 vs ruler .1660 / .1640 / .1652 / .1656. Neither converges; train
  photo fell .073 -> .032 (val .031, ~3x below ruler) while shape stayed put. Suspect anchor-w 0.3
  (L1 to the frozen warm-start) pins the shape; next A/B = same data, --anchor-w 0 / 0.1.
- A/B done (3 ep each, same data). proxy_gt ms_abs_rel ep0..3 / SCARED abs_rel at ep3:
  anchor 0.3 (26690383)  .1660 .1672 .1660 .1659 / .0542
  anchor 0.1 (26691199)  .1660 .1670 .1661 .1660 / .0542
  anchor 0   (26691198)  .1660 .2477 .2477 .2477 / .1270  -- COLLAPSED: ep1-3 identical to 4
  decimals, proxy_scale 1.03 -> 2.11, so the depth froze (saturated/constant) while train photo
  still fell .071 -> .031. best.pth = warm start.
  => the anchor is NOT what holds the shape: at 0.1 it is as flat as 0.3; at 0 it breaks. Photometric
  self-supervision on these clips does not move mono shape toward stereo in 3 epochs at all.
- Mask side-thread: coverage stayed min .676 after OR-ing black-in-every-frame, because the missed
  panels are TRANSIENT (15/28, 19/21, 13/15 frames, fixed position, box 77-86% black). bands=True
  fixes 2 of 3 but triples the mask (5.4% -> 16.7%), so not adopted. Masks are unused by training.
- FRAME STRIDE (anchor 0.3, 3 ep): proxy_gt ms_abs_rel ep1..3 / pose_trans ep3 / train,val photo ep3:
  stride 1 (26690383)  .1672 .1660 .1659 / .0011 / .032 .031
  stride 4 (26692024)  .1668 .1661 .1660 / .0054 / .153 .144
  stride 8 (26692025)  .1667 .1660 .1661 / .0039 / .205 .223
  More parallax is real (pose_trans 5x) but shape still does not move; photo loss stays 5-7x higher,
  i.e. the wider pairs are not explained by a rigid warp. Every sharpest-set run's best.pth = warm start.
- OPEN, raised by Nick: da Vinci DIGITAL ZOOM 2x/4x in some clips = focal x2/x4 while K is fixed
  (--no-learn-intrinsics). Breaks the rotation warp (shape error), makes mixed-zoom batches
  contradictory, and for the ruler/SUL work biases z_true = fx*mm/px by the zoom factor. Not yet
  measured how many clips are zoomed; options: detect + scale K, 1x-only subset, or learned K.
- ZOOM READ (by eye) for the 77 selected clips: the label sits right of the scope-angle readout in
  the bottom HUD, full 1920x1080 frame ~x1212-1262 y1034-1066 (`1x 0°`), only visible in the with-GUI
  originals `/home/nsmit2/data/UMCdissectionvid/<video>.mp4`. First/middle/last frame agree in every
  clip. 68 x 1x, 1 x 2x (4d8eca93), 8 x 4x (RARP_069, RARP_077, RARP_083, RARP_086, RARP_089,
  349725a5, 7d96d613, d7419222). 8 zoomed clips are in TRAIN, RARP_083 (4x) is in TEST; val is
  all 1x. 1x-only set = `../data/processed/depthclips_sharpest_1x` (symlinks, 59 train / 5 val /
  4 test clips), run job 26692937 -> outputs/depth_sharpest_1x, otherwise identical to 26690383.
  RESULT ep1..3: ms_abs_rel .1655 .1660 .1660 (best .1655, ep1) vs all-zoom .1672 .1660 .1659 --
  shape still flat. But the dynamics changed: pose_trans stays .0034-.0037 (all-zoom run collapsed
  to .0011) and photo loss no longer drops to ~.03 (train .079 -> .062, val .069 -> .051 on the SAME
  1x val clips). Reading: with zoomed clips in the batch the model escaped to ~zero camera motion,
  where depth does not affect the warp; without them it keeps moving the camera but 3 epochs don't
  move the shape. Candidate next: same 1x set, more epochs.
  12 EPOCHS (job 26696249, wandb umz8y8lp, 20 min): pose_trans .0034 -> .0009 by ep3 -> .0006-.0008
  after; val photo .070 -> .016 (ep3) -> .008 plateau from ep8. ms_abs_rel stays in .1653-.1663
  (warm .1660; best .1653 ep5, back to .1660 at ep12); SCARED .0539 -> .0554 at best.pth (worse).
  => the zero-motion escape is NOT a zoom artifact: 1x clips reach it too, 3 epochs later (the
  3-ep run's short cosine schedule stopped it early). Once pose ~0 the photometric loss has no
  depth gradient, so shape only random-walks within +-0.3% of the warm start. Photometric
  self-supervision on these short 17 ms-stride clips cannot move this model's shape.

### DEPTH: little-but-good vs lots-of-data (2026-09-15)
- Nick hand-picked camera-motion clips: local `data/depthclips_manual` -> Snellius
  `~/data/depthclips_manual_raw` (31 mp4, 840 frames, 11-38 frames each, 9 patients, none in
  staging / sharpest_1x val-test / proxy GT). Names = `<source video>-<t0>-<t1>[-segN].mp4`; sources
  are NOT on Snellius, but the GUI is still on screen, so `scripts/prep_manual_clips.py` runs
  full_gui_mask on each clip's own frames (black + `<i>_mask.png`).
- ZOOM DETECTOR `scripts/zoomdet.py` (templates `data/templates/zoom/*.npy`): label crop x1212-1262
  y1034-1066, high-pass, averaged over the clip (label fixed, tissue moves), NCC vs 1x/2x/4x class
  means from the 77 hand labels. Leave-one-out on those 77: 1x vs zoomed 100% (1x margin >= +.187,
  zoomed <= -.045); 2x/4x confusion only (one 2x example). Staging >1.5 MB: 796 1x / 22 2x / 64 4x,
  NO clip within +-0.15 of the boundary. Manual: 29 1x, 2 x 4x (06c50176) -> dropped.
- Both new sets share Validation/Test with depthclips_sharpest_1x (symlinks) so val numbers compare:
  `depthclips_manual` (manual, 1x) and `depthclips_all15_1x` (`prep_sharpest_clips.py --all --min-mb
  1.5 --zoom-csv`, every 1x staging clip, val/test patients excluded from Train). Runs: same as
  26696249 (12 ep, anchor .3, scale-w 0, stride 1).
- RESULTS. manual (26729785, 29 clips / 791 fr): pose_trans .005 -> .010 over ep1-5 (no collapse),
  ms_abs_rel best .1649 @ep5 (SCARED .0550 at best.pth vs warm .0539), then COLLAPSE at ep6
  (pose .0104 -> .0034 -> .0015, val photo .056 -> .019) and ms back to .1660. all15_1x (26729787,
  635 clips / 16622 fr / 59 patients): pose_trans EXPLODED .0137 -> .259 -> .615 (ep1-3) with
  proxy_scale fixed, ms .1650 flat, SCARED .0538 -> .0560 by ep4 -> CANCELLED after ep4 at Nick's
  request (best.pth = ep2 .1650). Neither little-good nor lots-of-data escapes the pose degeneracy:
  the pose net is the unconstrained variable (collapse on short clips, blow-up on big data).
- NEXT (Nick 2026-09-15): zero-shot benchmark on the proxy-GT before any more training --
  `scripts/zeroshot_proxy_gt.py` / `jobs/zeroshot_proxy_gt.sh`. transformers 5.17 is installed with
  `pip --target ~/pylibs/bench --no-deps` (NOT in the venv, hub 1.19 pinned there for FoundationStereo);
  weights pre-downloaded to ~/.cache/huggingface on the login node (DAv2-L 1.25 GB, DAv2 metric
  indoor L 1.25 GB, DepthPro 1.77 GB), job runs HF_HUB_OFFLINE=1.
- ZERO-SHOT RESULT (job 26731682, 2 min, outputs/zeroshot_proxy_gt/{results.json,per_frame.csv,grid.png};
  self-check ms_abs_rel == run_proxy_gt_eval 0.1660). ms = median-scaled depth (training metric),
  ssi = scale+shift in inverse depth; d = ssi vs warm-start, bootstrap CI over frames:
    model                     ms_abs_rel ms_a1 | ssi_abs_rel ssi_a1 | d_ssi [95% CI]     | per patient ssi 18de9c5a / 5e27066c
    EndoDAC warm-start          .1660   .724  |   .1527    .763   |                    | .156 / .147
    ft manual ep5               .1649   .728  |   .1519    .765   | -.0008 [-.001,-.0005]| .155 / .147
    ft sharpest1x ep5           .1653   .726  |   .1525    .763   | -.0002             | .155 / .147
    ft ruler sw05               .1666   .722  |   .1536    .761   | +.0009             | .156 / .149
    Depth Anything V2 L (rel)   .2998*  .548  |   .1059    .889   | -.0467 [-.057,-.037] | .119 / .082  (71/83 frames better)
    DAv2 Metric Indoor L        .1306   .819  |   .1181    .854   | -.0345 [-.043,-.026] | .127 / .103  (68/83)
    Depth Pro                   .1606   .757  |   .1459    .786   | -.0068 [-.021,+.007] | .160 / .121  (51/83)
  * DAv2-rel outputs affine-invariant DISPARITY: median scaling can't remove its shift, use ssi.
  => The proxy-GT is NOT at a floor: DAv2-L is 31% better on shape than EndoDAC, on BOTH patients.
  All EndoDAC fine-tunes sit within 1% of the warm start. grid.png: EndoDAC maps are blurry with a
  radial "bowl" (near at the rim, far in the centre) that follows the image, not the anatomy; DAv2
  reproduces the stereo GT's structures (tissue ridges, catheter tip) with sharp edges. CIs treat
  83 frames as independent but they come from 2 patients -> optimistic.
- ZERO-SHOT ROUND 2 (Nick's list, image models only + EndoUFM): `zeroshot_proxy_gt.py --all` runs each
  model in its own subprocess (`--model NAME`, per-model PYTHONPATH) then `--summarize`, because the
  packages clash (DA3 numpy<2 vs MoGe/UniDepth numpy>=2; EndoUFM's top-level `networks`/`utils`).
  Code, all `pip --target --no-deps` + only the modules the import actually missed (scratchpad
  autodeps loop), NEVER in the venv: ~/pylibs/{bench (transformers 5.17), da3 (depth-anything-3
  0.1.1 + moviepy==1.0.3, addict, plyfile, pycolmap, trimesh, evo), moge (MoGe git, utils3d-moge git),
  unidepth (git), metric3d (mmengine, yapf, addict) + metric3d_mmcvstub (mmcv.utils.collect_env stub:
  Metric3D imports it for env LOGGING only; real mmcv 1.7.2 does not build), endoufm_deps, EndoUFM (git
  clone)}. Weights: HF cache (DA3-LARGE/GIANT/METRIC-LARGE, moge-2-vitl, moge-vitl,
  unidepth-v2-vitl14), ~/.cache/torch/hub/checkpoints (Metric3D vit_large/giant2), repo code
  ~/.cache/torch/hub/yvanyin_metric3d_main (loaded source='local'), ~/backbones/EndoUFM/depth_model.pth
  (Google Drive id in its README). EndoOmni: NOT testable -- TianCuteQY/EndoOmni holds only a
  2-line README, no code/weights, nothing on HF. EndoUFM rvlora: random_1/2 are built but unused in
  RVLinear.forward, so loading is deterministic. Home quota ~170/200 GB after this.
- ROUND 2 RESULT (job 26733667, 7 min, all 16 ran, outputs/zeroshot_proxy_gt_v2/; round-1 models
  reproduce exactly). Sorted by ssi_abs_rel; d = ssi vs EndoDAC warm start [bootstrap 95% CI]:
    model                        ms_abs_rel  ssi_abs_rel ssi_a1  d_ssi [95% CI]        per patient   better
    UniDepthV2 ViT-L               .1187      .0991     .906   -.054 [-.063,-.044]   .106 / .087   74/83
    Depth Anything V2 L (rel)      .2998*     .1059     .889   -.047 [-.057,-.037]   .119 / .082   71/83
    MoGe-2 ViT-L                   .1638      .1120     .881   -.041 [-.050,-.032]   .122 / .095   68/83
    MoGe ViT-L                     .1413      .1148     .880   -.038 [-.049,-.027]   .122 / .103   64/83
    Metric3D v2 ViT-giant2         .1275      .1174     .878   -.035 [-.045,-.025]   .123 / .107   67/83
    DAv2 Metric Indoor L           .1306      .1181     .854   -.035 [-.043,-.026]   .127 / .103   68/83
    DA3 Metric-Large               .1673      .1335     .825   -.019 [-.029,-.009]   .137 / .127   55/83
    EndoUFM                        .1516      .1403     .789   -.012 [-.017,-.008]   .146 / .130   50/83
    Depth Pro                      .1606      .1459     .786   -.007 [-.021,+.008]   .160 / .121   51/83
    DA3 Giant                      .1937      .1500     .769   -.003 [-.014,+.009]   .170 / .115   39/83
    EndoDAC ft manual ep5          .1649      .1519     .765   -.001                 .155 / .147   70/83
    EndoDAC warm start             .1660      .1527     .762    0                    .156 / .147
    Metric3D v2 ViT-L              .1705      .1548     .763   +.002 [-.010,+.015]   .168 / .131   37/83
    DA3 Large                      .1861      .1585     .742   +.006 [-.006,+.018]   .176 / .128   34/83
  => UniDepthV2 is best on BOTH metrics (ms .119 = 28% below EndoDAC, ssi .099 = 35% below) and on
  both patients. Six general models beat EndoDAC by 23-35% with CIs far from 0. DA3 (all sizes) is
  mediocre here although its processor keeps the full frame (upper_bound_resize = longest side 504,
  each side resized to a multiple of 14, no crop/pad) -> not an alignment bug. Metric3D needs the
  giant2 backbone (ViT-L = EndoDAC level). EndoUFM is the best SCARED-trained model (+8%) but far
  behind the general ones. CIs treat 83 frames from 2 patients as independent -> optimistic.
- grid.png checks. Top 7 all reproduce the stereo ridge/fold/instrument/catheter tip; UniDepthV2
  cleanest, DA3-Metric-L patchy. Metric3D ViT-L shows a strong checkerboard -- NOT a load bug:
  checkpoint vs model = only `encoder.mask_token` missing (pretraining-only; giant2 also lacks the
  final encoder norm and still scores well), so its low rank is genuine. DA3's mediocre score is ALSO
  genuine: model/da3.py has _process_mono_sky_estimation (sky prob >= 0.3 -> 99th-percentile depth),
  and DA3-L frame 1 puts the yellow fat at the top FAR, so I suspected it. Tested with the step
  patched out (job 26734065, --only): identical to 4 decimals for L / Giant / Metric-L. Reason: none
  of the three checkpoints has a sky head (configs build DepthAnything3Net with DualDPT / plain DPT,
  no sky option), so the step returns at `if "sky" not in output`. The nosky variants were removed
  again; `--only` stays for running a subset into an existing --out.

### DEPTH: absolute mm after ONE ruler calibration, tested on the proxy-GT (2026-09-15)
- Nick: 6 models (EndoDAC warm start, UniDepthV2-L, DAv2-L, MoGe-2-L, Metric3D-g2, DAv2 Metric
  Indoor-L), calibrate scale+shift on the ruler set, test ABSOLUTE depth on the proxy-GT (not on
  a ruler hold-out). `scripts/metric_calib_proxy.py` / `jobs/metric_calib_proxy.sh`, job 26735955,
  out `outputs/metric_calib_proxy/`. Per object: z_true = mm / in-plane ray length (fixed K 0.82/1.02
  on 1340x1072), z_pred = mean predicted depth over its 5 points; robust (soft-l1, log) fit of
  1/z = s/z_pred + b, plus scale-only z_pred/s'. Frozen, applied to every proxy frame, no per-frame
  scaling. Also per-class fit residual and distance-tracking slope on the ruler objects.
- ZOOM, both sets. Ruler source videos (`~/data/UMCrulervid/<video>.mp4`, same names as the dump
  dirs; clips carry no source-frame index, but zoom was constant over 5 windows x 8 frames of each
  whole video): 15 x 1x, 349725a5 4x, 7d96d613 4x, 4d8eca93 2x -> passed as --exclude-videos
  (2002 of 2317 objects, 1249 frames remain). Proxy-GT: the `*.jpg` next to depth16 are viz panels
  (no GUI); raw SBS sources = `~/data/3D_ProxyGT/*.mp4` + `ruler/*.mp4` (5 clips). In SBS each eye is
  squeezed 2x, HUD label at full-frame y~995-1045, x~575-690 (left eye); read by eye on 6 frames
  per clip (30 crops): all `1x 0°`. zoomdet templates would need horizontal stretching for SBS.
- scale_objects.json facts: focal_px is null (so the fixed K is used), classes 1 Ruler (typed mm),
  2 Catheter tip 5.333, 3 Robot arm 8; sources manual 251 / tracked 1649 / measured 416 (arm) /
  hold 1; frame keys index the sorted images of the clip.
- RESULT (job 26735955, 16 min, all 6 ran; results.json in outputs/metric_calib_proxy/):
    PROXY-GT ABSOLUTE after one ruler calibration   abs_rel  rmse   a1   | scale-only abs_rel | d vs EndoDAC [95% CI]
    Metric3D v2 ViT-giant2                           .189  10.1mm .679 |  .191  | -.263 [-.302,-.223]
    UniDepthV2 ViT-L                                 .218  11.2mm .633 |  .222  | -.234 [-.277,-.192]
    DAv2 Metric Indoor L                             .278  12.8mm .549 |  .265  | -.174 [-.217,-.131]
    DAv2 L (relative)                                .300  15.5mm .452 |  .405  | -.152 [-.173,-.130]
    EndoDAC warm start                               .452  22.2mm .309 |  .403  |
    MoGe-2 ViT-L                                     .528  20.7mm .301 |  .580  | +.076 [+.039,+.114]
  RULER fit residual (median |ratio-1|) all/ruler/cath/arm + tracking slope (affine | scale-only):
    Metric3D-g2 .136/.124/.148/.182 .61|.50   UniDepthV2 .184/.187/.194/.155 .28|.41
    DAv2-MI .178/.177/.158/.218 .51|.65   DAv2-L .167/.160/.174/.187 .54|1.14
    EndoDAC .189/.191/.233/.145 .36|.17   MoGe-2 .193/.179/.277/.175 .29|.44
  => THE RANKING CHANGES with absolute scale: Metric3D-g2 (5th on per-frame shape) is best absolute;
  UniDepthV2 (1st on shape) 2nd; MoGe-2 (3rd on shape) falls BELOW EndoDAC -- its depth level jumps
  between scenes, so a frozen calibration misplaces the proxy frames. Metric models barely need the
  shift (scale-only within .004, DAv2-MI even better without); relative DAv2-L needs it (.405 -> .300).
  EndoDAC's negative shift hurt it (scale-only .403 < affine .452). Absolute error is ~2x the
  per-frame-scaled error for every model: that gap = scale inconsistency + ruler-vs-proxy scene
  difference. Tracking slopes are noisy (in-plane z_true is cos-biased, both axes noisy) -- read
  them as coarse, not as a ranking. Crops:
  ~/zoomcrop/zoom_grid_{a,b}.png on Snellius. Pixel MAD vs a 1x crop does NOT work as a detector
  (the label is translucent over tissue); a template/OCR detector would need 2x/4x glyph crops.
  Ruler-run split had 4d8eca93 + 7d96d613 as VALIDATION videos (other segments; zoom there unchecked).
- `finetune_depth.py`: `--scale-w 0` now selects best.pth on proxy_gt `ms_abs_rel`; epoch line
  prints `proxy_ms_abs_rel`. Overlap check: proxy-GT = 2 patients, in neither staging nor ruler.
- Baseline, `endodac-ruler-range-sw05-3ep` (job 26688975, wandb r3p74r8y), proxy_gt ms_abs_rel
  ep0..3 = .1660 / .1640 / .1652 / .1656 -- flat. Run = `jobs/finetune_depth_sharpest.sh`
  (sw05 settings, scale-w 0, 3 ep).

## 2026-09-15 — seg encoder variant: SurgeNet teacher is ReLU, seg runs built StarReLU
- `caformer_s18(pretrained="ImageNet")` = StarReLU (learnable scale+bias per act);
  `pretrained="SurgeNet"` = plain ReLU. The RARP teacher is the ReLU one. Loaded on Snellius:
  ImageNet variant -> missing 54 = 48 encoder (`stages.*.token_mixer.act1.{scale,bias}`,
  `stages.*.mlp.act.{scale,bias}`) + 6 `head.*`; SurgeNet variant -> missing 6 (`head.*` only,
  unused by the FPN), unexpected 0 for both. Every seg log so far says `missing=54` -> ALL existing
  seg models (rarp_tversky*, rarp_nick_*, ureth_*, bestseg, ...) started with StarReLU at default
  init (s=1, b=0 -> s*relu(x)^2, not relu(x)) instead of the pretrained network. Encoder was trainable, so
  they adapted, but the init was not the teacher.
- Fix: `metaformer.variant_for(sd)` ("ImageNet" iff `*mlp.act.scale` in sd). The seg fine-tune
  scripts build `pretrained="SurgeNet"`; `load_encoder` prints `(encoder N)` missing and asserts
  N == 0 unless the model is deliberately StarReLU. `finetune_seg_tversky.py --variant ImageNet`
  reproduces the old init (A/B only); its final/`--eval-only` eval rebuilds from the checkpoint's
  variant. Every script that loads a seg best.pth (overlay_dir, overlay_masks, run_rarp_seg,
  eval_urethra, slide_dice_examples, eval_sul_methods, urethra_cylinder, arch_tip_features tools)
  builds `pretrained=variant_for(sd)`, so old StarReLU and new ReLU checkpoints both load strict.
- A/B = `jobs/finetune_tversky_dice.sh` settings (keep 1,2,3,4, lr 1e-4, 50 ep, a=b=0.5, bg-in-loss)
  + `--data-root ../data/RARPSurgenet` (the job's `fold1` path no longer exists: flattened on Aug 3,
  378/97/60 frames) x `--variant {SurgeNet,ImageNet}` x seed {42,1,2}. Same fixed Test split.
  Outputs `outputs/ab_variant_{relu,starrelu}_s{seed}`, wandb `ab-dice-*`, jobs 26760474-80.
    test         seed42  seed1   seed2   mean     (val dice mean)
    ReLU dice    .8237   .8253   .8240   .8243    (.7893)
    StarReLU     .8129   .8080   .8104   .8104    (.7776)
    d dice      +.0108  +.0173  +.0136  +.0139
  mean mIoU .7125 vs .6931 (+.019); catheter .8544 vs .8348; urethra .8325 vs .8163; prostate
  .8047 vs .7896; apicalvesicle .6657 vs .6503 -- ReLU better on every seed AND every class.
  StarReLU s42 = .8129 reproduces the old rarp_tversky_dice (.8135) -> same data, bug is the only change.
  ReLU runs are also ~1 min faster (5:10 vs 6:30). Caveat: 60 test frames, 3 seeds, same frames
  per seed -> no CI, but the sign is consistent. => every existing seg model (incl. ureth_*,
  rarp_nick_*, bestseg) is worth a retrain with the fixed init; expect ~+1-2 dice points.
