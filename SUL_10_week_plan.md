# SUL estimation — 10-week implementation plan (v2)

Estimating Surgical Urethral Length (SUL) from **monocular** RARP video, on a dataset where every clip has **camera motion** and a visible **robot instrument** and/or **catheter** as the metric reference. A separate **ruler test set** of manual stump measurements is the only gold ground truth and stays walled off until final evaluation.

> **What changed in v2.** Folded in the preresearch. The decisive change: the clinical SUL (the ruler ground truth) is measured **after** the cut, on the urethral stump, where the **catheter is in frame** — so scale for the accuracy model is recovered directly per frame and the anchorless machinery is no longer on the critical path. The "optimally accurate" model is now a **confidence-weighted depth ensemble** (your Option 2) rather than a reconstruction-first approach, measuring a **chord between two endpoints**. Real-time is kept but explicitly **secondary**.

## Framing: what this project is actually for

The weak version of this project ("an AI that tells surgeons to cut low") is not worth doing — surgeons already know that. The strong version, supported by your clinician survey and references, is **automated, objective SUL measurement at scale**: replacing manual ruler studies so that technique → SUL → continence can be correlated across many surgeons and hospitals. That makes the **accurate retrospective measurement tool the primary deliverable**, and real-time intra-operative use a secondary, harder-to-justify extension.

## Two deliverables

- **Accuracy model (primary, "gold").** A confidence-weighted ensemble of endoscopic depth estimators, catheter-calibrated, measuring SUL on the post-cut stump. Maximised for accuracy and **uncertainty-awareness**, evaluated against the ruler test set, used only to output a SUL number with a confidence flag. Also the **teacher** that emits metric pseudo-labels.
- **Real-time model (secondary, deployed).** A single lightweight per-frame model **distilled** from the gold model, running on video at interactive frame rate with a live SUL + confidence readout.

## The three "scales" (resolves your per-video vs once question)

- **Intrinsics** (focal length, principal point, distortion): fixed by hardware → calibrate **once**, reuse across all videos.
- **Metric scale** (units → mm): set by camera-to-tissue distance, which changes every frame → recovered **per frame**, never a reusable constant.
- **Learned metric prior**: a *model* (not a number) that generalises across your narrow domain → only needed for the **anchorless** real-time/pre-cut case.

A metric measurement = intrinsics (once) + relative depth (per frame, shape) + one scale factor (per frame, size). With the catheter in frame post-cut, the scale factor comes free per frame from its known diameter.

## Phase philosophy

Each phase tests one design choice and ends in a **gate**: a measurable result that says "the assumption the next phase depends on holds — proceed," or triggers a named fallback. Ten weeks is a guideline; the gates are the real schedule and double as descope points.

Anchors throughout: the **catheter** (known French size, diameter = Fr ÷ 3 mm) is the primary anchor because it is present at the post-cut measurement moment; the **instrument shaft** (~8 mm for da Vinci) is the cross-check; the **prostate** from the GT mask (~3–4 cm) is a coarse sanity check only (patient-variable). Validating these against each other is the trick that checks scale internally without touching the ruler.

---

## Phase 0 — Foundations & definitions (Week 1)

**Tests:** Is the problem well-posed on this data? Three definitional answers here have outsized leverage on the whole project.

- **Settle the SUL definition with UMC** — the single highest-leverage step:
  - **Cut vs stretched length.** Stretched requires the surgeon to physically pull the stump (dynamic) and is hard for a passive video model; cut length is static. Target whichever your ruler set records.
  - **Confirm it's measured post-cut on the stump** (catheter in frame). This decides whether you need the anchorless path at all.
  - **Chord vs arc.** Your workflow measures the Euclidean distance between two endpoints (a chord). If that matches the ruler, you only need accurate metric 3D for **two points** — not a dense reconstruction. Confirm the proximal/distal landmarks (e.g. prostate apex to distal membranous urethra).
- **Data audit:** is the catheter French size recorded/standardised per patient? Instrument shaft diameter? Any kinematics logged? Identify the measurement moment, the span of camera motion, and confirm catheter visibility at measurement.
- **Intrinsics:** calibrate focal length, principal point, distortion once; verify optics are fixed across the dataset.
- **De-risk scale early on metric-GT data:** on **SCARED** (structured-light GT, ~1 mm) or **SERV-CT** (CT-registered, absolute metric), back-project two landmarks with K⁻¹ + GT depth, compute the true metric distance, and check your depth→metric estimate against it — using the 8 mm instrument shaft as the reference. This isolates the depth→metric step before touching UMC video.
- **Eval harness:** lock train/val/test splits with the ruler set isolated. Metrics: MAE (mm), bias, Bland-Altman. Targets: **±2–4 mm** (stretch), **±3–6 mm** (acceptable) — noting the manual ruler GT itself carries a few mm of error, so this is near the noise floor.

**Gate:** definitions pinned (cut/stretched, post-cut, chord/arc); catheter confirmed present at measurement; depth→metric validated on SCARED within tolerance; intrinsics stable. If endpoints/timing can't be pinned → resolve with UMC before modelling.

---

## Phase 1 — Segmentation (Week 2)

**Tests:** Can you get temporally stable masks for the urethra (centerline/endpoint source) and the anchors (scale source)? You likely don't need to train from scratch.

- Fine-tune **SurgeNetRARP** (its RARP downstream set already includes a urethra class) and/or **LemonFM** (stronger public backbone) on your RARP-annotated frames. Keep **nnU-Net** as a fallback baseline. Segment urethra + catheter + instrument.
- Temporal consistency via a video model (ATLAS-style) or **SAM2 / MA-SAM2** mask propagation across the measurement window.
- Extract the urethra **centerline** (skeletonisation) and the **two endpoints** (extreme points / a small landmark head).

**Gate:** urethra Dice and endpoint localisation clear an agreed threshold, and the catheter/instrument masks are clean enough to measure widths. If anchor masks are too noisy → that caps scale accuracy; fix here.

---

## Phase 2 — Metric scale from the in-frame anchor (Week 3)

**Tests:** The crux — can the in-frame catheter recover metric scale accurately, using only depth-free reasoning at this stage?

A known-size object plus calibrated intrinsics gives that object's **own** metric depth directly, no depth map needed: `Z = f·S / p` (physical size S, pixel size p, focal length f). This is the foundation of every later scale step. Because mm-per-pixel equals `Z/f`, it is depth-dependent — so a scale derived at one object is **not** transferable to another object at a different depth. That cross-object comparison needs depth and moves to Phase 3.

- Recover the catheter's depth and local scale per frame: segment it, measure its **sub-pixel silhouette width** perpendicular to its axis, and apply `Z = f·S/p` with the known diameter (Fr ÷ 3). Do the same for the **8 mm instrument shaft** to get its (closer) depth.
- **Depth-free sanity checks** (internal validation, no ruler leakage): each anchor's intrinsic-derived depth should land in a plausible endoscopic working range, and because the catheter and instrument are constant-diameter cylinders, their pixel-width gradient along their length should imply a consistent depth profile (within-object check). The prostate from the GT mask (~3–4 cm) is a coarse extra check (patient-variable).
- **2D baseline SUL:** Euclidean pixel distance between endpoints × catheter scale (no depth). Valid here **only because the catheter exits the stump at roughly the urethra's depth** — a co-location approximation, not a general one. First number + error bar.

**Gate:** the catheter's sub-pixel diameter is precise enough that its intrinsic-derived depth is stable across frames (a 1-px error on a ~30-px catheter is a few percent that multiplies straight into SUL) → improve sub-pixel edges / aggregate across frames before proceeding. The catheter–instrument cross-check is deferred to Phase 3, where depth makes it valid.

---

## Phase 3 — Add depth and anchor-calibrate it (Week 4)

**Tests:** Is the urethra curved enough out-of-plane that depth changes the number — and does the depth map, once anchored, reproduce a *second* known-size object it was not calibrated on?

- Fine-tune a **Depth Anything v2** backbone with an endoscopic adapter (**EndoDAC** or **DARES** vector-LoRA) for domain-robust relative depth. Add **segmentation-guided depth** (use the instrument/anatomy masks to guide prediction) since you're segmenting anyway.
- **Anchor-calibrate the relative depth (scale + shift fit).** For an affine-invariant depth map, true depth is `Z = a·d_rel + b`. Each anchor supplies both its relative depth `d_rel` (from the network) and its true `Z` (from `Z = f·S/p` in Phase 2), giving `Z_cath = a·d_cath + b` and `Z_inst = a·d_inst + b` — solvable for **both scale `a` and shift `b` precisely because the catheter and instrument sit at different depths** (a single anchor, or two at equal depth, leaves the shift undetermined). Over-determine the fit with multiple positions along each cylinder and across frames.
- **Cross-anchor check, now valid (ruler-free).** Calibrate the map using one anchor, then verify it reproduces the *other* anchor's known diameter at that anchor's own depth. Agreement validates both the scale and the relative geometry of the depth map; the disagreement is your true scale-error budget.
- Back-project the two endpoints (or the centerline) to metric 3D using intrinsics + the calibrated depth.
- Compare the 3D chord against the Phase 2 2D baseline.

**Gate (a fork that shapes both tracks):** the cross-anchor check agrees within a few percent (depth and scale trustworthy), and — if depth materially changes SUL → keep depth in both the gold ensemble and the real-time model; if the chord is near-planar and depth barely moves it → the real-time model can stay 2D + anchor + smoothing. Either result is a finding.

---

## Phase 4 — Accuracy-optimal model: confidence-weighted depth ensemble (Weeks 5–6)

**Tests:** Does fusing multiple depth estimators with confidence beat any single one? This is your gold, uncertainty-aware model (your Option 2, target ±2–4 mm).

- **Ensemble:** run **WS-SfMLearner** (gives depth + ego-motion + flow, exploiting your camera motion), **EndoDAC**, and **DARES**. Optionally fine-tune on RARP GT frames (weakly-supervised: photometric + GT depth) to sharpen metric scale.
- **Fuse** depths weighted by **catheter-validation confidence** (estimators that reproduce the known catheter diameter better get higher weight); keep a per-pixel confidence map.
- **Confidence-based filtering:** keep only pixels where depth-confidence × segmentation-confidence clears a threshold; **flag low-confidence measurements for manual review** (this uncertainty layer is a genuine contribution and what a clinical tool needs).
- **Temporal aggregation** across the motion window (robust/median over frames) for stability.
- **Measure** the chord between the two metric 3D endpoints (or arc length if Phase 0 demanded it); output SUL + an uncertainty estimate.

**Gate:** on the ruler test set, MAE within the ±2–4 mm target with **calibrated** uncertainty, and it beats the single-frame baseline. **Optional stretch only if needed:** deformable Gaussian splatting (**Deform3DGS / EndoGaussian**) for higher-fidelity 3D — but don't start here; it's a fallback for if the ensemble under-delivers on a curved arc.

---

## Phase 5 — Validate the gold model vs the ruler + emit pseudo-labels (Week 7)

**Tests:** Is the gold model good and reliable enough to (a) be the reported result and (b) teach the real-time model?

- Full evaluation on the **ruler test set**: MAE, bias, Bland-Altman, and a failure analysis (blood, smoke, occlusion, weak anchor/catheter visibility).
- **Ablations:** anchor (catheter vs instrument), with/without depth, single model vs ensemble, chord vs arc, with/without confidence filtering.
- Note the residual temporal offset between the model's measurement frames and the ruler placement (smaller now that both are post-cut on the stump).

**Gate:** accuracy + calibrated reliability acceptable → **freeze** the gold model and run it over all training videos to emit metric pseudo-labels (per-frame metric depth + SUL + confidence). This gate unlocks Phase 6.

---

## Phase 6 — Real-time model: distillation + deployment (Weeks 8–9, secondary)

**Tests:** Can a lightweight per-frame pipeline, taught by the gold model, hit real-time fps within an acceptable accuracy gap?

- Assemble the fast pipeline: a distilled/efficient segmenter + a single compact **metric depth** model (distilled from the ensemble) + online endpoint extraction + chord + **temporal smoothing** (EMA / Kalman) of the streaming SUL + a live confidence readout.
- Supervise with the Phase 5 pseudo-labels.
- Optimise for deployment: smaller backbone (e.g. Depth Anything v2 small / distilled), ONNX → **TensorRT**; measure latency and frames per second on HD.
- **Deploy on video:** streaming inference with a live SUL overlay + confidence. If you pursue the *pre-cut guidance* variant (no catheter in frame), this is where the **learned scale prior** is required and where the value proposition is weakest — flag that trade-off explicitly.

**Gate:** target frame rate met **and** accuracy gap to the gold model within tolerance on a held-out stream. Report the **fps-vs-error trade-off curve** — that curve is itself a thesis contribution.

---

## Phase 7 — Integration, demo, write-up (Week 10 + buffer)

- End-to-end **video demo**: real-time SUL overlay running on a clip, with the gold offline measurement + confidence reported alongside.
- Final tables and figures; assemble the TU/e-branded presentation.
- Thesis writing and buffer for slippage.
- Map outputs onto the **Preparation_Phase research-plan form**: the framing section → "research questions / AI&ES relevance," Phases 0–6 → "research methods," and the phase timeline → "time planning."

---

## Priority & descope (if you fall behind)

Protect the **gold ensemble** — it's the validated, scientifically valuable result. If time runs short, ship the **gold model fully evaluated with uncertainty** and deliver a **simpler real-time approximation** (2D chord + catheter scale + smoothing) rather than the full distilled-depth student. The Phase 3 fork tells you by week 4 whether that simplification is even lossy for your anatomy.

## Risk register (top items)

- **Scale precision** (Phases 2–3) — the catheter diameter is short, so any pixel error multiplies into SUL; mitigate with sub-pixel edges, multi-frame aggregation, and the Phase 3 cross-anchor check.
- **Definition ambiguity** (Phase 0) — cut vs stretched, chord vs arc, post-cut timing; resolve with UMC before modelling, because each reshapes the architecture.
- **Ensemble disagreement** (Phase 4) — treat WS-SfMLearner vs EndoDAC disagreement as a confidence signal, not noise.
- **Curved arc beyond ensemble fidelity** (Phase 4) — fall back to Gaussian splatting only if the chord/centerline error demands it.
- **Real-time budget** (Phase 6) — distillation + TensorRT; accept a defined fps/accuracy trade-off rather than chasing both.
- **Anchorless transfer** (Phase 6, pre-cut variant only) — anchor frames train the learned scale prior, anchorless frames use it; quantify the gap and question whether the use case is worth it.
