# Commands (Nick's cheatsheet)

Run from `code/` on Snellius unless noted.

## Submit jobs
```bash
sbatch jobs/finetune_tversky.sh                      # defaults
sbatch jobs/finetune_tversky.sh --epochs 50 --lr 1e-4 --alpha 0.3   # override via "$@"
sbatch jobs/finetune_tversky_fp.sh                   # H100 ablation: FP-biased
sbatch jobs/finetune_tversky_dice.sh                 # H100 ablation: symmetric Dice
sbatch jobs/overlay_masks.sh                         # CPU overlay render
```

## Depth (EndoDAC fine-tune — separate from segmentation)
```bash
sbatch jobs/finetune_depth.sh                                  # defaults (img 392x490, 20 ep)
sbatch jobs/finetune_depth.sh --epochs 30 --frame-stride 2     # wider motion baseline
sbatch jobs/finetune_depth.sh --image-shape 448 560            # higher train res (mult of 14)
sbatch jobs/finetune_depth.sh --intrinsics 0.9 1.2 0.5 0.5     # real da Vinci K (normalised)
sbatch jobs/finetune_depth.sh --no-refine                      # plain Monodepth2 (no AF-SfMLearner)
# frame-stride ablation (all with AF-SfMLearner refinement, default on):
for s in 1 2 3; do sbatch jobs/finetune_depth.sh --frame-stride $s \
    --out outputs/depth_s$s --run-name depth-refine-s$s; done
# verify the GUI 389-key contract (no GPU/data needed):
python scripts/finetune_depth.py --self-test
python scripts/finetune_depth.py --self-test --ckpt outputs/rarp_depth/best.pth
python scripts/finetune_depth.py --smoke                       # tiny synthetic fwd/bwd
```
Swap the result into the GUI (one line): copy `outputs/rarp_depth/best.pth` over
ATLAS-Interactive's `../backbones/EndoDAC/depth_model.pth` (or point `CHECKPOINT` at it).

## Monitor / cancel
```bash
squeue -u $USER                 # my queue
scancel <jobid>                 # cancel one
scancel -u $USER                # cancel all mine
tail -f logs/<jobid>.out        # live stdout
tail -n 50 logs/<jobid>.err     # errors
```

## Smoke test before a full run
```bash
sbatch jobs/finetune_tversky.sh --smoke    # tiny run to check it doesn't crash
```

## Sync code local -> Snellius (via GitHub)
```bash
# (local Windows)
git add -A && git commit -m "msg" && git push

# (Snellius, in code/)
git pull
```
Code only reaches Snellius after this round-trip — editing locally does nothing until pushed + pulled.

## Measure a distance on an image (local Windows GUI)

Click two points on a frame -> length in mm from the depth map. Runs on the laptop, CPU is fine
(~1-3 s per frame). Nothing here touches Snellius.

**One-time setup (PowerShell, in the repo):**
```powershell
# 1. copy the trained checkpoint down from Snellius (keep the folder name!)
mkdir outputs\depth_ruler_range_sw05
scp snellius:~/RARPai/outputs/depth_ruler_range_sw05/best.pth outputs\depth_ruler_range_sw05\

# 2. deps (Tkinter ships with python.org Python; the CPU torch wheel is enough)
pip install torch torchvision numpy pillow matplotlib opencv-python fvcore
#   fvcore is a hard import of the vendored EndoDAC backbone, not optional.
#   "xFormers is not available/disabled" warnings on start-up are expected and harmless.
```

**Run:**
```powershell
python scripts\gui_depth_measure.py --version               # which build + which file
python scripts\gui_depth_measure.py                          # browses ..\data
python scripts\gui_depth_measure.py --data-dir ..\data\fold1
python scripts\gui_depth_measure.py --image ..\data\some\frame_0001.jpg
python scripts\gui_depth_measure.py --self-test              # geometry checks, no torch needed
python scripts\gui_depth_measure.py --no-model               # UI only, synthetic depth
```

**Using it:** *Open folder* -> pick a frame -> left-click point A, left-click point B -> the mm
appears on the line. Wheel = zoom, right-drag = pan, `Esc` cancels a half-placed point,
`Ctrl-Z`/`<-`/`->` undo / prev / next image.

**Depth maps are computed once and kept.** Every map lands in
`outputs/depth_cache/<run>_<ckpt>_<fingerprint>/` as a `.npz` (float32 mm, key `depth`, plus a
`meta.json` describing the run) - so revisiting a frame, or reopening the app tomorrow, is
instant. The dropdown at the top lists every `outputs/*/*.pth` in the repo; picking one loads it
**and switches to that checkpoint's own cache folder**, so its maps are picked straight back up
and switching back and forth costs nothing. *Precompute depth for folder* fills the cache for the
whole file list in one go (`Esc` stops it); frames already on disk are marked `*` in the list.
The folder is keyed by the checkpoint's CONTENTS, so re-copying the same `best.pth` from Snellius
keeps the cache, while a genuinely different checkpoint gets its own folder and can never serve a
stale map. `--cache-dir` moves it, `--no-cache` turns it off.

Other scripts can read the dump directly: `np.load(f)["depth"]` is the depth in mm.

- **Check the aspect readout says ~1.250 before believing a number.** The model was trained on the
  5:4 ruler dumps, so a raw 16:9 console frame must have its black bars cropped off; *Auto bars*
  runs on load and usually gets it, the three frac boxes are the manual override.
- **Expect a few percent of scale error** (held-out test scale ratio 0.974). For better than that on
  a given scene, measure something of known size, select it in the list, hit *Calibrate from
  selected...* and type the true mm - every measurement rescales.
- Depth is sampled as a median over a small patch (`patch px`), so clicking near an edge is
  forgiving; drop it to 0 if you are measuring something genuinely thin.
- *Export CSV* / *Save PNG* write the measurements next to whatever you name them.

**Missing a button the notes mention?** The title bar and `--version` print a build number - if it
is lower than the change you are looking for, the running copy is an old one (wrong folder, or
`git pull` not done in the folder you launch from). The control column scrolls, so nothing is ever
hidden by a small screen.

**Still slow after the cache landed?** Every status-line message now ends `[tier NNN ms]` -
`[memory]`, `[cache]` or `[model]`. Read that before guessing at a cause:
- `[model 2000+ ms]` on a frame you have not visited before, or after a fresh `git pull` on
  Snellius (a checkpoint's cache lives under `outputs/depth_cache/<fingerprint>/`, which is not
  checked into git) - expected, that IS the one-time cost. Run *Precompute depth for folder*
  once per checkpoint and every later visit is `[cache]`/`[memory]`.
- `[cache 5-15 ms]` or `[memory <1 ms]` and it still feels slow - the depth prediction is not
  what is slow; something else in the click/redraw path is. Say what you clicked and the exact
  status-line text.
- The .npz files are NOT the bottleneck: writing one (448x560 float32, ~1 MB) is under a
  millisecond uncompressed, reading one is a few ms - both negligible next to the multi-second
  CPU forward pass. Storage format was never the cause; the timing tag now proves it either way.

## Temporal stereo on a 3D clip (depth from the video, not one frame)

```bash
# default: 5 s at 20 fps from the seg3 clip, FoundationStereo, +-8 frames fused
sbatch jobs/temporal_stereo.sh

# a different window of the same clip, and keep the fused depth as proxy GT
sbatch jobs/temporal_stereo.sh --start 30 --seconds 4 --save-depth

# another clip / output folder
CLIP=../data/3D_ProxyGT/<other>.mp4 OUT=outputs/temporal_stereo/other sbatch --export=ALL,CLIP,OUT jobs/temporal_stereo.sh
```

Writes into `$OUT`: `depth.mp4` (video | depth), `compare.mp4` (video | single pair | temporal,
white = filled from other frames, grey = still unsolved), `figure.png` (first frame of the same
three panels), `stats.json`.  `*_h264.mp4` are the same clips re-encoded so they play in a
browser — send those, not the `mp4v` originals.

**The matcher output is cached**, so re-running with different fusion settings does NOT need a
GPU. Tune on a CPU node:

```bash
sbatch --partition=genoa --gpus-per-node=0 --export=ALL,OUT=outputs/temporal_stereo/try1 jobs/temporal_stereo.sh --window 8 --fb-tol 3.0 --min-support 1
```

Read in the log: `single pair valid X%` vs `+ temporal valid Y%` (both over the geometrically
usable area, not the whole frame), `% of the holes closed`, and `agree ... by support` — the
leave-one-out error of the transferred values, split by how many frames agreed. Holes left are
attributed as `blind` (nothing in the window saw it — a floor, not a knob), `flow`, `thin`,
`disagree`; only the last three respond to tuning.

Coverage is what improves (84.1% -> 88.9% on the measured window). Depth where the single pair
already worked is left alone, and measured temporal jitter is only 0.13 mm, so do not expect the
map to get *sharper* — it gets *more complete*.

## Urethra end point / SUL from 3D (cylinder + where the roof covers it)

```bash
# 1. depth for a window where the urethra is visible, KEEPING the fused depth (images/ + depth/)
sbatch --export=ALL,OUT=outputs/temporal_stereo/seg3_t30 jobs/temporal_stereo.sh --start 30 --seconds 5 --save-depth
#    other clip: --export=ALL,CLIP=../data/3D_ProxyGT/<clip>.mp4,OUT=outputs/temporal_stereo/<name>

# 2. urethra mask -> cylinder -> roof end point -> SUL
sbatch --export=ALL,RUN=outputs/temporal_stereo/seg3_t30 jobs/urethra_cylinder.sh

# no model, no data: synthetic tube under a roof (runs locally)
python scripts/urethra_cylinder.py --self-test
```

Writes `$RUN/urethra_cyl/`: `overlay_h264.mp4` (mask yellow, cylinder cyan, start green, roof end
red, depth, and the gap profile underneath), `fig_3d.png` (section along the tube + 3D),
`sul_time.png`, `frames.csv`, `summary.json`. Quote `sul_median_mm` -- it uses only frames with
BOTH ends observed. A hollow green dot / "START HIDDEN" = an instrument over the proximal urethra,
that frame is not counted.

Hand masks instead of the model: labelling-tool PNGs named like the frames (`<frame>.png`, e.g.
from `OTHERS/annotate_urethra/<clip>/masks/`), uploaded next to the run. Writes
`$RUN/urethra_cyl_hand/`; the model run in `$RUN/urethra_cyl/` stays for comparison.

```bash
scp -r "OTHERS/annotate_urethra/short_seg3_14s/masks" snellius:~/RARPai/outputs/temporal_stereo/noocc_seg3_14s/hand_masks
sbatch --export=ALL,RUN=outputs/temporal_stereo/noocc_seg3_14s,SLOW=10 jobs/urethra_cylinder.sh --masks outputs/temporal_stereo/noocc_seg3_14s/hand_masks --fps 59.94
```

Add `--points outputs/temporal_stereo/<run>/hand_points.csv` (frame,ax,ay,bx,by -- your ruler start/end,
converted from `scale_objects.json`) for blue crosses, your SUL as a 3D chord, and start/end errors
along the fitted axis in `frames.csv` / `summary.json`.
