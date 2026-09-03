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
