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
