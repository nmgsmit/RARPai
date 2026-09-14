# Snellius operation guide (read me first)

This repo is **edited locally** (Windows / OneDrive) but **runs on Snellius** (SLURM).
Relative paths like `../data` and `../backbones` resolve on **Snellius**, not always on the
local mirror. When you write code, assume it will be launched with `sbatch` from the `code/`
dir on Snellius.

> Companion files: [COMMANDS.md](COMMANDS.md) = Nick's copy-paste commands.
> [CLAUDE_NOTES.md](CLAUDE_NOTES.md) = Claude's design log — **append to it** when you make a
> design choice or put something in a new place.

## Where things live (paths relative to `code/`)

| What | Path | Notes |
|------|------|-------|
| Datasets | `../data/RARPSurgenet/fold1` | Snellius layout. Local mirror has `../data/fold1` + `atlas120k_processed.zip` |
| Encoder backbone (DINO teacher) | `../backbones/RARP_checkpoint_epoch0050_teacher.pth` | frozen encoder init for finetuning |
| Trained checkpoints | `outputs/<run>/best.pth` | written by `scripts/finetune_*.py` |
| Job scripts | `jobs/*.sh` | one per experiment, submitted with `sbatch` |
| Train/infer code | `scripts/*.py` | |
| SLURM logs | `logs/<jobid>.out` / `.err` | |
| wandb creds | `.env` (gitignored) | project `rarp`, entity `nmgtue` |

## Environment

`venv` is current (not conda). In every job:
```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate
```
The old `jobs/run_seg.sh` + `environment_setup.md` still mention `conda activate rarp-seg` —
**deprecated**, use venv.

## Driving Snellius directly (Claude over SSH)

Claude can run cluster commands from its own shell via the `snellius` ssh alias (passphrase-less
key). Repo on Snellius = `~/RARPai` (this `code/` dir). Always `cd ~/RARPai` first:
```bash
ssh snellius 'cd ~/RARPai && git pull && sbatch jobs/finetune_tversky.sh --epochs 50'
ssh snellius 'cd ~/RARPai && squeue -u $USER'
ssh snellius 'tail -n 50 ~/RARPai/logs/<jobid>.out'
```
Login prints a harmless post-quantum warning — ignore / `grep -v` it.

## Running jobs

Each `jobs/*.sh` cds to `$SLURM_SUBMIT_DIR`, activates venv, and runs a script with fixed
args **plus `"$@"`** — so you override hyperparameters on the command line:
```bash
sbatch jobs/finetune_tversky.sh --epochs 50 --lr 1e-4 --alpha 0.3
```
New experiment = copy an existing `jobs/*.sh`, change `--out`, `--run-name`, and SBATCH header.

### Partitions
- **GPU training** → `--partition=gpu_h100 --gpus-per-node=1` (current; see `finetune_tversky_fp/dice/5class.sh`)
- **CPU-only** (overlays, viz) → `--partition=genoa`
- Some older finetune jobs still point at `genoa` (CPU) — switch to `gpu_h100` for real GPU runs.

## Frames, black bars and GUI overlays (agreed with Nick 2026-09-14)
- **Black side bars: crop once.** Every image past preprocessing is the 1340x1072 5:4 *content* frame
  (mono: `source_crop.json` x289 y4; SBS: the eye rectangle -> rectified). Intrinsics are defined on
  that frame. No pillarbox ever reaches a model, loss or eval; don't add new `--side-crop-frac` runs.
- **GUI (HUD banner + tab, popups, cue bars): black pixels AND a mask file.** Save `<stem>_mask.png`
  (255 = GUI) next to every image. Consumers read the mask; black is never the signal (stereo reads
  black as a flat ~47 mm surface; dark tissue fails `mean > 0.04`). Mono: `cut_cue_clips.full_gui_mask`;
  SBS: `scripts/mask_sbs_gui.py`.
- **Depth selection:** stereo proxy-GT (`--proxy-gt-dir`, metric `abs_rel`) is logged every epoch and
  picks `best.pth`. With `--scale-w 0` (no metric anchor, relative depth) it picks on `ms_abs_rel`
  (median-scaled shape) instead — compare such runs on `proxy_gt/ms_abs_rel` only.

## Gotchas
- Code reaches Snellius only via GitHub: `git push` (local) → `git pull` (Snellius). Local edits are **not** live on the cluster until then.
- Never train on the login node — always inside an sbatch job.
- `--num-classes 0` auto-detects from the checkpoint head bias.
- `--keep-classes` default `1,3` = instrument-class subset, not all classes.
- Local mirror lacks `../backbones` and the full dataset; don't expect training to run locally.
