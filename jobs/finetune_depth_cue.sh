#!/bin/bash
#SBATCH --job-name=depth-cue
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Self-supervised EndoDAC depth on the CUE clips: scripts/cut_cue_clips.py segments of the 7 raw
# console videos where a da Vinci "move cue" bar is up -- instruments parked, so the camera moves
# over a (near) static scene, which is exactly the rigid-scene assumption the photometric loss makes.
# GUI + cue bar are blacked out at cut time and drop out via the vignette/overlay valid mask.
# Small dataset (2709 frames / 109 clips) -> no --sample-frac. 10 epochs: this is a convergence
# check, and the val curve's shape is readable in the first few epochs.
# Split: Train 5 videos / Validation 749c8234 / Test RARP_092. Override via "$@".
python scripts/finetune_depth.py \
    --data-root ../data/CueClips \
    --init ../backbones/EndoDAC/depth_model.pth \
    --pose-init-dir ../backbones/EndoDAC \
    --out outputs/cue_depth \
    --run-name endodac-cue \
    --image-shape 392 490 \
    --epochs 10 \
    --frame-stride 5 \
    "$@"
