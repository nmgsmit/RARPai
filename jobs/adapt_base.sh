#!/bin/bash
#SBATCH --job-name=depth-adapt-base
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# BASE model for the per-video adaptation test (jobs/adapt_per_video_arm.sh).
# Trained on the 9 train videos with ALL anchor classes at FULL volume (no --anchor-balance --
# that flag exists for the per-class ablation, not for building the best model), at the operating
# point from CLAUDE_NOTES 2026-09-07: --scale-inplane with --anchor-w 0.1, 4 epochs.
# --seed 66 so the held-out videos are ee3be53d / 31e2c520 / feae9227 / 4d8eca93 / 7ee04683 --
# three of which carry BOTH robot-arm anchors (to adapt on) and ruler/catheter (to be scored on).
python scripts/finetune_depth.py \
    --data-root ../data/processed/depthclips_ruler_NoGUI \
    --init ../backbones/EndoDAC/depth_model.pth \
    --pose-init-dir ../backbones/EndoDAC \
    --out outputs/adapt_base \
    --run-name adapt-base \
    --image-shape 392 490 \
    --video-split 4 5 \
    --seed 66 \
    --scale-w 0.5 \
    --scale-inplane \
    --anchor-w 0.1 \
    --min-depth 20 \
    --max-depth 200 \
    --no-learn-intrinsics \
    --epochs 4 \
    "$@"
