#!/bin/bash
#SBATCH --job-name=depth-adapt-calib
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# THE CONTROL that per-video ARM FINETUNING has to beat: no training at all, just fit the
# calibration on the arm anchors of each held-out video and predict its ruler/catheter.
# --calib-classes 3 scores ONLY the non-arm objects, so it is the same held-out measurement
# jobs/adapt_per_video_arm.sh reports -- and it costs a CPU second per video.
# --seed 66 matches outputs/adapt_base (jobs/adapt_base.sh); without it the "held-out" videos
# would be a different five.
python scripts/fit_affine_scale.py \
    --dump outputs/adapt_base_test.npz \
    --ckpt outputs/adapt_base/best.pth \
    --split test \
    --video-split 4 5 \
    --seed 66 \
    --min-depth 20 \
    --max-depth 200 \
    --calib-classes 3 \
    "$@"
