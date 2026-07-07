#!/bin/bash
#SBATCH --job-name=eval-scared
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# SCARED test benchmark (calibrated-STEREO metric GT from the ds8/ds9 keyframe pairs -- the
# structured-light GT was withheld from the challenge test release). Step 1 builds the GT +
# rectified left frames; step 2 runs the trained depth model, median-scales, logs 7 metrics
# + overlays to wandb. Override --ckpt / paths via "$@".
CKPT=${CKPT:-outputs/depth_s1/best.pth}

python scripts/export_scared_stereo_gt.py \
    --scared-root ../data/SCARED --out ../data/SCARED/stereo_gt

python scripts/eval_scared.py \
    --ckpt "$CKPT" \
    --rgb-dir ../data/SCARED/stereo_gt/frames \
    --gt-npz ../data/SCARED/stereo_gt/gt_depths.npz \
    --image-shape 392 490 \
    --run-name "$(basename $(dirname $CKPT))-scared-stereo" \
    "$@"
