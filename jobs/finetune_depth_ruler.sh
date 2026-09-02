#!/bin/bash
#SBATCH --job-name=depth-ruler
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

# METRIC depth on the ruler clips: 1677 frames / 65 clips / 18 videos, ~89% of frames carrying
# known-size objects in scale_objects.json (Ruler = typed mm, Catheter tip = 5.333 mm, Robot
# arm = 8 mm). --scale-w turns those into a loss so the depth map converges to MILLIMETRES
# instead of an arbitrary scale; --video-split holds out whole surgeries (4 val / 5 test of 18)
# so the reported metric error can't be leaked by a sibling clip.
#
# K is FROZEN here (--no-learn-intrinsics): the focal and the depth scale trade off, so letting
# the IntrinsicsHead move while the scale loss pulls on depth makes the result unreadable.
# Run 2 flips it on to see whether the anchors calibrate f.
#
# Short convergence probe: 4 epochs. Watch train/scale_ratio -> 1.0 and metric_val/abs_rel.
# Frames are dumped at unknown fps, so --frame-stride 1 (clips are only ~26 frames); sweep it.
python scripts/finetune_depth.py \
    --data-root ../data/processed/depthclips_ruler_NoGUI \
    --init ../backbones/EndoDAC/depth_model.pth \
    --pose-init-dir ../backbones/EndoDAC \
    --out outputs/depth_ruler_scale \
    --run-name endodac-ruler-scale \
    --image-shape 392 490 \
    --video-split 4 5 \
    --scale-w 0.1 \
    --no-learn-intrinsics \
    --epochs 4 \
    "$@"
