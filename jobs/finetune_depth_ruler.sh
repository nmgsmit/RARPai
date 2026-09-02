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
# --min-depth/--max-depth ARE NOT OPTIONAL HERE. The 0.1/150 defaults are a KITTI/SCARED-unit
# convention: with them disp_to_depth squeezes the 35-120mm endoscopic range into 0.2% of the
# sigmoid's output range and the model cannot be metric (warm-start scale 0.005). At 20/200 the
# SAME untrained weights score scale 0.894. See CLAUDE_NOTES 2026-09-02.
#
# K is FROZEN (--no-learn-intrinsics): focal and depth scale trade off, so letting the
# IntrinsicsHead move while the scale loss pulls on depth makes the result unreadable. Run 2
# flips it on to see whether the anchors calibrate f now that the scale is pinned.
#
# Best config so far (endodac-ruler-range-sw05): test scale 0.974 on the 5 held-out videos,
# per-class 1.04 / 0.98 / 0.89, SCARED abs_rel 0.056. Residual ~20% is per-object, not scale.
# Frames are dumped at unknown fps, so --frame-stride 1 (clips are only ~26 frames); sweep it.
python scripts/finetune_depth.py \
    --data-root ../data/processed/depthclips_ruler_NoGUI \
    --init ../backbones/EndoDAC/depth_model.pth \
    --pose-init-dir ../backbones/EndoDAC \
    --out outputs/depth_ruler_metric \
    --run-name endodac-ruler-metric \
    --image-shape 392 490 \
    --video-split 4 5 \
    --scale-w 0.5 \
    --anchor-w 0.3 \
    --min-depth 20 \
    --max-depth 200 \
    --no-learn-intrinsics \
    --epochs 12 \
    "$@"
