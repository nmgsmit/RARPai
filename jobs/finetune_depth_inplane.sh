#!/bin/bash
#SBATCH --job-name=depth-inplane
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

# Does the depth map track DISTANCE? Copy of finetune_depth_ruler.sh with the two changes that
# CLAUDE_NOTES 2026-09-07 says are needed, and nothing else, so the A/B is readable:
#
#   --scale-inplane   the anchor's length is measured with ONE depth per segment. The 3D-polyline
#                     form is gameable: the sw05 run satisfied it by tilting segments out of the
#                     image plane (|dz| along the segment +72%) while the predicted DISTANCE
#                     stayed byte-for-byte the warm-start's (per-object ratio 1.005, corr .992).
#   --anchor-w 0      that term is an L1 pull toward a FROZEN warm-start copy over the whole
#                     image -- literally "do not change the depth". It has to go for this run.
#
# READ track_slope, NOT metric_scale. A model that emits one constant distance scores
# metric_scale 1.0 and track_slope 0.09 (what we have now). Success is track_slope -> 1.
# Expect metric_scale to drift off 1.0: in-plane length is mm*cos(theta), so oblique anchors bias
# the level ~1/cos. That is one constant, fixable by calibration; the tilt shortcut was not.
python scripts/finetune_depth.py \
    --data-root ../data/processed/depthclips_ruler_NoGUI \
    --init ../backbones/EndoDAC/depth_model.pth \
    --pose-init-dir ../backbones/EndoDAC \
    --out outputs/depth_ruler_inplane \
    --run-name endodac-ruler-inplane \
    --image-shape 392 490 \
    --video-split 4 5 \
    --scale-w 0.5 \
    --scale-inplane \
    --anchor-w 0 \
    --min-depth 20 \
    --max-depth 200 \
    --no-learn-intrinsics \
    --epochs 12 \
    "$@"
