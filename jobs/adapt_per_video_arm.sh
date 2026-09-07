#!/bin/bash
#SBATCH --job-name=depth-adapt-arm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# DOES PER-VIDEO ADAPTATION ON THE ROBOT ARM WORK?  (run jobs/adapt_base.sh first)
#
# The deployment case: a model trained on annotated surgeries meets a NEW one, where the robot
# arm is the only known-size object in frame. Adapt on it, then measure something else.
# Per held-out video, two passes over the SAME clips and the SAME objects:
#     --epochs 0   the base model, untouched          = baseline
#     --epochs 3   3 epochs on THAT video, arm only   = adapted
# Supervision is --scale-classes 3 (arm) throughout; the reported number is the RULER and
# CATHETER length error (c1_*/c2_*), which neither pass ever supervises. Adapting and scoring on
# one video is the point, not a leak -- the split is by CLASS here, not by frame.
#
# The three held-out videos of --seed 66 that carry arm AND something to score it against:
#   7ee04683  62 arm | 136 ruler + 104 catheter
#   ee3be53d  39 arm |  64 ruler +  22 catheter
#   31e2c520  32 arm | 114 ruler   (no catheter)
# (feae9227 and 4d8eca93 have no arm anchors at all -- nothing to adapt on.)
#
# READ c1_inplane_scale / c2_inplane_scale (1.0 = metric) adapted vs baseline. The bar to clear is
# NOT the baseline though, it is the no-training control -- one constant per video fitted on the
# arm, which costs a CPU second:
#   python scripts/fit_affine_scale.py --dump outputs/adapt_base_test.npz --split test --seed 66 \
#       --ckpt outputs/adapt_base/best.pth --calib-classes 3
# If per-video finetuning does not beat that row, it is not worth shipping.
for VID in 7ee04683 ee3be53d 31e2c520; do
  for EP in 0 3; do
    echo "=================== video=$VID epochs=$EP ==================="
    python scripts/finetune_depth.py \
        --data-root ../data/processed/depthclips_ruler_NoGUI \
        --init outputs/adapt_base/best.pth \
        --pose-init-dir ../backbones/EndoDAC \
        --out outputs/adapt_${VID}_ep${EP} \
        --run-name adapt-${VID}-ep${EP} \
        --image-shape 392 490 \
        --video-split 4 5 \
        --seed 66 \
        --only-videos $VID \
        --scale-classes 3 \
        --scale-w 0.5 \
        --scale-inplane \
        --anchor-w 0.1 \
        --min-depth 20 \
        --max-depth 200 \
        --no-learn-intrinsics \
        --no-scared \
        --epochs $EP \
        "$@"
  done
done
