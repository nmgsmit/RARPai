#!/bin/bash
#SBATCH --job-name=depth-clips
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Self-supervised EndoDAC depth on depth_clips: ALL 79 UMCdissectionvid videos cue-cut with the
# rewritten single-digit-inference detector (scripts/cut_cue_clips.py) -- 1034/26/61 Train/Val/Test
# clips, 27330 frames total -- vs cue_s3's 7 videos / 109 clips / 2709 frames from the old CueClips
# set. Same recipe as cue_s3 (frame-stride 3, fixed calibrated K: these clips are translation-heavy
# and focal length is only identifiable from camera ROTATION, so a learned K would drift and
# confound the size-of-data comparison) so outputs/cue_s3 and this run differ in ONE thing: how
# much data they saw.
#
# CAVEAT: the Train/Validation/Test split here is copied from UMCdissectionimg by
# assemble_depth_clips.py, which is a DIFFERENT split than cue_s3's own CueClips split (Val
# 749c8234, Test RARP_092). Val/test metrics are therefore not computed on the same held-out
# frames -- compare loss-curve SHAPE and convergence, don't diff the numbers directly.
#
# Override via "$@".
python scripts/finetune_depth.py \
    --data-root ../data/depth_clips \
    --init ../backbones/EndoDAC/depth_model.pth \
    --pose-init-dir ../backbones/EndoDAC \
    --out outputs/depthclips_s3 \
    --run-name depthclips-s3 \
    --image-shape 392 490 \
    --epochs 10 \
    --frame-stride 3 \
    --no-learn-intrinsics \
    "$@"
