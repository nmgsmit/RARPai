#!/bin/bash
#SBATCH --job-name=depth-anchorablation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# WHICH ANCHOR IS THE BEST SUPERVISOR -- at EQUAL supervision volume.
# Three arms, identical in every flag except --scale-classes:
#     sbatch jobs/ablate_anchor_class.sh --scale-classes 1 --out outputs/ablate_anchor_ruler --run-name ablate-anchor-ruler
#     sbatch jobs/ablate_anchor_class.sh --scale-classes 2 --out outputs/ablate_anchor_cath  --run-name ablate-anchor-cath
#     sbatch jobs/ablate_anchor_class.sh --scale-classes 3 --out outputs/ablate_anchor_arm   --run-name ablate-anchor-arm
# (1=Ruler, 2=Catheter tip 5.333 mm, 3=Robot arm 8 mm.) The real question is arm 3: it is the
# only anchor present in deployment, so "is arm-only enough" is what this answers.
#
# WHY --anchor-balance: the 2026-09-02 --scale-classes A/B was NOT a test of the anchors, it was
# a test of how many of each were annotated -- 1472 ruler objects over 18 videos vs 429 catheter
# over 9 and 416 arm over 15, and the notes concluded volume dominated. --anchor-balance 1 2 3
# keeps only TRAIN videos carrying all three, then subsamples each class to the same count:
# 5 videos x 135 objects per class at --seed 66. Val/test keep every annotation.
#
# WHY --seed 66: it is the split (searched over 200) where all three classes are well represented
# on the 5 HELD-OUT videos -- test n = 383 ruler / 143 catheter / 133 arm -- and on val, so each
# arm can select a checkpoint on its OWN class and still be scored on the other two.
#
# HOW TO READ IT: the eval scores EVERY class regardless of what was supervised, so for the
# arm-only run the ruler and catheter numbers are a held-out cross-object check -- predicted
# length vs true mm on an object the model never saw supervision for. Read the per-class
# in-plane numbers `c1_inplane_scale / c2_ / c3_` in `[metric/test]` (in-plane, because that is
# what --scale-inplane trains and the 3D-polyline form is gameable by tilt), plus track_slope.
#
# Base config = the current operating point (CLAUDE_NOTES 2026-09-07): --scale-inplane with
# --anchor-w 0.1. 3 epochs per the ablation brief; note the notes' warning that everything past
# ~ep 5 is the degenerate regime anyway, so 3 is not as short as it sounds.
python scripts/finetune_depth.py \
    --data-root ../data/processed/depthclips_ruler_NoGUI \
    --init ../backbones/EndoDAC/depth_model.pth \
    --pose-init-dir ../backbones/EndoDAC \
    --out outputs/ablate_anchor \
    --run-name ablate-anchor \
    --image-shape 392 490 \
    --video-split 4 5 \
    --seed 66 \
    --scale-w 0.5 \
    --scale-inplane \
    --anchor-w 0.1 \
    --anchor-balance 1 2 3 \
    --min-depth 20 \
    --max-depth 200 \
    --no-learn-intrinsics \
    --epochs 3 \
    "$@"
