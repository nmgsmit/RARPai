#!/bin/bash
#SBATCH --job-name=depth-umc
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Self-supervised EndoDAC depth on UMCdissectionHD -- real target-domain frames (5-10 fps HD dumps,
# 71 videos / 46k Train frames). --sample-frac 0.15 keeps every video but ~15% of the redundant
# triplet centers (~7k) so a refine run is ~3h not ~16h. SCARED metric eval auto-runs at the end.
# Override via "$@" (e.g. --frame-stride 2 --sample-frac 0.1).
python scripts/finetune_depth.py \
    --data-root ../data/UMCdissectionHD \
    --init ../backbones/EndoDAC/depth_model.pth \
    --pose-init-dir ../backbones/EndoDAC \
    --out outputs/umc_dissection_depth \
    --run-name endodac-umc-dissection \
    --image-shape 392 490 \
    --sample-frac 0.15 \
    --epochs 20 \
    "$@"
