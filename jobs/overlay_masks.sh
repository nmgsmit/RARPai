#!/bin/bash
#SBATCH --job-name=overlay-masks
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Usage:
#   sbatch jobs/overlay_masks.sh --video ../data/raw/RARP_example.mp4 \
#       --checkpoint outputs/rarp_higherlr/best.pth \
#       --output outputs/RARP_example_masked.mp4

python scripts/overlay_masks.py "$@"
