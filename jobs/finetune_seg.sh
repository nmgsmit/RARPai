#!/bin/bash
#SBATCH --job-name=rarp-finetune
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Adjust module versions to your cluster (Snellius)
module load 2023
module load Anaconda3/2023.09-0
module load CUDA/12.1.1

conda activate rarp
cd $SLURM_SUBMIT_DIR

# "$@" forwards any extra sbatch args to python, e.g.:
#   sbatch jobs/finetune_seg.sh --epochs 20
python scripts/finetune_segmentation.py \
    --data-root ../data/RARPSurgenet/fold1 \
    --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth \
    --out outputs/rarp_finetune \
    "$@"
