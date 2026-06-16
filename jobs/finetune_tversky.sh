#!/bin/bash
#SBATCH --job-name=tversky-ema
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

python scripts/finetune_seg_tversky.py \
    --data-root ../data/RARPSurgenet/fold1 \
    --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth \
    --out outputs/rarp_tversky \
    "$@"
