#!/bin/bash
#SBATCH --job-name=tversky-fp
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

# Ablation: flip alpha/beta (a=0.6, b=0.4) -> penalises FP more -> tighter borders
python scripts/finetune_seg_tversky.py \
    --data-root ../data/RARPSurgenet/fold1 \
    --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth \
    --out outputs/rarp_tversky_fp \
    --run-name tverskyall-fp \
    --keep-classes 1,2,3,4 \
    --batch-size 8 \
    --lr 1e-4 \
    --epochs 50 \
    --bg-in-loss \
    --alpha 0.6 --beta 0.4 \
    "$@"
