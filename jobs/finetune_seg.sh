#!/bin/bash
#SBATCH --job-name=rarp-finetune
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# "$@" forwards any extra sbatch args to python, e.g.:
#   sbatch jobs/finetune_seg.sh --img-size 768 --batch-size 4
python scripts/finetune_segmentation.py \
    --data-root ../data/RARPSurgenet/fold1 \
    --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth \
    --out outputs/rarp_finetune \
    "$@"
