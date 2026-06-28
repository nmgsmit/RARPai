#!/bin/bash
#SBATCH --job-name=depth-endodac
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

# Self-supervised EndoDAC depth fine-tune on RARP videos (mono, photometric loss).
# Warm-starts from EndoDAC's released depth_model.pth in ~/backbones/EndoDAC.
# Override hyperparams via "$@" (e.g. --image-shape 392 490 --epochs 30 --frame-stride 2).
python scripts/finetune_depth.py \
    --data-root ../data/RARPAtlas \
    --init ../backbones/EndoDAC/depth_model.pth \
    --pose-init-dir ../backbones/EndoDAC \
    --out outputs/rarp_depth \
    --run-name endodac-rarp \
    --image-shape 392 490 \
    --epochs 20 \
    "$@"
