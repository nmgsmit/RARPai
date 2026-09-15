#!/bin/bash
#SBATCH --job-name=arch_head
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Arch tip head (rgb, rgbd) over the 20 patient splits -> outputs/arch_tip_cv/head.
python scripts/arch_tip_head.py "$@"
