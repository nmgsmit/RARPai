#!/bin/bash
#SBATCH --job-name=arch_unidepth
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Ruler-calibrated UniDepth V2 depth for the best-half arch frames (scripts/arch_tip_prep.py output).
python scripts/arch_tip_unidepth.py "$@"
