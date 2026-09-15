#!/bin/bash
#SBATCH --job-name=arch_feats
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# SurgeNet encoder features (--what feats) or rarp_nick_fullres tool masks (--what tools) for arch_tip_all.
python scripts/arch_tip_features.py "$@"
