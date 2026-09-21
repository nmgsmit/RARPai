#!/bin/bash
#SBATCH --job-name=arch_noarm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Robot arm as void: predicted arm masks -> arm-blacked DINOv3 features -> baseline vs masked vs masked+void.
python scripts/arch_tip_noarm.py all "$@"
