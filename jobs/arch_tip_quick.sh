#!/bin/bash
#SBATCH --job-name=arch_quick
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Quick "does more data help" test: fixed test patients, equal frames per training patient, learning curve.
# GPU because it trains ~70 small heads; encoding the new frames is only ~160 images.
python scripts/arch_tip_quick.py "$@"
