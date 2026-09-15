#!/bin/bash
#SBATCH --job-name=arch_cv
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=128

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Improved arch fit + head refinement + temporal smoothing + 20-split report -> outputs/arch_tip_cv.
python scripts/arch_tip_cv.py --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
