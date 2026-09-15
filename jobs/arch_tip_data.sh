#!/bin/bash
#SBATCH --job-name=arch_data
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=128

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# All frames + GUI masks + labels of the 10 arch-annotated videos. Frames go to scratch (home ~93% full).
mkdir -p /scratch-shared/$USER/arch_tip/all
ln -sfn /scratch-shared/$USER/arch_tip/all ../data/processed/arch_tip_all
python scripts/arch_tip_data.py --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
