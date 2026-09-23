#!/bin/bash
#SBATCH --job-name=sul_arch_cyl
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=32

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Standard SUL: cylinder depth-step start + arch model (pure40) end, calibrated UniDepth mm.
# CPU: a handful of frames through UniDepth ViT-L + DINOv3 ViT-L.
python scripts/sul_arch_cyl.py "$@"
