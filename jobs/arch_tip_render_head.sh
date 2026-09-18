#!/bin/bash
#SBATCH --job-name=render_head
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=8

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Render held-out head predictions over a whole video (CPU). e.g. --video cada5bef --head outputs/arch_tip_pure/pred --variant tip_dinov3_surg
python scripts/arch_tip_render.py "$@"
