#!/bin/bash
#SBATCH --job-name=arch_consist
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Frame-to-frame tip consistency + offset/scatter: 14- vs 36-video training set, all frames of the 7 test videos.
python scripts/arch_tip_pure.py consistency --method arc7 "$@"
