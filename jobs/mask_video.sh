#!/bin/bash
#SBATCH --job-name=mask_video
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

python scripts/mask_video.py \
    data/raw/RARP_voorbeeld_A.mp4 \
    data/raw/RARP_voorbeeld_A_masked.mp4 \
    "${SLURM_CPUS_PER_TASK:-1}" "$@"
