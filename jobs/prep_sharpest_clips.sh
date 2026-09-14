#!/bin/bash
#SBATCH --job-name=prep-sharpest
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=32

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Sharpest (var of Laplacian) cue clip > 1 MB per patient from depth_clips_staging, cropped to
# 5:4 content, for the relative-depth run in jobs/finetune_depth_sharpest.sh.
python scripts/prep_sharpest_clips.py \
    --src /home/nsmit2/data/depth_clips_staging \
    --dst ../data/processed/depthclips_sharpest \
    --workers "${SLURM_CPUS_PER_TASK:-1}" \
    "$@"
