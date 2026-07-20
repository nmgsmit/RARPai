#!/bin/bash
#SBATCH --job-name=depth-clips
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Cue clips over ALL 79 UMCdissectionvid videos (628k frames, all 1920x1080 so the
# hardcoded cue geometry applies), then reshaped into the finetune_depth layout.
# CPU-only: template matching, no GPU. Override args via "$@".
cd "$SLURM_SUBMIT_DIR" || exit 1
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

STAGING=../data/depth_clips_staging
DEST=../data/depth_clips

python scripts/cut_cue_clips.py \
    --videos "/home/nsmit2/data/UMCdissectionvid/*.mp4" \
    --out "$STAGING" \
    --mask-gui \
    --frames \
    --stride 3 \
    --workers "${SLURM_CPUS_PER_TASK:-1}" \
    "$@"

# Split copied from the existing UMC dataset so val/test hold the SAME videos as the
# run we are comparing against; otherwise "more data helped" is unfalsifiable.
python scripts/assemble_depth_clips.py \
    --src "$STAGING" \
    --dst "$DEST" \
    --split-ref ../data/UMCdissectionvid/UMCdissectionimg
