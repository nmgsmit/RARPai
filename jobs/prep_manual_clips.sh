#!/bin/bash
#SBATCH --job-name=prep-manual
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Nick's hand-picked camera-motion clips (GUI still on screen) -> Train set that shares
# Validation/Test with depthclips_sharpest_1x. GUI blacked + <i>_mask.png from the templates.
python scripts/prep_manual_clips.py \
    --src ~/data/depthclips_manual_raw \
    --dst ../data/processed/depthclips_manual \
    --zoom-csv ~/zoomcrop/zoom_manual.csv \
    --val-test-from ../data/processed/depthclips_sharpest_1x \
    --workers "${SLURM_CPUS_PER_TASK:-1}" \
    "$@"
