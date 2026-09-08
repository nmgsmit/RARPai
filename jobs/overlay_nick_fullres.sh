#!/bin/bash
#SBATCH --job-name=overlay-nick-fullres
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

# Overlays for the full-res no-DVP run. Submit with a dependency so it fires by
# itself the moment training finishes, with nobody watching:
#   sbatch --dependency=afterok:<trainjobid> jobs/overlay_nick_fullres.sh
#
# --height/--width MUST match training (1072x1340 -> round32 -> 1088x1344): feeding
# a 512 square to a full-res model gets both the scale and the aspect wrong.
# --keep-classes MUST match too -- the model emits COMPACT ids, so without it
# compact 3 (catheter) would be drawn blue and labelled dorsal venous plexus.
# Same --images/--n/--seed as the 512 run's sheet, so the two are directly comparable.
# GPU because a 1088x1344 forward pass on CPU is minutes per image.
python scripts/overlay_dir.py \
    --images ../data/processed/UMCsulsnaps/sul_relaxed/images \
    --checkpoint outputs/rarp_nick_fullres/best.pth \
    --keep-classes 1,2,4,5 \
    --height 1072 --width 1340 \
    --n 6 --seed 0 \
    --out outputs/sul_relaxed_fullres.png \
    "$@"
