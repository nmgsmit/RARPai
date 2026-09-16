#!/bin/bash
#SBATCH --job-name=ruler-len
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Ruler LENGTH as each depth model measures it (scripts/ruler_length_models.py). One job per model:
#   for m in unidepth_v2_vitl metric3d_v2_vit_giant2 ...; do sbatch jobs/ruler_length_models.sh --model $m; done
# then CPU seconds anywhere: python scripts/ruler_length_models.py --summarize
# genoa: inference over ~1500 frames, GPU budget reserved for training.
export HF_HUB_OFFLINE=1
python scripts/ruler_length_models.py "$@"
