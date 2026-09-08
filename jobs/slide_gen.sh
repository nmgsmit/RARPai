#!/bin/bash
#SBATCH --job-name=slide-gen
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Candidate presentation slides: a generic prostate/urethra view -- no catheter
# labelled, instruments not dominating, urethra reasonably elongated.
#
# Every frame carries instruments (14-34% of the frame, p10-p90), so
# --max-instrument is a percentile choice, not "none".
#
# ONE ROW OF THREE, not six panels: that view exists in only 1 clip at the strict
# setting and 5 at the loosest, so six panels from six distinct surgeries is simply
# not available -- and three reads faster on a slide anyway.
run() {
    python scripts/slide_dice_examples.py \
        --checkpoint outputs/ureth_fn/best.pth \
        --keep-largest --per-group 1 "$@"
}

echo "### medium   (instr<=0.22, elong>=1.5)"
run --no-catheter --max-instrument 0.22 --min-elong 1.5 \
    --pick-seeds 0,1,2 --out outputs/slide_med.png

echo "### loose    (instr<=0.30, elong>=1.3)"
run --no-catheter --max-instrument 0.30 --min-elong 1.3 \
    --pick-seeds 0,1,2 --out outputs/slide_loose.png
