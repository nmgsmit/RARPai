#!/bin/bash
#SBATCH --job-name=tversky-nick
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Retrain the current best all-class config (rarp_tversky_dice: 512px, batch 8,
# lr 1e-4, 50 epochs, symmetric Tversky = Dice, bg in loss) on the NEW clip-layout
# annotations. Every hyperparameter below is copied from jobs/finetune_tversky_dice.sh;
# only the data and the label scheme change, so the comparison isolates the data.
#
# --keep-classes 1,2,3,4,5 = all available classes in the new scheme:
#   1 urethra  2 prostate  3 dorsal venous plexus  4 catheter  5 non-anatomical
# (legend: transfer_atlas_mod/gui/cutie/utils/palette.py)
#
# Two test numbers come out of this:
#   [test]    held-out CLIPS from the new data -- the honest number, not comparable
#   [compare] the old 60-frame RARPSurgenet test set, shared classes only
#             (urethra / prostate / catheter) -- comparable to rarp_tversky_dice,
#             which scored catheter=0.8567 urethra=0.8167 there.
# Walltime is 12h not 8h: 6584 frames vs the old 378 is ~17x the steps per epoch.
python scripts/finetune_seg_tversky.py \
    --clip-root ../data/processed/Segmentation/Nick \
    --label-scheme nick \
    --compare-test ../data/RARPSurgenet/Test \
    --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth \
    --out outputs/rarp_nick_dice \
    --run-name nick-dice-allclass \
    --keep-classes 1,2,3,4,5 \
    --batch-size 8 \
    --lr 1e-4 \
    --epochs 50 \
    --bg-in-loss \
    --alpha 0.5 --beta 0.5 \
    "$@"
