#!/bin/bash
#SBATCH --job-name=ureth-sweep
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# One arm of the urethra sweep. Submit as:
#   sbatch jobs/sweep_urethra.sh <tag> [extra args...]
#
# Everything is held at the 512px / DVP-excluded / 50-epoch config so the arms are
# mutually comparable; the extra args are the ONLY difference between them. Every
# arm selects its checkpoint on urethra dice (--select-on urethra), because that is
# the metric being optimised -- selecting on the mean would save an epoch that is
# better at prostate and worse at the class we care about.
#
# 512 not full-res: the full-res run bought nothing on urethra (0.7666 vs 0.7685)
# and cost 16 min/epoch instead of ~1.2. Resolution is a separate question; settle
# the loss first, then re-test resolution on the winner.
TAG="${1:?usage: sweep_urethra.sh <tag> [extra args]}"
shift

python scripts/finetune_seg_tversky.py \
    --clip-root ../data/processed/Segmentation/Nick \
    --label-scheme nick \
    --compare-test ../data/RARPSurgenet/Test \
    --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth \
    --out "outputs/ureth_$TAG" \
    --run-name "ureth-$TAG" \
    --keep-classes 1,2,4,5 \
    --img-size 512 \
    --batch-size 8 \
    --workers 16 \
    --lr 1e-4 \
    --epochs 50 \
    --bg-in-loss \
    --alpha 0.5 --beta 0.5 \
    --select-on urethra \
    "$@"

# Rank-line for the loop: dice / leak / U->P / P->U on the held-out clips.
python scripts/eval_urethra.py \
    --checkpoint "outputs/ureth_$TAG/best.pth" \
    --keep-classes 1,2,4,5 --img-size 512
