#!/bin/bash
#SBATCH --job-name=tversky-ema-5class
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Same script/loss/EMA/aug/ReduceLROnPlateau as tversky-ema, but all 5 classes,
# lr=1e-4, 50 epochs. keep-classes 1,2,3,4 = clean 5-class model (bg is class 0).
# --bg-in-loss: background IS included in the Tversky term here (sharper edges).
python scripts/finetune_seg_tversky.py \
    --data-root ../data/RARPSurgenet/fold1 \
    --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth \
    --out outputs/rarp_tversky_5class \
    --run-name tverskyall \
    --keep-classes 1,2,3,4 \
    --lr 1e-4 \
    --epochs 50 \
    --bg-in-loss \
    "$@"
