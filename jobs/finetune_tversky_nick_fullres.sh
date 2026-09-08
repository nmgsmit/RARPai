#!/bin/bash
#SBATCH --job-name=tversky-nick-fullres
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=18:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Follow-up to rarp_nick_dice (512px, all 5 classes): DVP dropped, native resolution.
#
# RESOLUTION: every one of the 56 clips is 1340x1072 (aspect exactly 1.2500 = 5:4).
# CAFormer downsamples by 32, so H,W must be multiples of 32 -- round32 takes
# 1072 -> 1088 and 1340 -> 1344. That is an UPSAMPLE, so no detail is discarded;
# the cost is a 1.2% aspect stretch (1.250 -> 1.235) applied identically to images
# and masks. The aspect-exact alternative on the /32 grid is 1024x1280, which would
# instead throw away 4.5% of the linear resolution -- the wrong trade when the point
# of this run is detail on thin structures.
#
# CLASSES: 1,2,4,5 = urethra, prostate, catheter, non-anatomical. Dorsal venous
# plexus (3) is EXCLUDED -- it scored 0.2065 test dice at 512px on 0.76% of pixels
# and never got off 0.0000 until late. Its pixels fall to background.
#
# BATCH: 1088x1344 is 5.6x the pixels of 512x512, so batch 2 x accum 4 keeps the
# effective batch at 8, identical to the 512 run -- the optimizer trajectory stays
# comparable and only the resolution changes. Workers 16 (up from 8) because
# decoding 1340x1072 JPEGs is the new bottleneck, not the GPU.
# Expect ~6h for 50 epochs (the 512 run did 50 in 1h01m); 18h walltime is slack.
python scripts/finetune_seg_tversky.py \
    --clip-root ../data/processed/Segmentation/Nick \
    --label-scheme nick \
    --compare-test ../data/RARPSurgenet/Test \
    --encoder-ckpt ../backbones/RARP_checkpoint_epoch0050_teacher.pth \
    --out outputs/rarp_nick_fullres \
    --run-name nick-fullres-noDVP \
    --keep-classes 1,2,4,5 \
    --height 1072 --width 1340 \
    --batch-size 2 \
    --accum-steps 4 \
    --workers 16 \
    --lr 1e-4 \
    --epochs 50 \
    --bg-in-loss \
    --alpha 0.5 --beta 0.5 \
    "$@"
