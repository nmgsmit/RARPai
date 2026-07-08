#!/bin/bash
#SBATCH --job-name=rescore
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Re-score the 4 crop/GUI best.pth on SCARED with the CORRECTED content-relative crop. SCARED is
# already 5:4 content, so we strip only the part of the training side-crop that goes beyond the UMC
# bar removal (bar_side=0.15): A/dropgui -> side 0; botcrop -> side 0.030; B -> side 0.129.
GT="../data/SCARED/stereo_gt/gt_depths.npz"
RGB="../data/SCARED/stereo_gt/frames"
COMMON="--gt-npz $GT --rgb-dir $RGB --image-shape 392 490 --no-wandb"

echo "===== A  umc_s1_cropLR  (side 0) ====="
python scripts/eval_scared.py --ckpt outputs/umc_s1_cropLR/best.pth $COMMON \
    --side-crop-frac 0.0 --bottom-crop-frac 0.0

echo "===== B  umc_s1_cropanat  (side 0.1286, bottom 0.222, top 0.037) ====="
python scripts/eval_scared.py --ckpt outputs/umc_s1_cropanat/best.pth $COMMON \
    --side-crop-frac 0.1286 --bottom-crop-frac 0.222 --top-crop-frac 0.037

echo "===== dropgui  umc_s1_dropgui  (side 0) ====="
python scripts/eval_scared.py --ckpt outputs/umc_s1_dropgui/best.pth $COMMON \
    --side-crop-frac 0.0 --bottom-crop-frac 0.0

echo "===== botcrop  umc_s1_dropgui_botcrop  (side 0.0303, bottom 0.0648) ====="
python scripts/eval_scared.py --ckpt outputs/umc_s1_dropgui_botcrop/best.pth $COMMON \
    --side-crop-frac 0.0303 --bottom-crop-frac 0.0648
