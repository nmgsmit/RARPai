#!/bin/bash
#SBATCH --job-name=scared-sweep
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

# Sweep the depth checkpoints on the same calibrated-stereo SCARED GT so "best" is decided by
# an independent metric (not the RARP photometric proxy). Assumes GT already built:
#   python scripts/export_scared_stereo_gt.py --scared-root ../data/SCARED --out ../data/SCARED/stereo_gt
GT=../data/SCARED/stereo_gt

for CK in rarp_depth depth_s1 depth_s2 depth_s3; do
    echo "===== $CK ====="
    python scripts/eval_scared.py --ckpt outputs/$CK/best.pth \
        --rgb-dir $GT/frames --gt-npz $GT/gt_depths.npz \
        --image-shape 392 490 --run-name $CK-scared-stereo
done

# depth_crop was trained with a hard 10% L/R/bottom crop -> feed the same crop for a fair score.
echo "===== depth_crop (10% L/R/B crop) ====="
python scripts/eval_scared.py --ckpt outputs/depth_crop/best.pth \
    --rgb-dir $GT/frames --gt-npz $GT/gt_depths.npz \
    --image-shape 392 490 --side-crop-frac 0.1 --bottom-crop-frac 0.1 \
    --run-name depth_crop-scared-stereo
