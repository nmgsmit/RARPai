#!/bin/bash
#SBATCH --job-name=depth-sharpest
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Does sharp, patient-diverse data make the mono depth converge to the stereo proxy-GT better
# than the ruler clips? Same settings as endodac-ruler-range-sw05 EXCEPT --scale-w 0: these clips
# carry no known-size objects, so the depth is RELATIVE and best.pth is picked on proxy_gt
# ms_abs_rel (median-scaled shape). Compare against the ruler run's proxy_gt/ms_abs_rel, not abs_rel.
python scripts/finetune_depth.py \
    --data-root ../data/processed/depthclips_sharpest \
    --init ../backbones/EndoDAC/depth_model.pth \
    --pose-init-dir ../backbones/EndoDAC \
    --out outputs/depth_sharpest \
    --run-name endodac-sharpest \
    --image-shape 392 490 \
    --scale-w 0 \
    --anchor-w 0.3 \
    --min-depth 20 \
    --max-depth 200 \
    --no-learn-intrinsics \
    --epochs 3 \
    "$@"
