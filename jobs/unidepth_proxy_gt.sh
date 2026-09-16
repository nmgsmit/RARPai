#!/bin/bash
#SBATCH --job-name=unidepth_pgt
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=32

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# UniDepth V2 on the stereo proxy-GT rectified LEFT eyes, next to them, so gui_depth_measure
# (precomputed/unidepth) shows UniDepth vs stereo on the same pixels. K = calib P1 normalised to
# 1340x1072, no crop: the map must cover the exact frame the stereo depth16 lives in.
# CPU (genoa): 83 frames of inference, GPU budget is reserved for training.
python scripts/unidepth_overlays.py --dir ../data/processed/proxy_gt_nogui_ffs --glob "*_left.png" \
    --no-crop --intrinsics 0.8530683 1.0663354 0.4503314 0.4980683 "$@"
