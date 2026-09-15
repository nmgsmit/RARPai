#!/bin/bash
#SBATCH --job-name=metric-calib
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Calibrate 6 depth models ONCE (scale+shift in inverse depth) on the ruler set's known-mm objects,
# then score ABSOLUTE mm depth on the stereo proxy-GT. Model code in ~/pylibs (per-model PYTHONPATH
# set inside the script), weights pre-downloaded -> offline. Pass --exclude-videos for zoomed clips.
export HF_HUB_OFFLINE=1
python scripts/metric_calib_proxy.py --all "$@"
