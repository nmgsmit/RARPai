#!/bin/bash
#SBATCH --job-name=proxy_gt_ffs
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Metric proxy-GT from the 3D SBS clips with C-Fast-FoundationStereo.
# Model + code live OUTSIDE the repo (they are large and not ours):
#   ~/Fast-FoundationStereo                       (NVlabs, git clone)
#   ~/Fast-FoundationStereo/weights/c-fast/       (nvidia/c-fast-foundationstereo on HF)
# Deps went into the existing venv; torch stayed at 2.12.0+cu130.
#
#   sbatch jobs/proxy_gt_ffs.sh                          # full run
#   sbatch jobs/proxy_gt_ffs.sh --stride 60              # quick smoke test

cd "$SLURM_SUBMIT_DIR" || exit 1
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

python scripts/make_stereo_proxy_gt.py \
    --matcher ffs \
    --src ../data/3D_ProxyGT \
    --dst ../data/processed/proxy_gt_ffs \
    --stride 6 \
    --scale 0.5 \
    "$@"
