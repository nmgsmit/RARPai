#!/bin/bash
#SBATCH --job-name=arch_pure40
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=03:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# The best arch-tip model (surgical DINOv3 + arc7; frozen and last-4-blocks fine-tuned, 3 seeds) redone on the
# 34-video Pure Arch set; test features/tokens are shared with the 14-video set via symlinks.
export ARCH_TIP_PURE=$HOME/data/processed/arch_tip_pure40
set -e
python scripts/arch_tip_pure.py feats --backbones dinov3_surg
python scripts/arch_tip_pure.py run --methods arc7 --backbones dinov3_surg
python scripts/arch_tip_ft.py cache
python scripts/arch_tip_ft.py ft --train-blocks 0 4
