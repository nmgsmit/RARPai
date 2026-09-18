#!/bin/bash
#SBATCH --job-name=pure_ablate
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Pure Arch input ablation (features / +depth / +anatomy / +both) on one backbone and target.
python scripts/arch_tip_pure.py ablate --backbones dinov3_surg "$@"
