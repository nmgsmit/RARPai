#!/bin/bash
#SBATCH --job-name=pure_predict
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Pure Arch: retrain (per seed) and predict every frame of one test video -> outputs/arch_tip_pure/pred.
python scripts/arch_tip_pure.py predict --backbones dinov3_surg "$@"
