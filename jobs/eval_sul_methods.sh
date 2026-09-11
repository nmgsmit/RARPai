#!/bin/bash
#SBATCH --job-name=eval_sul
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# SUL methods head to head on Nick's annotated frames (hand masks + ruler points under
# outputs/temporal_stereo/<run>/), with stereo and monocular depth. -> outputs/sul_eval/
#   sbatch jobs/eval_sul_methods.sh

cd "$SLURM_SUBMIT_DIR" || exit 1
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

python scripts/eval_sul_methods.py "$@"
