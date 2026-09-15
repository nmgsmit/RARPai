#!/bin/bash
#SBATCH --job-name=unidepth
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# UniDepth V2 metric depth + overlays next to the images. Override --dir via "$@".
python scripts/unidepth_overlays.py --dir ../data/others_ruler "$@"
