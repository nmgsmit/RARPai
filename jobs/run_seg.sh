#!/bin/bash
#SBATCH --job-name=rarp-seg
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Adjust module versions to your cluster
module load 2023
module load Anaconda3/2023.09-0
module load CUDA/12.1.1

conda activate rarp-seg
cd $SLURM_SUBMIT_DIR

python scripts/run_segmentation.py \
    --config configs/experiments/sam2_video.yaml \
    data.video_path=data/raw/RARP_voorbeeld_A.mp4 \
    output.dir=outputs/RARP_voorbeeld_A/
