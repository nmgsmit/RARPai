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
python scripts/viz_sul_methods.py || exit 1

module load FFmpeg/6.0-GCCcore-12.3.0
for f in outputs/sul_eval/*_methods.mp4; do
    ffmpeg -y -loglevel error -i "$f" -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart         "${f%.mp4}_h264.mp4"
    echo "encoded ${f%.mp4}_h264.mp4"
done
