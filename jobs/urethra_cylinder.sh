#!/bin/bash
#SBATCH --job-name=urethra_cyl
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Urethra cylinder + roof end point on a temporal-stereo run (needs its --save-depth output).
#   sbatch --export=ALL,RUN=outputs/temporal_stereo/seg3_t30 jobs/urethra_cylinder.sh
#   sbatch --export=ALL,RUN=outputs/temporal_stereo/seg3_t30 jobs/urethra_cylinder.sh --margin 2

cd "$SLURM_SUBMIT_DIR" || exit 1
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

RUN=${RUN:?set RUN=<temporal stereo output dir>}
python scripts/urethra_cylinder.py --run "$RUN" "$@" || exit 1

module load FFmpeg/6.0-GCCcore-12.3.0
# SLOW=N plays the video N x slower (encode only; the analysis and its time axis are untouched)
ffmpeg -y -loglevel error -i "$RUN/urethra_cyl/overlay.mp4" -vf "setpts=${SLOW:-1}*PTS" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$RUN/urethra_cyl/overlay_h264.mp4"
echo "encoded $RUN/urethra_cyl/overlay_h264.mp4"
