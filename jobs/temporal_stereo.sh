#!/bin/bash
#SBATCH --job-name=temporal_stereo
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Temporal stereo on ONE window of ONE 3D clip: 20 fps, per-frame FoundationStereo, then the
# neighbouring frames' disparity warped in along DIS optical flow to close the holes.
#
#   sbatch jobs/temporal_stereo.sh                                   # default window
#   sbatch jobs/temporal_stereo.sh --start 30 --seconds 4 --window 6
#   sbatch jobs/temporal_stereo.sh --matcher sgbm                    # no GPU needed
#
# Writes outputs/temporal_stereo/<name>/{depth,compare}.mp4 + figure.png + stats.json, then
# re-encodes both to H.264 (*_h264.mp4) because opencv only ships the mp4v encoder and mp4v
# does not play in a browser.

cd "$SLURM_SUBMIT_DIR" || exit 1
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

CLIP=${CLIP:-../data/3D_ProxyGT/18de9c5a-7710-438a-83ad-d61ae1582024-01.18.43.675-01.19.58.235-seg3.mp4}
OUT=${OUT:-outputs/temporal_stereo/seg3_t12}

python scripts/temporal_stereo_clip.py \
    --video "$CLIP" \
    --out "$OUT" \
    --start 12 --seconds 5 --fps 20 \
    --matcher ffs --scale 0.5 \
    --window 4 --min-support 2 \
    "$@"

module load FFmpeg/6.0-GCCcore-12.3.0
for f in depth compare; do
    [ -f "$OUT/$f.mp4" ] || continue
    ffmpeg -y -loglevel error -i "$OUT/$f.mp4" \
        -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$OUT/${f}_h264.mp4"
    echo "encoded $OUT/${f}_h264.mp4  $(du -h "$OUT/${f}_h264.mp4" | cut -f1)"
done
