#!/bin/bash
#SBATCH --job-name=mask_dir
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Mask + cut the da Vinci GUI out of every video in a directory.
#   sbatch jobs/mask_dir.sh <src-dir> <dst-dir>
# CPU-only (template matching). Already-written outputs are skipped, so a job
# that hits the walltime can just be resubmitted.
cd "$SLURM_SUBMIT_DIR" || exit 1
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

SRC="${1:?usage: mask_dir.sh <src-dir> <dst-dir>}"
DST="${2:?usage: mask_dir.sh <src-dir> <dst-dir>}"
mkdir -p "$DST"

shopt -s nullglob nocaseglob
for f in "$SRC"/*.mp4 "$SRC"/*.mov "$SRC"/*.avi "$SRC"/*.mkv; do
    out="$DST/$(basename "${f%.*}").mp4"
    if [ -s "$out" ]; then
        echo "skip $out"
        continue
    fi
    echo "=== $f -> $out"
    python scripts/mask_video.py "$f" "$out" "${SLURM_CPUS_PER_TASK:-1}" --crop
done
