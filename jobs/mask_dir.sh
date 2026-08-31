#!/bin/bash
#SBATCH --job-name=mask_dir
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Black out the da Vinci GUI in every video in a directory, and crop away the
# pillarbox black edges + bottom GUI bar (frame shrinks to the content box).
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

# Recursive: raw-videos holds one subdir of clips per segment, so mirror the tree
# under DST rather than flattening (clip_001.mp4 repeats across subdirs).
find "$SRC" -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.mkv' \) \
    | sort | while read -r f; do
    rel="${f#"$SRC"/}"
    out="$DST/${rel%.*}.mp4"
    if [ -s "$out" ]; then
        echo "skip $out"
        continue
    fi
    mkdir -p "$(dirname "$out")"
    echo "=== $f -> $out"
    python scripts/mask_video.py "$f" "$out" "${SLURM_CPUS_PER_TASK:-1}" --crop
done
