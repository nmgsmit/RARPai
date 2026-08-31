#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --partition=genoa
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -uo pipefail
module load 2023
module load FFmpeg/6.0-GCCcore-12.3.0

FPS=5
SRC=$HOME/data/UMCdissectionvid
DST=$HOME/data/UMCdissectionHD
export FPS DST

extract() {
  f="$1"; name=$(basename "$f" .mp4)
  out="$DST/Train/rarp/$name/clip_000/images"; mkdir -p "$out"
  ffmpeg -nostdin -loglevel error -i "$f" -vf fps=$FPS -q:v 2 "$out/frame_%06d.jpg" && echo "ok $name"
}
export -f extract
ls "$SRC"/*.mp4 | xargs -P 8 -I{} bash -c "extract \"\$@\"" _ {}

cd "$DST/Train/rarp"
mapfile -t vids < <(ls -d */ | sed "s#/##" | sort)
mkdir -p "$DST/Validation/rarp" "$DST/Test/rarp"
for v in "${vids[@]:0:4}"; do mv "$DST/Train/rarp/$v" "$DST/Validation/rarp/"; done
for v in "${vids[@]:4:4}"; do mv "$DST/Train/rarp/$v" "$DST/Test/rarp/"; done

echo "TRAIN $(ls -d $DST/Train/rarp/*/ | wc -l) vids / $(find $DST/Train -name "frame_*.jpg" | wc -l) frames"
echo "VAL   $(ls -d $DST/Validation/rarp/*/ | wc -l) vids"
echo "TEST  $(ls -d $DST/Test/rarp/*/ | wc -l) vids"
