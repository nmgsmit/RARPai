#!/bin/bash
#SBATCH --job-name=extract_ruler
#SBATCH --partition=genoa
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -uo pipefail
module load 2023
module load FFmpeg/6.0-GCCcore-12.3.0

FPS=5
SRC=$HOME/data/UMCrulervidNOgui
RUL=$HOME/data/UMCrulerHD
MIX=$HOME/data/UMCmixed
OLD=$HOME/data/UMCdissectionHD
export FPS RUL

extract() {
  f="$1"; name=$(basename "$f" .mp4)
  out="$RUL/$name/clip_000/images"; mkdir -p "$out"
  ffmpeg -nostdin -loglevel error -i "$f" -vf fps=$FPS -q:v 2 "$out/frame_%06d.jpg" && echo "ok $name"
}
export -f extract
ls "$SRC"/*.mp4 | xargs -P 8 -I{} bash -c "extract \"\$@\"" _ {}

# mixed root: Train = 14 ruler vids + 5 old vids (replay ~50/50 by frames)
# Validation = old 4 vids + 2 ruler vids (ruler in wandb qual panel)
# Test = old Test untouched (photo comparable to 0.0883 = forgetting check)
rm -rf "$MIX"; mkdir -p "$MIX/Train/rarp" "$MIX/Validation/rarp"
mapfile -t rul < <(ls -d "$RUL"/*/ | sed "s#/\$##" | sort)
n=${#rul[@]}
for v in "${rul[@]:0:n-2}"; do ln -s "$v" "$MIX/Train/rarp/"; done
for v in "${rul[@]:n-2:2}"; do ln -s "$v" "$MIX/Validation/rarp/"; done
for v in $(ls -d "$OLD/Train/rarp"/*/ | sed "s#/\$##" | head -5); do ln -s "$v" "$MIX/Train/rarp/"; done
for v in $(ls -d "$OLD/Validation/rarp"/*/ | sed "s#/\$##"); do ln -s "$v" "$MIX/Validation/rarp/"; done
ln -s "$OLD/Test" "$MIX/Test"

echo "TRAIN vids=$(ls "$MIX/Train/rarp" | wc -l) frames=$(find -L "$MIX/Train" -name "frame_*.jpg" | wc -l)"
echo "VAL   vids=$(ls "$MIX/Validation/rarp" | wc -l) frames=$(find -L "$MIX/Validation" -name "frame_*.jpg" | wc -l)"
echo "ruler frames total: $(find "$RUL" -name "frame_*.jpg" | wc -l)"
