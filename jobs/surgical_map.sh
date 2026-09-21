#!/bin/bash
#SBATCH --job-name=surg-map
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# 3D map of one video (DA3-L multi-view: shared depth + poses) -> <out-root>/<short>/map.html, foreground
# (ureth_fn: urethra/prostate/catheter/instruments, dilated) masked out of the cloud.
# DA3 code in ~/pylibs/da3, weights in the HF cache. CPU on purpose (GPU budget); 32 frames at 504.
export HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/pylibs/da3
for s in ${SHORTS:-749c8234 RARP_063 46867a8e}; do
  python scripts/surgical_map.py --short $s "$@"
done
