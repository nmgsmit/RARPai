#!/bin/bash
#SBATCH --job-name=arch_v2
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Re-annotated arches (json only) on the frames/features already packed for the 36-video set; frozen surgical DINOv3
# + arc7 (3 seeds, saved), predict every test frame, then score all training sets on the cutting windows only.
export ARCH_TIP_PURE=$HOME/data/processed/arch_tip_pure_v2
set -e
python scripts/arch_tip_pure.py relabel --from-pack $HOME/data/processed/arch_tip_pure40 --arches-dir $HOME/data/processed/arch_tip_v2_arches
python scripts/arch_tip_pure.py consistency --method arc7 --sets arch_tip_pure_v2 --save
python scripts/arch_tip_window.py --sets arch_tip_pure arch_tip_pure40 arch_tip_pure_v2
