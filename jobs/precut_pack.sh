#!/bin/bash
#SBATCH --job-name=precut-pack
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Footage before the arch-annotated Pure Arch clips (off-frame tips), packed for surgical_map --context.
python scripts/precut_pack.py --short ${SHORTS:-RARP_079 RARP_088 e97cc145 aec6ee2f 5c75328c RARP_087} "$@"
