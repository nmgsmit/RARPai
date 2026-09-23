#!/bin/bash
#SBATCH --job-name=goodruler_ud
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=32

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# GoodRulerTest (12 frames, ruler laid along the pre-cut urethra): UniDepth V2 3D length of the
# annotated line vs the ruler reading, one global scale factor. CPU: 12 frames of inference.
python scripts/goodruler_unidepth.py --root ../data/GoodRulerTest \
    --calib outputs/metric_calib_proxy/results.json "$@"
