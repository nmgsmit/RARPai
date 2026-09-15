#!/bin/bash
#SBATCH --job-name=zeroshot-proxy
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Zero-shot EndoDAC / Depth Anything V2 / Depth Pro on the stereo proxy-GT (83 frames).
# transformers lives in ~/pylibs/bench (pip --target --no-deps) so the training venv is untouched;
# weights were pre-downloaded into ~/.cache/huggingface on the login node -> run offline.
export PYTHONPATH="$HOME/pylibs/bench${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
python scripts/zeroshot_proxy_gt.py "$@"
