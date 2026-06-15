# Environment Setup

## Local (Windows/Linux with NVIDIA GPU)

```bash
# 1. Create and activate environment
conda env create -f environment.yml
conda activate rarp

# 2. Verify GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 3. Quick sanity check
python -c "from sam2.sam2_video_predictor import SAM2VideoPredictor; print('SAM2 OK')"
```

If CUDA version on your machine differs, edit `environment.yml`:
- CUDA 11.8 → replace `cu121` with `cu118`
- CUDA 12.4+ → `cu121` still works, or use `cu124`

Check your CUDA version with: `nvcc --version` or `nvidia-smi`

---

## Supercomputer (SLURM cluster, e.g. Snellius / TU/e HPC)

Supercomputer nodes typically have CUDA pre-installed as a module. Steps:

```bash
# 1. Load required modules (adjust versions to what your cluster offers)
module load 2023
module load Anaconda3/2023.09-0
module load CUDA/12.1.1

# 2. Create environment from file
conda env create -f environment.yml
conda activate rarp-seg

# 3. Verify
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### Example SLURM job script

Save as `jobs/run_seg.sh` and submit with `sbatch jobs/run_seg.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=rarp-seg
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load 2023
module load Anaconda3/2023.09-0
module load CUDA/12.1.1

conda activate rarp
cd $SLURM_SUBMIT_DIR

python scripts/run_segmentation.py \
    --config configs/experiments/sam2_video.yaml \
    data.video_path=data/raw/RARP_voorbeeld_A.mp4 \
    output.dir=outputs/RARP_voorbeeld_A/
```

Submit: `sbatch jobs/run_seg.sh`
Monitor: `squeue -u $USER`
Cancel: `scancel <job_id>`

---

## Updating the environment

```bash
conda activate rarp-seg
pip install -r requirements.txt
```

## Troubleshooting

**SAM2 CUDA extension build fails on install:**
Safe to ignore — SAM2 falls back to a pure Python implementation. Performance is slightly lower but it still runs.

**`torch.cuda.is_available()` returns False on cluster:**
The GPU node may not be allocated yet. Make sure you're running inside a SLURM job (not on the login node).

**Out of memory on high-res video:**
Reduce resolution in config: `data.resize: [360, 640]`
Or switch to a smaller model: `segmentation.params.hf_model_id: facebook/sam2-hiera-tiny`
