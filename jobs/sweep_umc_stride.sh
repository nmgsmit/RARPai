#!/bin/bash
# LAUNCHER -- run with `bash jobs/sweep_umc_stride.sh`, NOT sbatch.
#
# Credit-friendly frame-stride sweep on UMCdissectionimg to compare hyperparameters cheaply:
#   - short: --epochs 10 --sample-frac 0.1 (~5k triplets) -> ~40-50 min each on H100
#   - CHAINED via --dependency=afterany so only ONE GPU runs at a time (controlled burn;
#     scancel the pending jobs the moment a winner is clear)
#   - each auto-logs SCARED abs_rel -> that's the ranking metric (coarse screen, not final)
# Winner gets a full-length run afterwards (that's where an ensemble would slot in).
set -e
STRIDES="1 3 5"
DEP=""
for S in $STRIDES; do
    JID=$(sbatch $DEP --time=01:00:00 --parsable jobs/finetune_depth_umc.sh \
        --frame-stride $S --epochs 10 --sample-frac 0.1 \
        --out outputs/umc_s${S}_short --run-name umc-s${S}-short)
    echo "submitted stride $S -> job $JID  ${DEP:+(after prev)}"
    DEP="--dependency=afterany:$JID"
done
echo "sweep queued: 1 GPU at a time, ~40-50 min/job. Compare scared/abs_rel across the runs."
