#!/bin/bash
# LAUNCHER -- run with `bash jobs/sweep_cue_stride.sh`, NOT sbatch.
#
# The cue clips are 59.9 fps, so consecutive frames carry almost no camera baseline. Sweep the
# triplet stride to find one with real parallax that still fits the short clips (min 15 frames).
# CHAINED via --dependency=afterany so only ONE GPU runs at a time; scancel the pending jobs
# once a winner is clear. Each run logs per-epoch train/val photometric to wandb + SCARED abs_rel.
set -e
STRIDES="3 5 8"
DEP=""
for S in $STRIDES; do
    JID=$(sbatch $DEP --parsable jobs/finetune_depth_cue.sh \
        --frame-stride $S \
        --out outputs/cue_s${S} --run-name cue-s${S})
    echo "submitted stride $S -> job $JID  ${DEP:+(after prev)}"
    DEP="--dependency=afterany:$JID"
done
echo "sweep queued: 1 GPU at a time. Compare val/photo curves + scared/abs_rel across the runs."
