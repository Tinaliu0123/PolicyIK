#!/usr/bin/env bash
# Evaluate one checkpoint on the 60 evaluation tasks (three start-goal gap bins x 20 tasks) at the training control rate,
# without the collision guard, then compute the held-pose metrics.
# Ctrl-C stops after the current task (recorded as skipped); running the same command again resumes.
#
#   bash scripts/evaluate.sh <run_dir> [checkpoint_dir_name]
#   bash scripts/evaluate.sh runs/policyik_resumed                  # its final checkpoint
set -eu
[ $# -ge 1 ] || { sed -n 2,7p "$0"; exit 2; }
cd "$(dirname "$0")/.."
RUN=$1; CKPT=${2:-checkpoint}
TASKS=data/eval_tasks.json
for BIN in 0.02,0.05 0.05,0.10 0.10,1.0; do
    python3 -m policyik_hw.evaluate --run "$RUN" --checkpoint "$CKPT" --task-file "$TASKS" --gap-bin "$BIN" \
        --n-episodes 20 --max-steps 200 --no-collision-guard
done
# the 15 s and 25 s budgets are counted in steps at 8.13 Hz = 1 / 0.123 s, the training step period
python3 -m policyik_hw.metrics --run "$RUN" --checkpoint "$CKPT" --task-file "$TASKS" --eval-hz 8.13
