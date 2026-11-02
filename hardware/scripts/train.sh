#!/usr/bin/env bash
# Train on the real arm with the released recipe (config.yaml). Start the arm launch (openarm_right_bringup) first.
# Ctrl-C once: the trainer finishes the step and saves checkpoint + buffer, so the run can be resumed.
#
#   bash scripts/train.sh                                   # 120 k steps into runs/policyik
#   STEP=<global_steps_completed in runs/policyik/summary.json>              # continue a stopped run
#   NAME=policyik_resumed bash scripts/train.sh --init-from runs/policyik/checkpoint --init-buffer runs/policyik/buffer \
#        --step-offset $STEP --total-steps $((120000 - STEP))
# Extra arguments are passed to policyik_hw.train (see --help).
set -eu
cd "$(dirname "$0")/.."
NAME=${NAME:-policyik}
exec python3 -m policyik_hw.train --config config.yaml --name "$NAME" "$@"
