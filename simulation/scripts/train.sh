#!/usr/bin/env bash
# Train one seed in three PPO segments and average its last 20 checkpoints (usage: scripts/train.sh <seed>).
set -euo pipefail
cd "$(dirname "$0")/.."
# openarm_isaac_lab imports its own assets as `source.openarm...`, which resolves from its repository root.
export PYTHONPATH="$PWD/third_party/openarm_isaac_lab${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
SEED="${1:?usage: scripts/train.sh <seed>}"

segment() {   # <run name> <final epoch> [checkpoint to resume]
  "$PYTHON" -m policyik_sim.train --task PolicyIK-OpenArm --headless --num_envs 4096 --max_iterations "$2" \
      --seed "$SEED" ${3:+--checkpoint "$3"} agent.params.config.name="$1"
}
last() {      # <run name> <epoch>: the periodic checkpoint saved at that epoch
  ls logs/rl_games/"$1"/*/nn/last_"$1"_ep_"$2"_rew_*.pth | grep -E "_rew_-?[0-9.]+\.pth$" | tail -n 1
}

segment "policyik_s${SEED}_ep3500" 3500
ckpt=$(last "policyik_s${SEED}_ep3500" 3500)
segment "policyik_s${SEED}_ep8500" 8500 "$ckpt"
ckpt=$(last "policyik_s${SEED}_ep8500" 8500)
segment "policyik_s$SEED" 13000 "$ckpt"
"$PYTHON" -m policyik_sim.average_checkpoints --run_dir "logs/rl_games/policyik_s$SEED" \
    --out "logs/rl_games/policyik_s$SEED/checkpoint_average.pth"
