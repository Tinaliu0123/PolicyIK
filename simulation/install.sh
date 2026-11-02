#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
ISAACLAB_TAG=v2.3.2
OPENARM_COMMIT=bad82e23716e6941c2de78ccb978f57c78b37734

"$PYTHON" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$PYTHON" -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

[ -d third_party/IsaacLab ] || git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
(cd third_party/IsaacLab && ./isaaclab.sh --install rl_games)

[ -d third_party/openarm_isaac_lab ] || git clone https://github.com/enactic/openarm_isaac_lab.git third_party/openarm_isaac_lab
git -C third_party/openarm_isaac_lab checkout "$OPENARM_COMMIT"
"$PYTHON" -m pip install -e third_party/openarm_isaac_lab/source/openarm
"$PYTHON" -m pip install -r ../requirements/simulation.txt
