#!/usr/bin/env bash
# no -u: the ROS 2 setup scripts read unset variables
set -eo pipefail
cd "$(dirname "$0")"
WS="$(cd ../../.. && pwd)"  # this file is <workspace>/src/PolicyIK/hardware/install.sh
OPENARM_CAN_COMMIT=4063506136c60f62dbd159c0ce150094426279d7
OPENARM_ROS2_COMMIT=01b9f3815222b490a593f01379d8e2106c8da6a0
OPENARM_DESCRIPTION_COMMIT=846a82d8baf64efdbb7f55c7f64b9b004939061b
source /opt/ros/jazzy/setup.bash

for repo in openarm_can:$OPENARM_CAN_COMMIT openarm_ros2:$OPENARM_ROS2_COMMIT openarm_description:$OPENARM_DESCRIPTION_COMMIT; do
    name=${repo%%:*}
    [ -d "$WS/src/$name" ] || git clone "https://github.com/enactic/$name.git" "$WS/src/$name"
    git -C "$WS/src/$name" checkout "${repo#*:}"
done

(
    cd "$WS"
    rosdep update
    rosdep install --ignore-src -y --from-paths src/PolicyIK/hardware/ros2 src/openarm_ros2/openarm_hardware \
        src/openarm_can src/openarm_description
    colcon build --symlink-install --packages-up-to openarm_right_bringup
)
source "$WS/install/setup.bash"

# --system-site-packages keeps the ROS 2 Python packages visible; activating the venv also loads the workspace.
[ -d .venv ] || python3 -m venv --system-site-packages .venv
grep -qF "$WS/install/setup.bash" .venv/bin/activate || echo "source \"$WS/install/setup.bash\"" >> .venv/bin/activate
.venv/bin/pip install -r ../requirements/hardware.txt
