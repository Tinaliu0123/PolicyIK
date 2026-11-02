# Real Robot

We train PolicyIK directly on a physical [OpenArm](https://docs.openarm.dev/1.0/) and evaluate it on 60 fixed tasks. Training runs online on the arm with FlashSAC and Hindsight Experience Replay (HER). After the installation, the commands run from `hardware/` on the computer connected to the arm.

## Installation

The code needs ROS 2 Jazzy (Ubuntu 24.04). `install.sh` builds our bringup package in a ROS 2 workspace together with the [OpenArm packages](https://github.com/enactic) at the commits we used, and installs the Python packages into `.venv`.

```bash
mkdir -p ~/openarm_ws/src && cd ~/openarm_ws/src
git clone https://github.com/Tinaliu0123/PolicyIK.git
cd PolicyIK/hardware
bash install.sh
source .venv/bin/activate
```

> [!NOTE]
> Activating `.venv` also loads the ROS 2 workspace. Run `source .venv/bin/activate` in every new shell, including the shells of the motor monitor and the watchdog.

## Quick Start

```bash
sudo ip link set <iface> type can bitrate 1000000 fd off
sudo ip link set <iface> up
ros2 launch openarm_right_bringup openarm_right.launch.py can_interface:=<iface>  # shell 1: start the arm
python3 -m policyik_hw.robot.motor_health --iface <iface>                         # shell 2: motor monitor
bash scripts/train.sh                                                             # shell 3: 120,000 steps into runs/policyik
bash scripts/train_watchdog.sh runs/policyik                                      # shell 4: fault watchdog
bash scripts/evaluate.sh runs/policyik
```

> [!WARNING]
> The gravity feedforward assumes our mounting, in which the z axis of the arm base points down. Read [Robot Setup](#robot-setup) before starting an arm mounted differently.

## Robot Setup

`ros2/openarm_right_bringup` starts one OpenArm arm with ros2_control. It loads our motor gains (`config/control_gains.yaml`), accepts 7 joint positions on `/forward_position_controller/commands`, and runs a gravity feedforward node so that the arm holds its pose.

For a mounting other than ours, set `GRAVITY` in `ros2/openarm_right_bringup/scripts/gravity_node.py` to the gravity vector in the `openarm_link0` frame before the first launch. The training and evaluation scripts stop if they find no controller manager or more than one.

The motor monitor only reads the motor status frames on the CAN bus, and training and evaluation stop when a motor reports a fault.

## Data

| File | Content | Used by |
|---|---|---|
| `data/eval_tasks.json` | 60 tasks, 20 in each start-goal gap bin (20–50 mm, 50–100 mm, over 100 mm) | evaluation |
| `data/demo_tasks.json` | 150 large-rotation tasks | demonstration recording |
| `data/demos.npz` | the recorded demonstrations of these tasks (22,500 transitions) | training |
| `data/demos_meta.json` | the action and reward settings of the recording, which training checks against `config.yaml` | training |
| `data/task_pool.json` | 3,000 long-range tasks that training mixes into its resets | training |

## Demonstrations

Training draws 25% of every batch from scripted demonstrations of large-rotation moves. `data/demos.npz` holds our recording. To record a new one on your arm:

```bash
bash scripts/collect_demos.sh --dry-run     # print the plan without moving the arm
bash scripts/collect_demos.sh --overwrite   # replace data/demos.npz, about 1 h on the arm
```

## Training

The trainer saves a checkpoint and the replay buffer every 10,000 steps. After each save, the watchdog checks the preceding steps for signs of a motor fault, such as stuck joints. On a fault it stops the trainer and parks the arm.

To stop training by hand, press Ctrl-C once. After a stop by hand or by the watchdog, the trainer finishes the current step and saves `checkpoint/` and `buffer` in the run directory. To continue from there:

```bash
STEP=<global_steps_completed in runs/policyik/summary.json>
NAME=policyik_resumed bash scripts/train.sh --init-from runs/policyik/checkpoint --init-buffer runs/policyik/buffer \
    --step-offset $STEP --total-steps $((120000 - STEP))
```

## Evaluation

```bash
bash scripts/evaluate.sh runs/policyik checkpoint_step<N>   # a checkpoint saved during training
```

The script runs the 60 evaluation tasks for 200 steps each with the collision guard off and writes the metrics to `<run>/<checkpoint>_metrics.json`:

- **held**: the position error stays within 1 mm and the orientation error within 0.2° over the final 1, 2, 3 or 5 s of a 15 s or 25 s budget.
- **reached**: both errors stay within these tolerances for 0.5 s at any time in the budget.
- **median errors**: the median over tasks of the position and orientation errors averaged over the final 1 s of the 25 s budget.
