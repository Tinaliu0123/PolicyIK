"""Evaluate a policy on start and goal tasks, optionally commanding a transient goal first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="PolicyIK-OpenArm-Play")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task_file", required=True, help="tasks (.npz with q_start, goal_pos, goal_quat)")
parser.add_argument("--output", required=True, help="per-step errors and end-effector poses (.npz)")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--eval_seed", type=int, default=123)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--transient_seed", type=int, default=-1, help="transient-goal attempt with this seed (-1: off)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim is up."""
import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_mul,
)
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import policyik_sim.tasks  # noqa: F401

FREEZE = 1.0e6            # disables goal resampling during an attempt
ARM_EXPR = ["openarm_joint[1-7]"]
TRANSIENT_STEPS, TRANSIENT_OFFSET_M, TRANSIENT_ROT_DEG = 40, 0.30, 90.0


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.eval_seed
    agent_cfg["params"]["seed"] = args_cli.eval_seed

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: env)
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    resume_path = retrieve_file_path(args_cli.checkpoint)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    cmd = env.unwrapped.command_manager.get_term("ee_pose")
    pos_term = env.unwrapped.reward_manager.get_term_cfg("end_effector_position_tracking")
    robot = env.unwrapped.scene["robot"]
    arm_ids, _ = robot.find_joints(ARM_EXPR, preserve_order=True)
    arm_ids = list(arm_ids)
    device = env.unwrapped.device

    def pos_err_fn() -> torch.Tensor:
        return pos_term.func(env.unwrapped, **pos_term.params).detach()

    def ori_err_fn() -> torch.Tensor:
        ee_quat = cmd.robot.data.body_quat_w[:, cmd.body_idx]
        return quat_error_magnitude(ee_quat, cmd.pose_command_w[:, 3:]).detach()

    def refresh_manual_goal_world() -> None:
        """Synchronize the cached world-frame command after a direct base-frame command write."""
        cmd.pose_command_w[:, :3], cmd.pose_command_w[:, 3:] = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, cmd.pose_command_b[:, :3], cmd.pose_command_b[:, 3:])

    d = np.load(args_cli.task_file)
    q_start = d["q_start"]
    goal_pos = np.asarray(d["goal_pos"], dtype=np.float32)
    goal_quat = np.asarray(d["goal_quat"], dtype=np.float32)
    m_total = q_start.shape[0]
    goals = np.concatenate([goal_pos, goal_quat], axis=1).astype(np.float32)
    transient = args_cli.transient_seed >= 0
    if transient:   # per-task directions and axes, indexed by the task's position in the task file
        g = np.random.default_rng(args_cli.transient_seed)
        dirs = g.normal(size=(m_total, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        axes = g.normal(size=(m_total, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)

    n = args_cli.num_envs
    T = args_cli.steps
    d_pos = np.full((m_total, T), np.nan, dtype=np.float32)
    d_ori = np.full((m_total, T), np.nan, dtype=np.float32)
    d_ee_pos = np.full((m_total, T, 3), np.nan, dtype=np.float32)
    d_ee_quat = np.full((m_total, T, 4), np.nan, dtype=np.float32)
    d_goal_pos = np.full((m_total, 3), np.nan, dtype=np.float32)
    d_goal_quat = np.full((m_total, 4), np.nan, dtype=np.float32)
    with torch.inference_mode():
        obs0 = env.reset()
        if isinstance(obs0, dict):
            obs0 = obs0["obs"]
        agent.get_batch_size(obs0, 1)
        if agent.is_rnn:
            agent.init_rnn()

        for b in range(0, m_total, n):
            sl = slice(b, min(b + n, m_total))
            m = sl.stop - sl.start
            # rows beyond the end of the task file fill the last batch and are not scored
            goals_batch = np.zeros((n, 7), dtype=np.float32)
            goals_batch[:m] = goals[sl]
            if not transient:
                goals_batch[m:] = goals[sl.start]
            q_batch = np.zeros((n, 7), dtype=np.float32)
            q_batch[:m] = q_start[sl]
            q_batch[m:] = q_start[sl.start]
            goals_b = torch.from_numpy(goals_batch).to(device)
            q_b = torch.as_tensor(q_batch, device=device)
            if transient:
                g_tr = goals_batch.copy()
                g_tr[:m, :3] += (TRANSIENT_OFFSET_M * dirs[sl]).astype(np.float32)
                goals_b_tr = torch.from_numpy(g_tr).to(device)
                ang = torch.full((n,), float(np.deg2rad(TRANSIENT_ROT_DEG)), device=device)
                ax = torch.zeros((n, 3), device=device)
                ax[:m] = torch.as_tensor(axes[sl], device=device, dtype=torch.float32)
                ax[m:, 0] = 1.0
                dq = quat_from_angle_axis(ang, ax)
                goals_b_tr[:, 3:] = quat_mul(dq, goals_b_tr[:, 3:])

            def _pin(t: int) -> None:
                cmd.pose_command_b.copy_(goals_b_tr if (transient and t < TRANSIENT_STEPS) else goals_b)
                refresh_manual_goal_world()

            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            # start state of the task: written after the reset, zero velocity
            robot.write_joint_state_to_sim(q_b, torch.zeros_like(q_b), joint_ids=arm_ids)
            cmd.cfg.resampling_time_range = (FREEZE, FREEZE)
            cmd.time_left.fill_(FREEZE)
            _pin(0)
            # recompute the first observation after writing the start state and the goal
            env.unwrapped.sim.forward()
            fresh = env.unwrapped.observation_manager.compute()["policy"]
            obs = torch.clamp(fresh, -clip_obs, clip_obs)

            for t in range(T):
                _pin(t)
                obs_t = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_t, is_deterministic=True)
                obs, _, _, _ = env.step(actions)
                d_pos[sl, t] = pos_err_fn()[:m].cpu().numpy()
                d_ori[sl, t] = ori_err_fn()[:m].cpu().numpy()
                d_ee_pos[sl, t] = cmd.robot.data.body_pos_w[:, cmd.body_idx][:m].cpu().numpy()
                d_ee_quat[sl, t] = cmd.robot.data.body_quat_w[:, cmd.body_idx][:m].cpu().numpy()
                if t == 0:
                    d_goal_pos[sl] = cmd.pose_command_w[:, :3][:m].cpu().numpy()
                    d_goal_quat[sl] = cmd.pose_command_w[:, 3:][:m].cpu().numpy()
            print(f"[evaluate] tasks {sl.start}-{sl.stop} of {m_total} done", flush=True)

    np.savez(args_cli.output, pos_t=d_pos, ori_t=d_ori, ee_pos_t=d_ee_pos, ee_quat_t=d_ee_quat,
             goal_pos_w=d_goal_pos, goal_quat_w=d_goal_quat)
    print(f"[evaluate] wrote {args_cli.output}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
