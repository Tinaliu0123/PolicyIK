"""Start and goal sampling from pools of joint configurations and their end-effector poses."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

_ARM_EXPR = ["openarm_joint[1-7]"]
_CACHE: dict = {}   # pool joint positions (per file) and arm joint ids (per asset)


def reset_joints_from_pool(env, env_ids: torch.Tensor, pool_npz: str,
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Reset event: set the arm joints to configurations drawn uniformly from the pool, with zero velocity."""
    asset = env.scene[asset_cfg.name]
    qkey = ("q", pool_npz)
    if qkey not in _CACHE:
        q = np.load(pool_npz)["q"].astype(np.float32)
        _CACHE[qkey] = torch.as_tensor(q, device=env.device)
    akey = ("ids", asset_cfg.name)
    if akey not in _CACHE:
        arm_ids, _ = asset.find_joints(_ARM_EXPR, preserve_order=True)
        _CACHE[akey] = list(arm_ids)
    q_pool = _CACHE[qkey]
    arm_ids = _CACHE[akey]
    n = env_ids.shape[0] if torch.is_tensor(env_ids) else len(env_ids)
    idx = torch.randint(0, q_pool.shape[0], (n,), device=env.device)
    joint_pos = q_pool[idx]
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=arm_ids, env_ids=env_ids)


class PoolPoseCommand(UniformPoseCommand):
    """Pose command whose goals are rows drawn uniformly from a pool (position and, optionally, orientation of
    the same row)."""

    cfg: "PoolPoseCommandCfg"

    def __init__(self, cfg: "PoolPoseCommandCfg", env):
        super().__init__(cfg, env)
        pool = np.load(cfg.pool_npz)
        self._pool_pos, self._pool_quat = pool["ee_pos"], pool["ee_quat"]
        self._rng = np.random.default_rng(cfg.sampler_seed)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        rows = self._rng.integers(0, self._pool_pos.shape[0], size=len(env_ids))
        pos, quat = self._pool_pos[rows], self._pool_quat[rows]
        self.pose_command_b[env_ids, :3] = torch.as_tensor(pos, dtype=self.pose_command_b.dtype, device=self.device)
        if self.cfg.use_sampled_orientation:
            self.pose_command_b[env_ids, 3:7] = torch.as_tensor(
                quat, dtype=self.pose_command_b.dtype, device=self.device
            )


@configclass
class PoolPoseCommandCfg(UniformPoseCommandCfg):
    class_type: type = PoolPoseCommand
    pool_npz: str = MISSING
    sampler_seed: int = 7
    use_sampled_orientation: bool = False
