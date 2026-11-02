"""Accumulated joint-reference action with error-dependent action scaling."""

from __future__ import annotations

import math

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul


class AccumulatedJointReferenceAction(JointPositionAction):
    cfg: "AccumulatedJointReferenceActionCfg"

    def __init__(self, cfg: "AccumulatedJointReferenceActionCfg", env) -> None:
        super().__init__(cfg, env)
        body_ids, _ = self._asset.find_bodies(cfg.ee_body_name)
        self._ee_body_idx = int(body_ids[0])
        self._int_target = self._asset.data.joint_pos[:, self._joint_ids].clone()   # joint reference z
        self._needs_anchor = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        # the start configuration may be written after the reset, so the anchor is taken at the first action
        if env_ids is None:
            self._needs_anchor[:] = True
        else:
            self._needs_anchor[env_ids] = True

    def _gain(self) -> torch.Tensor:
        """Per-environment gain g(e_p, e_R) in [g_min, 1]."""
        cmd = self._env.command_manager.get_command(self.cfg.command_name)
        des_pos_w, _ = combine_frame_transforms(self._asset.data.root_pos_w, self._asset.data.root_quat_w, cmd[:, :3])
        ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_idx]
        d = torch.norm(ee_pos_w - des_pos_w, dim=1)
        t = ((d - self.cfg.d_lo) / max(self.cfg.d_hi - self.cfg.d_lo, 1e-6)).clamp(0.0, 1.0)
        gain = self.cfg.g_min + (1.0 - self.cfg.g_min) * t
        des_quat_w = quat_mul(self._asset.data.root_quat_w, cmd[:, 3:7])
        e_r = quat_error_magnitude(self._asset.data.body_quat_w[:, self._ee_body_idx], des_quat_w)
        lo = math.radians(self.cfg.ori_lo_deg)
        hi = math.radians(self.cfg.ori_hi_deg)
        t_r = ((e_r - lo) / max(hi - lo, 1e-9)).clamp(0.0, 1.0)
        return torch.maximum(gain, self.cfg.g_min + (1.0 - self.cfg.g_min) * t_r)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        gain = self._gain().unsqueeze(-1)
        q_now = self._asset.data.joint_pos[:, self._joint_ids]
        if bool(self._needs_anchor.any()):
            m = self._needs_anchor
            self._int_target = torch.where(m.unsqueeze(-1), q_now, self._int_target)
            self._needs_anchor = torch.zeros_like(self._needs_anchor)
        lim = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        delta = gain * self._raw_actions * self._scale
        self._int_target = torch.clamp(self._int_target + delta, lim[..., 0], lim[..., 1])
        self._processed_actions = self._int_target
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class AccumulatedJointReferenceActionCfg(JointPositionActionCfg):
    class_type: type = AccumulatedJointReferenceAction
    command_name: str = "ee_pose"
    ee_body_name: str = "openarm_hand"
    g_min: float = 0.1           # minimum gain
    d_lo: float = 0.0005         # position ramp (m)
    d_hi: float = 0.00931
    ori_lo_deg: float = 0.415    # orientation ramp (deg)
    ori_hi_deg: float = 2.328
