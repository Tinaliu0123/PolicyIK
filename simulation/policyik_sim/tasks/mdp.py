"""Observation and reward terms of the PolicyIK task."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_conjugate, quat_mul


def _quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_multiply_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    w = aw * bw - torch.sum(av * bv, dim=-1, keepdim=True)
    v = aw * bv + bw * av + torch.linalg.cross(av, bv, dim=-1)
    return torch.cat([w, v], dim=-1)


def quat_error_rotvec_body(curr_wxyz: torch.Tensor, goal_wxyz: torch.Tensor) -> torch.Tensor:
    """Shortest end-effector-frame rotation vector Log(R_curr^T R_goal)^vee."""
    rel = _quat_multiply_wxyz(_quat_conjugate_wxyz(curr_wxyz), goal_wxyz)
    rel = rel / torch.linalg.norm(rel, dim=-1, keepdim=True).clamp(min=1e-12)
    rel = torch.where(rel[..., :1] < 0.0, -rel, rel)
    vec = rel[..., 1:]
    nv = torch.linalg.norm(vec, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(nv, rel[..., :1].clamp(min=0.0))
    scale = torch.where(nv > 1e-8, angle / nv, torch.full_like(nv, 2.0))
    return vec * scale


def goal_position_obs(env, command_name: str) -> torch.Tensor:
    """Goal position in the robot base frame (3-D)."""
    return env.command_manager.get_command(command_name)[:, :3]


def keypoint_offset_obs(env, command_name: str, asset_cfg: SceneEntityCfg, kp_size: float = 0.30) -> torch.Tensor:
    """Goal-minus-current offsets of three keypoints rigidly attached to the end effector at kp_size * e_i,
    expressed in the robot base frame (9-D)."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b, des_quat_b = command[:, :3], command[:, 3:7]
    root_pos_w, root_quat_w = asset.data.root_pos_w, asset.data.root_quat_w
    des_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, des_pos_b)
    des_quat_w = quat_mul(root_quat_w, des_quat_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    offs = torch.eye(3, device=curr_pos_w.device, dtype=curr_pos_w.dtype) * kp_size
    root_quat_inv = quat_conjugate(root_quat_w)
    out = []
    for i in range(3):
        off = offs[i].expand(curr_pos_w.shape[0], 3)
        kp_cur = curr_pos_w + quat_apply(curr_quat_w, off)
        kp_des = des_pos_w + quat_apply(des_quat_w, off)
        offset_w = kp_des - kp_cur
        out.append(quat_apply(root_quat_inv, offset_w))
    return torch.cat(out, dim=-1)


def _keypoint_offsets(kp_size: float, zero_sum: bool, device, dtype) -> torch.Tensor:
    """Local keypoint offsets kp_size * e_i, optionally centered: kp_size * (e_i - 1/3)."""
    offs = torch.eye(3, device=device, dtype=dtype) * kp_size
    return offs - offs.mean(dim=0, keepdim=True) if zero_sum else offs


def keypoint_error_tanh(env, std: float, command_name: str, asset_cfg: SceneEntityCfg, kp_size: float = 0.30,
                        zero_sum: bool = False) -> torch.Tensor:
    """Keypoint tracking kernel 1 - tanh(eps / std), eps = mean keypoint distance to its goal position."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b, des_quat_b = command[:, :3], command[:, 3:7]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    offs = _keypoint_offsets(kp_size, zero_sum, curr_pos_w.device, curr_pos_w.dtype)
    errs = []
    for i in range(offs.shape[0]):
        off = offs[i].expand(curr_pos_w.shape[0], 3)
        kp_cur = curr_pos_w + quat_apply(curr_quat_w, off)
        kp_des = des_pos_w + quat_apply(des_quat_w, off)
        errs.append(torch.norm(kp_cur - kp_des, p=2, dim=-1))
    kpt_err = torch.stack(errs, dim=1).mean(dim=1)
    return 1.0 - torch.tanh(kpt_err / std)


def ee_position_error_obs(env, command_name: str, asset_cfg: SceneEntityCfg, d0: float) -> torch.Tensor:
    """clip(R_ee^T (p_goal - p_ee) / d0, -1, 1)."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    root_pos_w, root_quat_w = asset.data.root_pos_w, asset.data.root_quat_w
    des_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, command[:, :3])
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    err_body = quat_apply(quat_conjugate(curr_quat_w), des_pos_w - curr_pos_w)
    return torch.clamp(err_body / d0, -1.0, 1.0)


def ee_orientation_error_obs(env, command_name: str, asset_cfg: SceneEntityCfg, d0_rad: float) -> torch.Tensor:
    """clip(Log(R_ee^T R_goal)^vee / d0_rad, -1, 1)."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    root_quat_w = asset.data.root_quat_w
    des_quat_w = quat_mul(root_quat_w, command[:, 3:7])
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    phi = quat_error_rotvec_body(curr_quat_w, des_quat_w)
    return torch.clamp(phi / d0_rad, -1.0, 1.0)


_ERROR_STATE: dict = {}   # id(env) -> [best, command, step, rho, best_before, reset]


def tolerance_error_state(env, command_name: str, asset_cfg: SceneEntityCfg, tau_p: float, tau_r_rad: float):
    """Return (rho_t, b_{t-1}, reset) with rho = max(e_p / tau_p, e_R / tau_r) and b_t = min(b_{t-1}, rho_t).

    The state advances at most once per environment step (all callers of a step see the same values). It is
    re-initialized to the current rho on the first step of an episode and whenever the goal changes; `reset`
    marks those environments."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    root_pos_w, root_quat_w = asset.data.root_pos_w, asset.data.root_quat_w
    des_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, command[:, :3])
    des_quat_w = quat_mul(root_quat_w, command[:, 3:7])
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    e_p = torch.norm(des_pos_w - curr_pos_w, dim=-1)
    e_r = torch.norm(quat_error_rotvec_body(curr_quat_w, des_quat_w), dim=-1)
    rho = torch.maximum(e_p / tau_p, e_r / tau_r_rad)
    cmd = command[:, :7]
    if not hasattr(env, "common_step_counter"):
        raise AttributeError("tolerance_error_state needs env.common_step_counter")
    step = int(env.common_step_counter)
    st = _ERROR_STATE.get(id(env))
    if st is not None and st[2] == step and st[0].shape == rho.shape:
        return st[3], st[4], st[5]
    fresh = env.episode_length_buf <= 1
    if st is None or st[0].shape != rho.shape:
        reset = torch.ones_like(fresh)
        best = rho.clone()
    else:
        prev_cmd = st[1]
        reset = fresh | (cmd != prev_cmd).any(dim=1)
        best = torch.where(reset, rho, st[0])
    best_before = best.clone()
    _ERROR_STATE[id(env)] = [torch.minimum(best, rho).detach().clone(), cmd.detach().clone(),
                             step, rho, best_before, reset]
    return rho, best_before, reset


def error_history_obs(env, command_name: str, asset_cfg: SceneEntityCfg, tau_p: float = 0.001,
                      tau_r_deg: float = 0.2, d0: float = 20.0) -> torch.Tensor:
    """h_t = [clip(b_{t-1} / d0, 0, 1), clip((rho_t - b_{t-1}) / d0, 0, 1)]."""
    rho, best, _ = tolerance_error_state(env, command_name, asset_cfg, tau_p, math.radians(tau_r_deg))
    return torch.stack([torch.clamp(best / d0, 0.0, 1.0),
                        torch.clamp((rho - best) / d0, 0.0, 1.0)], dim=1)


def improvement_reward(env, command_name: str, asset_cfg: SceneEntityCfg, tau_p: float = 0.001,
                       tau_r_deg: float = 0.2, beta: float = 0.5) -> torch.Tensor:
    """[U(rho_t) - U(b_{t-1})]_+ with U(rho) = 0.5 * (1 + tanh((1 - rho) / beta)); zero when the goal changes."""
    rho, best_before, reset = tolerance_error_state(env, command_name, asset_cfg, tau_p, math.radians(tau_r_deg))
    u = 0.5 * (1.0 + torch.tanh((1.0 - rho) / beta))
    ub = 0.5 * (1.0 + torch.tanh((1.0 - best_before) / beta))
    f = (u - ub).clamp(min=0.0)
    return torch.where(reset, torch.zeros_like(f), f)


_POTENTIAL_PREV: dict = {}   # id(env) -> previous potential


def potential_shaping(env, command_name: str, asset_cfg: SceneEntityCfg, gamma: float = 0.99,
                      tau_p: float = 0.001, tau_r_deg: float = 0.2, cap: float = 50.0) -> torch.Tensor:
    """F = gamma * Phi_t - Phi_{t-1} with Phi = -min(cap, rho); zero when the goal changes."""
    rho, _, reset = tolerance_error_state(env, command_name, asset_cfg, tau_p, math.radians(tau_r_deg))
    phi = -torch.clamp(rho, max=cap)
    prev = _POTENTIAL_PREV.get(id(env))
    if prev is None or prev.shape != phi.shape:
        F = torch.zeros_like(phi)
    else:
        F = gamma * phi - prev
        F = torch.where(reset, torch.zeros_like(F), F)
    _POTENTIAL_PREV[id(env)] = phi.detach().clone()
    return F


def backslide_penalty(env, command_name: str, asset_cfg: SceneEntityCfg, tau_p: float = 0.001,
                      tau_r_deg: float = 0.2, cap: float = 3.0) -> torch.Tensor:
    """-min(cap, [rho_t - b_{t-1}]_+); zero when the goal changes."""
    rho, best, reset = tolerance_error_state(env, command_name, asset_cfg, tau_p, math.radians(tau_r_deg))
    pen = torch.clamp(rho - best, min=0.0).clamp(max=cap)
    return -torch.where(reset, torch.zeros_like(pen), pen)


def dwell_reward(env, command_name: str, asset_cfg: SceneEntityCfg, tau_p: float = 0.001,
                 tau_r_deg: float = 0.2, rho_max: float = 1.0) -> torch.Tensor:
    """1[rho_t <= rho_max]; zero when the goal changes."""
    rho, _, reset = tolerance_error_state(env, command_name, asset_cfg, tau_p, math.radians(tau_r_deg))
    r = (rho <= rho_max).to(rho.dtype)
    return torch.where(reset, torch.zeros_like(r), r)


def joint_margin_cost(env, asset_cfg: SceneEntityCfg, command_name: str = "", rho_gate: float = 0.0,
                      body_cfg: SceneEntityCfg | None = None, eps: float = 0.05) -> torch.Tensor:
    """clip(1 - min_j m_j / eps, 0, 1) with m_j the range-normalized distance of joint j to its nearer soft limit;
    applied only while rho > rho_gate when rho_gate > 0."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, ids]
    lim = asset.data.soft_joint_pos_limits[:, ids, :]
    lo, hi = lim[..., 0], lim[..., 1]
    rng = (hi - lo).clamp(min=1e-6)
    margin = (torch.minimum(q - lo, hi - q) / rng).min(dim=-1).values
    pen = (1.0 - margin / eps).clamp(min=0.0, max=1.0)
    if rho_gate > 0.0:
        rho, _, _ = tolerance_error_state(env, command_name, body_cfg, 0.001, math.radians(0.2))
        pen = pen * (rho > rho_gate).to(pen.dtype)
    return pen


def command_state_obs(env, action_name: str = "arm_action") -> torch.Tensor:
    """d_t = z_{t-1} - q_t: retained joint reference of the action term relative to the measured joints (7-D)."""
    term = env.action_manager.get_term(action_name)
    tgt = getattr(term, "_int_target", None)
    if tgt is None:
        raise RuntimeError(f"command_state_obs: action term {action_name!r} keeps no joint reference")
    return tgt - term._asset.data.joint_pos[:, term._joint_ids]
