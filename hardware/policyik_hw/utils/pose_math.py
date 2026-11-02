"""Rotation helpers shared by the env (numpy) and the HER buffer (torch)."""

from __future__ import annotations

import numpy as np

__all__ = ["rot6d_from_R", "angle_between", "rotvec_body",
           "keypoint_offsets", "keypoints", "keypoint_error",
           "t_R_from_rot6d", "t_angle_between", "t_rotvec_body", "t_R_from_rotvec", "t_keypoint_error"]


def rot6d_from_R(R: np.ndarray) -> np.ndarray:
    """(3,3) -> (6,): first two columns."""
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[:, 0], R[:, 1]])


def _skew_axis(Rd):
    return np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]])


def angle_between(Ra: np.ndarray, Rb: np.ndarray) -> float:
    Rd = np.asarray(Ra).T @ np.asarray(Rb)
    s = 0.5 * np.linalg.norm(_skew_axis(Rd))
    c = 0.5 * (np.trace(Rd) - 1.0)
    return float(np.arctan2(s, c))


def rotvec_body(R_ee: np.ndarray, R_goal: np.ndarray) -> np.ndarray:
    """Log(R_ee^T R_goal) as an axis-angle vector (3,), in the EE frame."""
    Rd = np.asarray(R_ee).T @ np.asarray(R_goal)
    axis = _skew_axis(Rd)
    s = np.linalg.norm(axis)
    c = np.trace(Rd) - 1.0
    th = np.arctan2(s, c)
    if th < 1e-9:
        return np.zeros(3)
    if s < 1e-6:
        w, V = np.linalg.eigh(Rd)
        axis = V[:, np.argmax(w)]
        return axis / np.linalg.norm(axis) * th
    return axis / s * th


def keypoint_offsets(kp_size: float) -> np.ndarray:
    """(3,3): the three offsets, one per EE axis."""
    return np.eye(3) * float(kp_size)


def keypoints(p: np.ndarray, R: np.ndarray, kp_size: float) -> np.ndarray:
    """(3,3): row k = p + R @ o_k."""
    return np.asarray(p)[None, :] + (np.asarray(R) @ keypoint_offsets(kp_size)).T


def keypoint_error(p, R, p_goal, R_goal, kp_size: float) -> float:
    """Mean over the 3 keypoints of the Euclidean distance, metres."""
    d = keypoints(p, R, kp_size) - keypoints(p_goal, R_goal, kp_size)
    return float(np.linalg.norm(d, axis=1).mean())


def t_R_from_rot6d(v):
    import torch
    a, b = v[:, :3], v[:, 3:6]
    x = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = b - (x * b).sum(dim=1, keepdim=True) * x
    y = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    z = torch.cross(x, y, dim=1)
    return torch.stack([x, y, z], dim=2)


def _t_skew_axis(Rd):
    import torch
    return torch.stack([Rd[:, 2, 1] - Rd[:, 1, 2], Rd[:, 0, 2] - Rd[:, 2, 0], Rd[:, 1, 0] - Rd[:, 0, 1]], dim=1)


def t_angle_between(Ra, Rb):
    import torch
    Rd = Ra.transpose(1, 2) @ Rb
    s = 0.5 * _t_skew_axis(Rd).norm(dim=1)
    c = 0.5 * (Rd.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0)
    return torch.atan2(s, c)


def t_rotvec_body(R_ee, R_goal):
    """[N,3]: Log(R_ee^T R_goal)."""
    import torch
    Rd = R_ee.transpose(1, 2) @ R_goal
    axis = _t_skew_axis(Rd)
    s = axis.norm(dim=1)
    c = Rd.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0
    th = torch.atan2(s, c)
    out = axis / s.clamp_min(1e-6).unsqueeze(1) * th.unsqueeze(1)
    out = torch.where((th < 1e-9).unsqueeze(1), torch.zeros_like(out), out)
    near_pi = s < 1e-6
    if bool(near_pi.any()):
        S = 0.5 * (Rd[near_pi] + Rd[near_pi].transpose(1, 2))
        w, V = torch.linalg.eigh(S)
        ax = V[:, :, -1]
        ax = ax / ax.norm(dim=1, keepdim=True).clamp_min(1e-12)
        out[near_pi] = ax * th[near_pi].unsqueeze(1)
    return out


def t_R_from_rotvec(v):
    """[N,3] axis-angle -> [N,3,3], Rodrigues (safe at theta = 0)."""
    import torch
    th = v.norm(dim=1, keepdim=True)
    k = v / th.clamp_min(1e-12)
    zero = torch.zeros_like(k[:, 0])
    K = torch.stack([torch.stack([zero, -k[:, 2], k[:, 1]], 1),
                     torch.stack([k[:, 2], zero, -k[:, 0]], 1),
                     torch.stack([-k[:, 1], k[:, 0], zero], 1)], 1)
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand(v.shape[0], 3, 3)
    s_, c_ = torch.sin(th).unsqueeze(2), (1 - torch.cos(th)).unsqueeze(2)
    return I + s_ * K + c_ * (K @ K)


def t_keypoint_error(p, R, p_goal, R_goal, kp_size: float):
    """[N]: mean keypoint distance."""
    import torch
    O = torch.eye(3, dtype=p.dtype, device=p.device) * kp_size
    kp = p.unsqueeze(1) + (R @ O).transpose(1, 2)
    kpg = p_goal.unsqueeze(1) + (R_goal @ O).transpose(1, 2)
    return (kp - kpg).norm(dim=2).mean(dim=1)
