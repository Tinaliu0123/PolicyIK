"""Dense pose-reaching reward and its batched torch version used by hindsight relabelling."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

__all__ = ["PoseRewardConfig", "PoseReward", "rho_best_channels", "u_of",
           "torch_pose_reward", "torch_rho_best_channels"]


def u_of(rho, tau_u: float = 0.5):
    return 0.5 * (1.0 + np.tanh((1.0 - np.asarray(rho, dtype=np.float64)) / tau_u))


def rho_best_channels(rho, rho_best_prev, d0: float = 20.0) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64); best = np.asarray(rho_best_prev, dtype=np.float64)
    return np.stack([np.clip(best / d0, 0.0, 1.0), np.clip((rho - best) / d0, 0.0, 1.0)], axis=-1)


@dataclass
class PoseRewardConfig:
    tau_p: float = 0.001
    tau_r_deg: float = 0.2
    tau_u: float = 0.5
    kp_size: float = 0.45

    w_pos_linear: float = -0.2
    w_tanh_fine: float = 0.1
    std_tanh_fine: float = 0.10
    w_kp_wide: float = 1.0
    std_kp_wide: float = 0.20
    w_kp_sharp: float = 1.0
    std_kp_sharp: float = 0.05
    w_action_rate: float = -0.05
    w_joint_vel: float = -0.0005

    w_bsp: float = 0.2
    bsp_cap: float = 3.0
    w_snlb: float = 26.831
    w_dwell: float = 1.0
    dwell_rho: float = 1.0
    w_pbrs: float = 1.0
    pbrs_cap: float = 500.0
    pbrs_gamma: float = 0.98
    w_limit_margin: float = 1.0
    limit_eps: float = 0.05
    limit_gate_rho: float = 10.0

    rho_obs_d0: float = 20.0
    scale_by_dt: bool = True
    dt: float = 0.1

    @property
    def tau_r(self) -> float:
        return float(np.radians(self.tau_r_deg))

    def validate(self, agent_gamma: Optional[float] = None) -> None:
        if min(self.tau_p, self.tau_r_deg, self.tau_u, self.kp_size) <= 0:
            raise ValueError("tau_p, tau_r_deg, tau_u, kp_size must be > 0")
        for n in ("std_tanh_fine", "std_kp_wide", "std_kp_sharp", "bsp_cap", "pbrs_cap",
                  "limit_eps", "dwell_rho"):
            if getattr(self, n) <= 0:
                raise ValueError(f"{n} must be > 0")
        for n in ("w_bsp", "w_snlb", "w_dwell", "w_pbrs", "w_limit_margin"):
            if getattr(self, n) < 0:
                raise ValueError(f"{n} is a magnitude (>= 0); the sign is inside the term")
        if self.w_pbrs and agent_gamma is not None and abs(self.pbrs_gamma - agent_gamma) > 1e-9:
            raise ValueError(f"pbrs_gamma {self.pbrs_gamma} != agent gamma {agent_gamma}")
        if self.scale_by_dt and self.dt <= 0:
            raise ValueError("dt must be > 0")

    def limit_gated(self, rho) -> bool:
        return rho > self.limit_gate_rho

    def rho_of(self, d, theta):
        return max(float(d) / self.tau_p, float(theta) / self.tau_r)

    def to_dict(self) -> dict:
        return asdict(self)


class PoseReward:
    def __init__(self, cfg: PoseRewardConfig, lim_lo, lim_hi):
        cfg.validate()
        self.cfg = cfg
        self.lo = np.asarray(lim_lo, dtype=np.float64)
        self.hi = np.asarray(lim_hi, dtype=np.float64)
        self.reset(rho_reset=0.0)

    def reset(self, rho_reset: float, prev_action=None) -> None:
        """New goal."""
        self.rho_best: Optional[float] = None
        self.prev_phi = -min(float(rho_reset), self.cfg.pbrs_cap)
        self.prev_action = None if prev_action is None else np.asarray(prev_action, dtype=np.float64).copy()

    def _weights(self) -> Dict[str, float]:
        c = self.cfg
        return {"pos_linear": c.w_pos_linear, "tanh_fine": c.w_tanh_fine, "kp_wide": c.w_kp_wide,
                "kp_sharp": c.w_kp_sharp, "action_rate": c.w_action_rate, "joint_vel": c.w_joint_vel,
                "bsp": c.w_bsp, "snlb": c.w_snlb, "dwell": c.w_dwell, "pbrs": c.w_pbrs,
                "limit_margin": c.w_limit_margin}

    def step(self, d, theta, kp_err, action=None, qdot=None, q=None):
        c = self.cfg
        d, theta, kp_err = float(d), float(theta), float(kp_err)
        rho = c.rho_of(d, theta)
        best_prev = rho if self.rho_best is None else self.rho_best

        raw: Dict[str, float] = {}
        raw["pos_linear"] = d
        raw["tanh_fine"] = 1.0 - np.tanh(d / c.std_tanh_fine)
        raw["kp_wide"] = 1.0 - np.tanh(kp_err / c.std_kp_wide)
        raw["kp_sharp"] = 1.0 - np.tanh(kp_err / c.std_kp_sharp)
        if action is None:
            raw["action_rate"] = 0.0
        else:
            a = np.asarray(action, dtype=np.float64)
            ap = np.zeros_like(a) if self.prev_action is None else self.prev_action
            raw["action_rate"] = float(np.sum((a - ap) ** 2))
            self.prev_action = a.copy()
        raw["joint_vel"] = 0.0 if qdot is None else float(np.sum(np.asarray(qdot, dtype=np.float64) ** 2))
        raw["bsp"] = -min(max(0.0, rho - best_prev), c.bsp_cap)
        raw["snlb"] = max(0.0, float(u_of(rho, c.tau_u) - u_of(best_prev, c.tau_u)))
        raw["dwell"] = 1.0 if rho <= c.dwell_rho else 0.0
        phi = -min(rho, c.pbrs_cap)
        raw["pbrs"] = c.pbrs_gamma * phi - self.prev_phi
        self.prev_phi = phi
        if q is None:
            raw["limit_margin"] = 0.0
        else:
            m = np.minimum(np.asarray(q, dtype=np.float64) - self.lo, self.hi - np.asarray(q, dtype=np.float64)) / (self.hi - self.lo)
            raw["limit_margin"] = (-float(np.clip((c.limit_eps - m.min()) / c.limit_eps, 0.0, 1.0))
                                   if c.limit_gated(rho) else 0.0)

        scale = c.dt if c.scale_by_dt else 1.0
        w = self._weights()
        weighted = {k: w[k] * raw[k] * scale for k in raw}
        reward = float(sum(weighted.values()))
        self.rho_best = min(best_prev, rho)
        info = {"d": d, "theta": theta, "kp_err": kp_err, "rho": rho,
                "rho_best_prev": best_prev, "rho_best": self.rho_best,
                "success": rho <= 1.0, "raw": raw, "weighted": weighted}
        return reward, info

    def channels(self, rho, rho_best_prev):
        return rho_best_channels(rho, rho_best_prev, self.cfg.rho_obs_d0)


def torch_rho_best_channels(rho, rho_best_prev, d0: float):
    import torch
    return torch.stack([torch.clamp(rho_best_prev / d0, 0, 1), torch.clamp((rho - rho_best_prev) / d0, 0, 1)], dim=1)


def torch_pose_reward(cfg: PoseRewardConfig, d, theta, kp_err, rho_prev, rho_best_prev,
                      action, prev_action, qdot, q, lim_lo, lim_hi):
    """Batched mirror of PoseReward.step."""
    import torch
    scale = cfg.dt if cfg.scale_by_dt else 1.0
    rho = torch.maximum(d / cfg.tau_p, theta / cfg.tau_r)
    r = torch.zeros_like(d)
    if cfg.w_pos_linear: r = r + cfg.w_pos_linear * d
    if cfg.w_tanh_fine:  r = r + cfg.w_tanh_fine * (1 - torch.tanh(d / cfg.std_tanh_fine))
    if cfg.w_kp_wide:    r = r + cfg.w_kp_wide * (1 - torch.tanh(kp_err / cfg.std_kp_wide))
    if cfg.w_kp_sharp:   r = r + cfg.w_kp_sharp * (1 - torch.tanh(kp_err / cfg.std_kp_sharp))
    if cfg.w_action_rate: r = r + cfg.w_action_rate * ((action - prev_action) ** 2).sum(dim=-1)
    if cfg.w_joint_vel:  r = r + cfg.w_joint_vel * (qdot * qdot).sum(dim=-1)
    if cfg.w_bsp:
        r = r - cfg.w_bsp * torch.clamp(torch.clamp(rho - rho_best_prev, min=0), max=cfg.bsp_cap)
    if cfg.w_snlb:
        u = 0.5 * (1 + torch.tanh((1 - rho) / cfg.tau_u)); ub = 0.5 * (1 + torch.tanh((1 - rho_best_prev) / cfg.tau_u))
        r = r + cfg.w_snlb * torch.clamp(u - ub, min=0)
    if cfg.w_dwell:      r = r + cfg.w_dwell * (rho <= cfg.dwell_rho).to(d.dtype)
    if cfg.w_pbrs:
        phi = -torch.clamp(rho, max=cfg.pbrs_cap); phi_prev = -torch.clamp(rho_prev, max=cfg.pbrs_cap)
        r = r + cfg.w_pbrs * (cfg.pbrs_gamma * phi - phi_prev)
    if cfg.w_limit_margin:
        m = torch.minimum(q - lim_lo, lim_hi - q) / (lim_hi - lim_lo)
        ramp = torch.clamp((cfg.limit_eps - m.min(dim=-1).values) / cfg.limit_eps, 0, 1)
        r = r - cfg.w_limit_margin * ramp * (rho > cfg.limit_gate_rho).to(d.dtype)
    return r * scale, rho <= 1.0
