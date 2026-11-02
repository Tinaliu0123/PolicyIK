"""Pose cloud of the reachable workspace, collision envelope, and the reverse start curriculum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.kinematics import ArmKinematics, N_JOINTS

__all__ = ["CloudConfig", "PoseCloud", "ReverseConfig", "ReverseCurriculum", "schedule_value"]


@dataclass
class CloudConfig:
    r_goal: float = 0.10
    n_samples: int = 20000
    joint_halfwidth_deg: float = 25.0
    seed: int = 0
    min_entries: int = 200
    centre_m: tuple = ()
    min_base_dist_m: float = 0.0
    link_min_x_m: float = -1e9
    link_min_y_m: float = -1e9
    self_clearance_m: float = 0.0
    self_pair_min_gap: int = 2
    link_bounds_from: int = 0
    tool_z_m: float = 0.0
    collision_tool_z_m: float = 0.0
    tool_min_x_m: float = -1e9
    tool_min_y_m: float = -1e9
    tool_min_base_dist_m: float = 0.0
    min_y_m: float = -1e9
    min_x_m: float = -1e9
    min_z_m: float = -1e9
    max_y_m: float = 1e9
    link_max_y_m: float = 1e9
    tool_max_y_m: float = 1e9
    link_min_z_m: float = -1e9
    link_max_z_m: float = 1e9
    tool_min_z_m: float = -1e9
    tool_max_z_m: float = 1e9
    post_clear_m: float = 0.0
    post_xz: tuple = (0.0, -0.031)
    post_from: int = None
    shoulder_clear_m: float = 0.0
    shoulder_z_range: tuple = (-0.25, 0.19)
    left_arm_clear_m: float = 0.0
    left_arm_xz: tuple = (0.0, -0.18)


class PoseCloud:
    """(q, p, R) triples near home, reachable by construction, sorted by |p - p_home|."""

    def __init__(self, q_home, cfg: CloudConfig, lim_lo, lim_hi):
        self.cfg = cfg
        self.kin = ArmKinematics(tool_z_m=cfg.tool_z_m, collision_tool_z_m=cfg.collision_tool_z_m)
        self.q_home = np.asarray(q_home, dtype=float).copy()
        self.p_home, self.R_home = self.kin.fk_pose(self.q_home)

        self.centre = (np.asarray(cfg.centre_m, dtype=float).reshape(3).copy()
                       if len(cfg.centre_m) == 3 else self.p_home.copy())

        rng = np.random.default_rng(cfg.seed)
        if cfg.joint_halfwidth_deg > 0:
            half = np.deg2rad(cfg.joint_halfwidth_deg)
            q = self.q_home[None, :] + rng.uniform(-half, half, size=(cfg.n_samples, N_JOINTS))
        else:
            q = rng.uniform(lim_lo, lim_hi, size=(cfg.n_samples, N_JOINTS))
        inside = np.all((q >= lim_lo[None, :]) & (q <= lim_hi[None, :]), axis=1)

        p = np.full((len(q), 3), np.nan); R = np.full((len(q), 3, 3), np.nan)
        for i in np.flatnonzero(inside):
            p[i], R[i] = self.kin.fk_pose(q[i])
        d = np.linalg.norm(p - self.centre[None, :], axis=1)
        with np.errstate(invalid="ignore"):
            keep = (inside & (d <= cfg.r_goal) & (np.linalg.norm(p, axis=1) >= cfg.min_base_dist_m)
                    & (p[:, 2] >= cfg.min_z_m) & (p[:, 0] >= cfg.min_x_m)
                    & (p[:, 1] >= cfg.min_y_m) & (p[:, 1] <= cfg.max_y_m))
        if self._any_link_bound():
            keep_idx = np.flatnonzero(keep)
            ok = np.array([self.config_ok(q[i]) for i in keep_idx])
            keep = np.zeros_like(keep); keep[keep_idx[ok]] = True
        if not self.config_ok(self.q_home):
            raise RuntimeError("the home CONFIGURATION violates the link envelope")
        if not self.in_workspace(self.p_home):
            raise RuntimeError("the home EE pose is outside the workspace "
                               f"(|p-centre| {np.linalg.norm(self.p_home-self.centre):.3f} m, "
                               f"|p| {np.linalg.norm(self.p_home):.3f} m, z {self.p_home[2]:.3f} m)")
        d_home = float(np.linalg.norm(self.p_home - self.centre))
        self.q = np.vstack([self.q_home[None], q[keep]])
        self.p = np.vstack([self.p_home[None], p[keep]])
        self.R = np.concatenate([self.R_home[None], R[keep]])
        self.d = np.concatenate([[d_home], d[keep]])
        order = np.argsort(self.d)
        self.q, self.p, self.R, self.d = self.q[order], self.p[order], self.R[order], self.d[order]
        if len(self.d) < cfg.min_entries:
            raise RuntimeError(f"pose cloud has only {len(self.d)} entries (< {cfg.min_entries})")

    def _any_link_bound(self) -> bool:
        c = self.cfg
        return (c.link_min_x_m > -1e8 or c.link_min_y_m > -1e8 or c.link_max_y_m < 1e8
                or c.link_min_z_m > -1e8 or c.link_max_z_m < 1e8
                or c.post_clear_m > 0 or c.shoulder_clear_m > 0 or c.left_arm_clear_m > 0)

    def _obstacle_slack(self, L) -> float:
        c = self.cfg; m = 1e9
        pts = L[(c.link_bounds_from if c.post_from is None else c.post_from):]
        if c.post_clear_m > 0:
            d = np.sqrt((pts[:, 0] - c.post_xz[0]) ** 2 + (pts[:, 2] - c.post_xz[1]) ** 2)
            m = min(m, float(d.min()) - c.post_clear_m)
        if c.left_arm_clear_m > 0:
            d = np.sqrt((pts[:, 0] - c.left_arm_xz[0]) ** 2 + (pts[:, 2] - c.left_arm_xz[1]) ** 2)
            m = min(m, float(d.min()) - c.left_arm_clear_m)
        if c.shoulder_clear_m > 0:
            fa = L[4:]
            zc = np.clip(fa[:, 2], c.shoulder_z_range[0], c.shoulder_z_range[1])
            d = np.sqrt(fa[:, 0] ** 2 + fa[:, 1] ** 2 + (fa[:, 2] - zc) ** 2)
            m = min(m, float(d.min()) - c.shoulder_clear_m)
        return m

    def config_margin(self, q) -> float:
        big = 1e9
        if not self._any_link_bound() and self.cfg.self_clearance_m <= 0 and self.cfg.collision_tool_z_m <= 0:
            return big
        L = self.kin.link_points(q)
        n_link = self.kin.chain.getNrOfSegments()
        links = L[self.cfg.link_bounds_from:n_link]
        m = min(links[:, 0].min() - self.cfg.link_min_x_m, links[:, 1].min() - self.cfg.link_min_y_m,
                self.cfg.link_max_y_m - links[:, 1].max(),
                links[:, 2].min() - self.cfg.link_min_z_m, self.cfg.link_max_z_m - links[:, 2].max())
        if self.cfg.collision_tool_z_m > 0:
            tip = L[-1]
            m = min(m, tip[0] - self.cfg.tool_min_x_m, tip[1] - self.cfg.tool_min_y_m,
                    self.cfg.tool_max_y_m - tip[1], tip[2] - self.cfg.tool_min_z_m, self.cfg.tool_max_z_m - tip[2],
                    np.linalg.norm(tip) - self.cfg.tool_min_base_dist_m)
        if self.cfg.self_clearance_m > 0:
            pts = np.vstack([np.zeros(3), L])
            d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            mask = (np.abs(np.arange(len(pts))[:, None] - np.arange(len(pts))[None, :])
                    >= self.cfg.self_pair_min_gap)
            m = min(m, d[mask].min() - self.cfg.self_clearance_m)
        m = min(m, self._obstacle_slack(L))
        return float(m)

    def config_ok(self, q) -> bool:
        """Whole-arm bounds: the link envelope and a self-folding check."""
        if not self._any_link_bound() and self.cfg.self_clearance_m <= 0 and self.cfg.collision_tool_z_m <= 0:
            return True
        L = self.kin.link_points(q)
        n_link = self.kin.chain.getNrOfSegments()
        links = L[self.cfg.link_bounds_from:n_link]
        if not (links[:, 0].min() >= self.cfg.link_min_x_m
                and links[:, 1].min() >= self.cfg.link_min_y_m
                and links[:, 1].max() <= self.cfg.link_max_y_m
                and links[:, 2].min() >= self.cfg.link_min_z_m
                and links[:, 2].max() <= self.cfg.link_max_z_m):
            return False
        if self.cfg.collision_tool_z_m > 0:
            tip = L[-1]
            if (tip[0] < self.cfg.tool_min_x_m or tip[1] < self.cfg.tool_min_y_m or tip[1] > self.cfg.tool_max_y_m
                    or tip[2] < self.cfg.tool_min_z_m or tip[2] > self.cfg.tool_max_z_m):
                return False
            if np.linalg.norm(tip) < self.cfg.tool_min_base_dist_m:
                return False
        if self.cfg.self_clearance_m > 0:
            pts = np.vstack([np.zeros(3), L])
            d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            m = (np.abs(np.arange(len(pts))[:, None] - np.arange(len(pts))[None, :])
                 >= self.cfg.self_pair_min_gap)
            if d[m].min() < self.cfg.self_clearance_m:
                return False
        if self._obstacle_slack(L) < 0:
            return False
        return True

    def in_workspace(self, p) -> bool:
        """Membership: inside the sphere and past both cut planes (their membership_rule)."""
        p = np.asarray(p, dtype=float)
        return bool(np.linalg.norm(p - self.centre) <= self.cfg.r_goal
                    and np.linalg.norm(p) >= self.cfg.min_base_dist_m
                    and p[2] >= self.cfg.min_z_m
                    and p[0] >= self.cfg.min_x_m
                    and p[1] >= self.cfg.min_y_m
                    and p[1] <= self.cfg.max_y_m)

    def _n_within(self, radius: float) -> int:
        if radius <= 1e-9:
            return 1
        return max(1, int(np.searchsorted(self.d, radius, side="right")))

    def sample_within(self, radius: float, rng):
        """(q, p, R) with |p - centre| <= radius."""
        i = int(rng.integers(self._n_within(radius)))
        return self.q[i].copy(), self.p[i].copy(), self.R[i].copy()


def schedule_value(schedule, frac: float) -> float:
    """Value of a piecewise-constant [[run fraction, value], ...] schedule at this fraction of the run."""
    v = float(schedule[0][1])
    for f, val in schedule:
        if frac >= float(f):
            v = float(val)
    return v


@dataclass
class ReverseConfig:
    schedule: list = None
    total_steps: int = 0
    sigma_lo_deg: float = 2.0
    rho_floor: float = 10.0
    max_start_tries: int = 20

    def validate(self) -> None:
        if not self.schedule:
            raise ValueError("the sigma schedule must not be empty")
        fr = [float(x[0]) for x in self.schedule]; sg = [float(x[1]) for x in self.schedule]
        if fr[0] != 0.0 or fr != sorted(fr) or max(fr) >= 1.0 or min(fr) < 0.0:
            raise ValueError("schedule fractions must start at 0, be increasing and stay < 1")
        if sg != sorted(sg) or min(sg) <= 0:
            raise ValueError("schedule sigmas must be positive and non-decreasing")
        if self.total_steps <= 0:
            raise ValueError("schedule needs total_steps")
        if not 0 < self.sigma_lo_deg <= sg[0]:
            raise ValueError("need 0 < sigma_lo <= the first scheduled sigma")
        if self.rho_floor < 0 or self.max_start_tries < 1:
            raise ValueError("rho_floor >= 0 and max_start_tries >= 1")


class ReverseCurriculum:
    """Start = goal_q + N(0, sigma) with sigma ~ U[sigma_lo, sigma_t] and sigma_t on a fixed schedule."""

    def __init__(self, cfg: ReverseConfig):
        cfg.validate()
        self.cfg = cfg
        self.sigma_deg = float(cfg.schedule[0][1])

    def sample_sigma_deg(self, rng) -> float:
        return float(rng.uniform(self.cfg.sigma_lo_deg, self.sigma_deg))

    def scheduled_sigma(self, step: int) -> float:
        return schedule_value(self.cfg.schedule, step / max(1, self.cfg.total_steps))

    def on_episode(self, step: int) -> str:
        s_new = self.scheduled_sigma(step)
        if s_new <= self.sigma_deg + 1e-9:
            return ""
        prev = self.sigma_deg
        self.sigma_deg = s_new
        return f"sigma {prev:.1f} -> {s_new:.1f} deg at {100 * step / max(1, self.cfg.total_steps):.0f}% of the run"
