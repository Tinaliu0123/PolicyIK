"""6-D pose reaching environment on the real OpenArm right arm."""

from __future__ import annotations

import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..robot.safety import joint_limits
from ..utils.kinematics import N_JOINTS
from ..utils.pose_math import angle_between, keypoint_error, keypoints, rot6d_from_R, rotvec_body
from .reward import PoseReward, PoseRewardConfig
from .workspace import CloudConfig, PoseCloud, ReverseConfig, ReverseCurriculum

__all__ = ["OpenArmPoseEnv", "EnvConfig", "action_gain", "build_layout"]


def build_layout(cfg: "EnvConfig") -> dict:
    """Observation column slices [lo, hi)."""
    lay = {}
    n = 0

    def add(name, k):
        nonlocal n
        lay[name] = (n, n + k); n += k

    add("q", 7); add("dq", 7); add("ee_pos", 3); add("ee_rot", 6)
    add("goal_pos", 3); add("goal_rot", 6); add("err_pos", 3); add("kp_err", 9)
    if cfg.zoom_obs:
        for k, _ in enumerate(cfg.zoom_pos_d0_m):
            add(f"zoom_pos_{k}", 3)
        for k, _ in enumerate(cfg.zoom_ori_d0_deg):
            add(f"zoom_ori_{k}", 3)
    add("rho_best", 2)
    if cfg.obs_prev_action:
        add("prev_action", 7)
    if cfg.obs_ref_lead:
        add("ref_lead", 7)
    lay["obs_dim"] = n
    return lay


@dataclass
class EnvConfig:
    r_goal: float = 0.10
    cloud_samples: int = 20000
    cloud_halfwidth_deg: float = 25.0
    cloud_seed: int = 0
    centre_m: tuple = ()
    min_base_dist_m: float = 0.0
    min_x_m: float = -1e9
    joint_lo_deg: tuple = ()
    joint_hi_deg: tuple = ()
    tool_z_m: float = 0.0
    collision_tool_z_m: float = 0.0
    tool_min_x_m: float = -1e9
    tool_min_y_m: float = -1e9
    tool_min_base_dist_m: float = 0.0
    self_pair_min_gap: int = 2
    link_bounds_from: int = 0
    self_clearance_m: float = 0.0
    link_min_x_m: float = -1e9
    link_min_y_m: float = -1e9
    min_y_m: float = -1e9
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
    home_pose_deg: tuple = (1.4, 57.5, 42.3, 131.1, 59.3, -8.4, -37.0)

    control_hz: float = 10.0
    max_episode_steps: int = 100

    action_scale_rad: float = 0.03
    max_joint_delta_rad: float = 0.05
    max_joint_velocity_rad_s: float = 0.4
    ref_band_rad: float = 0.25
    # the reference may lead past a joint bound by this much while the measured joint is still inside it,
    # so a goal at the bound is reachable despite the 1-2 deg tracking lag; 0 = hard clip at the bound
    ref_limit_relax_deg: float = 0.0

    zoom_obs: bool = True
    zoom_pos_d0_m: tuple = (0.01, 0.0005)
    zoom_ori_d0_deg: tuple = (20.0, 0.25)
    gain_schedule: bool = False
    gain_min: float = 0.1
    gain_d_lo_m: float = 0.0005
    gain_d_hi_m: float = 0.00931
    gain_th_lo_deg: float = 0.415
    gain_th_hi_deg: float = 2.328

    obs_prev_action: bool = True
    obs_ref_lead: bool = False

    obs_noise_q_rad: float = 0.01
    obs_noise_dq_rad_s: float = 0.01

    goal_jitter_deg: float = 0.0

    success_tau_m: float = 0.01
    success_tau_r_deg: float = 2.0
    settled_hold_steps: int = 10

    reset_ramp_s: float = 4.0
    reset_settle_s: float = 1.5
    drain_s: float = 0.02
    max_state_age_s: float = 0.3
    health_guard: bool = True
    health_required: bool = False
    collision_guard: bool = True
    collision_terminate: bool = False
    collision_block_steps: int = 20
    frozen_demand_rad: float = 0.02
    frozen_state_steps: int = 20
    min_post_learn_spin_s: float = 0.01
    reset_ramp_hz: float = 0.0

    def validate(self) -> None:
        if self.control_hz <= 0 or self.max_episode_steps <= 0:
            raise ValueError("control_hz and max_episode_steps must be > 0")
        if self.reset_ramp_hz < 0 or self.min_post_learn_spin_s < 0:
            raise ValueError("reset_ramp_hz and min_post_learn_spin_s must be >= 0")
        if self.settled_hold_steps > self.max_episode_steps:
            raise ValueError("settled_hold_steps exceeds the episode length")
        if not 0.0 <= self.ref_limit_relax_deg <= 5.0:
            raise ValueError("ref_limit_relax_deg must be in [0, 5]")
        if any(v <= 0 for v in (*self.zoom_pos_d0_m, *self.zoom_ori_d0_deg)):
            raise ValueError("zoom scales must be > 0")
        if self.gain_schedule:
            if not (0.0 < self.gain_min <= 1.0):
                raise ValueError("gain_min must be in (0, 1]")
            if not (0.0 <= self.gain_d_lo_m < self.gain_d_hi_m) or \
               not (0.0 <= self.gain_th_lo_deg < self.gain_th_hi_deg):
                raise ValueError("gain bands need lo < hi")
        if self.success_tau_m <= 0 or self.success_tau_r_deg <= 0:
            raise ValueError("success tolerances must be > 0")
        if self.goal_jitter_deg < 0:
            raise ValueError("goal_jitter_deg must be >= 0")
        if self.obs_noise_q_rad < 0 or self.obs_noise_dq_rad_s < 0:
            raise ValueError("obs noise bounds must be >= 0")
        if not (self.obs_prev_action or self.obs_ref_lead):
            print("[env] NOTE neither prev_action nor ref_lead in the obs: the integrator "
                  "setpoint is fully hidden from the policy")


def action_gain(d, th, c, xp=np):
    """Step gain g = max(g_pos(d), g_ori(theta)) in [gain_min, 1] (d in m, theta in rad); xp = numpy or torch."""
    gp = c.gain_min + (1 - c.gain_min) * xp.clip((d - c.gain_d_lo_m) / (c.gain_d_hi_m - c.gain_d_lo_m), 0, 1)
    go = c.gain_min + (1 - c.gain_min) * xp.clip(
        (xp.rad2deg(th) - c.gain_th_lo_deg) / (c.gain_th_hi_deg - c.gain_th_lo_deg), 0, 1)
    return xp.maximum(gp, go)


def cloud_config(cfg: "EnvConfig", **overrides) -> CloudConfig:
    """The workspace cloud and collision envelope an EnvConfig describes."""
    kw = dict(
        r_goal=cfg.r_goal, n_samples=cfg.cloud_samples,
        joint_halfwidth_deg=cfg.cloud_halfwidth_deg,
        seed=cfg.cloud_seed,
        centre_m=tuple(cfg.centre_m),
        min_base_dist_m=cfg.min_base_dist_m,
        min_x_m=cfg.min_x_m, min_y_m=cfg.min_y_m,
        link_min_x_m=cfg.link_min_x_m,
        link_min_y_m=cfg.link_min_y_m,
        self_clearance_m=cfg.self_clearance_m,
        self_pair_min_gap=cfg.self_pair_min_gap,
        link_bounds_from=cfg.link_bounds_from,
        tool_z_m=cfg.tool_z_m,
        collision_tool_z_m=cfg.collision_tool_z_m,
        tool_min_x_m=cfg.tool_min_x_m,
        tool_min_y_m=cfg.tool_min_y_m,
        tool_min_base_dist_m=cfg.tool_min_base_dist_m,
        min_z_m=cfg.min_z_m, max_y_m=cfg.max_y_m,
        link_max_y_m=cfg.link_max_y_m,
        tool_max_y_m=cfg.tool_max_y_m,
        link_min_z_m=cfg.link_min_z_m, link_max_z_m=cfg.link_max_z_m,
        tool_min_z_m=cfg.tool_min_z_m, tool_max_z_m=cfg.tool_max_z_m,
        post_clear_m=cfg.post_clear_m, post_xz=tuple(cfg.post_xz), post_from=cfg.post_from,
        shoulder_clear_m=cfg.shoulder_clear_m, shoulder_z_range=tuple(cfg.shoulder_z_range),
        left_arm_clear_m=cfg.left_arm_clear_m, left_arm_xz=tuple(cfg.left_arm_xz))
    kw.update(overrides)
    return CloudConfig(**kw)


def reflect_into(q, lo, hi):
    """Fold an out-of-range joint draw back inside the limits instead of snapping it onto them."""
    q = np.asarray(q, dtype=float)
    q = q - 2.0 * np.maximum(q - hi, 0.0) + 2.0 * np.maximum(lo - q, 0.0)
    return np.clip(q, lo, hi)


class OpenArmPoseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig, rew_cfg: PoseRewardConfig, curr_cfg=None,
                 io=None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.dt = 1.0 / cfg.control_hz
        rew_cfg.dt = self.dt
        self.rew_cfg = rew_cfg
        self.lim_lo, self.lim_hi = joint_limits(cfg.joint_lo_deg, cfg.joint_hi_deg)
        if np.any(self.lim_lo >= self.lim_hi):
            raise ValueError("joint sector bounds are empty after intersecting the URDF limits")
        self.reward_fn = PoseReward(rew_cfg, self.lim_lo, self.lim_hi)
        self.layout = build_layout(cfg)

        self.home_rad = np.deg2rad(np.asarray(cfg.home_pose_deg, dtype=float))
        self.cloud = PoseCloud(self.home_rad, cloud_config(cfg), self.lim_lo, self.lim_hi)
        self.kin = self.cloud.kin
        self.curriculum = ReverseCurriculum(curr_cfg) if isinstance(curr_cfg, ReverseConfig) else None
        self._ep_sigma_deg = float("nan")
        self._q_last = None
        self._q_frozen = 0
        self._rng = np.random.default_rng(cfg.cloud_seed)
        self.goal_p = self.cloud.p_home.copy()
        self.goal_R = self.cloud.R_home.copy()
        self._rho_state = (0.0, 0.0)

        self.obs_dim = self.layout["obs_dim"]
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), np.float32)

        if io is None:
            from ..robot.ros_io import ArmIO
            io = ArmIO()
        self.io = io
        if not self.io.wait_for_state(timeout=8.0):
            raise RuntimeError("no /joint_states within 8 s -- is the bringup launch running?")
        self._q_ref = np.zeros(N_JOINTS)
        self._n_collision_blocks = 0
        self._consec_blocks = 0
        self._blocked_now = False
        self._prev_action = np.zeros(N_JOINTS)
        self._t = 0
        self._ep = self._new_ep_acc()
        self._step_wall = []
        self.learn_fn = None; self.idle_fn = None

    def _new_ep_acc(self) -> dict:
        return {"ret": 0.0, "dists": [], "angs": [], "coarse": [], "strict": [], "terms": {}, "n": 0}

    def pose(self, q):
        """(p, R) of the EE from the measured joints (our own FK; no external tracker)."""
        return self.kin.fk_pose(q)

    def errors(self, p, R):
        """(d, theta, kp_err) of (p, R) against the current goal."""
        d = float(np.linalg.norm(self.goal_p - p))
        th = angle_between(R, self.goal_R)
        kp = keypoint_error(p, R, self.goal_p, self.goal_R, self.rew_cfg.kp_size)
        return d, th, kp

    def _obs(self, q, dq, p, R) -> np.ndarray:
        c = self.cfg
        err = self.goal_p - p
        kp_err = (keypoints(self.goal_p, self.goal_R, self.rew_cfg.kp_size)
                  - keypoints(p, R, self.rew_cfg.kp_size)).reshape(-1)
        q_o = q + self._rng.uniform(-c.obs_noise_q_rad, c.obs_noise_q_rad, N_JOINTS) if c.obs_noise_q_rad > 0 else q
        dq_o = dq + self._rng.uniform(-c.obs_noise_dq_rad_s, c.obs_noise_dq_rad_s, N_JOINTS) if c.obs_noise_dq_rad_s > 0 else dq
        parts = [q_o, dq_o, p, rot6d_from_R(R), self.goal_p, rot6d_from_R(self.goal_R), err, kp_err]
        if c.zoom_obs:
            rv = rotvec_body(R, self.goal_R)
            for d0 in c.zoom_pos_d0_m:
                parts.append(np.clip(err / d0, -1.0, 1.0))
            for d0 in c.zoom_ori_d0_deg:
                parts.append(np.clip(rv / np.radians(d0), -1.0, 1.0))
        rho, best_prev = self._rho_state
        parts.append(self.reward_fn.channels(rho, best_prev))
        if c.obs_prev_action:
            parts.append(self._prev_action)
        if c.obs_ref_lead:
            parts.append(self._q_ref - q)
        return np.concatenate(parts).astype(np.float32)

    def _seed_goal_state(self, p, R, prev_action=None) -> float:
        """New goal in the state (p, R): rho_best = current rho, PBRS potential seeded."""
        d, th, _ = self.errors(p, R)
        rho = self.rew_cfg.rho_of(d, th)
        self._rho_state = (rho, rho)
        self.reward_fn.reset(rho_reset=rho, prev_action=prev_action)
        return rho

    def _gain(self, d: float, th: float) -> float:
        """action_gain of the current error; 1.0 when the gain schedule is off."""
        return float(action_gain(d, th, self.cfg)) if self.cfg.gain_schedule else 1.0

    def _goto(self, q_goal: np.ndarray) -> None:
        ramp_hz = self.cfg.reset_ramp_hz if self.cfg.reset_ramp_hz > 0 else self.cfg.control_hz
        ramp_steps = max(1, int(round(self.cfg.reset_ramp_s * ramp_hz)))
        sleep = self.cfg.reset_ramp_s / ramp_steps
        q0, _ = self.io.get_state()
        for i in range(ramp_steps + 1):
            a = i / ramp_steps
            self.io.command_q(np.clip(q0 * (1 - a) + q_goal * a, self.lim_lo, self.lim_hi))
            self._spin_idle(sleep)
        self._spin_idle(self.cfg.reset_settle_s, timeout_sec=0.05)

    def _spin_idle(self, seconds: float, timeout_sec=None) -> None:
        kw = {} if timeout_sec is None else {"timeout_sec": timeout_sec}
        if self.idle_fn is None:
            self.io.spin_for(seconds, **kw); return
        t0 = time.monotonic()
        self.idle_fn(max(0.0, seconds - self.cfg.min_post_learn_spin_s))
        rem = seconds - (time.monotonic() - t0)
        self.io.spin_for(max(rem, self.cfg.min_post_learn_spin_s), **kw)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "q_start" in options:
            q_start = np.asarray(options["q_start"], dtype=float)
            self.goal_p = np.asarray(options["goal_p"], dtype=float).copy()
            self.goal_R = np.asarray(options["goal_R"], dtype=float).copy()
        elif self.curriculum is not None:
            q_goal, self.goal_p, self.goal_R = self._sample_goal()
            q_start = self._sample_reverse_start(q_goal, self.goal_p, self.goal_R)
        else:
            raise ValueError("reset() needs a curriculum or options with q_start, goal_p and goal_R")
        self._goto(q_start)

        q, dq = self.io.get_state()
        p, R = self.pose(q)
        self._q_ref = q.copy()
        self._prev_action = np.zeros(N_JOINTS)
        self._t = 0
        self._q_last = None
        self._q_frozen = 0
        self._n_collision_blocks = 0; self._consec_blocks = 0
        self._ep = self._new_ep_acc()
        self._seed_goal_state(p, R)
        return self._obs(q, dq, p, R), {}

    def _sample_goal(self):
        c = self.cfg
        r = self.cfg.r_goal
        q0, p0, R0 = self.cloud.sample_within(r, self._rng)
        if c.goal_jitter_deg <= 0:
            return q0, p0.copy(), R0.copy()
        for _ in range(10):
            q = reflect_into(q0 + self._rng.normal(0, np.radians(c.goal_jitter_deg), N_JOINTS), self.lim_lo, self.lim_hi)
            p, R = self.kin.fk_pose(q)
            if (self.cloud.in_workspace(p) and self.cloud.config_ok(q)
                    and np.linalg.norm(p - self.cloud.centre) <= r):
                return q, p, R
        return q0, p0.copy(), R0.copy()


    def _sample_reverse_start(self, q_goal, goal_p, goal_R):
        """goal_q + N(0, sigma_ep), resampled while rho < rho_floor (already at the goal)."""
        rc = self.curriculum.cfg
        sigma = self.curriculum.sample_sigma_deg(self._rng)
        self._ep_sigma_deg = sigma
        q_best, rho_best = None, -1.0
        for _ in range(rc.max_start_tries):
            q = reflect_into(q_goal + self._rng.normal(0, np.radians(sigma), N_JOINTS), self.lim_lo, self.lim_hi)
            p, R = self.kin.fk_pose(q)
            if not self.cloud.in_workspace(p) or not self.cloud.config_ok(q):
                continue
            d = float(np.linalg.norm(goal_p - p)); th = float(angle_between(R, goal_R))
            rho = self.rew_cfg.rho_of(d, th)
            if rho >= rc.rho_floor:
                return q
            if rho > rho_best:
                q_best, rho_best = q, rho
        if q_best is not None:
            return q_best
        q_c, _, _ = self.cloud.sample_within(self.cfg.r_goal, self._rng)
        return q_c

    def step(self, action):
        t_wall = time.monotonic()
        c = self.cfg
        action = np.asarray(action, dtype=float).reshape(N_JOINTS)

        self.io.drain(c.drain_s)
        q_start, _ = self.io.get_state()
        if c.health_guard and hasattr(self.io, "health_check"):
            ok, why = self.io.health_check()
            if ok is False:
                raise RuntimeError(f"MOTOR HEALTH FAULT: {why}. Stop, inspect, relaunch. "
                                   f"q = {np.round(np.degrees(q_start), 1).tolist()} deg")
            if ok is None and not getattr(self, "_health_warned", False):
                self._health_warned = True
                msg = f"[env] motor health monitor: {why}"
                if c.health_required:
                    raise RuntimeError(msg + " -- health_required is set")
                print(msg + " -- continuing without motor status (set health_required to refuse)")
        stale = self.io.state_age() >= c.max_state_age_s
        if stale:
            action = np.zeros(N_JOINTS)
        if c.frozen_state_steps > 0:
            _demanded = float(np.max(np.abs(self._q_ref - q_start)))
            if (self._q_last is not None and np.array_equal(q_start, self._q_last)
                    and _demanded > c.frozen_demand_rad
                    and getattr(self, "_consec_blocks", 0) == 0):
                self._q_frozen += 1
                if self._q_frozen >= c.frozen_state_steps:
                    raise RuntimeError(
                        f"joint state FROZEN: q identical for {self._q_frozen} steps "
                        f"({self._q_frozen / c.control_hz:.1f} s) while /joint_states keeps "
                        f"publishing. The CAN read has stalled -- the hardware interface is "
                        f"republishing its last frame. Stop the launch, bring the CAN interface down and up "
                        f"again, and relaunch to re-enable the motors. "
                        f"q = {np.round(np.degrees(q_start), 1).tolist()} deg")
            else:
                self._q_frozen = 0
            self._q_last = q_start.copy()

        p0, R0 = self.pose(q_start)
        d0, th0, _ = self.errors(p0, R0)
        gain = self._gain(d0, th0)
        a = np.clip(action, -1.0, 1.0)
        delta = gain * a * c.action_scale_rad
        delta = np.clip(delta, -c.max_joint_delta_rad, c.max_joint_delta_rad)
        vel_cap = c.max_joint_velocity_rad_s * self.dt
        delta = np.clip(delta, -vel_cap, vel_cap)
        _q_ref_prev = self._q_ref.copy()
        if c.ref_limit_relax_deg > 0:
            # a joint measured at its bound keeps the hard clip and refuses further outward steps
            rx = np.radians(c.ref_limit_relax_deg)
            at_lo, at_hi = q_start <= self.lim_lo, q_start >= self.lim_hi
            lo = np.where(at_lo, self.lim_lo, self.lim_lo - rx)
            hi = np.where(at_hi, self.lim_hi, self.lim_hi + rx)
            cand = np.clip(self._q_ref + delta, lo, hi)
            keep = (at_lo & (cand < self._q_ref)) | (at_hi & (cand > self._q_ref))
            self._q_ref = np.where(keep, self._q_ref, cand)
        else:
            self._q_ref = np.clip(self._q_ref + delta, self.lim_lo, self.lim_hi)
        self._q_ref = np.clip(self._q_ref, q_start - c.ref_band_rad, q_start + c.ref_band_rad)
        self._blocked_now = False
        if c.collision_guard and self.cloud is not None:
            m_new = self.cloud.config_margin(self._q_ref)
            if m_new < 0.0:
                m_old = self.cloud.config_margin(_q_ref_prev)
                if m_new > m_old + 1e-6:
                    pass
                else:
                    self._q_ref = _q_ref_prev
                    self._n_collision_blocks += 1
                    self._blocked_now = True
        self._consec_blocks = self._consec_blocks + 1 if self._blocked_now else 0
        q_ref_cmd = self._q_ref.copy()
        self.io.command_q(q_ref_cmd)

        if self.learn_fn is not None:
            t_cmd = time.monotonic()
            self.learn_fn()
            rem = self.dt - (time.monotonic() - t_cmd)
            self.io.spin_for(max(rem, c.min_post_learn_spin_s))
        else:
            self.io.spin_for(self.dt)
        q_end, dq_end = self.io.get_state()

        p, R = self.pose(q_end)
        d, th, kp = self.errors(p, R)
        r, rinfo = self.reward_fn.step(d, th, kp, action=action, qdot=dq_end, q=q_end)
        self._rho_state = (float(rinfo["rho"]), float(rinfo["rho_best_prev"]))
        rho = float(rinfo["rho"])
        coarse = (d <= c.success_tau_m) and (np.degrees(th) <= c.success_tau_r_deg)
        strict = rho <= 1.0

        self._t += 1
        self._prev_action = action.copy()
        ep = self._ep
        ep["ret"] += r
        ep["dists"].append(d); ep["angs"].append(th)
        ep["coarse"].append(bool(coarse)); ep["strict"].append(bool(strict))
        ep["n"] += 1
        for k, v in rinfo["weighted"].items():
            ep["terms"][k] = ep["terms"].get(k, 0.0) + float(v)

        terminated = False
        truncated = self._t >= c.max_episode_steps
        blocked_out = (c.collision_guard and c.collision_block_steps > 0
                       and self._consec_blocks >= c.collision_block_steps)
        if blocked_out:
            if c.collision_terminate:
                terminated, truncated = True, False
            else:
                truncated = True
            self._ep["collision_blocked"] = True
            print(f"[env] collision guard blocked {self._consec_blocks} consecutive steps at "
                  f"q = {np.round(np.degrees(q_end), 1).tolist()} deg (envelope slack "
                  f"{1000*self.cloud.config_margin(q_end):+.1f} mm) -- episode "
                  f"{'TERMINATED (no bootstrap)' if c.collision_terminate else 'truncated'}, full reset")
        obs = self._obs(q_end, dq_end, p, R)
        step_s = time.monotonic() - t_wall
        self._step_wall.append(step_s)
        info = {"step_wall_s": step_s}
        if terminated or truncated:
            info["episode"] = self._episode_summary()
        return obs, float(r), terminated, truncated, info

    def _episode_summary(self) -> dict:
        ep = self._ep
        dists = np.asarray(ep["dists"]); angs = np.degrees(np.asarray(ep["angs"]))
        band = np.asarray(ep["coarse"]); sband = np.asarray(ep["strict"])
        k = self.cfg.settled_hold_steps
        return {
            "r": ep["ret"], "l": ep["n"],
            "final_dist": float(dists[-1]), "min_dist": float(dists.min()), "final_ang_deg": float(angs[-1]),
            "settled_success": bool(band.size >= k and band[-k:].all()),
            "strict_settled": bool(sband.size >= k and sband[-k:].all()),
            "collision_blocked_out": bool(ep.get("collision_blocked", False)),
            "sigma_deg": float(self._ep_sigma_deg),
            "term_means": {kk: vv / max(1, ep["n"]) for kk, vv in ep["terms"].items()},
        }

    def timing_summary(self) -> dict:
        w = np.asarray(self._step_wall)
        if w.size == 0:
            return {}
        return {
            "n_steps": int(w.size),
            "declared_dt_s": self.dt,
            "step_wall_median_s": float(np.median(w)),
            "step_wall_p90_s": float(np.percentile(w, 90)),
            "effective_hz": float(1.0 / np.median(w)),
        }

    def close(self):
        try:
            q, _ = self.io.get_state()
            self.io.command_q(q)
        except Exception:
            pass
        self.io.destroy_node()
