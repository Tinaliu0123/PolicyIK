"""Demonstration file (<name>.npz + <name>_meta.json) and its check against the current settings."""

from __future__ import annotations

import json
import os

import numpy as np

KEYS = ("observation", "action", "reward", "terminated", "truncated", "next_observation", "ep_id", "ep_t")
# settings of the action law and the observation; demonstrations recorded under other values are refused
ENV_KEYS = ("gain_schedule", "gain_min", "gain_d_lo_m", "gain_d_hi_m", "gain_th_lo_deg", "gain_th_hi_deg",
            "action_scale_rad", "max_joint_delta_rad", "max_joint_velocity_rad_s", "ref_band_rad", "control_hz",
            "zoom_obs", "zoom_pos_d0_m", "zoom_ori_d0_deg", "obs_prev_action", "obs_ref_lead",
            "joint_lo_deg", "joint_hi_deg", "tool_z_m")
OBS_REWARD_KEYS = ("tau_p", "tau_r_deg", "kp_size", "rho_obs_d0")


def meta_path(path: str) -> str:
    return (path[:-4] if path.endswith(".npz") else path) + "_meta.json"


def norm(x):
    """JSON round-trip so tuples, lists and numpy scalars compare equal."""
    return json.loads(json.dumps(x, default=float))


def env_record(env: dict) -> dict:
    return norm({k: env[k] for k in ENV_KEYS})


def interface_diff(env_a: dict, env_b: dict, rew_a: dict, rew_b: dict):
    """Differences between a recorded (a) and a current (b) setting: (refused, reward-only)."""
    hard = [f"env.{k}: recorded {env_a.get(k)} vs now {env_b.get(k)}"
            for k in ENV_KEYS if norm(env_a.get(k)) != norm(env_b.get(k))]
    hard += [f"reward.{k}: recorded {rew_a.get(k)} vs now {rew_b.get(k)}"
             for k in OBS_REWARD_KEYS if norm(rew_a.get(k)) != norm(rew_b.get(k))]
    soft = [f"reward.{k}: recorded {rew_a.get(k)} vs now {rew_b.get(k)}" for k in sorted(rew_b)
            if k not in OBS_REWARD_KEYS and k != "dt" and norm(rew_a.get(k)) != norm(rew_b.get(k))]
    return hard, soft


def complete_episode_mask(ep_t, terminated, truncated):
    """True for rows of episodes whose last row is terminated or truncated."""
    ep_t = np.asarray(ep_t)
    n = ep_t.size
    keep = np.ones(n, bool)
    if n == 0:
        return keep
    starts = np.flatnonzero(ep_t == 0)
    if starts.size == 0 or starts[0] != 0:
        raise ValueError("demo rows must start with an ep_t == 0 row")
    ends = np.append(starts[1:], n)
    for s0, s1 in zip(starts, ends):
        if terminated[s1 - 1] < 0.5 and truncated[s1 - 1] < 0.5:
            keep[s0:s1] = False
    return keep


def load_demos(path: str, obs_dim: int, run_env: dict, run_reward: dict):
    """Complete episodes of a demonstration file, refused if recorded under another action law or observation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no demonstration file {path}")
    raw = np.load(path)
    missing = [k for k in KEYS if k not in raw.files]
    if missing:
        raise ValueError(f"{path} lacks {missing}")
    arrs = {k: np.asarray(raw[k]) for k in KEYS}
    if arrs["observation"].ndim != 2 or arrs["observation"].shape[1] != obs_dim:
        raise ValueError(f"demo observation width {arrs['observation'].shape[1:]} != {obs_dim}")
    keep = complete_episode_mask(arrs["ep_t"], arrs["terminated"], arrs["truncated"])
    arrs = {k: v[keep] for k, v in arrs.items()}
    if arrs["ep_t"].size == 0:
        raise ValueError(f"{path} holds no complete episode")
    with open(meta_path(path)) as f:
        m = json.load(f)
    hard, soft = interface_diff(m["env"], run_env, m["reward"], run_reward)
    if hard:
        raise ValueError("the demonstrations were collected under a different interface:\n  " + "\n  ".join(hard))
    return arrs, soft
