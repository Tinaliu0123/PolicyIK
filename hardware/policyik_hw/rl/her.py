"""Hindsight relabelling of 6-D pose goals, with a frozen demonstration store sampled next to the replay ring."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from flash_rl.buffers.base_buffer import Batch
from flash_rl.buffers.torch_buffer import TorchUniformBuffer

from ..utils.pose_math import t_R_from_rot6d, t_R_from_rotvec, t_angle_between, t_keypoint_error, t_rotvec_body
from ..envs.pose_env import action_gain
from ..envs.reward import PoseRewardConfig, torch_pose_reward, torch_rho_best_channels

__all__ = ["PoseHERBuffer", "PoseHERRLPDBuffer", "HERConfig"]


@dataclass
class HERConfig:
    """Goal-source probabilities of the online stream: stored goal, future pose, perturbed current pose, random pose."""

    p_orig: float = 0.4
    p_future: float = 0.4
    p_cur: float = 0.2
    p_rand: float = 0.0

    def validate(self) -> None:
        s = self.p_orig + self.p_future + self.p_cur + self.p_rand
        if not math.isclose(s, 1.0, abs_tol=1e-6):
            raise ValueError(f"HER goal-source probabilities must sum to 1, got {s}")
        if self.p_orig <= 0.0:
            raise ValueError("HER p_orig must be > 0")


class HERTorchBuffer(TorchUniformBuffer):
    """TorchUniformBuffer with per-episode bookkeeping, written one transition at a time."""

    def __init__(self, *args, her_cfg: HERConfig, **kwargs):
        super().__init__(*args, **kwargs)
        if self._n_step != 1:
            raise NotImplementedError("HERTorchBuffer supports n_step == 1 only.")
        her_cfg.validate()
        self.her_cfg = her_cfg
        self._cum = self._cumulative(her_cfg.p_orig, her_cfg.p_future, her_cfg.p_cur)

    @staticmethod
    def _cumulative(p_orig: float, p_future: float, p_cur: float) -> tuple[float, float, float]:
        return (p_orig, p_orig + p_future, p_orig + p_future + p_cur)

    def reset(self) -> None:
        super().reset()
        m = self._max_length
        dev = self._device
        self._ep_id = torch.full((m,), -1, dtype=torch.int64, device=dev)
        self._ep_t = torch.zeros((m,), dtype=torch.int64, device=dev)
        self._ep_len = torch.zeros((m,), dtype=torch.int64, device=dev)
        self._ep_start = torch.zeros((m,), dtype=torch.int64, device=dev)
        self._next_ep_id = 0
        self._cur_ep_id = 0
        self._cur_ep_t = 0
        self._cur_ep_start = 0

    def _squeeze(self, transition: Batch) -> dict[str, torch.Tensor]:
        return {k: self._to_tensor(v).reshape(-1)[0] if k in ("reward", "terminated", "truncated")
                else self._to_tensor(v).reshape(self._row_shape(k))
                for k, v in transition.items()}

    def _row_shape(self, key: str) -> tuple[int, ...]:
        if key in ("observation", "next_observation"):
            return (self._observations.shape[-1],)
        if key == "action":
            return (self._actions.shape[-1],)
        return (-1,)

    def add(self, transition: Batch) -> None:
        m = self._max_length
        slot = self._current_idx
        t = self._squeeze(transition)
        self._observations[slot] = t["observation"].to(self._observations.dtype)
        self._next_observations[slot] = t["next_observation"].to(self._next_observations.dtype)
        self._actions[slot] = t["action"].to(self._actions.dtype)
        self._rewards[slot] = t["reward"].to(self._rewards.dtype)
        self._terminateds[slot] = t["terminated"].to(self._terminateds.dtype)
        self._truncateds[slot] = t["truncated"].to(self._truncateds.dtype)
        self._ep_id[slot] = self._cur_ep_id
        self._ep_t[slot] = self._cur_ep_t
        self._ep_start[slot] = self._cur_ep_start
        self._ep_len[slot] = 0
        self._num_in_buffer = min(self._num_in_buffer + 1, m)
        self._current_idx = (slot + 1) % m
        self._cur_ep_t += 1
        if bool(t["terminated"].item()) or bool(t["truncated"].item()):
            L = self._cur_ep_t
            idxs = (torch.arange(L, device=self._device) + self._cur_ep_start) % m
            self._ep_len[idxs] = L
            self._cur_ep_id += 1
            self._cur_ep_t = 0
            self._cur_ep_start = self._current_idx

    def _sample_online_from(self, anchor: torch.Tensor) -> Batch:
        """Relabel a batch whose anchors come from the online ring buffer."""
        ep_len = self._ep_len[anchor]
        force_orig = ep_len == 0
        ep_len = ep_len.clamp_min(1)
        return self._relabel_slice(
            obs=self._observations[anchor],
            nobs=self._next_observations[anchor],
            act=self._actions[anchor],
            trunc=self._truncateds[anchor],
            stored_reward=self._rewards[anchor],
            stored_term=self._terminateds[anchor],
            ep_t=self._ep_t[anchor],
            ep_len=ep_len,
            ep_start=self._ep_start[anchor],
            eid=self._ep_id[anchor],
            src_obs=self._observations,
            src_ep_id=self._ep_id,
            src_n=self._num_in_buffer,
            cum=self._cum,
            use_stored_for_orig=True,
            force_orig=force_orig,
        )

    def save(self, path: str) -> None:
        super().save(path)
        n = self._num_in_buffer
        meta = {
            "ep_id": self._ep_id[:n],
            "ep_t": self._ep_t[:n],
            "ep_len": self._ep_len[:n],
            "ep_start": self._ep_start[:n],
            "next_ep_id": self._next_ep_id,
        }
        torch.save(meta, path + ".her")

    def load(self, path: str) -> None:
        super().load(path)
        her_path = path + ".her"
        if not os.path.exists(her_path):
            raise FileNotFoundError(f"HER episode metadata missing: {her_path}")
        meta = torch.load(her_path, map_location=self._device)
        n = self._num_in_buffer
        self._ep_id[:n] = meta["ep_id"]
        self._ep_t[:n] = meta["ep_t"]
        self._ep_len[:n] = meta["ep_len"]
        self._ep_start[:n] = meta["ep_start"]
        self._next_ep_id = meta["next_ep_id"]
        self._cur_ep_id = int(max(int(self._ep_id[:n].max().item()) + 1, int(self._next_ep_id)))
        self._next_ep_id = max(int(self._next_ep_id), self._cur_ep_id)
        self._cur_ep_t = 0
        self._cur_ep_start = int(self._current_idx)


class PoseHERBuffer(HERTorchBuffer):
    def __init__(self, *args, her_cfg: HERConfig, layout: dict, rew_cfg: PoseRewardConfig,
                 zoom_pos_d0_m=(), zoom_ori_d0_deg=(), max_episode_steps: int = 100,
                 lim_lo=None, lim_hi=None, near_sigma_p_m: float = 0.0,
                 near_sigma_r_deg: float = 0.0, near_zero_frac: float = 0.0,
                 near_anchor_frac: float = 0.0, near_anchor_rho: float = 10.0,
                 gain_cfg=None, **kwargs):
        super().__init__(*args, her_cfg=her_cfg, **kwargs)
        if not 0.0 <= near_anchor_frac <= 1.0:
            raise ValueError("near_anchor_frac in [0, 1]")
        if gain_cfg is None:
            raise ValueError("gain_cfg (the env's gain schedule) is required")
        self.near_anchor_frac = float(near_anchor_frac); self.near_anchor_rho = float(near_anchor_rho)
        self.gain_cfg = gain_cfg
        self._row_rho = torch.full((self._max_length,), float("inf"), device=self._device)
        self._nobs_p = torch.zeros((self._max_length, 3), device=self._device)
        self._nobs_R = torch.zeros((self._max_length, 3, 3), device=self._device)
        self._pm_memo = None
        self.layout = dict(layout)
        self.rc = rew_cfg
        self.near_sigma_p = float(near_sigma_p_m)
        self.near_sigma_r = math.radians(float(near_sigma_r_deg))
        if not 0.0 <= near_zero_frac <= 1.0:
            raise ValueError("near_zero_frac in [0, 1]")
        self.near_zero_frac = float(near_zero_frac)
        self.zoom_pos = tuple(float(v) for v in zoom_pos_d0_m)
        self.zoom_ori = tuple(math.radians(float(v)) for v in zoom_ori_d0_deg)
        self.T_max = int(max_episode_steps)
        self._lo = torch.as_tensor(lim_lo, dtype=torch.float32, device=self._device)
        self._hi = torch.as_tensor(lim_hi, dtype=torch.float32, device=self._device)
        for k in ("ee_pos", "ee_rot", "goal_pos", "goal_rot", "err_pos", "kp_err", "rho_best", "q", "dq", "prev_action"):
            if k not in layout:
                raise ValueError(f"layout lacks {k}")

    def _sl(self, x, name):
        a, b = self.layout[name]
        return x[:, a:b]

    def _pose_of(self, x):
        """(p [N,3], R [N,3,3]) from an obs tensor's ee slices."""
        return self._sl(x, "ee_pos"), t_R_from_rot6d(self._sl(x, "ee_rot"))

    def _rho(self, gp, gR, p, R):
        d = (gp - p).norm(dim=1)
        th = t_angle_between(R, gR)
        return d, th, torch.maximum(d / self.rc.tau_p, th / self.rc.tau_r)

    def _rho_of_rows(self, obs):
        """obs rho under the goal stored in the row (the goal the episode was run with)."""
        gp = self._sl(obs, "goal_pos"); gR = t_R_from_rot6d(self._sl(obs, "goal_rot"))
        p, R = self._pose_of(obs)
        return self._rho(gp, gR, p, R)[2]

    def _tag_new_rows(self, before_idx: int, before_n: int) -> None:
        m = self._max_length
        k = (int(self._current_idx) - before_idx) % m
        if k == 0 and int(self._num_in_buffer) > before_n:
            k = int(self._num_in_buffer) - before_n
        if k > 0:
            slots = (before_idx + torch.arange(k, device=self._device)) % m
            self._row_rho[slots] = self._rho_of_rows(self._observations[slots])
            self._nobs_p[slots], self._nobs_R[slots] = self._pose_of(self._next_observations[slots])

    def add(self, transition: Batch) -> None:
        before_idx, before_n = int(self._current_idx), int(self._num_in_buffer)
        super().add(transition)
        self._tag_new_rows(before_idx, before_n)

    def sample_anchors(self, n: int) -> torch.Tensor:
        anchor = torch.randint(0, self._num_in_buffer, (n,), device=self._device)
        k = int(round(n * self.near_anchor_frac))
        if k > 0:
            cand = torch.randint(0, self._num_in_buffer, (8 * k,), device=self._device)
            near = cand[self._row_rho[cand] <= self.near_anchor_rho]
            k_got = min(k, int(near.numel()))
            if k_got > 0:
                anchor[:k_got] = near[:k_got]
        return anchor

    def sample(self, sample_idxs=None) -> Batch:
        if sample_idxs is None:
            return self._sample_online_from(self.sample_anchors(self._sample_batch_size))
        return self._sample_online_from(torch.as_tensor(sample_idxs, device=self._device))

    def load(self, path: str) -> None:
        super().load(path)
        n = int(self._num_in_buffer)
        for b in range(0, n, 8192):
            e = min(n, b + 8192); self._row_rho[b:e] = self._rho_of_rows(self._observations[b:e])
            self._nobs_p[b:e], self._nobs_R[b:e] = self._pose_of(self._next_observations[b:e])

    def _write_goal_channels(self, x, gp, gR6, gR, p, R, rho, best_prev):
        """Rewrite every goal-dependent channel of x in place."""
        lay = self.layout
        x[:, lay["goal_pos"][0]:lay["goal_pos"][1]] = gp
        x[:, lay["goal_rot"][0]:lay["goal_rot"][1]] = gR6
        err = gp - p
        x[:, lay["err_pos"][0]:lay["err_pos"][1]] = err
        s = self.rc.kp_size
        O = torch.eye(3, dtype=x.dtype, device=x.device) * s
        kp = p.unsqueeze(1) + (R @ O).transpose(1, 2)
        kpg = gp.unsqueeze(1) + (gR @ O).transpose(1, 2)
        x[:, lay["kp_err"][0]:lay["kp_err"][1]] = (kpg - kp).reshape(x.shape[0], 9)
        if self.zoom_pos:
            rv = t_rotvec_body(R, gR)
            for k, d0 in enumerate(self.zoom_pos):
                a, b = lay[f"zoom_pos_{k}"]; x[:, a:b] = torch.clamp(err / d0, -1, 1)
            for k, d0 in enumerate(self.zoom_ori):
                a, b = lay[f"zoom_ori_{k}"]; x[:, a:b] = torch.clamp(rv / d0, -1, 1)
        a, b = lay["rho_best"]
        x[:, a:b] = torch_rho_best_channels(rho, best_prev, self.rc.rho_obs_d0)

    def _prefix_min_rho(self, gp, gR, ep_start, ep_t, eid, upto, src_nobs=None, src_ep_id=None):
        """min_{k < upto} pose-rho of next_obs_k under (gp, gR)."""
        dev = self._device
        src_nobs = self._next_observations if src_nobs is None else src_nobs
        src_ep_id = self._ep_id if src_ep_id is None else src_ep_id
        m = src_nobs.shape[0]
        N = gp.shape[0]
        k = torch.arange(self.T_max, device=dev)
        key = (gp.data_ptr(), gR.data_ptr(), ep_start.data_ptr(), eid.data_ptr(), ep_t.data_ptr(), N, src_nobs.data_ptr())
        memo = self._pm_memo; self._pm_memo = None
        if memo is not None and memo[0] == key and upto is not ep_t:
            rho_k, same_ep = memo[1], memo[2]
        else:
            slots = (ep_start[:, None] + k[None, :]) % m
            same_ep = src_ep_id[slots] == eid[:, None]
            flat = slots.reshape(-1)
            if src_nobs is self._next_observations:
                p_k, R_k = self._nobs_p[flat], self._nobs_R[flat]
            else:
                p_k, R_k = self._d_next_p[flat], self._d_next_R[flat]
            gp_r = gp.repeat_interleave(self.T_max, dim=0)
            gR_r = gR.repeat_interleave(self.T_max, dim=0)
            _, _, rho_k = self._rho(gp_r, gR_r, p_k, R_k)
            rho_k = rho_k.reshape(N, self.T_max)
            if upto is ep_t:
                self._pm_memo = (key, rho_k, same_ep)
        counted = k[None, :] < upto[:, None]
        broken = (counted & ~same_ep).any(dim=1)
        out = torch.where(counted & same_ep, rho_k, torch.full_like(rho_k, float("inf"))).min(dim=1).values
        return out, broken

    def sample_k(self, k: int) -> list:
        bs = self._sample_batch_size
        if k <= 1:
            return [self.sample()]
        self._sample_batch_size = k * bs
        try:
            big = self.sample()
        finally:
            self._sample_batch_size = bs
        perm = torch.randperm(k * bs, device=self._device)
        big = {kk: v[perm] for kk, v in big.items()}
        return [{kk: v[i * bs:(i + 1) * bs] for kk, v in big.items()} for i in range(k)]

    def _relabel_slice(self, obs, nobs, act, trunc, stored_reward, stored_term,
                       ep_t, ep_len, ep_start, eid, src_obs, src_ep_id, src_n,
                       cum, use_stored_for_orig, force_orig=None,
                       src_nobs=None) -> Batch:
        dev = self._device
        n = obs.shape[0]
        m_src = src_obs.shape[0]
        obs = obs.clone(); nobs = nobs.clone()
        obs_stored = obs.clone(); nobs_stored = nobs.clone()
        lay = self.layout

        p_obs, R_obs = self._pose_of(obs)
        p_next, R_next = self._pose_of(nobs)

        u = torch.rand(n, device=dev)
        is_orig = u < cum[0]
        is_future = (u >= cum[0]) & (u < cum[1])
        is_cur = (u >= cum[1]) & (u < cum[2])
        is_rand = u >= cum[2]

        gp = self._sl(obs, "goal_pos").clone(); gR6 = self._sl(obs, "goal_rot").clone()
        uu = torch.rand(n, device=dev).clamp_min(1e-9)
        offset = (torch.log(uu) / math.log(self._gamma)).ceil().long().clamp_min(1)
        future_t = torch.minimum(ep_t + offset, ep_len - 1)
        future_slot = (ep_start + future_t) % m_src
        future_valid = src_ep_id[future_slot] == eid
        fut = src_obs[future_slot]
        sel = (is_future & future_valid).unsqueeze(1)
        gp = torch.where(sel, self._sl(fut, "ee_pos"), gp); gR6 = torch.where(sel, self._sl(fut, "ee_rot"), gR6)
        a_p, a_R, a_R6 = p_obs, R_obs, self._sl(obs, "ee_rot")
        keep = (torch.rand(n, device=dev) < self.near_zero_frac).unsqueeze(1)
        near_p = torch.where(keep, a_p, a_p + self.near_sigma_p * torch.randn_like(a_p))
        if self.near_sigma_r > 0:
            R_near = a_R @ t_R_from_rotvec(self.near_sigma_r * torch.randn_like(a_p))
            near_R6 = torch.where(keep, a_R6, torch.cat([R_near[:, :, 0], R_near[:, :, 1]], dim=1))
        else:
            near_R6 = a_R6
        sel = is_cur.unsqueeze(1)
        gp = torch.where(sel, near_p, gp); gR6 = torch.where(sel, near_R6, gR6)
        rnd = src_obs[torch.randint(0, src_n, (n,), device=dev)]
        sel = is_rand.unsqueeze(1)
        gp = torch.where(sel, self._sl(rnd, "ee_pos"), gp); gR6 = torch.where(sel, self._sl(rnd, "ee_rot"), gR6)
        if force_orig is not None:
            sel = force_orig.unsqueeze(1)
            gp = torch.where(sel, self._sl(obs_stored, "goal_pos"), gp)
            gR6 = torch.where(sel, self._sl(obs_stored, "goal_rot"), gR6)
        gR = t_R_from_rot6d(gR6)

        d_obs, th_obs, rho_obs = self._rho(gp, gR, p_obs, R_obs)
        d_next, th_next, rho_next = self._rho(gp, gR, p_next, R_next)
        kp_next = t_keypoint_error(p_next, R_next, gp, gR, self.rc.kp_size)

        best_prev_next, broken = self._prefix_min_rho(gp, gR, ep_start, ep_t, eid, upto=ep_t,
                                                      src_nobs=src_nobs, src_ep_id=src_ep_id)
        best_prev_obs, _ = self._prefix_min_rho(gp, gR, ep_start, ep_t, eid, upto=(ep_t - 1).clamp_min(0),
                                                src_nobs=src_nobs, src_ep_id=src_ep_id)
        best_prev_next = torch.where(torch.isinf(best_prev_next), rho_next, best_prev_next)
        best_prev_obs = torch.where(torch.isinf(best_prev_obs), rho_obs, best_prev_obs)

        self._write_goal_channels(obs, gp, gR6, gR, p_obs, R_obs, rho_obs, best_prev_obs)
        self._write_goal_channels(nobs, gp, gR6, gR, p_next, R_next, rho_next, best_prev_next)

        pa = self._sl(obs, "prev_action")
        want = is_cur | is_rand | (is_future & future_valid)
        if force_orig is not None:
            want = want & ~force_orig
        gp0 = self._sl(obs_stored, "goal_pos"); gR0 = t_R_from_rot6d(self._sl(obs_stored, "goal_rot"))
        d0, th0, _ = self._rho(gp0, gR0, p_obs, R_obs)
        ratio = (action_gain(d0, th0, self.gain_cfg, torch) / action_gain(d_obs, th_obs, self.gain_cfg, torch)).unsqueeze(1)
        a_rel = torch.clamp(act * ratio, -1.0, 1.0)
        prev_slot = (ep_start + (ep_t - 1).clamp_min(0)) % m_src
        prev_valid = (ep_t > 0) & (src_ep_id[prev_slot] == eid)
        pobs = src_obs[prev_slot]; pp, pR = self._pose_of(pobs)
        dp0, thp0, _ = self._rho(self._sl(pobs, "goal_pos"), t_R_from_rot6d(self._sl(pobs, "goal_rot")), pp, pR)
        dpn, thpn, _ = self._rho(gp, gR, pp, pR)
        pratio = (action_gain(dp0, thp0, self.gain_cfg, torch) / action_gain(dpn, thpn, self.gain_cfg, torch)).unsqueeze(1)
        pa_rel = torch.where(prev_valid.unsqueeze(1), torch.clamp(pa * pratio, -1.0, 1.0), pa)
        w = want.unsqueeze(1)
        act_used = torch.where(w, a_rel, act); pa_used = torch.where(w, pa_rel, pa)
        a_, b_ = lay["prev_action"]; obs[:, a_:b_] = pa_used; nobs[:, a_:b_] = act_used
        reward, _ = torch_pose_reward(self.rc, d_next, th_next, kp_next, rho_obs, best_prev_next,
                                      act_used, pa_used, self._sl(nobs, "dq"), self._sl(nobs, "q"), self._lo, self._hi)
        terminated = stored_term.clone()

        use_stored = torch.zeros_like(is_orig)
        if use_stored_for_orig:
            use_stored = is_orig | (is_future & ~future_valid)
        if force_orig is not None:
            use_stored = use_stored | force_orig
        use_stored = use_stored | broken
        reward = torch.where(use_stored, stored_reward, reward)
        terminated = torch.where(use_stored, stored_term, terminated)
        obs = torch.where(use_stored.unsqueeze(1), obs_stored, obs)
        nobs = torch.where(use_stored.unsqueeze(1), nobs_stored, nobs)
        act = torch.where(use_stored.unsqueeze(1), act, act_used)

        return {"observation": obs, "action": act, "reward": reward, "terminated": terminated,
                "truncated": trunc, "next_observation": nobs}

class PoseHERRLPDBuffer(PoseHERBuffer):
    """PoseHERBuffer + a frozen demonstration store; each batch draws `demo_ratio` of its rows from the demonstrations."""

    def __init__(self, *args, demo_ratio: float = 0.5, demo_p_orig: float = 0.5,
                 demo_p_future: float = 0.3, demo_p_cur: float = 0.2, demo_p_rand: float = 0.0,
                 demo_near_anchor_frac: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= demo_near_anchor_frac <= 1.0:
            raise ValueError("demo_near_anchor_frac in [0, 1]")
        self.demo_near_anchor_frac = float(demo_near_anchor_frac); self._demo_rho = None
        if not 0.0 <= demo_ratio < 1.0:
            raise ValueError("demo_ratio in [0, 1)")
        mix = (demo_p_orig, demo_p_future, demo_p_cur, demo_p_rand)
        if min(mix) < 0 or abs(sum(mix) - 1.0) > 1e-6:
            raise ValueError(f"demo goal mix must be non-negative and sum to 1, got {mix}")
        self.demo_ratio = float(demo_ratio)
        self._demo_cum = self._cumulative(demo_p_orig, demo_p_future, demo_p_cur)
        self.n_demos = 0
        self._d_obs = self._d_next = self._d_act = None
        self._d_rew = self._d_term = self._d_trunc = None
        self._d_ep_id = self._d_ep_t = self._d_ep_len = self._d_ep_start = None
        self._d_next_p = self._d_next_R = None

    def set_demos(self, observation, next_observation, action, reward, terminated, truncated, ep_id, ep_t) -> None:
        obs = np.asarray(observation, np.float32); next_obs = np.asarray(next_observation, np.float32)
        act = np.asarray(action, np.float32)
        rew = np.asarray(reward, np.float32).reshape(-1)
        term = np.asarray(terminated, np.float32).reshape(-1)
        trunc = np.asarray(truncated, np.float32).reshape(-1)
        ep_id = np.asarray(ep_id, np.int64).reshape(-1); ep_t = np.asarray(ep_t, np.int64).reshape(-1)
        N = obs.shape[0]
        for name, arr in (("next_obs", next_obs), ("act", act), ("rew", rew), ("term", term),
                          ("trunc", trunc), ("ep_id", ep_id), ("ep_t", ep_t)):
            if arr.shape[0] != N:
                raise ValueError(f"demo {name} has {arr.shape[0]} rows, obs has {N}")
        if obs.shape[1] != self._observations.shape[1]:
            raise ValueError(f"demo obs dim {obs.shape[1]} != buffer obs dim {self._observations.shape[1]}")
        if act.shape[1] != self._actions.shape[1]:
            raise ValueError(f"demo action dim {act.shape[1]} != buffer action dim {self._actions.shape[1]}")
        if N == 0:
            raise ValueError("no demo rows")
        starts = np.flatnonzero(ep_t == 0)
        if starts.size == 0 or starts[0] != 0:
            raise ValueError("demo rows must start with an ep_t == 0 row")
        ends = np.append(starts[1:], N)
        blk_id = np.zeros(N, np.int64); blk_start = np.zeros(N, np.int64); blk_len = np.zeros(N, np.int64)
        for b, (s0, s1) in enumerate(zip(starts, ends)):
            L = s1 - s0
            if not np.array_equal(ep_t[s0:s1], np.arange(L)):
                raise ValueError(f"demo episode {b} (task {ep_id[s0]}): ep_t is not 0..{L - 1}")
            if L > self.T_max:
                raise ValueError(f"demo episode {b} (task {ep_id[s0]}) has {L} rows > max_episode_steps {self.T_max}")
            if term[s1 - 1] < 0.5 and trunc[s1 - 1] < 0.5:
                raise ValueError(f"demo episode {b} (task {ep_id[s0]}) is INCOMPLETE ({L} rows, last row not "
                                 f"terminated|truncated) -- drop it before loading")
            blk_id[s0:s1] = b; blk_start[s0:s1] = s0; blk_len[s0:s1] = L

        def t(x, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype, device=self._device)
        self._d_obs, self._d_next, self._d_act = t(obs), t(next_obs), t(act)
        self._d_rew, self._d_term, self._d_trunc = t(rew), t(term), t(trunc)
        self._d_ep_id, self._d_ep_t = t(blk_id, torch.int64), t(ep_t, torch.int64)
        self._d_ep_len, self._d_ep_start = t(blk_len, torch.int64), t(blk_start, torch.int64)
        self.n_demos = int(N)
        self._demo_rho = self._rho_of_rows(self._d_obs)
        self._d_next_p, self._d_next_R = self._pose_of(self._d_next)

    def _demo_anchors(self, n: int) -> torch.Tensor:
        anchor = torch.randint(0, self.n_demos, (n,), device=self._device)
        k = int(round(n * self.demo_near_anchor_frac))
        if k > 0 and self._demo_rho is not None:
            cand = torch.randint(0, self.n_demos, (8 * k,), device=self._device)
            near = cand[self._demo_rho[cand] <= self.near_anchor_rho]
            k_got = min(k, int(near.numel()))
            if k_got > 0:
                anchor[:k_got] = near[:k_got]
        return anchor

    def _sample_demo(self, n: int) -> Batch:
        anchor = self._demo_anchors(n)
        return self._relabel_slice(
            obs=self._d_obs[anchor], nobs=self._d_next[anchor], act=self._d_act[anchor],
            trunc=self._d_trunc[anchor], stored_reward=self._d_rew[anchor], stored_term=self._d_term[anchor],
            ep_t=self._d_ep_t[anchor], ep_len=self._d_ep_len[anchor], ep_start=self._d_ep_start[anchor],
            eid=self._d_ep_id[anchor],
            src_obs=self._d_obs, src_ep_id=self._d_ep_id, src_n=self.n_demos,
            cum=self._demo_cum, use_stored_for_orig=False,
            src_nobs=self._d_next)

    def sample(self, sample_idxs=None) -> Batch:
        if self.n_demos == 0 or self.demo_ratio <= 0.0 or sample_idxs is not None:
            return super().sample(sample_idxs)
        bs = self._sample_batch_size
        n_demo = int(round(bs * self.demo_ratio))
        n_online = bs - n_demo
        if self._num_in_buffer <= 0:
            return self._sample_demo(bs)
        parts = []
        if n_online > 0:
            anchor = self.sample_anchors(n_online)
            parts.append(self._sample_online_from(anchor))
        if n_demo > 0:
            parts.append(self._sample_demo(n_demo))
        return {k: torch.cat([q[k] for q in parts], dim=0) for k in parts[0]}
