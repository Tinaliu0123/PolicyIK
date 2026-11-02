"""Gradient updates inside the control period and inside the reset windows of the real-arm loop."""
from __future__ import annotations

import time


class LearnerHooks:
    def __init__(self, agent, utd: int, est_update_s: float = 0.07, warmup_updates: int = 3):
        self.agent = agent; self.utd = int(utd)
        self.max_chunk = 8
        self.last_info: dict = {}
        self._ema_one = float(est_update_s)
        self.warmup_updates = int(warmup_updates)
        self.n_total_upd = 0
        self._reset_episode()

    def _reset_episode(self):
        self.t_step_upd = 0.0; self.n_step_upd = 0; self.n_steps = 0
        self.t_reset_upd = 0.0; self.n_reset_upd = 0; self.n_reset_windows = 0
        self.step_wall_sum = 0.0; self.n_step_wall = 0

    def note_step(self, step_wall_s: float):
        if step_wall_s == step_wall_s:
            self.step_wall_sum += float(step_wall_s); self.n_step_wall += 1

    def pop_episode_timing(self) -> dict:
        n = max(1, self.n_steps)
        out = {"upd_step_s_per_step": self.t_step_upd / n, "n_upd_step_per_step": self.n_step_upd / n,
               "upd_reset_s": self.t_reset_upd, "n_upd_reset": self.n_reset_upd, "n_reset_windows": self.n_reset_windows,
               "step_wall_s_mean": (self.step_wall_sum / self.n_step_wall) if self.n_step_wall else float("nan"),
               "est_update_s": self._ema_one}
        self._reset_episode()
        return out

    def _updates(self, k: int) -> dict:
        buf = self.agent._replay_buffer; u = {}
        if hasattr(buf, "sample_k") and k > 1:
            for b in buf.sample_k(k):
                u = self.agent.update(batch=b)
        else:
            for _ in range(k):
                u = self.agent.update()
        self.n_total_upd += k
        return u

    def _fold(self, d_one: float, chunked: bool) -> None:
        """EMA of the wall time of one update."""
        self._ema_one = 0.8 * self._ema_one + 0.2 * (d_one * 1.5 if chunked else d_one)

    def learn_step(self) -> None:
        self.n_steps += 1
        if self.utd <= 0 or not self.agent.can_start_training():
            return
        t0 = time.perf_counter(); n_before = self.n_total_upd
        self.last_info = self._updates(self.utd)
        dt = time.perf_counter() - t0
        self.t_step_upd += dt; self.n_step_upd += self.utd
        if n_before >= self.warmup_updates:
            self._fold(dt / self.utd, self.utd > 1)

    def learn_idle(self, budget_s: float) -> None:
        """Budgeted single updates inside a reset window (ramp step or settle)."""
        self.n_reset_windows += 1
        if not self.agent.can_start_training():
            return
        if self.n_total_upd < self.warmup_updates:
            return
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) + self._ema_one <= budget_s:
            remaining = budget_s - (time.perf_counter() - t0)
            k = max(1, min(self.max_chunk, int(remaining / self._ema_one)))
            t1 = time.perf_counter()
            self.last_info = self._updates(k)
            d = (time.perf_counter() - t1) / k
            self._fold(d, k > 1)
            self.n_reset_upd += k
        self.t_reset_upd += time.perf_counter() - t0
