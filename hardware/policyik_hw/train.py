"""Train FlashSAC with hindsight relabelling and demonstrations on 6-D pose reaching on the real arm."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict

import numpy as np
import yaml

from .envs.workspace import schedule_value
from .rl.demos import interface_diff, load_demos
from .utils import kinematics

RUN_ROW_MEAN_KEYS = ("r", "l", "final_dist", "min_dist", "final_ang_deg")
RUN_ROW_RATE_KEYS = ("settled_success", "strict_settled")
TIMING_KEYS = ("step_wall_s_mean", "upd_step_s_per_step", "n_upd_step_per_step", "upd_reset_s", "n_upd_reset",
               "est_update_s")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--name", default=None, help="run name (default: pose_<time>)")
    p.add_argument("--total-steps", type=int, default=None, help="steps of this run segment")
    p.add_argument("--checkpoint-at", default=None, metavar="STEPS", help="comma-separated global steps at which to save a checkpoint")
    p.add_argument("--init-from", default=None, metavar="DIR", help="checkpoint directory to resume from")
    p.add_argument("--init-buffer", default=None, metavar="PATH", help="replay buffer saved with that checkpoint")
    p.add_argument("--step-offset", type=int, default=0, metavar="N",
                   help="global step of the resumed checkpoint; schedules continue from it")
    p.add_argument("--urdf", default=None, help="OpenArm v1 URDF (default: from the openarm_description package)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p


def resolve(args, cfg: dict, base: str) -> dict:
    """Apply the command line to the configuration."""
    a, d = cfg["agent"], cfg["demos"]
    if args.total_steps is not None:
        a["total_steps"] = args.total_steps
    if args.checkpoint_at is not None:
        a["checkpoint_at"] = sorted({int(x) for x in args.checkpoint_at.split(",") if x.strip()})
    if args.step_offset < 0:
        raise SystemExit("--step-offset must be >= 0")
    if args.step_offset > 0 and not (args.init_from and args.init_buffer):
        raise SystemExit("--step-offset is for resumed runs: pass --init-from and --init-buffer as well")
    if args.init_from:
        a["random_warmup"] = 0
    a["step_offset"] = args.step_offset
    a["buffer_size"] = a["total_steps"] + args.step_offset
    cfg["reward"]["pbrs_gamma"] = a["gamma"]
    d["p_rand"] = round(1.0 - d["p_orig"] - d["p_future"] - d["p_cur"], 9)
    d["p_rand"] = 0.0 if abs(d["p_rand"]) < 1e-9 else d["p_rand"]
    for k in ("file", "task_pool"):
        d[k] = d[k] if os.path.isabs(d[k]) else os.path.join(base, d[k])
    d["task_pool_frac_schedule"] = parse_schedule(d["task_pool_frac_schedule"], "demos.task_pool_frac_schedule")
    return cfg


def parse_schedule(spec, name: str) -> list:
    sch = [[float(f), float(v)] for f, v in spec]
    fr = [f for f, _ in sch]
    if not sch or fr[0] != 0.0 or any(b <= a for a, b in zip(fr, fr[1:])) or fr[-1] >= 1.0:
        raise SystemExit(f"{name}: fractions must start at 0, strictly increase and stay < 1, got {fr}")
    if any(not 0.0 <= v <= 1.0 for _, v in sch):
        raise SystemExit(f"{name}: values must lie in [0, 1]")
    return sch


def load_task_pool(path: str) -> list:
    """Start/goal tasks mixed into training: a list of {q_start, goal_p, goal_R}."""
    with open(path) as f:
        tasks = [t for t in json.load(f)["tasks"] if all(k in t for k in ("q_start", "goal_p", "goal_R"))]
    if not tasks:
        raise SystemExit(f"{path} has no task with q_start/goal_p/goal_R")
    return tasks


class RunLogger:
    def __init__(self, out_dir: str):
        os.makedirs(os.path.join(out_dir, "checkpoint"), exist_ok=True)
        self._m = open(os.path.join(out_dir, "metrics.jsonl"), "w")
        self._e = open(os.path.join(out_dir, "episodes.jsonl"), "w")
        self._c = open(os.path.join(out_dir, "console.log"), "a", buffering=1)
        self._orig = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self._c)
        sys.stderr = _Tee(sys.stderr, self._c)

    def metric(self, row: dict) -> None:
        self._m.write(json.dumps(row) + "\n")
        self._m.flush()

    def episode(self, row: dict) -> None:
        self._e.write(json.dumps(row) + "\n")
        self._e.flush()

    def close(self) -> None:
        self._m.close()
        self._e.close()
        sys.stdout, sys.stderr = self._orig
        self._c.close()


class _Tee:
    """Write to the terminal and a log file."""

    def __init__(self, a, b):
        self._a, self._b = a, b

    def write(self, x):
        self._a.write(x)
        self._b.write(x)
        return len(x)

    def flush(self):
        self._a.flush()
        self._b.flush()

    def __getattr__(self, k):
        return getattr(self._a, k)


def _fmt(x, spec=".3f"):
    return "  n/a" if x is None or not np.isfinite(x) else format(x, spec)


def build_agent(cfg: dict, env, device: str, training: bool = True):
    """FlashSAC agent; with training=False it only loads a checkpoint for deterministic actions."""
    from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig
    from flash_rl.agents.flashSAC.lan_critic import lan_cols_from_layout

    ac = cfg["agent"]
    fs_cfg = FlashSACConfig(
        seed=ac["seed"], normalize_reward=ac["normalize_reward"], normalized_G_max=ac["reward_g_max"],
        asymmetric_observation=False, device_type=device,
        buffer_max_length=ac["buffer_size"], buffer_min_length=ac["buffer_min"],
        buffer_device_type=device, sample_batch_size=ac["batch_size"],
        learning_rate_init=ac["lr"], learning_rate_peak=ac["lr"], learning_rate_end=ac["lr"],
        learning_rate_warmup_rate=0.0, learning_rate_warmup_step=0, learning_rate_decay_rate=1.0,
        learning_rate_decay_step=max(1, ac["total_steps"] * ac["utd"]),
        actor_num_blocks=ac["actor_num_blocks"], actor_hidden_dim=ac["actor_hidden_dim"],
        actor_bc_alpha=0.0, actor_noise_zeta_mu=2.0, actor_noise_zeta_max=16,
        actor_update_period=ac["actor_update_period"],
        critic_num_blocks=ac["critic_num_blocks"], critic_hidden_dim=ac["critic_hidden_dim"],
        critic_num_bins=ac["critic_num_bins"], critic_min_v=-ac["reward_g_max"], critic_max_v=ac["reward_g_max"],
        critic_target_update_tau=ac["tau"],
        temp_initial_value=0.01, temp_target_sigma=ac["temp_target_sigma"], temp_target_entropy=None,
        gamma=ac["gamma"], n_step=1, use_compile=training and bool(ac["use_compile"]), compile_mode="auto",
        use_amp=False, load_optimizer=training, load_reward_normalizer=True,
        lan_cols=lan_cols_from_layout(env.layout),
    )
    return FlashSACAgent(env.observation_space, env.action_space, env_info={}, cfg=fs_cfg), fs_cfg


def build_buffer(cfg: dict, layout: dict, env, rew_cfg, fs_cfg):
    """Replay ring that relabels pose goals, with the frozen demonstration store beside it."""
    from .rl.her import HERConfig, PoseHERRLPDBuffer

    h, d, ec = cfg["her"], cfg["demos"], env.cfg
    her_cfg = HERConfig(p_orig=h["p_orig"], p_future=h["p_future"], p_cur=h["p_cur"], p_rand=h["p_rand"])
    return PoseHERRLPDBuffer(
        observation_space=env.observation_space, action_space=env.action_space, n_step=1, gamma=fs_cfg.gamma,
        max_length=fs_cfg.buffer_max_length, min_length=fs_cfg.buffer_min_length,
        sample_batch_size=fs_cfg.sample_batch_size, device_type=fs_cfg.buffer_device_type,
        her_cfg=her_cfg, layout=layout, rew_cfg=rew_cfg,
        zoom_pos_d0_m=ec.zoom_pos_d0_m, zoom_ori_d0_deg=ec.zoom_ori_d0_deg,
        max_episode_steps=ec.max_episode_steps, lim_lo=env.lim_lo, lim_hi=env.lim_hi,
        near_sigma_p_m=h["near_sigma_p_m"], near_sigma_r_deg=h["near_sigma_r_deg"],
        near_zero_frac=h["near_zero_frac"], near_anchor_frac=h["near_anchor_frac"],
        near_anchor_rho=h["near_anchor_rho"],
        gain_cfg=ec,
        demo_ratio=d["ratio"], demo_p_orig=d["p_orig"], demo_p_future=d["p_future"], demo_p_cur=d["p_cur"],
        demo_p_rand=d["p_rand"], demo_near_anchor_frac=d["near_anchor_frac"])


def check_resume(ck: str, env_d: dict, rew_d: dict, layout: dict) -> None:
    """Refuse a checkpoint trained on a different action law or observation."""
    with open(os.path.join(os.path.dirname(os.path.normpath(ck)), "config.json")) as f:
        src = json.load(f)
    if src["layout"]["obs_dim"] != layout["obs_dim"]:
        raise SystemExit(f"--init-from: checkpoint obs dim {src['layout']['obs_dim']} != {layout['obs_dim']}")
    mism, _ = interface_diff(src["env"], env_d, src["reward"], rew_d)
    if mism:
        raise SystemExit("--init-from: the checkpoint was trained on a different interface:\n  " + "\n  ".join(mism))


def log_row(step: int, el: float, eps: list, new_eps: int, cfg: dict, agent, last_info: dict, off: int,
            run_frac: float, pool_frac: float) -> dict:
    recent = eps[-cfg["log"]["window"]:]
    row = {"step": step, "wall_s": el, "n_episodes": len(eps), "new_episodes": int(new_eps)}
    if recent:
        for k in RUN_ROW_MEAN_KEYS:
            vals = [e[k] for e in recent if k in e and e[k] is not None and np.isfinite(e[k])]
            row[k] = float(np.mean(vals)) if vals else float("nan")
        for k in RUN_ROW_RATE_KEYS:
            row[k] = float(np.mean([float(e[k]) for e in recent]))
        row["term_means"] = {k: float(np.mean([e["term_means"].get(k, 0.0) for e in recent]))
                             for k in recent[-1]["term_means"]}
    for k in ("actor/loss", "critic/loss", "actor/entropy", "temperature/value"):
        if k in last_info:
            row[k] = float(last_info[k])
    buf = agent._replay_buffer
    row.update(replay=len(buf), global_step=step + off, run_frac=run_frac, pool_frac_target=pool_frac,
               steps_per_s=step / max(el, 1e-9))
    if recent:
        for k in TIMING_KEYS:
            vals = [e[k] for e in recent if k in e and e[k] == e[k]]
            row[k] = float(np.mean(vals)) if vals else float("nan")
        row["pool_frac"] = float(np.mean([float(e["from_pool"]) for e in recent]))
    return row


def print_row(row: dict, total: int, off: int) -> None:
    print(f"[{row['step']:>5}/{total}{' g' + str(row['global_step']) if off else ''}] {row['wall_s'] / 60:5.1f}m  "
          f"ep{row['n_episodes']:>4}  ret {_fmt(row.get('r'), '+7.2f')}  "
          f"d_fin {_fmt(row.get('final_dist', np.nan) * 1000, '5.1f')}mm "
          f"th_fin {_fmt(row.get('final_ang_deg'), '5.2f')}deg  "
          f"strict settled {_fmt(row.get('strict_settled'), '.0%')}  "
          f"coarse settled {_fmt(row.get('settled_success'), '.0%')}  "
          f"critic {_fmt(row.get('critic/loss'), '.3f')}  temp {_fmt(row.get('temperature/value'), '.4f')}")
    print(f"{'':>7}   timing  step {_fmt(row.get('step_wall_s_mean'), '.3f')} s, "
          f"{_fmt(row.get('n_upd_step_per_step'), '.1f')} updates per step, "
          f"{_fmt(row.get('n_upd_reset'), '.0f')} updates per reset   pool {_fmt(row.get('pool_frac'), '.0%')} "
          f"(target {_fmt(row.get('pool_frac_target'), '.0%')})")


def make_env(cfg: dict, io=None):
    from .envs.pose_env import EnvConfig, OpenArmPoseEnv
    from .envs.reward import PoseRewardConfig
    from .envs.workspace import ReverseConfig

    env_cfg = EnvConfig(**cfg["env"])
    env_cfg.validate()
    rew_cfg = PoseRewardConfig(**cfg["reward"])
    rew_cfg.dt = 1.0 / env_cfg.control_hz
    rew_cfg.validate(agent_gamma=cfg["agent"]["gamma"])
    ac = cfg["agent"]
    curr_cfg = ReverseConfig(**cfg["curriculum"], total_steps=ac["total_steps"] + ac["step_offset"])
    curr_cfg.validate()
    env = OpenArmPoseEnv(env_cfg, rew_cfg, curr_cfg=curr_cfg, io=io)
    if ac["step_offset"] > 0:
        env.curriculum.sigma_deg = env.curriculum.scheduled_sigma(ac["step_offset"])
    return env, rew_cfg


def write_summary(out_dir: str, env, eps: list, step: int, off: int, t_start: float) -> None:
    timing = env.timing_summary()
    summary = {"steps_completed": step, "global_steps_completed": step + off, "n_episodes": len(eps),
               "wall_s": time.time() - t_start, "timing": timing}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if timing:
        print(f"  median step {timing['step_wall_median_s']:.3f} s: aggregate the evaluation with "
              f"--eval-hz {timing['effective_hz']:.2f}")
    print(f"  artifacts in {out_dir}")


def save_all(agent, out_dir: str, suffix: str) -> None:
    ck = os.path.join(out_dir, f"checkpoint{suffix}")
    os.makedirs(ck, exist_ok=True)
    agent.save(ck)
    agent._replay_buffer.save(os.path.join(out_dir, f"buffer{suffix}"))
    print(f"[save] {ck} and buffer{suffix}")


def main(io=None) -> None:
    args = build_parser().parse_args()
    cfg_path = os.path.abspath(args.config)
    with open(cfg_path) as f:
        cfg = resolve(args, yaml.safe_load(f), os.path.dirname(cfg_path))
    kinematics.set_urdf_path(args.urdf)

    import torch

    from .envs.pose_env import build_layout
    from .rl.learner_hooks import LearnerHooks

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    ac, d = cfg["agent"], cfg["demos"]
    off, total = ac["step_offset"], ac["total_steps"]
    name = args.name or f"pose_{time.strftime('%m%d_%H%M')}"
    out_dir = os.path.join(os.path.dirname(cfg_path), cfg["log"]["save_dir"], name)

    if io is None:
        import rclpy

        from .robot.ros_io import check_single_control_stack
        check_single_control_stack(raise_on_fail=True)
        rclpy.init()
    env, rew_cfg = make_env(cfg, io)
    layout = build_layout(env.cfg)
    try:
        demo_arrs, reward_diffs = load_demos(d["file"], layout["obs_dim"], asdict(env.cfg), rew_cfg.to_dict())
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"[demos] refused: {e}")
    pool = load_task_pool(d["task_pool"])
    print(f"[env] observation {env.observation_space.shape[0]}-D, action {env.action_space.shape[0]}-D, "
          f"dt {env.dt:.3f} s, device {device}, out {out_dir}")
    print(f"[demos] {demo_arrs['ep_t'].size} rows / {int((demo_arrs['ep_t'] == 0).sum())} episodes, "
          f"{d['ratio']:.0%} of every batch; task pool {len(pool)} tasks")
    if reward_diffs:
        print("[demos] recorded under other reward settings, rewards recomputed: " + "; ".join(reward_diffs))

    agent, fs_cfg = build_agent(cfg, env, device)
    if args.init_from:
        check_resume(args.init_from, asdict(env.cfg), rew_cfg.to_dict(), layout)
        agent.load(args.init_from)
        print(f"[init] loaded {args.init_from}")
    agent._replay_buffer = build_buffer(cfg, layout, env, rew_cfg, fs_cfg)
    try:
        agent._replay_buffer.set_demos(**demo_arrs)
    except ValueError as e:
        raise SystemExit(f"[demos] refused by the buffer: {e}")
    if args.init_buffer:
        agent._replay_buffer.load(args.init_buffer)
        print(f"[init] replay buffer {args.init_buffer} ({int(agent._replay_buffer._num_in_buffer)} transitions)")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"env": asdict(env.cfg), "reward": rew_cfg.to_dict(), "layout": layout, "agent": ac,
                   "her": cfg["her"], "curriculum": cfg["curriculum"], "demos": d, "init_from": args.init_from},
                  f, indent=2)
    logger = RunLogger(out_dir)

    state = {"stop": False}

    def _sigint(_sig, _frm):
        if state["stop"]:
            print("\n[stop] second Ctrl-C, exiting now")
            sys.exit(1)
        state["stop"] = True
        print("\n[stop] Ctrl-C: finishing this step, then saving. Press again to abort.")

    signal.signal(signal.SIGINT, _sigint)

    pool_rng = np.random.default_rng(ac["seed"] + 777)
    ep_src = {"pool": None}
    run_step = {"step": 0}

    def run_frac() -> float:
        return (run_step["step"] + off) / max(1, total + off)

    def pool_frac_now() -> float:
        return schedule_value(d["task_pool_frac_schedule"], run_frac())

    def reset_env():
        if pool_rng.random() < pool_frac_now():
            i = int(pool_rng.integers(len(pool)))
            t = pool[i]
            ep_src["pool"] = {"pool_task": i, "pool_trial_id": t.get("trial_id")}
            return env.reset(options={"q_start": np.asarray(t["q_start"], float),
                                      "goal_p": np.asarray(t["goal_p"], float),
                                      "goal_R": np.asarray(t["goal_R"], float)})[0]
        ep_src["pool"] = None
        return env.reset()[0]

    hooks = LearnerHooks(agent, utd=ac["utd"])
    env.learn_fn = hooks.learn_step
    env.idle_fn = hooks.learn_idle

    eps: list = []
    last_info: dict = {}
    t_start = time.time()
    obs = reset_env()
    step = 0
    last_logged_eps = 0
    ck_at = set(ac["checkpoint_at"])
    try:
        for step in range(1, total + 1):
            run_step["step"] = step
            if step <= ac["random_warmup"] or not agent.can_start_training():
                action = env.action_space.sample()
            else:
                prev = {"next_observation": obs[None].astype(np.float32)}
                action = agent.sample_actions(step, prev_transition=prev, training=True)[0]
            action = np.asarray(action, dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.process_transition({
                "observation": obs[None].astype(np.float32),
                "action": action[None],
                "reward": np.array([reward], dtype=np.float32),
                "terminated": np.array([float(terminated)], dtype=np.float32),
                "truncated": np.array([float(truncated)], dtype=np.float32),
                "next_observation": next_obs[None].astype(np.float32),
            })
            hooks.note_step(info.get("step_wall_s", float("nan")))
            last_info.update(hooks.last_info)
            obs = next_obs

            if terminated or truncated:
                s = info["episode"]
                s.update({"step": step, "wall_s": time.time() - t_start, "terminated": bool(terminated),
                          "from_pool": ep_src["pool"] is not None})
                s.update(hooks.pop_episode_timing())
                if s["from_pool"]:
                    s.update(ep_src["pool"])
                    s["sigma_deg"] = float("nan")
                else:
                    ev = env.curriculum.on_episode(step + off)
                    if ev:
                        print(f"[curriculum] ep{len(eps)} {ev}")
                s["sigma_after"] = env.curriculum.sigma_deg
                eps.append(s)
                logger.episode(s)
                obs = reset_env()

            if step + off in ck_at:
                save_all(agent, out_dir, f"_step{step + off}")

            new_eps = len(eps) - last_logged_eps
            if (step % cfg["log"]["every"] == 0 and new_eps >= 1) or state["stop"]:
                last_logged_eps = len(eps)
                row = log_row(step, time.time() - t_start, eps, new_eps, cfg, agent, last_info, off,
                              run_frac(), pool_frac_now())
                logger.metric(row)
                print_row(row, total, off)
            if state["stop"]:
                print(f"[stop] stopping at step {step}")
                break
    finally:
        save_all(agent, out_dir, "")
        write_summary(out_dir, env, eps, step, off, t_start)
        logger.close()
        env.close()
        if io is None:
            import rclpy
            rclpy.shutdown()


if __name__ == "__main__":
    main()
