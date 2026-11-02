"""Evaluate a trained checkpoint on the real arm on fixed start/goal tasks."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .utils import kinematics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", required=True, help="run directory written by training")
    p.add_argument("--task-file", required=True, help="evaluation tasks (q_start, goal_p, goal_R, gap_m)")
    p.add_argument("--gap-bin", default=None, metavar="LO,HI", help="metres: only tasks whose start-goal gap lies here")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None, help="episode length (default: the training value)")
    p.add_argument("--no-collision-guard", action="store_true", help="do not block commands at the envelope")
    p.add_argument("--checkpoint", default="checkpoint", metavar="DIR", help="checkpoint directory inside the run")
    p.add_argument("--tag", default=None, help="extra token for the output file name")
    p.add_argument("--out", default=None, help="json path (default: <run>/eval_<settings>.json)")
    p.add_argument("--overwrite", action="store_true",
                   help="start the result file fresh; by default a file with the same settings is resumed and a "
                        "task interrupted with Ctrl-C is recorded as skipped")
    p.add_argument("--urdf", default=None, help="OpenArm v1 URDF (default: from the openarm_description package)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p


def make_env_cfg(saved: dict, args):
    from .envs.pose_env import EnvConfig

    env_cfg = EnvConfig(**saved["env"])
    env_cfg.goal_jitter_deg = 0.0
    env_cfg.frozen_state_steps = 0
    env_cfg.obs_noise_q_rad = 0.0
    env_cfg.obs_noise_dq_rad_s = 0.0
    env_cfg.collision_guard = not args.no_collision_guard
    if args.max_steps is not None:
        env_cfg.max_episode_steps = args.max_steps
    env_cfg.validate()
    return env_cfg


def load_tasks(path: str, gap_bin) -> list:
    with open(path) as f:
        tasks = [t for t in json.load(f)["tasks"] if gap_bin is None or gap_bin[0] <= t["gap_m"] <= gap_bin[1]]
    if not tasks:
        raise SystemExit("--task-file has no task in the gap bin")
    return tasks


def output_path(args, run_dir: str, env_cfg, gap_bin) -> str:
    if args.out:
        return args.out
    stem = "eval"
    if gap_bin is not None:
        stem += f"_gap{gap_bin[0] * 1000:g}-{gap_bin[1] * 1000:g}mm"
    if args.max_steps is not None:
        stem += f"_steps{env_cfg.max_episode_steps}"
    if args.checkpoint != "checkpoint":
        stem += f"_{args.checkpoint.replace('checkpoint_', '')}"
    if args.tag:
        stem += f"_{args.tag}"
    return os.path.join(run_dir, stem + ".json")


def run_episode(env, agent, task: dict) -> dict:
    """One deterministic episode; the per-step position (m) and orientation (deg) errors."""
    obs, _ = env.reset(options={"q_start": np.asarray(task["q_start"], float),
                                "goal_p": np.asarray(task["goal_p"], float),
                                "goal_R": np.asarray(task["goal_R"], float)})
    done = False
    while not done:
        prev = {"next_observation": obs[None].astype(np.float32)}
        a = agent.sample_actions(0, prev_transition=prev, training=False)[0]
        obs, _, term, trunc, _ = env.step(np.asarray(a, dtype=np.float32))
        done = term or trunc
    return {"trial_id": task["trial_id"],
            "per_step": {"d_m": [float(v) for v in env._ep["dists"]],
                         "theta_deg": [float(np.degrees(v)) for v in env._ep["angs"]]}}


def main(io=None) -> None:
    p = build_parser()
    args = p.parse_args()
    gap_bin = [float(v) for v in args.gap_bin.split(",")] if args.gap_bin else None
    if gap_bin is not None and not 0 <= gap_bin[0] < gap_bin[1]:
        p.error("--gap-bin needs 0 <= LO < HI")
    run_dir = os.path.abspath(args.run)
    if not os.path.isdir(os.path.join(run_dir, args.checkpoint)):
        p.error(f"no checkpoint dir {os.path.join(run_dir, args.checkpoint)}")
    with open(os.path.join(run_dir, "config.json")) as f:
        saved = json.load(f)
    kinematics.set_urdf_path(args.urdf)

    import torch

    from .envs.pose_env import OpenArmPoseEnv
    from .envs.reward import PoseRewardConfig
    from .train import build_agent

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    env_cfg = make_env_cfg(saved, args)
    rew_cfg = PoseRewardConfig(**saved["reward"])
    tasks = load_tasks(args.task_file, gap_bin)
    plan = [tasks[k % len(tasks)] for k in range(args.n_episodes)]
    if len({t["trial_id"] for t in plan}) != len(plan):
        p.error(f"--n-episodes {args.n_episodes} repeats tasks: the bin holds {len(tasks)}")
    path = output_path(args, run_dir, env_cfg, gap_bin)
    print(f"[eval] {run_dir} {args.checkpoint}: {len(plan)} tasks, {env_cfg.max_episode_steps} steps, "
          f"collision guard {env_cfg.collision_guard}")

    if io is None:
        import rclpy

        from .robot.ros_io import check_single_control_stack
        check_single_control_stack(raise_on_fail=True)
        rclpy.init()
    env = OpenArmPoseEnv(env_cfg, rew_cfg, curr_cfg=None, io=io)
    agent, _ = build_agent(saved, env, device, training=False)
    agent.load(os.path.join(run_dir, args.checkpoint))

    settings = {"checkpoint": args.checkpoint, "gap_bin_m": gap_bin, "max_steps": env_cfg.max_episode_steps,
                "collision_guard": env_cfg.collision_guard}
    rows, skipped = [], []
    if os.path.exists(path) and not args.overwrite:
        with open(path) as f:
            prev = json.load(f)
        if prev["settings"] != settings:
            env.close()
            raise SystemExit(f"[eval] {path} was produced with settings {prev['settings']}, not {settings}; "
                             f"pass --overwrite or --tag")
        rows, skipped = prev["episodes"], prev["skipped"]
    done_ids = {r["trial_id"] for r in rows} | set(skipped)
    remaining = [(k, t) for k, t in enumerate(plan) if t["trial_id"] not in done_ids]
    if rows or skipped:
        print(f"[eval] resuming {path}: {len(rows)} done, {len(skipped)} skipped, {len(remaining)} remaining")
    plan_order = {t["trial_id"]: k for k, t in enumerate(plan)}

    def save() -> None:
        """Write everything completed so far, atomically."""
        rows.sort(key=lambda r: plan_order[r["trial_id"]])
        with open(path + ".tmp", "w") as f:
            json.dump({"task_file": os.path.basename(args.task_file), "settings": settings,
                       "episodes": rows, "skipped": skipped}, f, indent=2)
        os.replace(path + ".tmp", path)

    current = None
    try:
        for k, t in remaining:
            current = t["trial_id"]
            s = run_episode(env, agent, t)
            rows.append(s)
            current = None
            print(f"  [{s['trial_id']}] task {k + 1:>3}/{len(plan)}  d_fin {s['per_step']['d_m'][-1] * 1000:6.2f}mm "
                  f"th_fin {s['per_step']['theta_deg'][-1]:5.2f}deg")
            save()
    except KeyboardInterrupt:
        if current is not None:
            skipped.append(current)
            print(f"\n[eval] interrupted during {current}: recorded as skipped; the same command continues")
    finally:
        save()
        print(f"  wrote {path}")
        env.close()
        if io is None:
            import rclpy
            rclpy.shutdown()


if __name__ == "__main__":
    main()
