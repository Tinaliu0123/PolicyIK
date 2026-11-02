"""Collect scripted large-rotation demonstrations on the real arm through the training environment."""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict

import numpy as np
import yaml

from .rl.demos import KEYS, complete_episode_mask, env_record, interface_diff, meta_path, norm
from .utils import kinematics

SETTLE_KI = 0.5
ARRIVE_TOL_RAD = 2e-3


def main(io=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task-file", required=True, help="demonstration task file written by select_tasks")
    ap.add_argument("--out", default=None, help="output .npz (default: <tasks dir>/demos.npz)")
    ap.add_argument("--start", type=int, default=None,
                    help="first task index; default = resume after the last task already in --out")
    ap.add_argument("--n", type=int, default=None, help="how many tasks from the start (default: all remaining)")
    ap.add_argument("--overwrite", action="store_true", help="ignore an existing --out (the default appends to it)")
    ap.add_argument("--urdf", default=None, help="OpenArm v1 URDF (default: from the openarm_description package)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; move nothing")
    a = ap.parse_args()
    kinematics.set_urdf_path(a.urdf)

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    from .envs.pose_env import EnvConfig
    from .envs.reward import PoseRewardConfig
    env_d = {**cfg["env"], **cfg["collect"]["env"]}
    rew_d, gamma = dict(cfg["reward"]), cfg["agent"]["gamma"]
    rew_d["pbrs_gamma"] = gamma
    with open(a.task_file) as f:
        all_tasks = json.load(f)["tasks"]
    tasks_file = os.path.basename(a.task_file)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.task_file)), "demos.npz")
    meta_file = meta_path(out)
    env_cfg = EnvConfig(**env_d)
    rew_cfg = PoseRewardConfig(**rew_d)
    rew_cfg.dt = 1.0 / env_cfg.control_hz
    T = env_cfg.max_episode_steps

    prior, done_through, n_dropped_prior = None, 0, 0
    if os.path.exists(out) and not a.overwrite:
        prior = {k: v for k, v in np.load(out).items()}
        missing = [k for k in KEYS if k not in prior]
        if missing:
            raise SystemExit(f"{out} lacks {missing}: not a demonstration file. Use another --out, or --overwrite.")
        if not os.path.exists(meta_file):
            raise SystemExit(f"{out} has no {os.path.basename(meta_file)}, so its action law is unknown. "
                             f"Use --overwrite or another --out.")
        with open(meta_file) as f:
            prior_meta = json.load(f)
        if prior_meta["tasks_file"] != tasks_file:
            raise SystemExit(f"{out} was collected from {prior_meta['tasks_file']}, not {tasks_file}. "
                             f"Use another --out, or --overwrite.")
        hard, soft = interface_diff(prior_meta["env"], asdict(env_cfg), prior_meta["reward"], rew_cfg.to_dict())
        if hard or soft:
            raise SystemExit(f"{out} was collected under a DIFFERENT interface/reward than this run:\n  "
                             + "\n  ".join(hard + soft) + "\n  Use --overwrite or another --out.")
        keep = complete_episode_mask(prior["ep_t"], prior["terminated"], prior["truncated"])
        n_dropped_prior = int((~keep).sum())
        if n_dropped_prior:
            dropped_tasks = sorted(set(prior["ep_id"][~keep].tolist()))
            prior = {k: v[keep] for k, v in prior.items()}
            print(f"[collect] {out} held {n_dropped_prior} rows of an INCOMPLETE episode "
                  f"(task {dropped_tasks}) -- dropped; that task is redone")
        done_through = int(prior["ep_id"].max()) + 1 if prior["ep_id"].size else 0
    start = a.start if a.start is not None else done_through
    n_all = len(all_tasks)
    tasks = all_tasks[start: start + (a.n if a.n is not None else n_all)]

    print("=" * 84)
    print("DEMO COLLECTION  (scripted large-rotation moves through the training env)")
    if prior is not None:
        print(f"  RESUMING       {out} already holds {len(np.unique(prior['ep_id']))} complete episodes "
              f"/ {int(prior['ep_id'].size)} transitions (tasks 0..{done_through - 1}); appending"
              + (f"  [{n_dropped_prior} rows of a cut-off episode dropped]" if n_dropped_prior else ""))
    if not tasks:
        print(f"  nothing to do: tasks {start}.. are beyond the {n_all} in {tasks_file}")
        print("=" * 84)
        return
    rots = np.array([t["rot_deg"] for t in tasks]); gaps = np.array([t["gap_m"] for t in tasks])
    est_s = len(tasks) * (T / 8.2 + env_cfg.reset_ramp_s + env_cfg.reset_settle_s + 1.0)
    print(f"  tasks          {len(tasks)} of {n_all} in {tasks_file} [{start}:{start + len(tasks)}]")
    print(f"  rotation       p10 {np.percentile(rots, 10):.0f} / p50 {np.percentile(rots, 50):.0f} "
          f"/ p90 {np.percentile(rots, 90):.0f} deg     gap p50 {1000 * np.median(gaps):.0f} mm")
    print(f"  scripted law   approach on q_ref, then settle with ki={SETTLE_KI:g} on the measured joint error, "
          f"to the {T}-step horizon ({T / env_cfg.control_hz:.0f} s)")
    print(f"  action law     gain_schedule {env_cfg.gain_schedule}, gain floor {env_cfg.gain_min:g}, "
          f"obs noise q +-{env_cfg.obs_noise_q_rad:g}")
    print(f"  guard          collision_guard={env_cfg.collision_guard}  self-pair gap>={env_cfg.self_pair_min_gap} "
          f"from link {env_cfg.link_bounds_from}   health guard={env_cfg.health_guard}")
    print(f"  action caps    {env_cfg.action_scale_rad:g} rad/step x gain, |dq| <= {env_cfg.max_joint_delta_rad:g}, "
          f"{env_cfg.max_joint_velocity_rad_s:g} rad/s, band {env_cfg.ref_band_rad:g}")
    print(f"  output         {out}   (interrupt-safe: a cut-off episode is dropped and redone on resume)")
    print(f"  est. arm time  ~{est_s / 60:.0f} min")
    print("=" * 84)
    if a.dry_run:
        print("--dry-run: nothing was built and the arm did not move.")
        return

    if io is None:
        import rclpy

        from .robot.ros_io import check_single_control_stack
        check_single_control_stack(raise_on_fail=True)
        rclpy.init()
    from .envs.pose_env import OpenArmPoseEnv
    env_cfg.validate()
    rew_cfg.validate(agent_gamma=gamma)
    env = OpenArmPoseEnv(env_cfg, rew_cfg, curr_cfg=None, io=io)
    scale = env_cfg.action_scale_rad
    tau_p, tau_r = rew_cfg.tau_p, rew_cfg.tau_r

    O, A, R, TE, TR, NO, EP, ET = [], [], [], [], [], [], [], []
    episodes = []
    committed = 0
    t0 = time.time()
    aborted = None
    try:
        for k, t in enumerate(tasks):
            task_idx = start + k
            q_goal = np.asarray(t["q_goal"], float)
            obs, _ = env.reset(options={"q_start": np.asarray(t["q_start"], float),
                                        "goal_p": np.asarray(t["goal_p"], float),
                                        "goal_R": np.asarray(t["goal_R"], float)})
            arrived_ref = None; first_strict = None; n_strict = 0
            blocks0 = env._n_collision_blocks
            for step in range(T):
                q_meas, _ = env.io.get_state()
                p, Rm = env.pose(q_meas)
                d, th, _ = env.errors(p, Rm)
                g = env._gain(d, th)
                if arrived_ref is None and np.abs(q_goal - env._q_ref).max() < ARRIVE_TOL_RAD:
                    arrived_ref = step
                if arrived_ref is None or SETTLE_KI <= 0:
                    act = (q_goal - env._q_ref) / max(g * scale, 1e-9)
                else:
                    act = SETTLE_KI * (q_goal - q_meas) / max(g * scale, 1e-9)
                act = np.clip(act, -1.0, 1.0).astype(np.float32)

                next_obs, reward, term, trunc, info = env.step(act)
                O.append(obs.astype(np.float32)); A.append(act); R.append(np.float32(reward))
                TE.append(np.float32(term)); TR.append(np.float32(trunc))
                NO.append(next_obs.astype(np.float32)); EP.append(task_idx); ET.append(step)
                obs = next_obs
                if d <= tau_p and th <= tau_r:
                    n_strict += 1
                    if first_strict is None:
                        first_strict = step
                if term or trunc:
                    break

            s = info["episode"]
            ep = {"ref_arrived_at": arrived_ref, "first_strict_at": first_strict,
                  "strict_dwell": n_strict / (step + 1),
                  "final_dist_mm": 1000 * s["final_dist"], "final_ang_deg": s["final_ang_deg"],
                  "guard_blocks": env._n_collision_blocks - blocks0,
                  "truncated_by_guard": bool(s.get("collision_blocked_out", False)),
                  "strict_settled": bool(s.get("strict_settled", False))}
            episodes.append(ep)
            committed = len(O)
            print(f"  [{task_idx:>3}] {t['trial_id']:<20} rot {t['rot_deg']:>4.0f}d gap {1000*t['gap_m']:>4.0f}mm | "
                  f"ref@{'%3d' % arrived_ref if arrived_ref is not None else '  -'}  "
                  f"strict@{'%3d' % first_strict if first_strict is not None else '  -'}  "
                  f"dwell {100*ep['strict_dwell']:>3.0f}%  final {ep['final_dist_mm']:>5.2f}mm/{ep['final_ang_deg']:>5.2f}d  "
                  f"guard {ep['guard_blocks']:>2}{'  TRUNC' if ep['truncated_by_guard'] else ''}")
    except KeyboardInterrupt:
        aborted = "KeyboardInterrupt"
        print("\n[collect] interrupted -- saving the completed episodes")
    except Exception as e:
        aborted = f"{type(e).__name__}: {e}"
        print(f"\n[collect] ABORTED: {aborted}\n[collect] saving the completed episodes")
        if not isinstance(e, RuntimeError):
            traceback.print_exc()
    finally:
        try:
            env.close()
        finally:
            if io is None:
                import rclpy
                rclpy.shutdown()

    dropped = len(O) - committed
    O, A, R, TE, TR, NO, EP, ET = (x[:committed] for x in (O, A, R, TE, TR, NO, EP, ET))
    if dropped:
        print(f"[collect] dropped {dropped} transitions of the interrupted episode (task {start + len(episodes)}); "
              f"the next run redoes it")
    if not O:
        print("[collect] nothing new recorded"); return

    new = {"observation": np.stack(O), "action": np.stack(A), "reward": np.array(R, np.float32),
           "terminated": np.array(TE, np.float32), "truncated": np.array(TR, np.float32),
           "next_observation": np.stack(NO), "ep_id": np.array(EP, np.int64), "ep_t": np.array(ET, np.int64)}
    if prior is not None:
        new = {k: np.concatenate([prior[k], new[k]]) for k in KEYS}
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **new)
    os.replace(tmp, out)
    with open(meta_file, "w") as f:
        json.dump({"tasks_file": tasks_file, "env": env_record(asdict(env_cfg)), "reward": norm(rew_cfg.to_dict())},
                  f, indent=1)

    e = episodes
    ok = [x for x in e if not x["truncated_by_guard"]]
    n_on_disk = int((new["ep_t"] == 0).sum())
    print("\n" + "=" * 84)
    print(f"this batch: {committed} transitions over {len(e)} episodes in {(time.time()-t0)/60:.0f} min"
          + (f"   [ABORTED: {aborted}]" if aborted else ""))
    print(f"on disk:    {int(new['ep_id'].size)} transitions over {n_on_disk} episodes "
          f"(tasks 0..{int(new['ep_id'].max())} of {n_all})")
    if ok:
        arr = [x['ref_arrived_at'] for x in ok if x['ref_arrived_at'] is not None]
        print(f"  reference arrived      {len(arr)}/{len(ok)}   median step {np.median(arr) if arr else float('nan'):.0f}")
        print(f"  reached STRICT         {sum(1 for x in ok if x['first_strict_at'] is not None)}/{len(ok)}   "
              f"strict dwell median {100*np.median([x['strict_dwell'] for x in ok]):.0f}%   "
              f"strict_settled {sum(1 for x in ok if x['strict_settled'])}/{len(ok)}")
        print(f"  final error            median {np.median([x['final_dist_mm'] for x in ok]):.2f} mm / "
              f"{np.median([x['final_ang_deg'] for x in ok]):.2f} deg")
    print(f"  guard truncations      {len(e) - len(ok)}/{len(e)}")
    print(f"  wrote {out}\n        {meta_file}")
    if start + len(episodes) < n_all:
        print(f"  {n_all - start - len(episodes)} tasks remain -- run the same command again to continue")
    print("  a demo that never reaches STRICT teaches the approach but not the endgame -- "
          "check 'reached STRICT' before training on this file")


if __name__ == "__main__":
    main()
