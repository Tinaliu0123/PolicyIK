"""Select a stratified, pre-validated list of large-rotation demonstration tasks from the workspace cloud."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from .utils import kinematics

TASK_KEYS = ("trial_id", "q_start", "q_goal", "goal_p", "goal_R", "gap_m", "rot_deg")
REJECT_KEYS = ("rot band", "gap", "dq", "goal too close", "quota full", "joint margin", "envelope", "left workspace",
               "too long")


def build_parser() -> argparse.ArgumentParser:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=os.path.join(here, "config.yaml"))
    p.add_argument("--n", type=int, default=150, help="how many demos to select")
    p.add_argument("--out", default=os.path.join(here, "data", "demo_tasks.json"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rot-bands", default="60,90,120,180", help="rotation band edges in degrees; N edges = N-1 bands")
    p.add_argument("--shells", type=int, default=3, help="radial shells of the goal about the workspace centre")
    p.add_argument("--min-gap-m", type=float, default=0.02, help="reject pairs closer than this")
    p.add_argument("--min-goal-sep-m", type=float, default=0.03,
                   help="reject a goal within this of an already-accepted one (spreads coverage)")
    p.add_argument("--limit-margin", type=float, default=0.0,
                   help="degrees every waypoint must keep from every joint stop")
    p.add_argument("--clearance", type=float, default=0.025, help="metres of envelope slack every waypoint must keep")
    p.add_argument("--max-steps", type=int, default=120, help="reject a task whose approach needs more steps")
    p.add_argument("--max-draws", type=int, default=5000000)
    p.add_argument("--min-dq-rad", type=float, default=3.0,
                   help="reject pairs whose joint-space distance |q_goal - q_start| (L2, rad) is below this")
    p.add_argument("--id-prefix", default="demo", help="trial id prefix: <prefix>-<index>-rot<deg>")
    p.add_argument("--urdf", default=None, help="OpenArm v1 URDF (default: from the openarm_description package)")
    p.add_argument("--dry-run", action="store_true")
    return p


def make_rollout(cfg, cl, lo, hi, a):
    """Offline replay of the env's action law at gain 1 with perfect tracking, from q0 toward q_goal."""
    kin = cl.kin
    scale, dmax = float(cfg.action_scale_rad), float(cfg.max_joint_delta_rad)
    vcap = float(cfg.max_joint_velocity_rad_s) / float(cfg.control_hz)
    lim_margin = np.radians(a.limit_margin)

    def rollout(q0, q_goal):
        q_ref = q0.copy()
        for n in range(1, a.max_steps + 1):
            act = np.clip((q_goal - q_ref) / scale, -1.0, 1.0)
            delta = np.clip(np.clip(act * scale, -dmax, dmax), -vcap, vcap)
            q_ref = np.clip(q_ref + delta, lo, hi)
            if np.min(np.minimum(q_ref - lo, hi - q_ref)) < lim_margin:
                return n, "joint margin"
            if cl.config_margin(q_ref) < a.clearance:
                return n, "envelope"
            if not cl.in_workspace(kin.fk_pose(q_ref)[0]):
                return n, "left workspace"
            if np.abs(q_goal - q_ref).max() < 1e-4:
                return n, None
        return a.max_steps, "too long"

    return rollout


def draw_tasks(cl, bands, shell_edges, quota, rollout, a):
    from .utils.pose_math import angle_between

    rng = np.random.default_rng(a.seed)
    N = len(cl.q)
    centre = np.asarray(cl.cfg.centre_m, float)
    picked = {k: [] for k in quota}
    accepted_goals = []
    rej = {k: 0 for k in REJECT_KEYS}
    draws = 0
    while sum(len(v) for v in picked.values()) < a.n and draws < a.max_draws:
        draws += 1
        i, j = rng.integers(1, N, 2)
        if i == j:
            continue
        rot = float(np.degrees(angle_between(cl.R[i], cl.R[j])))
        bi = next((k for k, (blo, bhi) in enumerate(bands) if blo <= rot < bhi), None)
        if bi is None:
            rej["rot band"] += 1
            continue
        gap = float(np.linalg.norm(cl.p[j] - cl.p[i]))
        if gap < a.min_gap_m:
            rej["gap"] += 1
            continue
        dq = float(np.linalg.norm(cl.q[j] - cl.q[i]))
        if dq < a.min_dq_rad:
            rej["dq"] += 1
            continue
        si = int(np.clip(np.searchsorted(shell_edges, np.linalg.norm(cl.p[j] - centre)) - 1, 0, a.shells - 1))
        if len(picked[(bi, si)]) >= quota[(bi, si)]:
            rej["quota full"] += 1
            continue
        if accepted_goals and np.min(np.linalg.norm(np.array(accepted_goals) - cl.p[j], axis=1)) < a.min_goal_sep_m:
            rej["goal too close"] += 1
            continue
        n, why = rollout(cl.q[i], cl.q[j])
        if why is not None:
            rej[why] += 1
            continue
        picked[(bi, si)].append({
            "trial_id": f"{a.id_prefix}-{len(accepted_goals):03d}-rot{rot:.0f}",
            "q_start": cl.q[i].tolist(), "q_goal": cl.q[j].tolist(), "goal_p": cl.p[j].tolist(),
            "goal_R": cl.R[j].tolist(), "gap_m": gap, "rot_deg": rot, "dq_rad": dq, "approach_steps": int(n)})
        accepted_goals.append(cl.p[j])
    return picked, rej, draws


def report(tasks, picked, quota, rej, draws, bands, cl, a) -> None:
    print(f"\nselected {len(tasks)} of {a.n} after {draws} draws")
    if len(tasks) < a.n:
        print("  NOT FULL -- raise --max-draws, loosen --min-goal-sep-m, or widen the bands")
    print("\nrejections:")
    for k, v in sorted(rej.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"   {k:<16} {v:>7}")
    print(f"\n{'stratum':<22}{'picked':>8}{'quota':>7}{'rot p50':>9}{'gap p50':>9}{'steps p50':>11}")
    for (bi, si), g in picked.items():
        head = f"   band {bands[bi][0]:.0f}-{bands[bi][1]:.0f}, shell {si:<7}"
        if not g:
            print(f"{head}{0:>8}{quota[(bi, si)]:>7}{'-':>9}{'-':>9}{'-':>11}")
            continue
        r = np.median([t["rot_deg"] for t in g])
        gg = np.median([t["gap_m"] for t in g])
        st = np.median([t["approach_steps"] for t in g])
        print(f"{head}{len(g):>8}{quota[(bi, si)]:>7}{r:>8.0f}d{1000 * gg:>8.0f}mm{st:>11.0f}")
    if not tasks:
        return
    rr = np.array([t["rot_deg"] for t in tasks])
    gg = np.array([t["gap_m"] for t in tasks])
    ss = np.array([t["approach_steps"] for t in tasks])
    dd = np.array([t["dq_rad"] for t in tasks])
    vox = len(set(map(tuple, np.floor(np.array([t["goal_p"] for t in tasks]) / 0.05).astype(int))))
    allv = len(set(map(tuple, np.floor(cl.p / 0.05).astype(int))))
    print(f"\noverall: rotation p10 {np.percentile(rr, 10):.0f} / p50 {np.percentile(rr, 50):.0f} "
          f"/ p90 {np.percentile(rr, 90):.0f} deg")
    print(f"         gap      p10 {1000 * np.percentile(gg, 10):.0f} / p50 {1000 * np.percentile(gg, 50):.0f} "
          f"/ p90 {1000 * np.percentile(gg, 90):.0f} mm")
    print(f"         approach p50 {np.percentile(ss, 50):.0f} / p90 {np.percentile(ss, 90):.0f} steps")
    print(f"         |dq|     min {dd.min():.2f} / p50 {np.percentile(dd, 50):.2f} / p90 {np.percentile(dd, 90):.2f} rad")
    print(f"         goals occupy {vox} of the workspace's {allv} 5-cm voxels ({100 * vox / allv:.0f}%)")


def main() -> None:
    a = build_parser().parse_args()
    kinematics.set_urdf_path(a.urdf)
    from .envs.pose_env import EnvConfig, cloud_config
    from .envs.workspace import PoseCloud
    from .robot.safety import joint_limits

    with open(a.config) as f:
        cfg = EnvConfig(**yaml.safe_load(f)["env"])
    home = np.deg2rad(np.array(cfg.home_pose_deg))
    lo, hi = joint_limits(cfg.joint_lo_deg, cfg.joint_hi_deg)
    cl = PoseCloud(home, cloud_config(cfg, min_entries=10), lim_lo=lo, lim_hi=hi)

    edges = [float(x) for x in a.rot_bands.split(",")]
    bands = list(zip(edges[:-1], edges[1:]))
    shell_edges = np.linspace(0.0, cl.cfg.r_goal, a.shells + 1)
    strata = [(bi, si) for bi in range(len(bands)) for si in range(a.shells)]
    quota = {k: a.n // len(strata) for k in strata}
    for k in strata[: a.n - sum(quota.values())]:
        quota[k] += 1

    print("=" * 84)
    print("DEMO TASK SELECTION  (large-rotation demonstrations)")
    print(f"  cloud             {len(cl.q)} entries   guard gap>={cfg.self_pair_min_gap} from link {cfg.link_bounds_from}")
    print(f"  rotation bands    {', '.join(f'{b[0]:.0f}-{b[1]:.0f}' for b in bands)} deg")
    print(f"  radial shells     {', '.join(f'{shell_edges[i]:.3f}-{shell_edges[i + 1]:.3f}' for i in range(a.shells))}"
          " m from the centre")
    print(f"  quota             {a.n} demos over {len(strata)} strata ({min(quota.values())}-{max(quota.values())} each)")
    print(f"  waypoint safety   >= {a.limit_margin:g} deg from every joint stop, "
          f">= {1000 * a.clearance:.0f} mm envelope slack, EE inside the workspace")
    print(f"  spread            goals >= {1000 * a.min_goal_sep_m:.0f} mm apart, "
          f"start->goal gap >= {1000 * a.min_gap_m:.0f} mm, approach <= {a.max_steps} steps")
    if a.min_dq_rad > 0:
        print(f"  joint distance    |q_goal - q_start| >= {a.min_dq_rad:g} rad")
    print("=" * 84)
    if a.dry_run:
        print("--dry-run: nothing was selected or written.")
        return

    picked, rej, draws = draw_tasks(cl, bands, shell_edges, quota, make_rollout(cfg, cl, lo, hi, a), a)
    tasks = [t for k in strata for t in picked[k]]
    report(tasks, picked, quota, rej, draws, bands, cl, a)

    out = {"tasks": [{k: t[k] for k in TASK_KEYS} for t in tasks]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}  ({len(tasks)} tasks)")


if __name__ == "__main__":
    main()
