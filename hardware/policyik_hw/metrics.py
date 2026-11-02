"""Real-time held-pose metrics of one checkpoint from its three gap-bin evaluations."""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

BINS = {(0.02, 0.05): "medium", (0.05, 0.10): "long", (0.10, 1.0): "extended"}
TP_MM, TR_DEG = 1.0, 0.2
REACH_HOLD_S = 0.5
BUDGETS_S = (15, 25)
HOLDS_S = (1, 2, 3, 5)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", required=True, help="run directory holding the eval_*.json files")
    p.add_argument("--checkpoint", default="checkpoint", help="checkpoint directory the evaluations ran")
    p.add_argument("--tag", default=None, help="the evaluator --tag, if one was given")
    p.add_argument("--task-file", required=True, help="the task file the evaluations ran on")
    p.add_argument("--eval-hz", type=float, default=8.13,
                   help="step rate used to convert the time budgets to steps (1 / the median training step)")
    p.add_argument("--out", default=None, help="default: <run>/<checkpoint>[_<tag>]_metrics.json")
    return p


def load_bins(run_dir: str, checkpoint: str, tag) -> list:
    """The one evaluation file of each gap bin for this checkpoint."""
    found = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "eval_*.json"))):
        if tag and not f.endswith(f"_{tag}.json"):
            continue
        with open(f) as fh:
            d = json.load(fh)
        s = d["settings"]
        b = BINS.get(tuple(s["gap_bin_m"] or ()))
        if s["checkpoint"] != checkpoint or b is None:
            continue
        if b in found:
            raise SystemExit(f"two evaluations of the {b} bin: {found[b][0]} and {f}; pass --tag")
        found[b] = (f, d)
    missing = [b for b in BINS.values() if b not in found]
    if missing:
        raise SystemExit(f"no evaluation of the {missing} bin(s) of {checkpoint} in {run_dir}")
    if any(d["settings"]["collision_guard"] for _, d in found.values()):
        raise SystemExit("an evaluation ran with the collision guard on")
    if len({d["settings"]["max_steps"] for _, d in found.values()}) != 1:
        raise SystemExit("the bins ran with different episode lengths")
    return [d for _, d in found.values()]


def metrics(rows: list, n_killed: int, budget_s: float, dt: float) -> dict:
    """Held and reached counts within the first `budget_s` seconds; interrupted tasks count as failures."""
    n = int(round(budget_s / dt))
    k1, kr = int(round(1 / dt)), int(round(REACH_HOLD_S / dt))
    held = {h: 0 for h in HOLDS_S}
    reached, mp, mr = 0, [], []
    for e in rows:
        dm = np.array(e["per_step"]["d_m"])[:n] * 1000
        th = np.array(e["per_step"]["theta_deg"])[:n]
        ok = (dm <= TP_MM) & (th <= TR_DEG)
        for h in held:
            held[h] += bool(ok[-int(round(h / dt)):].all())
        run = 0
        for o in ok:
            run = run + 1 if o else 0
            if run >= kr:
                reached += 1
                break
        mp.append(dm[-k1:].mean())
        mr.append(th[-k1:].mean())
    # the medians run over the completed tasks
    return {"held": held, "reached": reached, "median_pos_mm": float(np.median(mp)),
            "median_rot_deg": float(np.median(mr)), "n_tasks": len(rows) + n_killed, "killed": n_killed}


def main() -> None:
    a = build_parser().parse_args()
    run_dir = os.path.abspath(a.run)
    with open(a.task_file) as f:
        task_ids = {t["trial_id"] for t in json.load(f)["tasks"]}
    evals = load_bins(run_dir, a.checkpoint, a.tag)
    rows = [e for d in evals for e in d["episodes"]]
    skipped = [s for d in evals for s in d["skipped"]]
    ids = [e["trial_id"] for e in rows] + skipped
    if len(ids) != len(set(ids)) or set(ids) != task_ids:
        raise SystemExit(f"the evaluations cover {len(set(ids))} of the {len(task_ids)} evaluation tasks")

    dt = 1 / a.eval_hz
    res = {H: metrics(rows, len(skipped), H, dt) for H in BUDGETS_S}
    out = a.out or os.path.join(run_dir, f"{a.checkpoint}{'_' + a.tag if a.tag else ''}_metrics.json")
    with open(out, "w") as f:
        json.dump({"checkpoint": a.checkpoint, "eval_hz": a.eval_hz, "budgets_s": res}, f, indent=1)

    print(f"| metric | {os.path.basename(run_dir)} {a.checkpoint} |\n|---|---|")
    n_tasks = len(ids)
    for H in BUDGETS_S:
        for h in HOLDS_S:
            print(f"| held final {h} s, {H} s budget | {res[H]['held'][h]}/{n_tasks} |")
    for H in BUDGETS_S:
        print(f"| reached ({REACH_HOLD_S:g} s), {H} s budget | {res[H]['reached']}/{n_tasks} |")
    last = res[BUDGETS_S[-1]]
    print(f"| median pos err, final 1 s of {BUDGETS_S[-1]} s (mm) | {last['median_pos_mm']:.2f} |")
    print(f"| median rot err, final 1 s of {BUDGETS_S[-1]} s (deg) | {last['median_rot_deg']:.2f} |")
    print(f"interrupted tasks (counted as failures): {len(skipped)}\nwrote {out}")


if __name__ == "__main__":
    main()
