"""Build the training pools and the evaluation tasks from the source set of joint configurations."""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.spatial import cKDTree

N_EVAL = 4120
GOAL_SEED, START_SEED = 260919, 260920
# deployment workspace: cylinder about the base axis minus a keep-out region around the base links (m)
R_MAX, Z_MIN, Z_MAX = 0.49, 0.15, 0.50
R_KEEPOUT, Z_KEEPOUT = 0.128, 0.222
# a training goal duplicates an evaluation goal if it lies within both tolerances
DUP_POS, DUP_ORI = 0.001, np.deg2rad(0.2)


def in_workspace(ee_pos: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(ee_pos[:, :2], axis=1)
    z = ee_pos[:, 2]
    inside = (r <= R_MAX) & (z >= Z_MIN) & (z <= Z_MAX)
    return inside & ~((r < R_KEEPOUT) & (z < Z_KEEPOUT))


def _quat_angle(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(qa * qb, axis=-1)) / (np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def split(src: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the evaluation goal and start rows and the boolean masks of the two training pools."""
    valid = ~src["floor_contact"] & ~src["self_collision"]
    inw = in_workspace(src["ee_pos"].astype(np.float64))
    used = np.zeros(len(valid), dtype=bool)
    goals = np.random.default_rng(GOAL_SEED).choice(np.where(valid & inw)[0], size=N_EVAL, replace=False)
    used[goals] = True
    starts = np.random.default_rng(START_SEED).choice(np.where(valid & ~used)[0], size=N_EVAL, replace=False)
    used[starts] = True
    train = valid & ~used
    return goals, starts, train, train & inw


def check(src: dict, goals: np.ndarray, starts: np.ndarray, train: np.ndarray) -> None:
    """Training and evaluation share no configuration and no evaluation goal duplicates a training goal."""
    rows = np.concatenate([goals, starts])
    q_eval = {r.tobytes() for r in src["q"][rows]}
    if len(q_eval) != len(rows) or any(r.tobytes() in q_eval for r in src["q"][train]):
        raise AssertionError("a configuration is shared between training and evaluation")
    train_goals = np.where(train & in_workspace(src["ee_pos"].astype(np.float64)))[0]
    tree = cKDTree(src["ee_pos"][train_goals].astype(np.float64))
    for i, near in zip(goals, tree.query_ball_point(src["ee_pos"][goals].astype(np.float64), DUP_POS)):
        if near and (_quat_angle(src["ee_quat"][train_goals[near]], src["ee_quat"][i]) < DUP_ORI).any():
            raise AssertionError(f"evaluation goal of row {i} duplicates a training goal")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/source_set.npz")
    ap.add_argument("--out_dir", default="data")
    a = ap.parse_args()
    with np.load(a.source) as z:
        src = {k: z[k] for k in z.files}
    goals, starts, train, train_goal = split(src)
    check(src, goals, starts, train)

    def rows(mask: np.ndarray) -> dict:
        return {k: src[k][mask] for k in ("q", "ee_pos", "ee_quat")}

    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(a.out_dir, "train_start_pool.npz"), **rows(train))
    np.savez_compressed(os.path.join(a.out_dir, "train_goal_pool.npz"), **rows(train_goal))
    np.savez_compressed(os.path.join(a.out_dir, "eval_tasks.npz"), q_start=src["q"][starts],
                        goal_pos=src["ee_pos"][goals], goal_quat=src["ee_quat"][goals])
    print(f"training starts {int(train.sum())}, training goals {int(train_goal.sum())}, evaluation tasks {N_EVAL}")


if __name__ == "__main__":
    main()
