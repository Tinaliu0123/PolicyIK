"""Build the error-corrected goals of the deployment protocol from a single-attempt evaluation."""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

SP, SO = 0.001, np.deg2rad(0.2)


def build(single_p: str, tasks_p: str, out_p: str) -> None:
    z, b = np.load(single_p), np.load(tasks_p)
    sp, so = z["pos_t"][:, -30:].mean(1), z["ori_t"][:, -30:].mean(1)
    fail = np.maximum(sp / SP, so / SO) > 1.0

    gp = b["goal_pos"].astype(np.float64).copy()
    gq = b["goal_quat"].astype(np.float64).copy()
    e = z["ee_pos_t"][:, -1, :].astype(np.float64) - z["goal_pos_w"].astype(np.float64)
    q_ee = z["ee_quat_t"][:, -1, :].astype(np.float64)
    Rg = Rotation.from_quat(gq[:, [1, 2, 3, 0]])     # wxyz -> xyzw
    Re = Rotation.from_quat(q_ee[:, [1, 2, 3, 0]])
    v = (Rg.inv() * Re).as_rotvec()                  # Log(R_goal^T R_ee)
    q_corr = (Rg * Rotation.from_rotvec(-v)).as_quat()[:, [3, 0, 1, 2]]   # xyzw -> wxyz
    gp[fail] -= e[fail]
    gq[fail] = q_corr[fail]

    out = {k: b[k] for k in b.files}
    out.update(goal_pos=gp.astype(np.float32), goal_quat=gq.astype(np.float32), corrected=fail)
    np.savez(out_p, **out)
    print(f"{int(fail.sum())} of {len(fail)} goals corrected -> {out_p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single", required=True, help="single-attempt evaluation (.npz written by evaluate.py)")
    ap.add_argument("--task_file", required=True, help="tasks that were evaluated")
    ap.add_argument("--output", required=True, help="tasks with the corrected goals")
    a = ap.parse_args()
    build(a.single, a.task_file, a.output)


if __name__ == "__main__":
    main()
