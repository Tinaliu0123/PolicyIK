"""Compute the single-attempt and deployment-protocol metrics from the evaluation dumps."""
from __future__ import annotations

import argparse
import os

import numpy as np

SP, SO = 0.001, np.deg2rad(0.2)      # strict thresholds (m, rad)
CP, CO = 0.01, np.deg2rad(2.0)       # coarse thresholds
GATE = 0.01                          # error-corrected goal only if the terminal-window position error <= 1 cm
TAIL = 30                            # terminal window (policy steps)
POLICY_HZ = 30.0
N_TRANSIENT = 8
SINGLE, CORRECTED, TRANSIENT = "single.npz", "corrected.npz", "transient{}.npz"


def _qang(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Rotation angle of qa * conj(qb) (w, x, y, z), as Isaac Lab's quat_error_magnitude."""
    qa = qa / np.linalg.norm(qa, axis=-1, keepdims=True)
    qb = qb / np.linalg.norm(qb, axis=-1, keepdims=True)
    aw, av = qa[..., :1], qa[..., 1:]
    bw, bv = qb[..., :1], -qb[..., 1:]
    w = aw * bw - (av * bv).sum(-1, keepdims=True)
    v = aw * bv + bw * av + np.cross(av, bv)
    return 2.0 * np.arctan2(np.linalg.norm(v, axis=-1), np.abs(w[..., 0]))


def _tail_means(z) -> tuple[np.ndarray, np.ndarray]:
    return z["pos_t"][:, -TAIL:].mean(1), z["ori_t"][:, -TAIL:].mean(1)


def single_attempt(z) -> dict:
    p, o = _tail_means(z)
    strict = (p < SP) & (o < SO)
    inst = (z["pos_t"] < SP) & (z["ori_t"] < SO)
    entered, contained = inst.any(1), inst[:, -TAIL:].all(1)
    first = np.argmax(inst, 1)
    ee_p, ee_q = z["ee_pos_t"].astype(np.float64), z["ee_quat_t"].astype(np.float64)
    drift_p = np.linalg.norm(ee_p[:, -1] - ee_p[:, -TAIL - 1], axis=1)
    drift_o = _qang(ee_q[:, -1], ee_q[:, -TAIL - 1])
    return dict(strict=100 * strict.mean(), coarse=100 * ((p < CP) & (o < CO)).mean(),
                entry=100 * entered.mean(), containment=100 * contained.mean(),
                uncontained=100 * (entered & ~contained).sum() / entered.sum(),
                entry_time_s=np.median((first[entered] + 1) / POLICY_HZ),
                drift_um=1e6 * np.median(drift_p[strict]), drift_deg=np.degrees(np.median(drift_o[strict])),
                pos_mean_mm=1e3 * p.mean(), pos_median_mm=1e3 * np.median(p),
                ori_mean_deg=np.degrees(o.mean()), ori_median_deg=np.degrees(np.median(o)))


def corrected_vs_original(body, corr) -> tuple[np.ndarray, np.ndarray]:
    """Terminal-window errors of the corrected attempt, measured against the original goal."""
    g_pos = body["goal_pos_w"].astype(np.float64)
    g_quat = body["goal_quat_w"].astype(np.float64)
    ee = corr["ee_pos_t"][:, -TAIL:, :].astype(np.float64)
    eq = corr["ee_quat_t"][:, -TAIL:, :].astype(np.float64)
    p = np.linalg.norm(ee - g_pos[:, None, :], axis=2).mean(1)
    o = _qang(eq, g_quat[:, None, :]).mean(1)
    return p, o


def deployment(body, corr, transients) -> dict:
    pb, ob = _tail_means(body)
    pc, oc = corrected_vs_original(body, corr)
    pos_err = np.stack([pb, pc] + [_tail_means(w)[0] for w in transients])      # (10, n)
    ori_err = np.stack([ob, oc] + [_tail_means(w)[1] for w in transients])
    strict = (pos_err < SP) & (ori_err < SO)
    coarse = (pos_err < CP) & (ori_err < CO)
    n = pos_err.shape[1]
    executed = np.zeros_like(strict)
    covered = np.zeros(n, bool)
    executed[0] = True
    covered |= strict[0]

    def _row(label: str, k: int):
        return (label, 100 * covered.mean(), 100 * (executed[:k + 1] & coarse[:k + 1]).any(0).mean(),
                executed[:k + 1].sum(0).mean())

    steps = [_row("single attempt", 0)]
    call = (~covered) & (pb <= GATE)
    executed[1] = call
    covered |= call & strict[1]
    steps.append(_row("+ error-corrected goal", 1))
    for k in range(2, pos_err.shape[0]):
        executed[k] = ~covered
        covered |= executed[k] & strict[k]
        steps.append(_row(f"+ transient goal {k - 2}", k))
    first_ok = np.where((executed & strict).any(0), np.argmax(executed & strict, 0), -1)
    last_exec = pos_err.shape[0] - 1 - np.argmax(executed[::-1], 0)
    d = np.where(first_ok >= 0, first_ok, last_exec)
    idx = np.arange(n)
    pd, od = pos_err[d, idx], ori_err[d, idx]
    return dict(steps=steps, strict=100 * covered.mean(), coarse_delivered=100 * coarse[d, idx].mean(),
                pos_mean_mm=1e3 * pd.mean(), pos_median_mm=1e3 * np.median(pd),
                ori_mean_deg=np.degrees(od.mean()), ori_median_deg=np.degrees(np.median(od)),
                EN=executed.sum(0).mean(), corr_calls=int(call.sum()), corr_rescued=int((call & strict[1]).sum()))


def ms(v, digits: int = 2) -> str:
    v = np.asarray(v, float)
    if len(v) == 1:
        return f"{v[0]:7.{digits}f} (n=1)"
    return f"{v.mean():7.{digits}f} +- {v.std(ddof=1):4.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed_dir", action="append", required=True, help="dumps of one training seed")
    ap.add_argument("--single_only", action="store_true", help="single-attempt metrics only")
    a = ap.parse_args()
    singles, deploys = [], []
    for sd in a.seed_dir:
        body = np.load(os.path.join(sd, SINGLE))
        singles.append(single_attempt(body))
        if not a.single_only:
            corr = np.load(os.path.join(sd, CORRECTED))
            transients = [np.load(os.path.join(sd, TRANSIENT.format(k))) for k in range(N_TRANSIENT)]
            deploys.append(deployment(body, corr, transients))

    print(f"{len(body['pos_t'])} tasks, {len(a.seed_dir)} seed(s); mean +- SD over seeds")
    print("\nSingle attempt")
    for k, lab, dg in [("strict", "strict (%)", 2), ("coarse", "coarse (%)", 2), ("entry", "entry (%)", 2),
                       ("containment", "containment (%)", 2), ("uncontained", "uncontained after entry (%)", 2),
                       ("entry_time_s", "entry time (s)", 2), ("drift_um", "final-second drift (um)", 2),
                       ("drift_deg", "final-second drift (deg)", 4),
                       ("pos_mean_mm", "position mean (mm)", 2), ("pos_median_mm", "position median (mm)", 2),
                       ("ori_mean_deg", "orientation mean (deg)", 2), ("ori_median_deg", "orientation median (deg)", 2)]:
        print(f"  {lab:28s} {ms([s[k] for s in singles], dg)}")
    if a.single_only:
        return
    print("\nDeployment protocol (cumulative strict and coarse success over executed attempts; E[N])")
    for i, (lab, _, _, _) in enumerate(deploys[0]["steps"]):
        st = [d["steps"][i][1] for d in deploys]
        co = [d["steps"][i][2] for d in deploys]
        en = [d["steps"][i][3] for d in deploys]
        print(f"  {lab:24s} strict {ms(st)}   coarse {ms(co)}   E[N] {np.mean(en):.3f}")
    print("\nDelivered attempt (first strict success, otherwise the last executed attempt)")
    for k, lab in [("strict", "strict (%)"), ("coarse_delivered", "coarse (%)"),
                   ("pos_mean_mm", "position mean (mm)"), ("pos_median_mm", "position median (mm)"),
                   ("ori_mean_deg", "orientation mean (deg)"), ("ori_median_deg", "orientation median (deg)"),
                   ("EN", "E[N]")]:
        print(f"  {lab:28s} {ms([d[k] for d in deploys])}")
    print(f"  error-corrected goal: attempts {[d['corr_calls'] for d in deploys]}, "
          f"successes {[d['corr_rescued'] for d in deploys]} per seed")


if __name__ == "__main__":
    main()
