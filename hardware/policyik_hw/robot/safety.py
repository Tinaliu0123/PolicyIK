"""Joint limits of the OpenArm right arm with a 2 degree safety margin."""
import numpy as np

LIM_RAD = np.array([
    (-1.3963, 3.4907),
    (-0.1745, 3.3161),
    (-1.5708, 1.5708),
    (0.0000,  2.4435),
    (-1.5708, 1.5708),
    (-0.7854, 0.7854),
    (-1.5708, 1.5708),
], dtype=float)

MARGIN_RAD = np.deg2rad(2.0)

LIM_LO = LIM_RAD[:, 0] + MARGIN_RAD
LIM_HI = LIM_RAD[:, 1] - MARGIN_RAD


def joint_limits(lo_deg=(), hi_deg=()):
    """These limits intersected with a joint sector [lo_deg, hi_deg] (degrees; empty = no sector)."""
    lo, hi = LIM_LO.copy(), LIM_HI.copy()
    if len(lo_deg) == len(lo):
        lo = np.maximum(lo, np.deg2rad(np.asarray(lo_deg, float)))
    if len(hi_deg) == len(hi):
        hi = np.minimum(hi, np.deg2rad(np.asarray(hi_deg, float)))
    return lo, hi
