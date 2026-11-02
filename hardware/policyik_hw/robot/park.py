"""Ramp all 7 arm joints to 0 (hanging straight down) over the forward_position_controller, then exit."""
import argparse, time, sys
import numpy as np

from ..utils.kinematics import JOINTS
from .safety import LIM_RAD

LIM = np.degrees(LIM_RAD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=15.0, help="ramp cap, deg/s of the largest joint")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--dry-run", action="store_true", help="read the state and print the plan, publish nothing")
    a = ap.parse_args()
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    class N(Node):
        def __init__(self):
            super().__init__("park")
            self.cur = {}; self.stamps = 0
            self.create_subscription(JointState, "/joint_states", self._cb, 10)
            self.pub = self.create_publisher(Float64MultiArray, "/forward_position_controller/commands", 10)
        def _cb(self, msg):
            for nme, p in zip(msg.name, msg.position):
                self.cur[nme] = p
            self.stamps += 1
        def q(self):
            return np.array([self.cur[j] for j in JOINTS])

    rclpy.init(); n = N(); t0 = time.time()
    while time.time() - t0 < 5 and not all(j in n.cur for j in JOINTS):
        rclpy.spin_once(n, timeout_sec=0.1)
    if not all(j in n.cur for j in JOINTS):
        print("[park] no /joint_states: is the launch running?"); rclpy.shutdown(); sys.exit(3)
    q0 = n.q(); deg = np.degrees(q0)
    print("[park] arm now: " + " ".join(f"J{i+1}={v:+.1f}" for i, v in enumerate(deg)))
    for i, (lo, hi) in enumerate(LIM):
        if not (lo - 3 <= deg[i] <= hi + 3):
            print(f"[park] REFUSED: J{i+1} reads {deg[i]:+.1f} deg, outside the URDF range [{lo:.0f}, {hi:.0f}]: unhealthy reading"); rclpy.shutdown(); sys.exit(2)
    s0 = n.stamps; t1 = time.time()
    while time.time() - t1 < 1.0:
        rclpy.spin_once(n, timeout_sec=0.05)
    if n.stamps == s0:
        print("[park] REFUSED: /joint_states stopped"); rclpy.shutdown(); sys.exit(2)
    dist = float(np.abs(deg).max()); steps = max(1, int(max(1.0, dist / a.speed) * a.rate))
    print(f"[park] ramp to all-zero in {steps/a.rate:.1f} s at <= {a.speed:g} deg/s" + ("  (dry run: not published)" if a.dry_run else ""))
    if a.dry_run:
        rclpy.shutdown(); return
    for s in range(steps + 1):
        f = s / steps; m = Float64MultiArray(); m.data = [float(v) for v in q0 * (1 - f)]
        n.pub.publish(m); rclpy.spin_once(n, timeout_sec=0.0); time.sleep(1.0 / a.rate)
    t2 = time.time()
    while time.time() - t2 < 2.0:
        rclpy.spin_once(n, timeout_sec=0.05)
    print("[park] arm now: " + " ".join(f"J{i+1}={v:+.1f}" for i, v in enumerate(np.degrees(n.q()))))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
