"""ROS 2 I/O for the real arm: read /joint_states, write /forward_position_controller/commands."""

from __future__ import annotations

import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ..utils.kinematics import JOINTS

__all__ = ["ArmIO", "check_single_control_stack"]


def check_single_control_stack(raise_on_fail: bool = True) -> bool:
    """Refuse to start if two controller_managers are up."""
    try:
        # A ROS 2 daemon that has just started can list only part of the graph.
        out = subprocess.check_output(
            ["bash", "-lc", "ros2 node list --no-daemon --spin-time 3"],
            stderr=subprocess.DEVNULL, timeout=15).decode()
    except Exception as e:
        print(f"[preflight] could not list ROS nodes ({e}); skipping the "
              "dual-stack check -- verify manually with 'ros2 node list'")
        return True
    names = [n.strip() for n in out.splitlines() if n.strip()]
    n_cm = sum(1 for n in names if "controller_manager" in n)
    if n_cm > 1:
        msg = (f"{n_cm} controller_manager nodes are running. Two control stacks "
               "on one CAN bus fight each other and the arm jumps. Kill the stale "
               "launch and start exactly one.")
        if raise_on_fail:
            raise RuntimeError(msg)
        print(f"[preflight] WARNING: {msg}")
        return False
    if n_cm == 0:
        msg = ("no controller_manager found: start the arm with 'ros2 launch "
               "openarm_right_bringup openarm_right.launch.py can_interface:=<iface>'")
        if raise_on_fail:
            raise RuntimeError(msg)
        print(f"[preflight] WARNING: {msg}")
        return False
    has_grav = any("gravity" in n for n in names)
    print(f"[preflight] 1 controller_manager, gravity node "
          f"{'present' if has_grav else 'ABSENT -- the arm will droop'}")
    return True


class ArmIO(Node):
    """Reads /joint_states, publishes joint position commands."""

    def __init__(self, node_name: str = "openarm_reach_io"):
        super().__init__(node_name)
        self._pos: dict = {}
        self._vel: dict = {}
        self._last_stamp: float | None = None
        self._n_msgs = 0

        self.sub = self.create_subscription(JointState, "/joint_states", self._cb, 10)
        self.pub = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 10)
        from std_msgs.msg import String
        self._health: dict | None = None
        self._health_stamp: float | None = None
        self._disabled_strikes = [0] * 8
        self.sub_health = self.create_subscription(String, "/motor_health", self._cb_health, 10)


    def _cb(self, msg: JointState) -> None:
        for name, p, v in zip(msg.name, msg.position, msg.velocity):
            self._pos[name] = p
            self._vel[name] = v
        self._last_stamp = time.time()
        self._n_msgs += 1

    def _cb_health(self, msg) -> None:
        import json
        try:
            self._health = json.loads(msg.data); self._health_stamp = time.time()
        except Exception:
            pass

    def health_check(self, max_reply_age_s: float = 0.5, max_temp_c: float = 60.0,
                     max_data_age_s: float = 1.0) -> tuple[bool, str]:
        """(ok, reason)."""
        if self._health is None or self._health_stamp is None:
            return None, "no /motor_health data (motor_health.py not running?)"
        if time.time() - self._health_stamp > max_data_age_s:
            return None, f"/motor_health stale ({time.time() - self._health_stamp:.1f} s)"
        h = self._health
        for i in range(8):
            j = i + 1; e = int(h["err"][i]); name = h["err_name"][i]
            if h["age_s"][i] >= max_reply_age_s:
                return False, f"J{j} SILENT: no CAN reply for {h['age_s'][i]:.2f} s (last status {name})"
            if e not in (0, 1):
                return False, f"J{j} motor fault: {name} (ERR=0x{e:X})"
            if max(h["tmos"][i], h["trotor"][i]) >= max_temp_c:
                return False, f"J{j} temperature {max(h['tmos'][i], h['trotor'][i]):.0f} C >= {max_temp_c:g}"
            self._disabled_strikes[i] = self._disabled_strikes[i] + 1 if e == 0 else 0
            if self._disabled_strikes[i] >= 3:
                return False, f"J{j} repeatedly reports DISABLED (ignoring commands; relaunch re-enables)"
        return True, "ok"

    def has_state(self) -> bool:
        return all(j in self._pos for j in JOINTS)

    def wait_for_state(self, timeout: float = 8.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.has_state():
                return True
        return False

    def get_state(self):
        """(q, dq) in rad, rad/s."""
        q = np.array([self._pos[j] for j in JOINTS])
        dq = np.array([self._vel.get(j, 0.0) for j in JOINTS])
        return q, dq

    def state_age(self) -> float:
        if self._last_stamp is None:
            return float("inf")
        return time.time() - self._last_stamp

    def command_q(self, q_target: np.ndarray) -> None:
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q_target]
        self.pub.publish(msg)


    def spin_for(self, seconds: float, timeout_sec: float = 0.002) -> None:
        """Process callbacks for a wall-clock interval."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=timeout_sec)

    def drain(self, seconds: float = 0.02) -> int:
        """Process any backlogged /joint_states without blocking."""
        before = self._n_msgs
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            rclpy.spin_once(self, timeout_sec=0.0)
        return self._n_msgs - before
