#!/usr/bin/env python3
"""Gravity feedforward: publishes the gravity torques G(q) of the arm to the forward effort controller."""

import PyKDL as kdl
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from urdf_parser_py.urdf import URDF

ROOT_LINK = "openarm_link0"
TIP_LINK = "openarm_link7"
JOINTS = [f"openarm_joint{i}" for i in range(1, 8)]
# Gravity in the openarm_link0 frame of our mounting (m/s^2): +x back, +y right, +z down.
GRAVITY = (2.93, 1.19, 9.29)
RATE_HZ = 100.0


def _kdl_pose(pose):
    if pose is not None and pose.xyz and pose.rpy:
        return kdl.Frame(kdl.Rotation.RPY(*pose.rpy), kdl.Vector(*pose.xyz))
    return kdl.Frame()


def _kdl_inertia(i):
    origin = _kdl_pose(i.origin)
    ix = i.inertia
    return origin.M * kdl.RigidBodyInertia(
        i.mass, origin.p,
        kdl.RotationalInertia(ix.ixx, ix.iyy, ix.izz, ix.ixy, ix.ixz, ix.iyz))


def _kdl_joint(jnt):
    F = _kdl_pose(jnt.origin)
    if jnt.type in ("revolute", "continuous"):
        return kdl.Joint(jnt.name, F.p, F.M * kdl.Vector(*jnt.axis), kdl.Joint.RotAxis)
    if jnt.type == "prismatic":
        return kdl.Joint(jnt.name, F.p, F.M * kdl.Vector(*jnt.axis), kdl.Joint.TransAxis)
    return kdl.Joint(jnt.name, kdl.Joint.Fixed)


def _add_children(robot, root_name, tree):
    for (jname, cname) in robot.child_map.get(root_name, []):
        child = robot.link_map[cname]
        inert = _kdl_inertia(child.inertial) if child.inertial else kdl.RigidBodyInertia()
        jnt = robot.joint_map[jname]
        tree.addSegment(kdl.Segment(cname, _kdl_joint(jnt), _kdl_pose(jnt.origin), inert), root_name)
        _add_children(robot, cname, tree)


class GravityModel:
    """Gravity torques of the link0 -> link7 chain by recursive Newton-Euler."""

    def __init__(self, urdf_xml, gravity=GRAVITY):
        robot = URDF.from_xml_string(urdf_xml)
        tree = kdl.Tree(robot.get_root())
        _add_children(robot, robot.get_root(), tree)
        # ChainDynParam keeps a reference to the chain, so the chain must live as long as the model.
        self._chain = tree.getChain(ROOT_LINK, TIP_LINK)
        self.n = self._chain.getNrOfJoints()
        self._dyn = kdl.ChainDynParam(self._chain, kdl.Vector(*gravity))
        self._g = kdl.JntArray(self.n)

    def torques(self, q):
        qa = kdl.JntArray(self.n)
        for i in range(self.n):
            qa[i] = float(q[i])
        self._dyn.JntToGravity(qa, self._g)
        return [self._g[i] for i in range(self.n)]


class GravityFeedforward(Node):
    """Adds G(q) to the position PD of the motors through /forward_effort_controller/commands."""

    def __init__(self):
        super().__init__("gravity_ff")
        urdf_xml = self.declare_parameter("robot_description", "").value
        if not urdf_xml:
            raise RuntimeError("robot_description is empty: start this node with openarm_right.launch.py")
        self._model = GravityModel(urdf_xml)
        self._q = None
        self._pub = self.create_publisher(Float64MultiArray, "/forward_effort_controller/commands", 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        self.create_timer(1.0 / RATE_HZ, self._publish)
        self.get_logger().info("gravity feedforward running")

    def _on_joint_states(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(j in idx for j in JOINTS):
            self._q = [msg.position[idx[j]] for j in JOINTS]

    def _publish(self):
        if self._q is not None:
            self._pub.publish(Float64MultiArray(data=self._model.torques(self._q)))

    def release(self):
        """Zero the effort command so that the position PD alone holds the arm."""
        self._pub.publish(Float64MultiArray(data=[0.0] * len(JOINTS)))


def main():
    # Ctrl-C must leave the context valid so that release() can still publish.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = GravityFeedforward()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
