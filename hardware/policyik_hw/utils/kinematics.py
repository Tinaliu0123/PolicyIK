"""Forward kinematics of the OpenArm right arm from the openarm_description v1 URDF."""

from __future__ import annotations

import os
import subprocess

import numpy as np
import PyKDL as kdl
from urdf_parser_py.urdf import URDF

ROOT_LINK = "openarm_right_link0"
TIP_LINK = "openarm_right_link7"
JOINTS = [f"openarm_joint{i}" for i in range(1, 8)]
N_JOINTS = 7
URDF_PATH: str | None = None

__all__ = ["ArmKinematics", "JOINTS", "N_JOINTS", "ROOT_LINK", "TIP_LINK", "get_urdf_xml", "set_urdf_path"]


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
    if jnt.type == "fixed":
        return kdl.Joint(jnt.name, kdl.Joint.Fixed)
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
        seg = kdl.Segment(cname, _kdl_joint(jnt), _kdl_pose(jnt.origin), inert)
        tree.addSegment(seg, root_name)
        _add_children(robot, cname, tree)


def _tree_from_urdf(robot):
    root = robot.get_root()
    tree = kdl.Tree(root)
    _add_children(robot, root, tree)
    return tree


def set_urdf_path(path: str | None) -> None:
    """Use this URDF file instead of the one installed with openarm_description."""
    global URDF_PATH
    URDF_PATH = path


def get_urdf_xml() -> str:
    """The v1 URDF (its right arm is the modelled chain)."""
    path = URDF_PATH
    if path is None:
        desc = subprocess.check_output(
            ["bash", "-lc", "ros2 pkg prefix openarm_description"]).decode().strip()
        path = os.path.join(desc, "share", "openarm_description", "assets", "robot", "openarm_v1.0", "urdf",
                            "example", "v1.urdf")
    with open(path) as f:
        return f.read()


class ArmKinematics:
    """Forward kinematics of the right arm, link0 -> link7, with optional tool points along the link7 z axis."""

    def __init__(self, urdf_xml: str | None = None, tool_z_m: float = 0.0, collision_tool_z_m: float = 0.0):
        root, tip = ROOT_LINK, TIP_LINK
        xml = urdf_xml if urdf_xml is not None else get_urdf_xml()
        robot = URDF.from_xml_string(xml)
        self.tree = _tree_from_urdf(robot)
        self.chain = self.tree.getChain(root, tip)
        self.n = self.chain.getNrOfJoints()
        if self.n != N_JOINTS:
            raise RuntimeError(
                f"expected a {N_JOINTS}-DoF chain {root}->{tip}, got {self.n}")
        self._fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self.tool_z_m = float(tool_z_m)
        self.collision_tool_z_m = float(collision_tool_z_m)

    def _qa(self, q) -> kdl.JntArray:
        a = kdl.JntArray(self.n)
        for i in range(self.n):
            a[i] = float(q[i])
        return a

    def fk_pose(self, q):
        """(p (3,), R (3,3)) of the tool point in link0 (link7 itself when tool_z_m = 0)."""
        frame = kdl.Frame()
        self._fk.JntToCart(self._qa(q), frame)
        p = np.array([frame.p[0], frame.p[1], frame.p[2]])
        R = np.array([[frame.M[i, j] for j in range(3)] for i in range(3)])
        if self.tool_z_m:
            p = p + R @ np.array([0.0, 0.0, self.tool_z_m])
        return p, R

    def link_points(self, q):
        """Origins of every chain segment, then the tool and collision-tool points."""
        qa = self._qa(q); out = []
        for i in range(1, self.chain.getNrOfSegments() + 1):
            f = kdl.Frame(); self._fk.JntToCart(qa, f, i)
            out.append((f.p[0], f.p[1], f.p[2]))
        out = np.asarray(out)
        extra = {round(z, 6) for z in (self.tool_z_m, self.collision_tool_z_m) if z}
        if extra:
            frame = kdl.Frame(); self._fk.JntToCart(qa, frame)
            p7 = np.array([frame.p[0], frame.p[1], frame.p[2]])
            R7 = np.array([[frame.M[i, j] for j in range(3)] for i in range(3)])
            for z in sorted(extra):
                out = np.vstack([out, p7 + R7 @ np.array([0.0, 0.0, z])])
        return out
