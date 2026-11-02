# Copyright 2025 Enactic, Inc.
# Copyright 2024 Stogl Robotics Consulting UG (haftungsbeschränkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changed: one arm on classic CAN with a required interface, the motor gains of
# config/control_gains.yaml, and the gravity feedforward node.

import os

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = "openarm_right_bringup"
GAIN_PARAMS = [f"{k}{i}" for i in range(1, 8) for k in ("kp", "kd")]


def _can_driver(name):
    """Kernel driver of a CAN interface ("gs_usb", ...), or None if unknown."""
    link = os.path.join("/sys/class/net", name, "device", "driver")
    try:
        return os.path.basename(os.path.realpath(link))
    except OSError:
        return None


def _can_is_up(name):
    """True if the interface is up, None if its flags cannot be read."""
    try:
        with open(os.path.join("/sys/class/net", name, "flags")) as fh:
            return bool(int(fh.read().strip(), 16) & 0x1)
    except (OSError, ValueError):
        return None


def _usb_can_interfaces():
    """CAN interfaces backed by the gs_usb USB adapter."""
    try:
        names = sorted(os.listdir("/sys/class/net"))
    except OSError:
        return []
    return [n for n in names if n.startswith("can") and _can_driver(n) == "gs_usb"]


def _validate_can_interface(name):
    """Refuse a missing, onboard or down CAN interface instead of driving the wrong one."""
    found = ", ".join(_usb_can_interfaces()) or "none (adapter unplugged?)"
    hint = (f"  USB CAN interfaces now: {found}\n"
            f"  Then: ros2 launch {PACKAGE} openarm_right.launch.py can_interface:=<iface>")
    if not name:
        raise RuntimeError(
            "can_interface is required. Interface numbers can change between boots, "
            "so name the interface of the arm explicitly.\n" + hint)
    if not os.path.exists(os.path.join("/sys/class/net", name)):
        raise RuntimeError(f"can_interface:={name} does not exist.\n" + hint)
    driver = _can_driver(name)
    if driver is not None and driver != "gs_usb":
        raise RuntimeError(
            f"can_interface:={name} is driven by '{driver}', not the USB CAN adapter (gs_usb). "
            "An onboard CAN controller without an arm comes up healthy and transmits into nothing.\n"
            + hint)
    if _can_is_up(name) is False:
        raise RuntimeError(
            f"can_interface:={name} is down. Bring it up first:\n"
            f"  sudo ip link set {name} type can bitrate 1000000 fd off\n"
            f"  sudo ip link set {name} up")


def _robot_description(can_interface):
    """URDF of the arm with the motor gains of config/control_gains.yaml."""
    xacro_path = os.path.join(get_package_share_directory("openarm_description"),
                              "assets", "robot", "openarm_v1.0", "urdf", "openarm_v10.urdf.xacro")
    doc = xacro.process_file(xacro_path, mappings={
        "arm_type": "v10",
        "bimanual": "false",
        "use_fake_hardware": "false",
        "ros2_control": "true",
        "can_interface": can_interface,
        # The hardware opens a CAN-FD socket unless this is false.
        "can_fd": "false",
        "arm_prefix": "",
    })
    with open(os.path.join(get_package_share_directory(PACKAGE), "config", "control_gains.yaml")) as f:
        gains = yaml.safe_load(f)
    params = [p for p in doc.getElementsByTagName("param") if p.getAttribute("name") in GAIN_PARAMS]
    if sorted(p.getAttribute("name") for p in params) != sorted(GAIN_PARAMS):
        raise RuntimeError("the openarm_description hardware block lacks one kp and one kd per joint")
    for p in params:
        name = p.getAttribute("name")
        p.firstChild.data = str(gains[f"joint{name[2:]}"][name[:2]])
    return doc.toprettyxml(indent="  ")


def _robot_nodes(context):
    can_interface = LaunchConfiguration("can_interface").perform(context)
    _validate_can_interface(can_interface)
    description = {"robot_description": _robot_description(can_interface)}
    controllers = os.path.join(get_package_share_directory(PACKAGE), "config", "controllers.yaml")
    return [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             name="robot_state_publisher", output="screen", parameters=[description]),
        Node(package="controller_manager", executable="ros2_control_node",
             output="both", parameters=[description, controllers]),
        TimerAction(period=5.0, actions=[
            Node(package=PACKAGE, executable="gravity_node.py", output="screen", parameters=[description])]),
    ]


def _spawner(period, controller):
    """Spawn a controller after the controller manager is up."""
    return TimerAction(period=period, actions=[
        Node(package="controller_manager", executable="spawner", arguments=[controller, "-c", "/controller_manager"])])


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "can_interface", default_value="",
            description="SocketCAN interface of the arm, e.g. can0 (required)"),
        OpaqueFunction(function=_robot_nodes),
        _spawner(2.0, "joint_state_broadcaster"),
        _spawner(3.0, "forward_position_controller"),
        _spawner(3.2, "forward_effort_controller"),
        _spawner(3.5, "gripper_controller"),
    ])
