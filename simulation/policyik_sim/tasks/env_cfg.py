"""PolicyIK training and evaluation tasks built on the OpenArm reach task of openarm_isaac_lab."""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.unimanual.reach.config.joint_pos_env_cfg import (
    OpenArmReachEnvCfg,
    OpenArmReachEnvCfg_PLAY,
)

from . import mdp
from .actions import AccumulatedJointReferenceActionCfg
from .sampling import PoolPoseCommandCfg, reset_joints_from_pool

TRAIN_START_POOL = "data/train_start_pool.npz"   # training start configurations
TRAIN_GOAL_POOL = "data/train_goal_pool.npz"     # training goals (deployment workspace)
SOURCE_SET = "data/source_set.npz"               # evaluation environment (start and goal are set by the evaluator)


def _configure(cfg, start_pool: str, goal_pool: str, train: bool) -> None:
    ee = cfg.rewards.end_effector_position_tracking.params["asset_cfg"]   # end-effector body (openarm_hand)
    cmd = "ee_pose"

    # start and goal sampling
    base = cfg.commands.ee_pose
    cfg.commands.ee_pose = PoolPoseCommandCfg(
        asset_name=base.asset_name,
        body_name=base.body_name,
        resampling_time_range=base.resampling_time_range,
        make_quat_unique=base.make_quat_unique,
        ranges=base.ranges,
        debug_vis=base.debug_vis,
        pool_npz=goal_pool,
    )
    cfg.events.reset_robot_joints = EventTerm(
        func=reset_joints_from_pool, mode="reset", params={"pool_npz": start_pool, "asset_cfg": SceneEntityCfg("robot")}
    )

    if train:
        cfg.commands.ee_pose.use_sampled_orientation = True
        # motion costs: 5x the OpenArm reach defaults, and 5x the targets of their weight schedule
        cfg.rewards.action_rate.weight = -5.0e-4
        cfg.rewards.joint_vel.weight = -5.0e-4
        cfg.curriculum.action_rate.params["weight"] = -0.025
        cfg.curriculum.joint_vel.params["weight"] = -0.005
        # tracking reward: keypoint kernels replace the orientation term (wide kernel here, sharp kernel below)
        cfg.rewards.end_effector_keypoint_tracking = RewTerm(
            func=mdp.keypoint_error_tanh, weight=1.0,
            params={"asset_cfg": ee, "command_name": cmd, "std": 0.20, "kp_size": 0.45, "zero_sum": True},
        )
        cfg.rewards.end_effector_orientation_tracking = None

    # observation: [q - q_default, qdot, goal position, keypoint offsets, coarse and fine pose channels,
    #               error history h_t, command-state feedback d_t, previous action]   (54-D)
    pol = cfg.observations.policy
    last_action = pol.actions
    del pol.pose_command
    del pol.actions
    pol.goal_position = ObsTerm(func=mdp.goal_position_obs, params={"command_name": cmd})
    pol.keypoint_offsets = ObsTerm(
        func=mdp.keypoint_offset_obs,
        params={"command_name": cmd, "asset_cfg": SceneEntityCfg("robot", body_names=["openarm_hand"]), "kp_size": 0.30},
    )
    body = SceneEntityCfg("robot", body_names=["openarm_hand"])
    pol.position_coarse = ObsTerm(
        func=mdp.ee_position_error_obs, params={"command_name": cmd, "asset_cfg": body, "d0": 0.01}
    )
    pol.orientation_coarse = ObsTerm(
        func=mdp.ee_orientation_error_obs, params={"command_name": cmd, "asset_cfg": body, "d0_rad": math.radians(20.0)}
    )
    pol.position_fine = ObsTerm(
        func=mdp.ee_position_error_obs, params={"command_name": cmd, "asset_cfg": body, "d0": 0.0005}
    )
    pol.orientation_fine = ObsTerm(
        func=mdp.ee_orientation_error_obs, params={"command_name": cmd, "asset_cfg": body, "d0_rad": math.radians(0.25)}
    )

    # progress reward (backslide, potential-based shaping, improvement), dwell reward and joint-margin cost
    cfg.rewards.backslide = RewTerm(
        func=mdp.backslide_penalty, weight=0.2, params={"command_name": cmd, "asset_cfg": ee, "cap": 3.0}
    )
    pol.error_history = ObsTerm(func=mdp.error_history_obs, params={"command_name": cmd, "asset_cfg": ee})
    cfg.rewards.potential_shaping = RewTerm(
        func=mdp.potential_shaping, weight=1.0, params={"command_name": cmd, "asset_cfg": ee}
    )
    cfg.rewards.dwell = RewTerm(func=mdp.dwell_reward, weight=1.0, params={"command_name": cmd, "asset_cfg": ee})
    cfg.rewards.joint_margin = RewTerm(
        func=mdp.joint_margin_cost, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_joint[1-7]"], preserve_order=True),
                "command_name": cmd, "rho_gate": 10.0, "body_cfg": ee},
    )
    cfg.rewards.improvement = RewTerm(
        func=mdp.improvement_reward, weight=26.831, params={"command_name": cmd, "asset_cfg": ee}
    )

    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 32

    # action: accumulated joint reference with error-dependent scaling
    base_action = cfg.actions.arm_action
    cfg.actions.arm_action = AccumulatedJointReferenceActionCfg(
        asset_name=base_action.asset_name, joint_names=base_action.joint_names, scale=0.0625,
        g_min=0.1, d_lo=0.0005, d_hi=0.00931, ori_lo_deg=0.415, ori_hi_deg=2.328,
    )
    pol.command_state = ObsTerm(func=mdp.command_state_obs, params={"action_name": "arm_action"})
    pol.actions = last_action

    if train:
        cfg.rewards.end_effector_keypoint_tracking_sharp = RewTerm(
            func=mdp.keypoint_error_tanh, weight=1.0,
            params={"asset_cfg": ee, "command_name": cmd, "std": 0.05, "kp_size": 0.45, "zero_sum": True},
        )


@configclass
class PolicyIKEnvCfg(OpenArmReachEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure(self, TRAIN_START_POOL, TRAIN_GOAL_POOL, train=True)


@configclass
class PolicyIKEnvCfg_PLAY(OpenArmReachEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure(self, SOURCE_SET, SOURCE_SET, train=False)

