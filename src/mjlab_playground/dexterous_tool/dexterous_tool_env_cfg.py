"""Base factory for the dexterous tool manipulation task."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_playground.dexterous_tool import mdp as dex_mdp
from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommandCfg


def make_dexterous_tool_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base dexterous tool manipulation task configuration."""

  actor_terms = {
    # Arm proprioception.
    "arm_joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=())},  # Set per-robot.
      noise=Unoise(n_min=-0.01, n_max=0.01),  # Override per-robot.
    ),
    "arm_joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=())},  # Set per-robot.
      noise=Unoise(n_min=-0.5, n_max=0.5),  # Override per-robot.
    ),
    # Hand proprioception.
    "hand_joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=())},  # Set per-robot.
      noise=Unoise(n_min=-0.01, n_max=0.01),  # Override per-robot.
    ),
    "hand_joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=())},  # Set per-robot.
      noise=Unoise(n_min=-0.5, n_max=0.5),  # Override per-robot.
    ),
    "fingertip_pos_in_palm": ObservationTermCfg(
      func=dex_mdp.fingertip_pos_in_palm,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Other.
    "actions": ObservationTermCfg(func=mdp.last_action),
    # Exteroception: tool and goal expressed in the palm's local frame, plus a
    # redundant pre-computed SE(3) error in the tool frame. All terms anchor
    # to a frame whose pose responds directly to the policy's actions; world
    # frame does not appear anywhere.
    "tool_pose_in_palm": ObservationTermCfg(
      func=dex_mdp.tool_pose_in_palm,
      params={"asset_cfg": SceneEntityCfg("robot")},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "goal_pose_in_palm": ObservationTermCfg(
      func=dex_mdp.goal_pose_in_palm,
      params={
        "command_name": "tool_goal",
        "asset_cfg": SceneEntityCfg("robot"),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "goal_pose_in_tool": ObservationTermCfg(
      func=dex_mdp.goal_pose_in_tool,
      params={"command_name": "tool_goal"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # "object_scales": ObservationTermCfg(
    #   func=dex_mdp.object_scales,
    #   params={
    #     "asset_cfg": SceneEntityCfg("tool", geom_names=()),  # Set per-robot.
    #   },
    # ),
  }

  critic_terms = {**actor_terms}

  observations = {
    # TODO: Re-enable noise once we can solve the task in the noise-free setting.
    "actor": ObservationGroupCfg(
      actor_terms, enable_corruption=False, nan_policy="warn"
    ),
    "critic": ObservationGroupCfg(
      critic_terms, enable_corruption=False, nan_policy="warn"
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "arm_joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(),  # Set per-robot.
    ),
    "hand_joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(),  # Set per-robot.
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    "tool_goal": ToolGoalPoseCommandCfg(
      entity_name="tool",
      resampling_time_range=(1e9, 1e9),  # Don't resample by time; use success.
      debug_vis=True,
    ),
  }

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}},
    ),
    "reset_table": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("table"),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # "tool_geometry": EventTermCfg(
    #   func=dex_mdp.randomize_tool_geometry,
    #   mode="reset",
    #   params={
    #     "asset_cfg": SceneEntityCfg("tool"),
    #     "handle_scale_range": (0.7, 1.25),
    #     "head_scale_range": (0.7, 1.25),
    #   },
    # ),
  }

  rewards = {
    # Reach phase: long-range signal toward grasp_center.
    "approach": RewardTermCfg(
      func=dex_mdp.approach_reward,
      weight=1.0,
      params={
        "command_name": "tool_goal",
        "std": 0.4,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "approach_precise": RewardTermCfg(
      func=dex_mdp.approach_reward,
      weight=1.0,
      params={
        "command_name": "tool_goal",
        "std": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Lift-off phase: smooth gradient from "tool on table" to "tool in air".
    # target_height must sit *above* the command's lift threshold so saturation
    # happens after `lifted_object` flips to True (≈ 0.41 reset z + 0.15 lift
    # threshold = 0.56 for the default kuka_sharpa setup). Set per-robot if
    # the reset z or lift threshold differ.
    "tool_above_table": RewardTermCfg(
      func=dex_mdp.tool_above_table_reward,
      weight=1.0,
      params={"target_height": 0.6, "std": 0.1},
    ),
    # Pose tracking: 3-tier coarse → mid → precise on full SE(3) error.
    "pose_position_coarse": RewardTermCfg(
      func=dex_mdp.pose_position_reward,
      weight=1.0,
      params={"command_name": "tool_goal", "std": 0.3},
    ),
    "pose_position": RewardTermCfg(
      func=dex_mdp.pose_position_reward,
      weight=1.0,
      params={"command_name": "tool_goal", "std": 0.1},
    ),
    "pose_position_precise": RewardTermCfg(
      func=dex_mdp.pose_position_reward,
      weight=1.0,
      params={"command_name": "tool_goal", "std": 0.03},
    ),
    "pose_orientation_coarse": RewardTermCfg(
      func=dex_mdp.pose_orientation_reward,
      weight=1.0,
      params={"command_name": "tool_goal", "ori_std": math.radians(60.0)},
    ),
    "pose_orientation": RewardTermCfg(
      func=dex_mdp.pose_orientation_reward,
      weight=1.0,
      params={"command_name": "tool_goal", "ori_std": math.radians(20.0)},
    ),
    "pose_orientation_precise": RewardTermCfg(
      func=dex_mdp.pose_orientation_reward,
      weight=1.0,
      params={"command_name": "tool_goal", "ori_std": math.radians(5.0)},
    ),
    # Regularization rewards.
    "arm_posture": RewardTermCfg(
      func=mdp.posture,
      weight=0.01,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
        "std": {},  # Set per-robot.
      },
    ),
    "arm_action_rate": RewardTermCfg(
      func=dex_mdp.action_rate_l2,
      weight=-0.001,
      params={"action_name": "arm_joint_pos"},
    ),
    "hand_action_rate": RewardTermCfg(
      func=dex_mdp.action_rate_l2,
      weight=-0.0001,
      params={"action_name": "hand_joint_pos"},
    ),
    "arm_joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=())},  # Set per-robot.
    ),
    "hand_joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=())},  # Set per-robot.
    ),
    "arm_joint_vel_hinge": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_hinge_penalty,
      weight=-0.001,
      params={
        "max_vel": 0.5,  # Override per-robot.
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "hand_joint_vel_hinge": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_hinge_penalty,
      weight=-0.001,
      params={
        "max_vel": 0.5,  # Override per-robot.
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "hand_table_collision": RewardTermCfg(
      func=dex_mdp.contact_force_penalty,
      weight=-0.01,
      params={"sensor_name": "hand_table_collision"},
    ),
  }

  # Collision sensors.
  arm_collision_cfg = ContactSensorCfg(
    name="arm_collision",
    primary=ContactMatch(
      mode="body",
      pattern=(),  # Set per-robot (arm body names, e.g., link3-7).
      entity="robot",
    ),
    secondary=None,  # Any contact.
    secondary_policy="any",
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,  # Match decimation.
  )
  hand_table_collision_cfg = ContactSensorCfg(
    name="hand_table_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern="",  # Set per-robot (hand subtree root, e.g., "left_hand_C_MC").
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="table", entity="table"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,  # Match decimation.
  )

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "object_fallen": TerminationTermCfg(
      func=dex_mdp.object_fallen,
      params={"object_name": "tool", "min_z": 0.32},
    ),
    # Stronger drop guard: terminates if the tool was lifted at any point in the
    # episode and is now back at or below its reset height. Forces the policy
    # to commit to a stable grasp instead of "lift briefly then drop".
    # "object_dropped_after_lift": TerminationTermCfg(
    #   func=dex_mdp.object_dropped_after_lift,
    #   params={"command_name": "tool_goal", "object_name": "tool"},
    # ),
    "object_velocity_exceeded": TerminationTermCfg(
      func=dex_mdp.object_velocity_exceeded,
      params={"object_name": "tool", "max_lin_vel": 5.0, "max_ang_vel": 20.0},
    ),
    "hand_too_far": TerminationTermCfg(
      func=dex_mdp.hand_too_far,
      params={
        "command_name": "tool_goal",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "max_distance": 0.45,
      },
    ),
    "arm_collision": TerminationTermCfg(
      func=manipulation_mdp.illegal_contact,
      params={"sensor_name": "arm_collision", "force_threshold": 1.0},
    ),
  }

  metrics: dict[str, MetricsTermCfg] = {
    "object_lin_speed": MetricsTermCfg(
      func=dex_mdp.object_lin_speed,
      params={"object_name": "tool"},
    ),
    "object_ang_speed": MetricsTermCfg(
      func=dex_mdp.object_ang_speed,
      params={"object_name": "tool"},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=None,
      num_envs=1,
      env_spacing=1.5,
      sensors=(arm_collision_cfg, hand_table_collision_cfg),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=1.5,
      elevation=-15.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        enableflags=("multiccd",),
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
