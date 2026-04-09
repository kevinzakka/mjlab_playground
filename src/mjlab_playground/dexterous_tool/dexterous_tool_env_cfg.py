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
    # Reach phase: multi-scale Gaussian on fingertip → grasp_center distance.
    # Wide std (0.4 m) gives the long-range pre-grasp signal; narrow std
    # (0.1 m) provides the close-range alignment signal.
    "approach": RewardTermCfg(
      func=dex_mdp.approach_reward,
      weight=1.0,
      params={
        "command_name": "tool_goal",
        "stds": (0.4, 0.1),
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
    # Pose tracking: a single multi-scale Gaussian per category. Each tuple
    # contains (wide, narrow) stds — wide gives shaping at long range, narrow
    # gives precision near the goal. The reward function averages them so
    # range stays in [0, 1] regardless of how many scales are stacked.
    "pose_position": RewardTermCfg(
      func=dex_mdp.pose_position_reward,
      weight=1.0,
      params={"command_name": "tool_goal", "stds": (0.3, 0.03)},
    ),
    # Orientation gate uses a tool↔table contact sensor: the reward is exactly
    # zero whenever any tool geom touches the table, and (1 + ori_gauss)/2
    # otherwise. This is the principled "the tool is held in the air" signal —
    # no height thresholds, no sticky flags, no magic numbers tied to tool
    # geometry. See `pose_orientation_reward` docstring for the rationale.
    "pose_orientation": RewardTermCfg(
      func=dex_mdp.pose_orientation_reward,
      weight=1.0,
      params={
        "command_name": "tool_goal",
        "ori_stds": (math.radians(60.0), math.radians(5.0)),
        "table_contact_sensor_name": "tool_table_collision",
      },
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
  # Tool↔table contact sensor: the *principled* "is the tool airborne" signal
  # used by the orientation reward gate. The tool is considered held in the air
  # iff zero contacts are reported between any tool geom and the table body.
  # This is geometry-independent and impossible to exploit (you cannot touch
  # and not-touch the table simultaneously).
  tool_table_collision_cfg = ContactSensorCfg(
    name="tool_table_collision",
    primary=ContactMatch(mode="body", pattern="tool", entity="tool"),
    secondary=ContactMatch(mode="body", pattern="table", entity="table"),
    fields=("found",),
    reduce="none",
    num_slots=1,
    history_length=1,
  )

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "object_fallen": TerminationTermCfg(
      func=dex_mdp.object_fallen,
      params={"object_name": "tool", "min_z": 0.32},
    ),
    # Drop termination kept around but disabled — the contact-based
    # orientation gate already provides a sharp per-step penalty for
    # dropping, so this is redundant.
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
      sensors=(
        arm_collision_cfg,
        hand_table_collision_cfg,
        tool_table_collision_cfg,
      ),
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
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
