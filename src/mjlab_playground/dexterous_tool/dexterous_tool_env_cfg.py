"""Base factory for the dexterous tool manipulation task."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_playground.dexterous_tool import mdp as dex_mdp
from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommandCfg


def make_dexterous_tool_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base dexterous tool manipulation task configuration."""

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "palm_pose": ObservationTermCfg(
      func=dex_mdp.palm_pose,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "fingertip_pos_rel_palm": ObservationTermCfg(
      func=dex_mdp.fingertip_pos_rel_palm,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "object_orientation": ObservationTermCfg(
      func=dex_mdp.object_orientation,
      params={"object_name": "tool"},
    ),
    "keypoints_rel_palm": ObservationTermCfg(
      func=dex_mdp.keypoints_rel_palm,
      params={
        "command_name": "tool_goal",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "keypoints_rel_goal": ObservationTermCfg(
      func=dex_mdp.keypoint_errors,
      params={"command_name": "tool_goal"},
    ),
    "object_scales": ObservationTermCfg(
      func=dex_mdp.object_scales,
      params={
        "asset_cfg": SceneEntityCfg("tool", geom_names=()),  # Set per-robot.
      },
    ),
  }

  critic_terms = {
    **actor_terms,
    "palm_velocity": ObservationTermCfg(
      func=dex_mdp.palm_velocity,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "object_velocity": ObservationTermCfg(
      func=dex_mdp.object_velocity,
      params={"object_name": "tool"},
    ),
    "closest_keypoint_max_dist": ObservationTermCfg(
      func=dex_mdp.closest_keypoint_max_dist,
      params={"command_name": "tool_goal"},
    ),
    "closest_fingertip_dist": ObservationTermCfg(
      func=dex_mdp.closest_fingertip_distances,
      params={"command_name": "tool_goal"},
    ),
    "lifted_object": ObservationTermCfg(
      func=dex_mdp.lifted_object,
      params={"command_name": "tool_goal"},
    ),
    "progress": ObservationTermCfg(func=dex_mdp.progress),
    "successes": ObservationTermCfg(
      func=dex_mdp.successes,
      params={"command_name": "tool_goal"},
    ),
    "reward": ObservationTermCfg(func=dex_mdp.reward),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "arm_joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(),
      scale=0.0125,
    ),
    "hand_joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(),
      scale=0.025,
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
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "tool_geometry": EventTermCfg(
      func=dex_mdp.randomize_tool_geometry,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("tool"),
        "handle_scale_range": (0.7, 1.25),
        "head_scale_range": (0.7, 1.25),
      },
    ),
  }

  rewards = {
    "fingertip_approach": RewardTermCfg(
      func=dex_mdp.fingertip_approach,
      weight=1.0,
      params={
        "command_name": "tool_goal",
        "object_name": "tool",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "lift_object": RewardTermCfg(
      func=dex_mdp.lift_object,
      weight=1.0,
      params={
        "command_name": "tool_goal",
        "object_name": "tool",
        "lift_bonus": 300.0,
      },
    ),
    "keypoint_goal": RewardTermCfg(
      func=dex_mdp.keypoint_goal,
      weight=1.0,
      params={"command_name": "tool_goal"},
    ),
    "goal_success_bonus": RewardTermCfg(
      func=dex_mdp.goal_success_bonus,
      weight=1.0,
      params={"command_name": "tool_goal", "bonus": 1000.0},
    ),
    "object_lin_vel_penalty": RewardTermCfg(
      func=dex_mdp.object_lin_velocity_penalty,
      weight=0.0,
      params={"object_name": "tool"},
    ),
    "object_ang_vel_penalty": RewardTermCfg(
      func=dex_mdp.object_ang_velocity_penalty,
      weight=0.0,
      params={"object_name": "tool"},
    ),
    "arm_velocity_penalty": RewardTermCfg(
      func=dex_mdp.joint_velocity_penalty,
      weight=-0.03,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "hand_velocity_penalty": RewardTermCfg(
      func=dex_mdp.joint_velocity_penalty,
      weight=-0.003,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "arm_dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "hand_dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "object_fallen": TerminationTermCfg(
      func=dex_mdp.object_fallen,
      params={"object_name": "tool", "min_z": 0.32},
    ),
    "object_dropped": TerminationTermCfg(
      func=dex_mdp.object_dropped_after_lift,
      params={"command_name": "tool_goal", "object_name": "tool"},
    ),
    "hand_too_far": TerminationTermCfg(
      func=dex_mdp.hand_too_far,
      params={
        "object_name": "tool",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "max_distance": 0.45,
      },
    ),
  }

  mj_cfg = MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
    impratio=10,
    cone="elliptic",
  )
  decimation = 4
  episode_length_s = 10.0

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane", textures=(), materials=()),
      num_envs=1,
      env_spacing=1.5,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=1.5,
      elevation=-15.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(mujoco=mj_cfg),
    decimation=decimation,
    episode_length_s=episode_length_s,
  )
