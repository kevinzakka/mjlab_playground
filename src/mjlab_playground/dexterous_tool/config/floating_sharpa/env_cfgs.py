"""Floating SHARPA hand configuration for dexterous tool task."""

import math

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg

from mjlab_playground.asset_zoo.robots.floating_sharpa import (
  get_floating_sharpa_robot_cfg,
)
from mjlab_playground.asset_zoo.robots.floating_sharpa.floating_sharpa_constants import (
  FINGERTIP_SITE_NAMES,
  HAND_JOINT_NAMES,
  SHARPA_ACTION_SCALE,
)
from mjlab_playground.dexterous_tool.dexterous_tool_env_cfg import (
  make_dexterous_tool_env_cfg,
)
from mjlab_playground.dexterous_tool.mdp.actions import RelativeMocapActionCfg
from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommandCfg
from mjlab_playground.dexterous_tool.mdp.events import reset_floating_mocap_root
from mjlab_playground.dexterous_tool.mdp.observations import base_pose


def get_tool_spec(
  handle_half_length: float = 0.1,
  handle_radius: float = 0.015,
  head_size: tuple[float, float, float] = (0.04, 0.03, 0.03),
) -> mujoco.MjSpec:
  """Create a procedural hammer-like tool.

  ``handle_half_length`` is the half of the *total* handle length including
  the rounded caps, so the geom's outer z-extent is ±``handle_half_length``
  and the head sits flush with the top.
  """
  if handle_half_length <= handle_radius:
    raise ValueError(
      f"handle_half_length ({handle_half_length}) must exceed handle_radius "
      f"({handle_radius}); the capsule cylinder body would be non-positive."
    )
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="tool")
  body.add_freejoint(name="tool_joint")

  capsule_cylinder_half_length = handle_half_length - handle_radius
  body.add_geom(
    name="handle",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=[handle_radius, capsule_cylinder_half_length, 0],
    pos=[0, 0, 0],
    density=500.0,
    rgba=[0.6, 0.4, 0.2, 0.65],
    solref=[0.01, 1],
  )

  body.add_geom(
    name="head",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(head_size),
    pos=[0, 0, handle_half_length + head_size[2]],
    density=1500.0,
    rgba=[0.4, 0.4, 0.4, 0.65],
    solref=[0.01, 1],
  )

  body.add_site(
    name="grasp_center",
    pos=[0, 0, 0],
    size=[0.006],
    rgba=[0.1, 0.3, 1.0, 1.0],
    group=5,
  )

  return spec


def get_table_spec(
  size: tuple[float, float, float] = (0.3, 0.2, 0.19),
) -> mujoco.MjSpec:
  """Create a static table."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  body.pos[:] = [0.55, 0, size[2]]
  body.add_geom(
    name="table_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(size),
    rgba=[0.34, 0.39, 0.45, 1.0],
    friction=[1.0, 0.005, 0.0001],
    solref=[0.01, 1],
  )
  for i in range(4):
    body.add_site(
      name=f"support_corner_{i}",
      pos=[0.0, 0.0, size[2] + 0.005],
      size=[0.006],
      rgba=[0.0, 0.9, 0.9, 0.8],
      group=5,
    )
  return spec


def floating_sharpa_dexterous_tool_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_dexterous_tool_env_cfg()

  cfg.sim.njmax = 500
  cfg.sim.nconmax = 100

  cfg.scene.entities = {
    "robot": get_floating_sharpa_robot_cfg(),
    "tool": EntityCfg(
      spec_fn=get_tool_spec,
      init_state=EntityCfg.InitialStateCfg(
        pos=(0.55, 0.0, 0.41),
        rot=(1.0, 1.0, 0.0, 0.0),
      ),
    ),
    "table": EntityCfg(spec_fn=get_table_spec),
  }

  ##
  # Events.
  ##

  cfg.events["reset_base"] = EventTermCfg(
    func=reset_floating_mocap_root,
    mode="reset",
    params={
      "pose_range": {
        "x": (0.30, 0.40),
        "y": (-0.10, 0.10),
        "z": (0.60, 0.60),
      },
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("robot"),
      "mocap_body_name": "hand_mocap",
    },
  )

  ##
  # Observations.
  ##

  del cfg.observations["actor"].terms["arm_joint_pos"]
  del cfg.observations["actor"].terms["arm_joint_vel"]
  del cfg.observations["critic"].terms["arm_joint_pos"]
  del cfg.observations["critic"].terms["arm_joint_vel"]

  cfg.observations["actor"].terms["hand_joint_pos"].params[
    "asset_cfg"
  ].joint_names = HAND_JOINT_NAMES
  cfg.observations["actor"].terms["hand_joint_vel"].params[
    "asset_cfg"
  ].joint_names = HAND_JOINT_NAMES

  cfg.observations["actor"].terms["fingertip_pos_in_palm"].params[
    "asset_cfg"
  ].site_names = FINGERTIP_SITE_NAMES

  # Base pose so the policy knows where the hand is in world space.
  cfg.observations["actor"].terms["base_pose"] = ObservationTermCfg(
    func=base_pose,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.observations["critic"].terms["base_pose"] = ObservationTermCfg(
    func=base_pose,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  ##
  # Actions.
  ##

  del cfg.actions["arm_joint_pos"]

  cfg.actions["base_mocap"] = RelativeMocapActionCfg(
    entity_name="robot",
    mocap_body_name="hand_mocap",
    pos_scale=0.01,
    rot_scale=0.05,
  )

  hand_action_cfg = cfg.actions["hand_joint_pos"]
  assert isinstance(hand_action_cfg, RelativeJointPositionActionCfg)
  hand_action_cfg.actuator_names = HAND_JOINT_NAMES
  hand_action_cfg.scale = SHARPA_ACTION_SCALE

  ##
  # Rewards.
  ##

  del cfg.rewards["arm_action_rate"]
  del cfg.rewards["arm_joint_pos_limits"]
  del cfg.rewards["arm_joint_vel_hinge"]

  cfg.rewards["task"].params["asset_cfg"].site_names = FINGERTIP_SITE_NAMES
  cfg.metrics["approach_gauss"].params["asset_cfg"].site_names = FINGERTIP_SITE_NAMES

  cfg.rewards["hand_joint_pos_limits"].params[
    "asset_cfg"
  ].joint_names = HAND_JOINT_NAMES
  cfg.rewards["hand_joint_vel_hinge"].params["asset_cfg"].joint_names = HAND_JOINT_NAMES

  ##
  # Sensors.
  ##

  cfg.scene.sensors = tuple(
    s
    for s in cfg.scene.sensors
    if not (isinstance(s, ContactSensorCfg) and s.name == "arm_collision")
  )
  for sensor in cfg.scene.sensors:
    if isinstance(sensor, ContactSensorCfg) and sensor.name == "hand_table_collision":
      sensor.primary.pattern = "left_hand_C_MC"

  ##
  # Commands.
  ##

  tool_goal_cfg = cfg.commands["tool_goal"]
  if not isinstance(tool_goal_cfg, ToolGoalPoseCommandCfg):
    raise TypeError("Expected 'tool_goal' to use ToolGoalPoseCommandCfg.")

  tool_goal_cfg.workspace_mins = (0.30, -0.15, 0.65)
  tool_goal_cfg.workspace_maxs = (0.80, 0.15, 0.80)
  tool_goal_cfg.footprint_entity_name = "table"
  tool_goal_cfg.footprint_site_names = (
    "support_corner_0",
    "support_corner_1",
    "support_corner_2",
    "support_corner_3",
  )

  tool_goal_cfg.object_pose_range = tool_goal_cfg.ObjectPoseRangeCfg(
    x=(0.40, 0.70),
    y=(-0.10, 0.10),
    z=(0.41, 0.41),
    roll=(math.pi / 2, math.pi / 2),
    pitch=(0.0, 0.0),
    yaw=(-math.pi, math.pi),
  )

  ##
  # Viewer.
  ##

  cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
  cfg.viewer.entity_name = None
  cfg.viewer.body_name = None
  cfg.viewer.lookat = (0.25, 0.0, 0.32)
  cfg.viewer.distance = 1.45
  cfg.viewer.elevation = -28.0
  cfg.viewer.azimuth = 160.0
  cfg.viewer.fovy = 50.0

  # ========================================== #

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    del cfg.terminations["object_velocity_exceeded"]

  return cfg
