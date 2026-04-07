"""KUKA iiwa14 + SHARPA hand configuration for dexterous tool task."""

import math

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_playground.asset_zoo.robots.kuka_sharpa import get_kuka_sharpa_robot_cfg
from mjlab_playground.asset_zoo.robots.kuka_sharpa.kuka_sharpa_constants import (
  ARM_JOINT_NAMES,
  HAND_JOINT_NAMES,
)
from mjlab_playground.dexterous_tool.dexterous_tool_env_cfg import (
  make_dexterous_tool_env_cfg,
)
from mjlab_playground.dexterous_tool.mdp.actions import DeltaJointPositionActionCfg
from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommandCfg


def get_tool_spec(
  handle_half_length: float = 0.1,
  handle_radius: float = 0.015,
  head_size: tuple[float, float, float] = (0.04, 0.03, 0.03),
) -> mujoco.MjSpec:
  """Create a procedural hammer-like tool."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="tool")
  body.add_freejoint(name="tool_joint")

  # Handle: cylinder along z-axis.
  body.add_geom(
    name="handle",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[handle_radius, handle_half_length, 0],
    pos=[0, 0, 0],
    density=500.0,
    rgba=[0.6, 0.4, 0.2, 0.65],
  )

  # Head: box at the top of the handle.
  body.add_geom(
    name="head",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(head_size),
    pos=[0, 0, handle_half_length + head_size[2]],
    density=1500.0,
    rgba=[0.4, 0.4, 0.4, 0.65],
  )

  # Keypoint sites at bounding box corners.
  offsets = [
    (1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (-1, -1, -1),
  ]
  keypoint_half_extents = (0.015, 0.015, 0.07)
  for i, (ox, oy, oz) in enumerate(offsets):
    body.add_site(
      name=f"keypoint_{i}",
      pos=[
        ox * keypoint_half_extents[0],
        oy * keypoint_half_extents[1],
        oz * keypoint_half_extents[2],
      ],
      size=[0.01],
      rgba=[1, 0.1, 0.1, 1.0],
      group=4,
    )

  # Grasp bounding box center.
  body.add_site(
    name="grasp_center",
    pos=[0, 0, 0],
    size=[0.006],
    rgba=[0.1, 0.3, 1.0, 1.0],
    group=4,
  )

  return spec


def get_table_spec(
  size: tuple[float, float, float] = (0.3, 0.2, 0.19),
) -> mujoco.MjSpec:
  """Create a static table."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  body.pos[:] = [0.55, 0, size[2]]  # Position in front of robot.
  body.add_geom(
    name="table_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(size),
    mass=50.0,
    rgba=[0.34, 0.39, 0.45, 1.0],
    friction=[1.0, 0.005, 0.0001],
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


# Fingertip site names (must match what get_kuka_sharpa_spec adds).
_FINGERTIP_SITES = (
  "fingertip_thumb",
  "fingertip_index",
  "fingertip_middle",
  "fingertip_ring",
  "fingertip_pinky",
)


def kuka_sharpa_dexterous_tool_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_dexterous_tool_env_cfg()

  cfg.sim.njmax = 180

  cfg.scene.entities = {
    "robot": get_kuka_sharpa_robot_cfg(arm_collisions=False),
    "tool": EntityCfg(
      spec_fn=get_tool_spec,
      init_state=EntityCfg.InitialStateCfg(
        pos=(0.55, 0.0, 0.41),
        rot=(0.70710678, 0.70710678, 0.0, 0.0),
      ),
    ),
    "table": EntityCfg(spec_fn=get_table_spec),
  }

  ##
  # Observations.
  ##

  # Configure palm site for observations.
  cfg.observations["actor"].terms["palm_pose"].params["asset_cfg"].site_names = (
    "palm_center",
  )

  # Configure fingertip sites for observations.
  cfg.observations["actor"].terms["fingertip_pos_rel_palm"].params[
    "asset_cfg"
  ].site_names = _FINGERTIP_SITES

  # Keypoints relative to palm.
  cfg.observations["actor"].terms["keypoints_rel_palm"].params[
    "asset_cfg"
  ].site_names = ("palm_center",)

  # Object scales: use the handle geom as the grasp-scale observation.
  cfg.observations["actor"].terms["object_scales"].params["asset_cfg"].geom_names = (
    "handle",
  )

  # Critic: palm velocity body.
  cfg.observations["critic"].terms["palm_velocity"].params["asset_cfg"].body_names = (
    "left_hand_C_MC",
  )
  cfg.observations["critic"].terms["closest_fingertip_dist"].params[
    "asset_cfg"
  ].site_names = _FINGERTIP_SITES

  ##
  # Rewards.
  ##

  arm_action_cfg = cfg.actions["arm_joint_pos"]
  hand_action_cfg = cfg.actions["hand_joint_pos"]
  if not isinstance(arm_action_cfg, DeltaJointPositionActionCfg):
    raise TypeError("Expected 'arm_joint_pos' to use DeltaJointPositionActionCfg.")
  if not isinstance(hand_action_cfg, DeltaJointPositionActionCfg):
    raise TypeError("Expected 'hand_joint_pos' to use DeltaJointPositionActionCfg.")
  arm_action_cfg.actuator_names = ARM_JOINT_NAMES
  hand_action_cfg.actuator_names = HAND_JOINT_NAMES

  # fingertip approach sites.
  cfg.rewards["fingertip_approach"].params["asset_cfg"].site_names = _FINGERTIP_SITES

  # arm and hand velocity penalties.
  cfg.rewards["arm_velocity_penalty"].params["asset_cfg"].joint_names = ARM_JOINT_NAMES
  cfg.rewards["hand_velocity_penalty"].params[
    "asset_cfg"
  ].joint_names = HAND_JOINT_NAMES
  cfg.rewards["arm_dof_pos_limits"].params["asset_cfg"].joint_names = ARM_JOINT_NAMES
  cfg.rewards["hand_dof_pos_limits"].params["asset_cfg"].joint_names = HAND_JOINT_NAMES

  ##
  # Terminations.
  ##

  # hand too far (palm site).
  cfg.terminations["hand_too_far"].params["asset_cfg"].site_names = _FINGERTIP_SITES

  ##
  # Events.
  ##

  tool_goal_cfg = cfg.commands["tool_goal"]
  if not isinstance(tool_goal_cfg, ToolGoalPoseCommandCfg):
    raise TypeError("Expected 'tool_goal' to use ToolGoalPoseCommandCfg.")

  # Keep sampled goals over the table for every env origin.
  tool_goal_cfg.workspace_mins = (0.30, -0.15, 0.41)
  tool_goal_cfg.workspace_maxs = (0.80, 0.15, 0.52)
  tool_goal_cfg.num_fingertips = len(_FINGERTIP_SITES)
  tool_goal_cfg.footprint_entity_name = "table"
  tool_goal_cfg.footprint_site_names = (
    "support_corner_0",
    "support_corner_1",
    "support_corner_2",
    "support_corner_3",
  )

  # Mirror lift_cube: own object placement in the command term only.
  tool_goal_cfg.object_pose_range = tool_goal_cfg.ObjectPoseRangeCfg(
    x=(0.25, 0.85),  # full tabletop x-bounds; command shrinks by tool footprint
    y=(-0.20, 0.20),  # full tabletop y-bounds; command shrinks by tool footprint
    z=(0.41, 0.41),  # Default flat-on-table root z; command adjusts for DR support.
    roll=(math.pi / 2, math.pi / 2),
    pitch=(0.0, 0.0),
    yaw=(-math.pi, math.pi),
  )

  ##
  # Misc.
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

  return cfg
