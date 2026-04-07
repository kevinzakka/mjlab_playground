"""Observation functions for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.dexterous_tool.mdp.actions import DeltaJointPositionAction
from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_GEOM_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE.value
_GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER.value


def palm_pose(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Palm position (3) + quaternion (4) in world frame. Shape: (B, 7)."""
  entity: Entity = env.scene[asset_cfg.name]
  pos = entity.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)  # (B, 3)
  quat = entity.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)  # (B, 4)
  return torch.cat([pos, quat], dim=-1)


def palm_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Palm linear (3) + angular (3) velocity in world frame. Shape: (B, 6)."""
  entity: Entity = env.scene[asset_cfg.name]
  lin_vel = entity.data.body_link_lin_vel_w[:, asset_cfg.body_ids].squeeze(1)  # (B, 3)
  ang_vel = entity.data.body_link_ang_vel_w[:, asset_cfg.body_ids].squeeze(1)  # (B, 3)
  return torch.cat([lin_vel, ang_vel], dim=-1)


def prev_action_targets(
  env: ManagerBasedRlEnv,
  arm_action_name: str = "arm_joint_pos",
  hand_action_name: str = "hand_joint_pos",
) -> torch.Tensor:
  """Previous commanded joint targets for arm and hand. Shape: (B, J)."""
  arm_term = env.action_manager.get_term(arm_action_name)
  hand_term = env.action_manager.get_term(hand_action_name)
  if not isinstance(arm_term, DeltaJointPositionAction):
    raise ValueError(f"Expected DeltaJointPositionAction, got {type(arm_term)}")
  if not isinstance(hand_term, DeltaJointPositionAction):
    raise ValueError(f"Expected DeltaJointPositionAction, got {type(hand_term)}")
  return torch.cat([arm_term.target_command, hand_term.target_command], dim=-1)


def fingertip_pos_rel_palm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  palm_site_name: str = "palm_center",
) -> torch.Tensor:
  """5 fingertip positions relative to palm. Shape: (B, 15).

  asset_cfg should have site_names for the 5 fingertip sites.
  """
  entity: Entity = env.scene[asset_cfg.name]

  # Get palm position.
  palm_ids, _ = entity.find_sites((palm_site_name,))
  palm_pos = entity.data.site_pos_w[:, palm_ids[0]]  # (B, 3)

  # Get fingertip positions relative to palm.
  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]  # (B, 5, 3)
  rel_pos = fingertip_pos - palm_pos.unsqueeze(1)
  return rel_pos.reshape(env.num_envs, -1)  # (B, 15)


def object_orientation(
  env: ManagerBasedRlEnv,
  object_name: str,
) -> torch.Tensor:
  """Object quaternion (w,x,y,z). Shape: (B, 4)."""
  entity: Entity = env.scene[object_name]
  return entity.data.root_link_quat_w


def object_velocity(
  env: ManagerBasedRlEnv,
  object_name: str,
) -> torch.Tensor:
  """Object linear (3) + angular (3) velocity. Shape: (B, 6)."""
  entity: Entity = env.scene[object_name]
  lin_vel = entity.data.root_link_lin_vel_w
  ang_vel = entity.data.root_link_ang_vel_w
  return torch.cat([lin_vel, ang_vel], dim=-1)


def keypoints_rel_palm(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  palm_site_name: str = "palm_center",
) -> torch.Tensor:
  """4 object keypoints relative to palm position. Shape: (B, 12)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  entity: Entity = env.scene[asset_cfg.name]

  palm_ids, _ = entity.find_sites((palm_site_name,))
  palm_pos = entity.data.site_pos_w[:, palm_ids[0]]  # (B, 3)

  # Object keypoints from the command term.
  kp_w = command.object_keypoints_w  # (B, 4, 3)
  rel_kp = kp_w - palm_pos.unsqueeze(1)
  return rel_kp.reshape(env.num_envs, -1)  # (B, 12)


def keypoint_errors(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Object keypoints - goal keypoints, flattened. Shape: (B, 12)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  return (command.object_keypoints_w - command.goal_keypoints_w).reshape(
    env.num_envs, -1
  )


def object_scales(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Tool grasp bounding box dimensions from geom sizes. Shape: (B, 3).

  Reads the actual geom_size from the model so it reflects DR changes.
  asset_cfg should select the geoms whose scale should be exposed.
  """
  entity: Entity = env.scene[asset_cfg.name]
  global_geom_ids = entity.indexing.geom_ids[asset_cfg.geom_ids]
  sizes = env.sim.model.geom_size[:, global_geom_ids]  # (B, G, 3)
  geom_types = torch.as_tensor(
    env.sim.model.geom_type,
    device=env.device,
    dtype=torch.int32,
  )[global_geom_ids.long()]

  full_sizes = 2.0 * sizes.clone()
  is_cylinder = geom_types[None, :] == _GEOM_CYLINDER
  is_sphere = geom_types[None, :] == _GEOM_SPHERE
  full_sizes[..., 0] = torch.where(
    is_sphere | is_cylinder, 2.0 * sizes[..., 0], full_sizes[..., 0]
  )
  full_sizes[..., 1] = torch.where(
    is_sphere | is_cylinder, 2.0 * sizes[..., 0], full_sizes[..., 1]
  )
  full_sizes[..., 2] = torch.where(is_sphere, 2.0 * sizes[..., 0], full_sizes[..., 2])
  full_sizes[..., 2] = torch.where(is_cylinder, 2.0 * sizes[..., 1], full_sizes[..., 2])
  return full_sizes.max(dim=1).values


def closest_keypoint_max_dist(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Best max-keypoint distance achieved for the current goal. Shape: (B, 1)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  return command.min_keypoint_max_dist.unsqueeze(-1)


def closest_fingertip_distances(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Best fingertip-to-object distances achieved this episode. Shape: (B, F)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  entity: Entity = env.scene[asset_cfg.name]
  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]
  grasp_pos = command.grasp_pos_w.unsqueeze(1)
  current_dists = torch.norm(fingertip_pos - grasp_pos, dim=-1)
  return torch.where(
    torch.isfinite(command.min_fingertip_dists),
    command.min_fingertip_dists,
    current_dists,
  )


def lifted_object(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Whether the object has crossed the lift threshold. Shape: (B, 1)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  return command.lifted_object.float().unsqueeze(-1)


def progress(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Log-scaled episode progress. Shape: (B, 1)."""
  return torch.log(env.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1)


def successes(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Log-scaled number of goals achieved this episode. Shape: (B, 1)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  return torch.log(command.num_goal_resets.float() + 1.0).unsqueeze(-1)


def reward(
  env: ManagerBasedRlEnv,
  scale: float = 0.01,
) -> torch.Tensor:
  """Scaled current reward for privileged critic observations. Shape: (B, 1)."""
  reward_buf = getattr(env, "reward_buf", None)
  if reward_buf is None:
    return torch.zeros(env.num_envs, 1, device=env.device)
  return (reward_buf * scale).unsqueeze(-1)
