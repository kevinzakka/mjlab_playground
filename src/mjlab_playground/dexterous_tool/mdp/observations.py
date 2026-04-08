"""Observation functions for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_GEOM_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE.value
_GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER.value


def palm_pose(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  palm_site_name: str = "palm_center",
) -> torch.Tensor:
  """Palm position (in robot base frame) and quaternion. Shape: (B, 7)."""
  entity: Entity = env.scene[asset_cfg.name]
  palm_ids, _ = entity.find_sites((palm_site_name,))
  palm_pos_w = entity.data.site_pos_w[:, palm_ids[0]]  # (B, 3)
  palm_quat_w = entity.data.site_quat_w[:, palm_ids[0]]  # (B, 4)

  # Express position in robot base frame.
  base_pos_w = entity.data.root_link_pos_w  # (B, 3)
  base_quat_w = entity.data.root_link_quat_w  # (B, 4)
  palm_pos_b = quat_apply(quat_inv(base_quat_w), palm_pos_w - base_pos_w)

  return torch.cat([palm_pos_b, palm_quat_w], dim=-1)


def fingertip_pos_rel_palm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  palm_site_name: str = "palm_center",
) -> torch.Tensor:
  """Fingertip positions relative to palm. Shape: (B, F*3)."""
  entity: Entity = env.scene[asset_cfg.name]
  palm_ids, _ = entity.find_sites((palm_site_name,))
  palm_pos = entity.data.site_pos_w[:, palm_ids[0]]  # (B, 3)
  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]  # (B, F, 3)
  rel_pos = fingertip_pos - palm_pos.unsqueeze(1)
  return rel_pos.reshape(env.num_envs, -1)


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
  """Tool bounding box dimensions from geom sizes. Shape: (B, 3)."""
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
