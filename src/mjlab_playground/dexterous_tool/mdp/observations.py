"""Observation functions for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_GEOM_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE.value
_GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER.value


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
