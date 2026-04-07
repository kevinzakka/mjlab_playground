"""Termination conditions for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def object_fallen(
  env: ManagerBasedRlEnv,
  object_name: str,
  min_z: float = 0.1,
) -> torch.Tensor:
  """Terminate if object z-position is below threshold. Shape: (B,)."""
  obj: Entity = env.scene[object_name]
  obj_z = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return obj_z < min_z


def object_dropped_after_lift(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  """Terminate if object was lifted but then dropped. Shape: (B,)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  obj: Entity = env.scene[object_name]
  obj_z = obj.data.root_link_pos_w[:, 2]
  return command.lifted_object & (obj_z < command.object_initial_pos_w[:, 2])


def hand_too_far(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  max_distance: float = 1.5,
) -> torch.Tensor:
  """Terminate if any fingertip is too far from the object. Shape: (B,)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  entity: Entity = env.scene[asset_cfg.name]
  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]
  grasp_pos = command.grasp_pos_w.unsqueeze(1)
  dists = torch.norm(fingertip_pos - grasp_pos, dim=-1)
  return dists.max(dim=-1).values > max_distance
