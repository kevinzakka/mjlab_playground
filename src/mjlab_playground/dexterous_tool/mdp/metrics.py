"""Metrics for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def object_lin_speed(
  env: ManagerBasedRlEnv,
  object_name: str,
) -> torch.Tensor:
  """Object linear speed in world frame (m/s). Shape: (B,)."""
  obj: Entity = env.scene[object_name]
  return torch.norm(obj.data.root_link_lin_vel_w, dim=-1)


def object_ang_speed(
  env: ManagerBasedRlEnv,
  object_name: str,
) -> torch.Tensor:
  """Object angular speed in world frame (rad/s). Shape: (B,)."""
  obj: Entity = env.scene[object_name]
  return torch.norm(obj.data.root_link_ang_vel_w, dim=-1)


def approach_gauss(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  std: float,
) -> torch.Tensor:
  """Gaussian on mean fingertip-to-grasp distance. Shape: (B,)."""
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, ToolGoalPoseCommand)
  entity: Entity = env.scene[asset_cfg.name]
  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]
  grasp_pos = command.grasp_pos_w.unsqueeze(1)
  mean_dist = torch.norm(fingertip_pos - grasp_pos, dim=-1).mean(dim=-1)
  return torch.exp(-(mean_dist**2) / std**2)


def position_gauss(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Gaussian on position error to goal. Shape: (B,)."""
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, ToolGoalPoseCommand)
  tool: Entity = env.scene["tool"]
  err = torch.norm(tool.data.root_link_pos_w - command.goal_pos, dim=-1)
  return torch.exp(-(err**2) / std**2)


def orientation_gauss(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Gaussian on orientation error to goal. Shape: (B,)."""
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, ToolGoalPoseCommand)
  tool: Entity = env.scene["tool"]
  ori_err = quat_error_magnitude(command.goal_quat, tool.data.root_link_quat_w)
  return torch.exp(-(ori_err**2) / std**2)
