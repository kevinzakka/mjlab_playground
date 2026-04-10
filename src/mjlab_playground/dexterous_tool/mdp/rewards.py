"""Reward functions for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def task_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  approach_std: float,
  position_std: float,
  orientation_std: float,
) -> torch.Tensor:
  """``approach * (1 + position + orientation) / 3``. Shape: (B,). Range: [0, 1].

  Approach gates tracking: position and orientation only pay out while the grasp is
  maintained.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  entity: Entity = env.scene[asset_cfg.name]
  tool: Entity = env.scene["tool"]

  # Approach: Gaussian on mean fingertip→grasp distance.
  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]  # (B, F, 3)
  grasp_pos = command.grasp_pos_w.unsqueeze(1)  # (B, 1, 3)
  mean_dist = torch.norm(fingertip_pos - grasp_pos, dim=-1).mean(dim=-1)
  approach = torch.exp(-(mean_dist**2) / approach_std**2)

  # Position: Gaussian on position error to goal.
  pos_err = torch.norm(tool.data.root_link_pos_w - command.goal_pos, dim=-1)
  position = torch.exp(-(pos_err**2) / position_std**2)

  # Orientation: Gaussian on orientation error to goal.
  ori_err = quat_error_magnitude(command.goal_quat, tool.data.root_link_quat_w)
  orientation = torch.exp(-(ori_err**2) / orientation_std**2)

  return approach * (1.0 + position + orientation) / 3.0


def action_rate_l2(
  env: ManagerBasedRlEnv,
  action_name: str,
) -> torch.Tensor:
  """L2 squared action rate for a single action term. Shape: (B,)."""
  am = env.action_manager
  offset = 0
  for name in am.active_terms:
    term = am.get_term(name)
    if name == action_name:
      curr = am.action[:, offset : offset + term.action_dim]
      prev = am.prev_action[:, offset : offset + term.action_dim]
      return torch.sum(torch.square(curr - prev), dim=-1)
    offset += term.action_dim
  raise ValueError(f"Action term '{action_name}' not found.")


def contact_force_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Max contact force magnitude from a contact sensor. Shape: (B,)."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return force_mag.max(dim=-1).values.max(dim=-1).values
  if data.force is not None:
    return torch.norm(data.force, dim=-1).max(dim=-1).values
  return torch.zeros(env.num_envs, device=env.device)
