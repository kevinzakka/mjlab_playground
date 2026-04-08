"""Reward functions for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def approach_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  std: float,
) -> torch.Tensor:
  """Gaussian on mean fingertip-to-grasp distance. Shape: (B,). Range: [0, 1]."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  entity: Entity = env.scene[asset_cfg.name]

  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]  # (B, F, 3)
  grasp_pos = command.grasp_pos_w.unsqueeze(1)  # (B, 1, 3)
  mean_dist = torch.norm(fingertip_pos - grasp_pos, dim=-1).mean(dim=-1)  # (B,)
  return torch.exp(-(mean_dist**2) / std**2)


def lift_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Gaussian on object height error relative to goal height. Shape: (B,). Range: [0, 1]."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")

  obj_z = env.scene["tool"].data.root_link_pos_w[:, 2]
  goal_z = command.goal_pos[:, 2]
  height_error = torch.abs(obj_z - goal_z)
  return torch.exp(-(height_error**2) / std**2)


def alignment_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Gaussian on max keypoint distance to goal. Shape: (B,). Range: [0, 1]."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")

  kp_dists = torch.norm(
    command.goal_keypoints_w - command.object_keypoints_w, dim=-1
  )  # (B, 4)
  max_kp_dist = kp_dists.max(dim=-1).values  # (B,)
  return torch.exp(-(max_kp_dist**2) / std**2)


def action_rate_l2(
  env: ManagerBasedRlEnv,
  action_name: str,
) -> torch.Tensor:
  """L2 squared action rate for a single action term. Shape: (B,)."""
  am = env.action_manager
  # Find the slice for this action term in the flat action buffer.
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
  from mjlab.sensor import ContactSensor

  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return force_mag.max(dim=-1).values.max(dim=-1).values  # [B]
  if data.force is not None:
    # force: [B, N, 3]
    return torch.norm(data.force, dim=-1).max(dim=-1).values  # [B]
  return torch.zeros(env.num_envs, device=env.device)
