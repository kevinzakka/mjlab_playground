"""Reward functions for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def staged_goal_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  reaching_std: float,
  lifting_std: float,
  bringing_std: float,
) -> torch.Tensor:
  """3-stage Markovian reward: approach * (1 + lift * (1 + alignment)). Shape: (B,).

  All terms are stateless Gaussian kernels over current-state distances:

  - approach: Gaussian on mean fingertip-to-grasp distance.
  - lift: Gaussian on object height error relative to goal height.
  - alignment: Gaussian on max keypoint distance to goal.

  The nested product naturally stages learning:
    1. Reach toward the object (approach saturates near 1).
    2. Lift the object toward the goal height (lift gets gradient).
    3. Align keypoints to the goal pose (alignment gets gradient).
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  entity: Entity = env.scene[asset_cfg.name]

  # Approach: mean fingertip distance to grasp site.
  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]  # (B, F, 3)
  grasp_pos = command.grasp_pos_w.unsqueeze(1)  # (B, 1, 3)
  mean_dist = torch.norm(fingertip_pos - grasp_pos, dim=-1).mean(dim=-1)  # (B,)
  approach = torch.exp(-(mean_dist**2) / reaching_std**2)

  # Lift: object height error relative to goal height.
  obj_z = env.scene["tool"].data.root_link_pos_w[:, 2]
  goal_z = command.goal_pos[:, 2]
  height_error = torch.abs(obj_z - goal_z)
  lift = torch.exp(-(height_error**2) / lifting_std**2)

  # Alignment: max keypoint distance to goal.
  kp_dists = torch.norm(
    command.goal_keypoints_w - command.object_keypoints_w, dim=-1
  )  # (B, 4)
  max_kp_dist = kp_dists.max(dim=-1).values  # (B,)
  alignment = torch.exp(-(max_kp_dist**2) / bringing_std**2)

  return approach * (1.0 + lift * (1.0 + alignment))


def goal_precision_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Tight Gaussian on max keypoint distance. Shape: (B,).

  Provides a sharper gradient near the goal for precise alignment.
  """
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
