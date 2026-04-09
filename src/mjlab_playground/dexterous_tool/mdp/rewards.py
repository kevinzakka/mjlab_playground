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


def tool_above_table_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
) -> torch.Tensor:
  """Smooth shaping reward for getting the tool off the table. Shape: (B,).

  Returns a Gaussian on the env-local height *deficit*
  ``max(0, target_height − (obj_z − env_origin_z))``, saturating at 1.0 once
  the tool reaches ``target_height`` above its env origin and providing a
  smooth gradient below it. Provides the long-range "lift off the table"
  signal that bridges from "fingers on tool" to "tool in the air."

  ``target_height`` should be set just *above* the command's
  ``object_pose_range.z + lift_threshold`` so the saturation point is past
  the threshold at which ``ToolGoalPoseCommand.lifted_object`` flips. This
  ensures the agent has shaping gradient all the way through the lift gate.
  """
  tool: Entity = env.scene["tool"]
  obj_z = tool.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  deficit = torch.clamp(target_height - obj_z, min=0.0)
  return torch.exp(-(deficit**2) / std**2)


def pose_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Gaussian on object position error to goal. Shape: (B,). Range: [0, 1]."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  tool: Entity = env.scene["tool"]

  err = torch.norm(tool.data.root_link_pos_w - command.goal_pos, dim=-1)
  return torch.exp(-(err**2) / std**2)


def pose_orientation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  ori_std: float,
) -> torch.Tensor:
  """Lift-gated Gaussian on orientation error. Shape: (B,). Range: [0, 1].

  Returns ``lifted * (1 + ori_gauss) / 2`` where ``lifted`` is the sticky
  per-episode flag set by ``ToolGoalPoseCommand`` once the tool has crossed
  ``lift_threshold`` above its reset height. The reward is zero until the
  agent has *actually* lifted the tool — preventing the "spin flat on the
  table" cheat — and once lifted gives a baseline 0.5 plus an alignment bonus
  up to 0.5. Uses ``quat_error_magnitude`` (frame-invariant, double-cover safe).
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  tool: Entity = env.scene["tool"]

  ori_err = quat_error_magnitude(command.goal_quat, tool.data.root_link_quat_w)
  ori = torch.exp(-(ori_err**2) / ori_std**2)
  return command.lifted_object.float() * (1.0 + ori) / 2.0


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
