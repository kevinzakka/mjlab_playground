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


def _multiscale_gaussian(err: torch.Tensor, stds: tuple[float, ...]) -> torch.Tensor:
  """Average of Gaussians ``exp(-err² / std²)`` over the given std scales.

  Bakes a multi-scale curriculum into a single bounded ``[0, 1]`` reward:
  wide stds give long-range shaping gradient, narrow stds give precision near
  the goal. Pass a length-1 tuple for a plain Gaussian. The result is the
  arithmetic mean so the reward stays in ``[0, 1]`` regardless of how many
  scales are stacked.
  """
  if not stds:
    raise ValueError("stds must be non-empty")
  err_sq = err**2
  per_scale = torch.stack([torch.exp(-err_sq / s**2) for s in stds], dim=0)
  return per_scale.mean(dim=0)


def approach_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  stds: tuple[float, ...],
) -> torch.Tensor:
  """Multi-scale Gaussian on mean fingertip-to-grasp distance. Shape: (B,)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  entity: Entity = env.scene[asset_cfg.name]

  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]  # (B, F, 3)
  grasp_pos = command.grasp_pos_w.unsqueeze(1)  # (B, 1, 3)
  mean_dist = torch.norm(fingertip_pos - grasp_pos, dim=-1).mean(dim=-1)  # (B,)
  return _multiscale_gaussian(mean_dist, stds)


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
  stds: tuple[float, ...],
) -> torch.Tensor:
  """Multi-scale Gaussian on object position error to goal. Shape: (B,). Range: [0, 1]."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  tool: Entity = env.scene["tool"]

  err = torch.norm(tool.data.root_link_pos_w - command.goal_pos, dim=-1)
  return _multiscale_gaussian(err, stds)


def pose_orientation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  ori_stds: tuple[float, ...],
  table_contact_sensor_name: str,
) -> torch.Tensor:
  """Airborne-gated multi-scale Gaussian on orientation error.

  Shape: (B,). Range: [0, 1].

  Returns ``is_airborne * (1 + multiscale_ori_gauss) / 2`` where:

  - ``is_airborne`` reads a tool↔table contact sensor and is True iff zero
    contacts between any tool geom and the table body are reported this
    step. This is the **principled** "tool is held in the air" signal:
    geometry-independent (works for any tool shape), state-independent (no
    sticky flags), and impossible to exploit (you can't be both touching and
    not touching the table). Dropping the tool *immediately* closes the
    gate.
  - ``multiscale_ori_gauss`` is the average of ``exp(-ori_err² / s²)`` over
    each ``s`` in ``ori_stds``. Wide stds give shaping gradient at large
    misalignment; narrow stds give precision near the goal.
  - The ``(1 + ...) / 2`` baseline gives a 0.5 floor when the gate is open,
    providing the "be airborne" bootstrap signal so the agent has incentive
    to pick the tool up before it has learned to align.

  Uses ``quat_error_magnitude`` (frame-invariant, double-cover safe).
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  tool: Entity = env.scene["tool"]
  sensor: ContactSensor = env.scene[table_contact_sensor_name]

  found = sensor.data.found
  if found is None:
    raise ValueError(
      f"Sensor {table_contact_sensor_name!r} must request the 'found' field "
      f"for pose_orientation_reward to read it."
    )
  in_contact = (found > 0).any(dim=-1)  # (B,)
  is_airborne = (~in_contact).float()

  ori_err = quat_error_magnitude(command.goal_quat, tool.data.root_link_quat_w)
  ori = _multiscale_gaussian(ori_err, ori_stds)
  return is_airborne * (1.0 + ori) / 2.0


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
