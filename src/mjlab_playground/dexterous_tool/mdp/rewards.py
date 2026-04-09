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


def staged_track_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  fingertip_stds: tuple[float, ...],
  height_target: float,
  height_std: float,
  pos_stds: tuple[float, ...],
  ori_stds: tuple[float, ...],
  table_contact_sensor_name: str,
) -> torch.Tensor:
  """Single staged reward bridging reach → lift → 6-DoF tracking. Shape: (B,).

  Range: ``[0, 3]``. Form::

      approach · (1 + height · (1 + airborne · track))

  with each factor a bounded ``[0, 1]`` shaping signal:

  - ``approach`` is the multi-scale Gaussian on the mean fingertip→
    ``grasp_center`` distance over ``fingertip_stds``. Wide stds give the
    long-range "go to the tool" gradient; narrow stds the close-range
    grasp-alignment gradient. Always on.

  - ``height`` is a single Gaussian on the env-local height deficit
    ``max(0, height_target − (obj_z − env_origin_z))``, saturating at 1.0
    once the tool reaches ``height_target`` and providing a smooth "lift the
    tool higher" gradient below it. ``height_target`` should sit just above
    the command's ``object_pose_range.z + lift_threshold`` so the saturation
    point is past the lift gate.

  - ``airborne`` is a hard ``{0, 1}`` indicator from a tool↔table contact
    sensor: ``1`` iff zero contacts are reported between any tool geom and
    the table body this step. This is the principled "tool is held in the
    air" signal — geometry-independent, state-independent, and unexploitable
    (you cannot be both touching and not touching the table). Closes the
    slide-along-table and stand-the-tool-on-its-head exploits that any
    purely-geometric proxy admits.

  - ``track`` averages multi-scale position and orientation Gaussians,
    ``(pos_gauss + ori_gauss) / 2``. Position uses ``pos_stds`` (meters);
    orientation uses ``ori_stds`` (radians, via ``quat_error_magnitude``,
    which is frame-invariant and double-cover safe). Position and
    orientation are summed (not multiplied) because they are two projections
    of the same SE(3) error, not separate phases.

  The multiplicative staging is the same trick mjlab's lift-cube task uses
  (``reach · (1 + bring)``), recursed one level for the extra lift phase. It
  has two key properties:

  1. **No phase plateau.** ``approach`` carries gradient until grasp;
     ``height`` carries gradient through the lift transition; ``track``
     carries gradient through 6-DoF alignment. There is no flat region the
     policy can park on.

  2. **No free constants.** After grasp ``approach ≈ 1`` looks constant, but
     it is the multiplier that unlocks the ``(1 + height · …)`` bonus —
     without it the downstream stages collapse. Same for ``height`` after
     lift. Every saturated factor is doing the gating job for the stage
     above it.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  entity: Entity = env.scene[asset_cfg.name]
  tool: Entity = env.scene["tool"]
  sensor: ContactSensor = env.scene[table_contact_sensor_name]

  # Stage 1: approach — multi-scale Gaussian on mean fingertip→grasp distance.
  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]  # (B, F, 3)
  grasp_pos = command.grasp_pos_w.unsqueeze(1)  # (B, 1, 3)
  fingertip_dist = torch.norm(fingertip_pos - grasp_pos, dim=-1).mean(dim=-1)
  approach = _multiscale_gaussian(fingertip_dist, fingertip_stds)

  # Stage 2: height — Gaussian on env-local height deficit (clamped at 0 above
  # target so going higher than target_height does not get penalized).
  obj_z = tool.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  deficit = torch.clamp(height_target - obj_z, min=0.0)
  height = torch.exp(-(deficit**2) / height_std**2)

  # Stage 3: track — multi-scale Gaussians on pos & ori, hard-gated on
  # tool↔table contact (zero contacts ⇔ airborne).
  found = sensor.data.found
  if found is None:
    raise ValueError(
      f"Sensor {table_contact_sensor_name!r} must request the 'found' field "
      f"for staged_track_reward to read it."
    )
  is_airborne = (~(found > 0).any(dim=-1)).float()

  pos_err = torch.norm(tool.data.root_link_pos_w - command.goal_pos, dim=-1)
  pos = _multiscale_gaussian(pos_err, pos_stds)
  ori_err = quat_error_magnitude(command.goal_quat, tool.data.root_link_quat_w)
  ori = _multiscale_gaussian(ori_err, ori_stds)
  track = is_airborne * (pos + ori) / 2.0

  return approach * (1.0 + height * (1.0 + track))


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
