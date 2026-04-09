"""Observation functions for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_GEOM_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE.value
_GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER.value


def _palm_pose_w(
  robot: Entity, palm_site_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return (palm_pos_w, palm_quat_w) for the palm site on the given entity."""
  palm_ids, _ = robot.find_sites((palm_site_name,))
  return (
    robot.data.site_pos_w[:, palm_ids[0]],
    robot.data.site_quat_w[:, palm_ids[0]],
  )


def _pos_and_ori_6d(
  pos: torch.Tensor, ori_quat: torch.Tensor, num_envs: int
) -> torch.Tensor:
  """Concatenate a (B, 3) position and (B, 4) quat into (B, 9) [pos(3), 6d(6)]."""
  mat = matrix_from_quat(ori_quat)
  ori_6d = mat[..., :2, :].reshape(num_envs, 6)
  return torch.cat([pos, ori_6d], dim=-1)


def fingertip_pos_in_palm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  palm_site_name: str = "palm_center",
) -> torch.Tensor:
  """Fingertip positions expressed in the palm's local frame. Shape: (B, F*3)."""
  robot: Entity = env.scene[asset_cfg.name]
  palm_pos_w, palm_quat_w = _palm_pose_w(robot, palm_site_name)
  fingertip_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (B, F, 3)

  # Rotate the world-frame displacement into the palm's local frame.
  delta_w = fingertip_pos_w - palm_pos_w.unsqueeze(1)  # (B, F, 3)
  F = delta_w.shape[1]
  palm_quat_expanded = palm_quat_w.unsqueeze(1).expand(-1, F, -1)
  rel_in_palm = quat_apply_inverse(
    palm_quat_expanded.reshape(-1, 4), delta_w.reshape(-1, 3)
  ).reshape(env.num_envs, F, 3)
  return rel_in_palm.reshape(env.num_envs, -1)


def tool_pose_in_palm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  palm_site_name: str = "palm_center",
) -> torch.Tensor:
  """Tool pose expressed in the palm's local frame. Shape: (B, 9).

  This is what the policy uses to locate the tool relative to its end-effector for
  reaching and grasping.
  """
  robot: Entity = env.scene[asset_cfg.name]
  tool: Entity = env.scene["tool"]
  palm_pos_w, palm_quat_w = _palm_pose_w(robot, palm_site_name)

  pos_b, ori_b = subtract_frame_transforms(
    palm_pos_w,
    palm_quat_w,
    tool.data.root_link_pos_w,
    tool.data.root_link_quat_w,
  )
  return _pos_and_ori_6d(pos_b, ori_b, env.num_envs)


def goal_pose_in_palm(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  palm_site_name: str = "palm_center",
) -> torch.Tensor:
  """Goal pose expressed in the palm's local frame. Shape: (B, 9).

  Tells the policy where to take the hand for the tool to land at the goal.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  robot: Entity = env.scene[asset_cfg.name]
  palm_pos_w, palm_quat_w = _palm_pose_w(robot, palm_site_name)

  pos_b, ori_b = subtract_frame_transforms(
    palm_pos_w,
    palm_quat_w,
    command.goal_pos,
    command.goal_quat,
  )
  return _pos_and_ori_6d(pos_b, ori_b, env.num_envs)


def goal_pose_in_tool(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Goal pose expressed in the tool's local frame. Shape: (B, 9).

  This is the SE(3) tracking error feature: a redundant pre-computed signal that tells
  the policy how to twist the tool toward the goal, regardless of where the hand
  happens to be.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  tool: Entity = env.scene["tool"]

  pos_b, ori_b = subtract_frame_transforms(
    tool.data.root_link_pos_w,
    tool.data.root_link_quat_w,
    command.goal_pos,
    command.goal_quat,
  )
  return _pos_and_ori_6d(pos_b, ori_b, env.num_envs)


# TODO: To be used when we enable tool size randomization.
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
