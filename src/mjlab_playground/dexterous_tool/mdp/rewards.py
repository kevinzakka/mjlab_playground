"""Reward functions for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def fingertip_approach(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Delta reward for fingertips approaching the object. Shape: (B,).

  Rewards reduction in mean fingertip-to-object distance. Uses the command
  term's min_fingertip_dist for stateful tracking.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  entity: Entity = env.scene[asset_cfg.name]

  # Get fingertip positions.
  fingertip_pos = entity.data.site_pos_w[:, asset_cfg.site_ids]  # (B, 5, 3)

  # Grasp reference position.
  grasp_pos = command.grasp_pos_w.unsqueeze(1)  # (B, 1, 3)

  # Mean distance from fingertips to the tool grasp site.
  dists = torch.norm(fingertip_pos - grasp_pos, dim=-1)  # (B, 5)
  fingertip_improvement = torch.clamp(command.min_fingertip_dists - dists, min=0.0)

  # Update tracker.
  command.min_fingertip_dists = torch.minimum(command.min_fingertip_dists, dists)

  return fingertip_improvement.sum(dim=-1)


def lift_object(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  lift_bonus: float = 300.0,
) -> torch.Tensor:
  """Dense height reward + sparse bonus when lifted. Shape: (B,)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  obj: Entity = env.scene[object_name]

  obj_delta_z = obj.data.root_link_pos_w[:, 2] - command.object_initial_pos_w[:, 2]
  lifted_now = command.lifted_object | (obj_delta_z > command.cfg.lift_threshold)

  # Dense: reward lift above the reset pose, but only before the object is counted as
  # lifted.
  dense = torch.clamp(obj_delta_z, min=0.0, max=0.5) * (~lifted_now).float()

  # Sparse bonus when first lifted (only once per episode via lifted_object flag).
  just_lifted = (~command.lifted_object) & lifted_now
  sparse = just_lifted.float() * lift_bonus

  return dense + sparse


def keypoint_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Delta reward for keypoint distance reduction to goal. Shape: (B,).

  Only active when the object has been lifted.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")

  # Max keypoint distance to goal.
  kp_dists = torch.norm(
    command.goal_keypoints_w - command.object_keypoints_w, dim=-1
  )  # (B, 4)
  max_kp_dist = kp_dists.max(dim=-1).values  # (B,)

  # Delta reward: improvement over best so far.
  reward = torch.clamp(command.min_keypoint_max_dist - max_kp_dist, min=0.0)

  # Only active when lifted.
  reward = reward * command.lifted_object.float()

  return reward


def goal_success_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  bonus: float = 1000.0,
) -> torch.Tensor:
  """Sparse bonus when within tolerance. Shape: (B,).

  Spreads the bonus over success_steps to avoid a single spike.
  """
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ToolGoalPoseCommand):
    raise ValueError(f"Expected ToolGoalPoseCommand, got {type(command)}")
  max_kp_dist = command.metrics["max_keypoint_dist"]
  at_goal = (max_kp_dist < command.cfg.success_tolerance).float()
  per_step_bonus = bonus / command.cfg.success_steps
  return at_goal * per_step_bonus * command.lifted_object.float()


def joint_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """L1 norm of joint velocities. Shape: (B,).

  Use weight to control arm vs hand penalty scale.
  """
  entity: Entity = env.scene[asset_cfg.name]
  vel = entity.data.joint_vel[:, asset_cfg.joint_ids]
  return vel.abs().sum(dim=-1)
