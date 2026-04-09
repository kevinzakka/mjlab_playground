"""Metrics for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity

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
