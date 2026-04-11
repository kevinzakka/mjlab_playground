"""Custom action terms for dexterous tool manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RelativeMocapActionCfg(ActionTermCfg):
  """Action that applies relative position and orientation deltas to a mocap body.

  The policy outputs a 6D vector: [dx, dy, dz, droll, dpitch, dyaw].
  Position deltas are added to the current mocap position.
  Orientation deltas (Euler angles) are composed with the current mocap quaternion.
  """

  entity_name: str = "robot"
  """Entity that contains the mocap body."""

  mocap_body_name: str = "hand_mocap"
  """Name of the mocap body inside the entity."""

  pos_scale: float = 0.01
  """Scale for position deltas (meters per unit action)."""

  rot_scale: float = 0.05
  """Scale for orientation deltas (radians per unit action)."""

  def build(self, env: ManagerBasedRlEnv) -> RelativeMocapAction:
    return RelativeMocapAction(self, env)


class RelativeMocapAction(ActionTerm):
  """Apply relative pose deltas to an internal mocap body."""

  cfg: RelativeMocapActionCfg

  def __init__(self, cfg: RelativeMocapActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    self._raw_actions = torch.zeros(env.num_envs, 6, device=env.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    # Resolve the mocap body ID in the simulation.
    asset: Entity = env.scene[cfg.entity_name]
    local_ids, _ = asset.find_bodies((cfg.mocap_body_name,), preserve_order=True)
    global_body_id = asset.indexing.body_ids[local_ids[0]]
    self._mocap_id = int(env.sim.model.body_mocapid[global_body_id].item())
    if self._mocap_id < 0:
      raise ValueError(
        f"Body '{cfg.mocap_body_name}' is not a mocap body (body_mocapid < 0)."
      )

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._processed_actions[:, :3] = actions[:, :3] * self.cfg.pos_scale
    self._processed_actions[:, 3:] = actions[:, 3:] * self.cfg.rot_scale

  def apply_actions(self) -> None:
    pos_delta = self._processed_actions[:, :3]
    rot_delta = self._processed_actions[:, 3:]

    # Current mocap pose.
    cur_pos = self._env.sim.data.mocap_pos[:, self._mocap_id, :]  # (N, 3)
    cur_quat = self._env.sim.data.mocap_quat[:, self._mocap_id, :]  # (N, 4)

    # Apply position delta.
    new_pos = cur_pos + pos_delta

    # Apply orientation delta (Euler XYZ → quaternion, composed with current).
    delta_quat = quat_from_euler_xyz(rot_delta[:, 0], rot_delta[:, 1], rot_delta[:, 2])
    new_quat = quat_mul(delta_quat, cur_quat)

    # Write back.
    self._env.sim.data.mocap_pos[:, self._mocap_id, :] = new_pos
    self._env.sim.data.mocap_quat[:, self._mocap_id, :] = new_quat

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
