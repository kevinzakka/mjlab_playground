"""Task-local action terms for dexterous tool manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DeltaJointPositionActionCfg(BaseActionCfg):
  """Joint position control relative to the previous commanded target.

  The commanded target is updated once per policy step as:

  `target_next = clamp(target_prev + action * scale)`

  The stored target is then re-applied on every physics substep until the next
  policy action arrives.
  """

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT
    if self.offset != 0.0:
      raise ValueError(
        "DeltaJointPositionActionCfg does not support 'offset'. "
        "The target is previous_target + action * scale."
      )

  def build(self, env: ManagerBasedRlEnv) -> DeltaJointPositionAction:
    return DeltaJointPositionAction(self, env)


class DeltaJointPositionAction(BaseAction):
  """Control joints via deltas on the previous commanded target."""

  def __init__(self, cfg: BaseActionCfg, env: ManagerBasedRlEnv):
    if not isinstance(cfg, DeltaJointPositionActionCfg):
      raise TypeError("Expected DeltaJointPositionActionCfg.")
    super().__init__(cfg=cfg, env=env)
    self._target_command = self._entity.data.joint_pos[:, self._target_ids].clone()
    joint_limits = self._entity.data.joint_pos_limits[:, self._target_ids]
    self._lower = joint_limits[..., 0]
    self._upper = joint_limits[..., 1]

  @property
  def target_command(self) -> torch.Tensor:
    """Most recent commanded joint target."""
    return self._target_command

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    self._target_command = torch.clamp(
      self._target_command + self._processed_actions,
      min=self._lower,
      max=self._upper,
    )

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(
      self._target_command, joint_ids=self._target_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids=env_ids)
    self._target_command[env_ids] = self._entity.data.joint_pos[env_ids][
      :, self._target_ids
    ]
