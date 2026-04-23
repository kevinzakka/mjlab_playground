"""Action terms for the getup task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.actions import (
  IntegratedJointPositionAction,
  IntegratedJointPositionActionCfg,
  RelativeJointPositionAction,
  RelativeJointPositionActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class SettleRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
  """RelativeJointPositionActionCfg that disables actions for the first N steps.

  Since the robot is dropped from a height in a random configuration, actions are
  suppressed until ``settle_steps`` env steps have passed, allowing the robot to land
  and settle before the policy takes over.
  """

  settle_steps: int = 0
  """Number of env steps after reset during which the policy action is ignored and the
  robot holds its current position. Set to 0 to disable."""

  def build(self, env: ManagerBasedRlEnv) -> SettleRelativeJointPositionAction:
    return SettleRelativeJointPositionAction(self, env)


class SettleRelativeJointPositionAction(RelativeJointPositionAction):
  """RelativeJointPositionAction that disables actions for the first N steps."""

  def __init__(
    self,
    cfg: SettleRelativeJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg=cfg, env=env)
    self._settle_steps = cfg.settle_steps

  def apply_actions(self) -> None:
    current_pos = self._entity.data.joint_pos[:, self._target_ids]
    current_pos_ref = current_pos
    if self._pos_noise is not None:
      current_pos_ref = current_pos_ref + self._pos_noise_sample
    target = current_pos_ref + self._processed_actions
    if self._settle_steps > 0:
      in_window = self._env.episode_length_buf < self._settle_steps
      was_fallen = self._env.extras.get("settle_mask", in_window)
      settling = (in_window & was_fallen).unsqueeze(-1)
      # During settle: command the motor to hold the true current pose so
      # actuator error (and thus force) is zero while the robot lands.
      target = torch.where(settling, current_pos, target)
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class SettleIntegratedJointPositionActionCfg(IntegratedJointPositionActionCfg):
  """IntegratedJointPositionActionCfg that disables actions for the first N steps.

  While the robot is settling after a random drop, the integrator state is
  held at the current measured joint position so the dispatched actuator
  command produces zero stiffness error (the joint is free to fall under
  gravity and damping). When the window closes, the integrator continues from
  the robot's actual resting pose.
  """

  settle_steps: int = 0
  """Number of env steps after reset during which the policy delta is ignored
  and the actuator holds the current measured joint position. Set to 0 to
  disable."""

  def build(self, env: ManagerBasedRlEnv) -> SettleIntegratedJointPositionAction:
    return SettleIntegratedJointPositionAction(self, env)


class SettleIntegratedJointPositionAction(IntegratedJointPositionAction):
  """IntegratedJointPositionAction with a post-reset settle window."""

  def __init__(
    self,
    cfg: SettleIntegratedJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg=cfg, env=env)
    self._settle_steps = cfg.settle_steps

  def apply_actions(self) -> None:
    if self._settle_steps > 0:
      in_window = self._env.episode_length_buf < self._settle_steps
      was_fallen = self._env.extras.get("settle_mask", in_window)
      settling = (in_window & was_fallen).unsqueeze(-1)
      measured = (
        self._entity.data.joint_pos[:, self._target_ids]
        + self._entity.data.encoder_bias[:, self._target_ids]
      )
      self._target = torch.where(settling, measured, self._target)
    super().apply_actions()
