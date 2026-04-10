"""Curriculum functions for dexterous tool manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from .commands import ToolGoalPoseCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _apply_command_stages(
  cfg: Any,
  step_counter: int,
  stages: Sequence[dict],
) -> dict[str, torch.Tensor]:
  """Apply staged updates to a command cfg and return a logging snapshot."""
  logged_keys: set[str] = set()
  for stage in stages:
    if step_counter >= stage["step"]:
      for key, value in stage.items():
        if key != "step":
          setattr(cfg, key, value)
          logged_keys.add(key)
    else:
      for key in stage:
        if key != "step":
          logged_keys.add(key)
  result: dict[str, torch.Tensor] = {}
  for key in logged_keys:
    value = getattr(cfg, key)
    if isinstance(value, (int, float)):
      result[key] = torch.tensor(value)
  return result


class command_tolerance_curriculum:
  """Tighten a ToolGoalPoseCommand's success tolerances over training.

  Each stage specifies a ``step`` threshold and one or both of
  ``pos_tolerance`` (meters) and ``ori_tolerance`` (radians).  When
  ``env.common_step_counter`` reaches a stage's ``step``, the
  corresponding values are applied to the live command config.

  Example::

    CurriculumTermCfg(
      func=dex_mdp.command_tolerance_curriculum,
      params={
        "command_name": "tool_goal",
        "stages": [
          {"step": 0, "pos_tolerance": 0.10, "ori_tolerance": math.radians(45)},
          {"step": 3000 * 24, "pos_tolerance": 0.05, "ori_tolerance": math.radians(25)},
          {"step": 10000 * 24, "pos_tolerance": 0.025, "ori_tolerance": math.radians(15)},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    command_name: str = cfg.params["command_name"]
    stages: list[dict] = cfg.params["stages"]
    command_term = env.command_manager.get_term(command_name)
    if not isinstance(command_term, ToolGoalPoseCommand):
      raise TypeError(f"Expected ToolGoalPoseCommand, got {type(command_term)}")
    self._cfg = command_term.cfg
    self._stages = stages
    # Validate that stage keys are real fields on the command cfg.
    for stage in stages:
      for key in stage:
        if key != "step" and not hasattr(self._cfg, key):
          raise AttributeError(
            f"Field '{key}' does not exist on ToolGoalPoseCommandCfg."
          )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, stages
    return _apply_command_stages(self._cfg, env.common_step_counter, self._stages)
