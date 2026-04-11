"""Unit tests for dexterous tool task reward."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_playground.dexterous_tool.mdp.rewards import task_reward


@dataclass
class _StubData:
  root_link_pos_w: torch.Tensor
  root_link_quat_w: torch.Tensor
  site_pos_w: torch.Tensor


@dataclass
class _StubEntity:
  data: _StubData


@dataclass
class _StubCommand:
  goal_pos: torch.Tensor
  goal_quat: torch.Tensor
  grasp_pos_w: torch.Tensor


class _StubScene:
  def __init__(self, robot: _StubEntity, tool: _StubEntity, num_envs: int = 1):
    self._robot = robot
    self._tool = tool
    self.env_origins = torch.zeros(num_envs, 3)

  def __getitem__(self, key: str):
    if key == "robot":
      return self._robot
    if key == "tool":
      return self._tool
    raise KeyError(key)


class _StubCommandManager:
  def __init__(self, command: _StubCommand):
    self._command = command

  def get_term(self, _name: str) -> _StubCommand:
    return self._command


@dataclass
class _StubEnv:
  scene: _StubScene
  command_manager: _StubCommandManager


_IDENTITY_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
_FLIP_X_180 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

_ASSET_CFG = SceneEntityCfg("robot", site_names=("fingertip",))
_ASSET_CFG.site_ids = slice(0, 1)

_DEFAULTS = {
  "approach_std": 0.1,
  "position_std": 0.1,
  "orientation_std": math.radians(30.0),
}


@pytest.fixture(autouse=True)
def _bypass_command_typecheck(monkeypatch):
  from mjlab_playground.dexterous_tool.mdp import rewards as m

  monkeypatch.setattr(m, "ToolGoalPoseCommand", _StubCommand)


def _make_env(
  obj_pos: torch.Tensor,
  obj_quat: torch.Tensor,
  goal_pos: torch.Tensor,
  goal_quat: torch.Tensor,
  fingertip_pos: torch.Tensor | None = None,
  grasp_pos: torch.Tensor | None = None,
) -> ManagerBasedRlEnv:
  num_envs = int(obj_pos.shape[0])
  if fingertip_pos is None:
    fingertip_pos = obj_pos.clone()
  if grasp_pos is None:
    grasp_pos = obj_pos.clone()
  robot = _StubEntity(
    data=_StubData(
      root_link_pos_w=torch.zeros(num_envs, 3),
      root_link_quat_w=_IDENTITY_QUAT.repeat(num_envs, 1),
      site_pos_w=fingertip_pos.unsqueeze(1),
    )
  )
  tool = _StubEntity(
    data=_StubData(
      root_link_pos_w=obj_pos,
      root_link_quat_w=obj_quat,
      site_pos_w=torch.zeros(num_envs, 0, 3),
    )
  )
  command = _StubCommand(goal_pos=goal_pos, goal_quat=goal_quat, grasp_pos_w=grasp_pos)
  stub = _StubEnv(
    scene=_StubScene(robot, tool, num_envs=num_envs),
    command_manager=_StubCommandManager(command),
  )
  return cast(ManagerBasedRlEnv, stub)


def _call(env: ManagerBasedRlEnv, **overrides) -> torch.Tensor:
  params = {**_DEFAULTS, **overrides}
  return task_reward(env, command_name="tool_goal", asset_cfg=_ASSET_CFG, **params)


def test_perfect_alignment():
  """Fingers on grasp, at goal, perfect ori → reward = 1.0."""
  pos = torch.tensor([[0.3, 0.0, 0.5]])
  env = _make_env(pos, _IDENTITY_QUAT, pos.clone(), _IDENTITY_QUAT)
  # approach=1, pos=1, ori=1 → 1*(1+1+1)/3 = 1
  assert _call(env).item() == pytest.approx(1.0, abs=1e-6)


def test_far_from_grasp():
  """Fingers far from tool → approach ≈ 0, whole reward ≈ 0."""
  obj_pos = torch.tensor([[0.3, 0.0, 0.5]])
  fingertip_pos = torch.tensor([[1.0, 0.0, 0.5]])
  env = _make_env(
    obj_pos, _IDENTITY_QUAT, obj_pos.clone(), _IDENTITY_QUAT, fingertip_pos
  )
  assert _call(env).item() < 0.01


def test_grasped_but_far_from_goal():
  """Fingers on grasp, but tool far from goal → pos/ori contribute little."""
  obj_pos = torch.tensor([[0.0, 0.0, 0.5]])
  goal_pos = torch.tensor([[1.0, 0.0, 0.5]])
  env = _make_env(obj_pos, _IDENTITY_QUAT, goal_pos, _IDENTITY_QUAT)
  r = _call(env).item()
  # approach=1, pos≈0, ori=1 → 1*(1+0+1)/3 ≈ 0.67
  assert 0.5 < r < 0.8


def test_approach_gates_tracking():
  """Even with perfect pos+ori, poor approach kills the reward."""
  obj_pos = torch.tensor([[0.3, 0.0, 0.5]])
  fingertip_pos = torch.tensor([[0.8, 0.0, 0.5]])  # far from grasp
  env = _make_env(
    obj_pos, _IDENTITY_QUAT, obj_pos.clone(), _IDENTITY_QUAT, fingertip_pos
  )
  r_far = _call(env).item()
  fingertip_pos2 = obj_pos.clone()
  env2 = _make_env(
    obj_pos, _IDENTITY_QUAT, obj_pos.clone(), _IDENTITY_QUAT, fingertip_pos2
  )
  r_close = _call(env2).item()
  assert r_close > r_far * 2


def test_orientation_180():
  """180° misalignment, but grasped and at goal pos → partial reward."""
  pos = torch.tensor([[0.3, 0.0, 0.5]])
  env = _make_env(pos, _IDENTITY_QUAT, pos.clone(), _FLIP_X_180)
  r = _call(env).item()
  # approach=1, pos=1, ori≈0 → 1*(1+1+0)/3 ≈ 0.67
  assert 0.5 < r < 0.8


def test_double_cover():
  """q and -q represent the same rotation; reward must be invariant."""
  pos = torch.tensor([[0.3, 0.0, 0.5]])
  q = torch.tensor([[0.7071, 0.0, 0.7071, 0.0]])
  r_pos = _call(_make_env(pos, q, pos.clone(), q))
  r_neg = _call(_make_env(pos, -q, pos.clone(), q))
  assert torch.allclose(r_pos, r_neg, atol=1e-5)


def test_bounded_zero_one():
  """Reward must lie in [0, 1] across a sweep of configurations."""
  for err in (0.0, 0.05, 0.1, 0.5, 1.0):
    for finger_dist in (0.0, 0.1, 0.5):
      obj_pos = torch.tensor([[err, 0.0, 0.5]])
      fingertip_pos = torch.tensor([[finger_dist, 0.0, 0.5]])
      for ori in (_IDENTITY_QUAT, _FLIP_X_180):
        env = _make_env(obj_pos, _IDENTITY_QUAT, torch.zeros(1, 3), ori, fingertip_pos)
        r = _call(env).item()
        assert 0.0 <= r <= 1.0, f"reward {r} out of [0,1]"
