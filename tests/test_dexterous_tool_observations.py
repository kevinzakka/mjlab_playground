"""Unit tests for the palm-anchored pose observations.

All spatial observations live in the palm frame: ``fingertip_pos_in_palm``,
``tool_pose_in_palm``, ``goal_pose_in_palm``. ``goal_pose_in_tool`` is the
redundant SE(3) error feature in the tool frame. These tests verify identity,
non-trivial geometry, and global transform invariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_from_angle_axis,
  quat_inv,
  quat_mul,
)
from mjlab_playground.dexterous_tool.mdp.observations import (
  goal_pose_in_palm,
  goal_pose_in_tool,
  tool_pose_in_palm,
)

# Stub env exposing only what the observation functions read.


@dataclass
class _StubData:
  root_link_pos_w: torch.Tensor
  root_link_quat_w: torch.Tensor
  site_pos_w: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, 3))
  site_quat_w: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, 4))


@dataclass
class _StubEntity:
  data: _StubData

  def find_sites(self, names):
    # Single-site entity; always return id 0.
    ids = torch.zeros(len(names), dtype=torch.long)
    return ids, list(names)


@dataclass
class _StubCommand:
  goal_pos: torch.Tensor
  goal_quat: torch.Tensor


class _StubScene:
  def __init__(self, tool: _StubEntity, robot: _StubEntity):
    self._tool = tool
    self._robot = robot

  def __getitem__(self, key: str) -> _StubEntity:
    if key == "tool":
      return self._tool
    if key == "robot":
      return self._robot
    raise KeyError(key)


class _StubCommandManager:
  def __init__(self, command: _StubCommand):
    self._command = command

  def get_term(self, name: str) -> _StubCommand:
    assert name == "tool_goal"
    return self._command


@dataclass
class _StubEnv:
  scene: _StubScene
  command_manager: _StubCommandManager
  num_envs: int = 1


def _make_env(
  *,
  tool_pos: torch.Tensor,
  tool_quat: torch.Tensor,
  goal_pos: torch.Tensor,
  goal_quat: torch.Tensor,
  palm_pos: torch.Tensor | None = None,
  palm_quat: torch.Tensor | None = None,
) -> ManagerBasedRlEnv:
  """Build a stub env. Palm defaults to identity pose at origin."""
  palm_pos = torch.zeros_like(tool_pos) if palm_pos is None else palm_pos
  palm_quat = (
    torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(tool_pos.shape[0], 4)
    if palm_quat is None
    else palm_quat
  )
  tool = _StubEntity(data=_StubData(tool_pos, tool_quat))
  robot = _StubEntity(
    data=_StubData(
      root_link_pos_w=torch.zeros_like(tool_pos),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(
        tool_pos.shape[0], 4
      ),
      site_pos_w=palm_pos.unsqueeze(1),  # (B, 1, 3)
      site_quat_w=palm_quat.unsqueeze(1),  # (B, 1, 4)
    )
  )
  command = _StubCommand(goal_pos=goal_pos, goal_quat=goal_quat)
  stub = _StubEnv(
    scene=_StubScene(tool, robot),
    command_manager=_StubCommandManager(command),
    num_envs=int(tool_pos.shape[0]),
  )
  return cast(ManagerBasedRlEnv, stub)


@pytest.fixture(autouse=True)
def _bypass_command_typecheck(monkeypatch):
  """Stub command isn't a real ToolGoalPoseCommand; patch the isinstance check."""
  from mjlab_playground.dexterous_tool.mdp import observations as obs_module

  monkeypatch.setattr(obs_module, "ToolGoalPoseCommand", _StubCommand)


# Stub for the asset_cfg arg of the palm-frame functions. The functions only
# read ``asset_cfg.name``; site lookup goes through the entity's find_sites stub.
@dataclass
class _StubAssetCfg:
  name: str = "robot"


_IDENTITY = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
_IDENTITY_9D = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])


def _quat_axis_angle(axis: tuple[float, float, float], angle: float) -> torch.Tensor:
  ax = torch.tensor([list(axis)], dtype=torch.float)
  return quat_from_angle_axis(torch.tensor([angle]), ax)


# tool_pose_in_palm: identity & non-trivial.


def test_tool_pose_in_palm_identity_when_palm_eq_tool():
  pos = torch.tensor([[0.5, -0.2, 0.4]])
  q = _quat_axis_angle((0.0, 1.0, 0.0), math.pi / 5)
  env = _make_env(
    tool_pos=pos, tool_quat=q, goal_pos=pos, goal_quat=q, palm_pos=pos, palm_quat=q
  )
  out = tool_pose_in_palm(env, _StubAssetCfg())  # type: ignore[arg-type]
  assert torch.allclose(out, _IDENTITY_9D, atol=1e-6)


def test_tool_pose_in_palm_translation_only():
  """Tool 0.1 m above palm in world; palm identity → expect (0, 0, 0.1) + identity 6D."""
  palm_pos = torch.zeros(1, 3)
  tool_pos = torch.tensor([[0.0, 0.0, 0.1]])
  env = _make_env(
    tool_pos=tool_pos,
    tool_quat=_IDENTITY,
    goal_pos=tool_pos,
    goal_quat=_IDENTITY,
    palm_pos=palm_pos,
    palm_quat=_IDENTITY,
  )
  out = tool_pose_in_palm(env, _StubAssetCfg())  # type: ignore[arg-type]
  expected = torch.tensor([[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
  assert torch.allclose(out, expected, atol=1e-6)


def test_tool_pose_in_palm_rotated_palm():
  """Palm yawed +90° around z; tool at world +x → tool in palm frame is local -y."""
  palm_pos = torch.zeros(1, 3)
  palm_quat = _quat_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
  tool_pos = torch.tensor([[1.0, 0.0, 0.0]])
  env = _make_env(
    tool_pos=tool_pos,
    tool_quat=_IDENTITY,
    goal_pos=tool_pos,
    goal_quat=_IDENTITY,
    palm_pos=palm_pos,
    palm_quat=palm_quat,
  )
  out = tool_pose_in_palm(env, _StubAssetCfg())  # type: ignore[arg-type]
  # Position in palm frame: (0, -1, 0).
  assert torch.allclose(out[:, 0:3], torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6)


# goal_pose_in_palm: identity & invariance.


def test_goal_pose_in_palm_identity():
  pos = torch.zeros(1, 3)
  env = _make_env(
    tool_pos=pos,
    tool_quat=_IDENTITY,
    goal_pos=pos,
    goal_quat=_IDENTITY,
    palm_pos=pos,
    palm_quat=_IDENTITY,
  )
  out = goal_pose_in_palm(env, "tool_goal", _StubAssetCfg())  # type: ignore[arg-type]
  assert torch.allclose(out, _IDENTITY_9D, atol=1e-6)


def test_goal_pose_in_palm_invariant_under_global_rotation():
  """Rotating the entire scene (palm + tool + goal) must not change the obs."""
  palm_pos = torch.tensor([[0.1, 0.0, 0.2]])
  goal_pos = torch.tensor([[0.4, -0.1, 0.5]])
  palm_quat = _quat_axis_angle((0.0, 0.0, 1.0), 0.3)
  goal_quat = _quat_axis_angle((1.0, 0.0, 0.0), 0.4)

  env_a = _make_env(
    tool_pos=palm_pos,
    tool_quat=_IDENTITY,
    goal_pos=goal_pos,
    goal_quat=goal_quat,
    palm_pos=palm_pos,
    palm_quat=palm_quat,
  )
  out_a = goal_pose_in_palm(env_a, "tool_goal", _StubAssetCfg())  # type: ignore[arg-type]

  R = _quat_axis_angle((0.5, 0.3, 0.8), 1.1)
  palm_pos_b = quat_apply(R, palm_pos)
  goal_pos_b = quat_apply(R, goal_pos)
  palm_quat_b = quat_mul(R, palm_quat)
  goal_quat_b = quat_mul(R, goal_quat)

  env_b = _make_env(
    tool_pos=palm_pos_b,
    tool_quat=_IDENTITY,
    goal_pos=goal_pos_b,
    goal_quat=goal_quat_b,
    palm_pos=palm_pos_b,
    palm_quat=palm_quat_b,
  )
  out_b = goal_pose_in_palm(env_b, "tool_goal", _StubAssetCfg())  # type: ignore[arg-type]
  assert torch.allclose(out_a, out_b, atol=1e-5)


# goal_pose_in_tool: identity, 180°, double cover, invariance.


def test_goal_pose_in_tool_identity():
  pos = torch.zeros(1, 3)
  env = _make_env(tool_pos=pos, tool_quat=_IDENTITY, goal_pos=pos, goal_quat=_IDENTITY)
  out = goal_pose_in_tool(env, "tool_goal")
  assert torch.allclose(out, _IDENTITY_9D, atol=1e-6)


def test_goal_pose_in_tool_180deg():
  """Tool at identity, goal 180° around x → ori_6d = [1,0,0, 0,-1,0]."""
  pos = torch.zeros(1, 3)
  q_goal = _quat_axis_angle((1.0, 0.0, 0.0), math.pi)
  env = _make_env(tool_pos=pos, tool_quat=_IDENTITY, goal_pos=pos, goal_quat=q_goal)
  out = goal_pose_in_tool(env, "tool_goal")
  expected_ori = torch.tensor([[1.0, 0.0, 0.0, 0.0, -1.0, 0.0]])
  assert torch.allclose(out[:, 3:9], expected_ori, atol=1e-5)


def test_goal_pose_in_tool_double_cover_obj():
  pos = torch.zeros(1, 3)
  q = _quat_axis_angle((0.0, 1.0, 0.0), math.pi / 3)
  goal_q = _quat_axis_angle((0.0, 0.0, 1.0), math.pi / 4)
  env_pos = _make_env(tool_pos=pos, tool_quat=q, goal_pos=pos, goal_quat=goal_q)
  env_neg = _make_env(tool_pos=pos, tool_quat=-q, goal_pos=pos, goal_quat=goal_q)
  out_pos = goal_pose_in_tool(env_pos, "tool_goal")
  out_neg = goal_pose_in_tool(env_neg, "tool_goal")
  assert torch.allclose(out_pos, out_neg, atol=1e-5)


def test_goal_pose_in_tool_double_cover_goal():
  pos = torch.zeros(1, 3)
  q = _quat_axis_angle((0.0, 1.0, 0.0), math.pi / 3)
  goal_q = _quat_axis_angle((0.0, 0.0, 1.0), math.pi / 4)
  env_pos = _make_env(tool_pos=pos, tool_quat=q, goal_pos=pos, goal_quat=goal_q)
  env_neg = _make_env(tool_pos=pos, tool_quat=q, goal_pos=pos, goal_quat=-goal_q)
  out_pos = goal_pose_in_tool(env_pos, "tool_goal")
  out_neg = goal_pose_in_tool(env_neg, "tool_goal")
  assert torch.allclose(out_pos, out_neg, atol=1e-5)


def test_goal_pose_in_tool_invariant_under_global_rotation():
  pos = torch.tensor([[0.1, 0.2, 0.3]])
  obj_quat = _quat_axis_angle((0.0, 0.0, 1.0), 0.3)
  goal_quat = _quat_axis_angle((1.0, 0.0, 0.0), 0.4)
  env_a = _make_env(tool_pos=pos, tool_quat=obj_quat, goal_pos=pos, goal_quat=goal_quat)
  out_a = goal_pose_in_tool(env_a, "tool_goal")

  R = _quat_axis_angle((0.5, 0.3, 0.8), 1.1)
  pos_b = quat_apply(R, pos)
  obj_quat_b = quat_mul(R, obj_quat)
  goal_quat_b = quat_mul(R, goal_quat)
  env_b = _make_env(
    tool_pos=pos_b, tool_quat=obj_quat_b, goal_pos=pos_b, goal_quat=goal_quat_b
  )
  out_b = goal_pose_in_tool(env_b, "tool_goal")
  assert torch.allclose(out_a, out_b, atol=1e-5)


# Cross-frame consistency: when palm == tool (zero grasp offset), the tool-frame
# and palm-frame goal observations must agree.


def test_palm_and_tool_frames_agree_when_palm_eq_tool():
  pos = torch.tensor([[0.2, -0.1, 0.4]])
  q = _quat_axis_angle((0.0, 1.0, 0.0), 0.5)
  goal_pos = torch.tensor([[0.5, 0.1, 0.3]])
  goal_q = _quat_axis_angle((0.0, 0.0, 1.0), 0.7)

  env = _make_env(
    tool_pos=pos,
    tool_quat=q,
    goal_pos=goal_pos,
    goal_quat=goal_q,
    palm_pos=pos,
    palm_quat=q,
  )
  out_palm = goal_pose_in_palm(env, "tool_goal", _StubAssetCfg())  # type: ignore[arg-type]
  out_tool = goal_pose_in_tool(env, "tool_goal")
  assert torch.allclose(out_palm, out_tool, atol=1e-5)


# 6D rep validity: rows unit-norm and orthogonal.


def test_tool_pose_6d_is_valid_rotation_matrix():
  pos = torch.zeros(1, 3)
  q_tool = _quat_axis_angle((0.3, 0.5, 0.8), 0.7)
  env = _make_env(
    tool_pos=pos,
    tool_quat=q_tool,
    goal_pos=pos,
    goal_quat=_IDENTITY,
    palm_pos=pos,
    palm_quat=_IDENTITY,
  )
  out = tool_pose_in_palm(env, _StubAssetCfg())  # type: ignore[arg-type]
  row1 = out[:, 3:6]
  row2 = out[:, 6:9]
  assert torch.allclose(torch.norm(row1, dim=-1), torch.ones(1), atol=1e-5)
  assert torch.allclose(torch.norm(row2, dim=-1), torch.ones(1), atol=1e-5)
  dot = (row1 * row2).sum(dim=-1)
  assert torch.allclose(dot, torch.zeros(1), atol=1e-5)


def test_goal_pose_in_tool_6d_recovers_full_matrix():
  """Reconstructing the 3rd row from cross product matches matrix_from_quat."""
  pos = torch.zeros(1, 3)
  q_obj = _quat_axis_angle((0.3, 0.5, 0.8), 0.7)
  q_goal = _quat_axis_angle((0.9, 0.2, 0.4), 1.3)
  env = _make_env(tool_pos=pos, tool_quat=q_obj, goal_pos=pos, goal_quat=q_goal)
  out = goal_pose_in_tool(env, "tool_goal")
  row1, row2 = out[:, 3:6], out[:, 6:9]
  row3 = torch.cross(row1, row2, dim=-1)
  reconstructed = torch.stack([row1, row2, row3], dim=1)
  q12 = quat_mul(quat_inv(q_obj), q_goal)
  expected = matrix_from_quat(q12)
  assert torch.allclose(reconstructed, expected, atol=1e-5)
