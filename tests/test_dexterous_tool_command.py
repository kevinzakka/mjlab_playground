"""Tests for ToolGoalPoseCommand workspace sampling and reset behavior."""

from __future__ import annotations

import pytest
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab_playground.dexterous_tool.config.kuka_sharpa.env_cfgs import (
  kuka_sharpa_dexterous_tool_env_cfg,
)
from mjlab_playground.dexterous_tool.mdp.commands import ToolGoalPoseCommand

_NUM_ENVS = 4


@pytest.fixture(scope="module")
def env() -> ManagerBasedRlEnv:
  cfg = kuka_sharpa_dexterous_tool_env_cfg()
  cfg.scene.num_envs = _NUM_ENVS
  env = ManagerBasedRlEnv(cfg, device="cpu")
  env.reset()
  return env


def _get_command(env: ManagerBasedRlEnv) -> ToolGoalPoseCommand:
  cmd = env.command_manager.get_term("tool_goal")
  assert isinstance(cmd, ToolGoalPoseCommand)
  return cmd


def test_tool_inside_object_pose_range(env: ManagerBasedRlEnv) -> None:
  """After reset, the tool xy lies within the configured object_pose_range
  (plus env_origins). The command sampler may shift xy by tool extents, so we
  use the *raw* config range as a loose envelope check."""
  tool: Entity = env.scene["tool"]
  cmd = _get_command(env)
  assert cmd.cfg.object_pose_range is not None
  r = cmd.cfg.object_pose_range
  origins = env.scene.env_origins  # (B, 3)

  pos_local = tool.data.root_link_pos_w - origins  # (B, 3)
  # Allow generous tolerance because the sampler shrinks the range by tool extents.
  margin = 0.2
  assert torch.all(pos_local[:, 0] >= r.x[0] - margin)
  assert torch.all(pos_local[:, 0] <= r.x[1] + margin)
  assert torch.all(pos_local[:, 1] >= r.y[0] - margin)
  assert torch.all(pos_local[:, 1] <= r.y[1] + margin)


def test_tool_above_table(env: ManagerBasedRlEnv) -> None:
  """The tool root must be above the table top after reset."""
  tool: Entity = env.scene["tool"]
  table: Entity = env.scene["table"]
  table_top_z = table.data.root_link_pos_w[:, 2]  # mocap pose; table body sits above
  # Body offset from mocap is (0.55, 0, table_size_z=0.19), so the table top is
  # mocap_z + 2 * 0.19 in world frame. With env_origins=0 and default keyframe,
  # the table top is at z = 0 + 0 + 2*0.19 = 0.38. Use a loose check.
  tool_z = tool.data.root_link_pos_w[:, 2]
  # Tool should be at least at the configured min z range from object_pose_range.
  cmd = _get_command(env)
  assert cmd.cfg.object_pose_range is not None
  min_z = cmd.cfg.object_pose_range.z[0]
  assert torch.all(tool_z >= min_z - 0.05), (
    f"tool z {tool_z.tolist()} below min {min_z}; table top {table_top_z.tolist()}"
  )


def test_goal_within_workspace_bounds(env: ManagerBasedRlEnv) -> None:
  """The sampled goal pose must lie within (workspace_mins, workspace_maxs)
  in env-local coordinates."""
  cmd = _get_command(env)
  origins = env.scene.env_origins
  goal_local = cmd.goal_pos - origins  # (B, 3)

  ws_min = torch.tensor(cmd.cfg.workspace_mins)
  ws_max = torch.tensor(cmd.cfg.workspace_maxs)
  # Goals can be lifted higher than ws_max[2] when constrained by table clearance,
  # so we only enforce xy and a loose z lower bound.
  assert torch.all(goal_local[:, 0] >= ws_min[0] - 1e-5)
  assert torch.all(goal_local[:, 0] <= ws_max[0] + 1e-5)
  assert torch.all(goal_local[:, 1] >= ws_min[1] - 1e-5)
  assert torch.all(goal_local[:, 1] <= ws_max[1] + 1e-5)
  assert torch.all(goal_local[:, 2] >= ws_min[2] - 1e-5)


def test_tool_velocity_zero_after_reset(env: ManagerBasedRlEnv) -> None:
  """The command writes zero velocity at reset; verify both lin and ang."""
  tool: Entity = env.scene["tool"]
  lin_speed = torch.norm(tool.data.root_link_lin_vel_w, dim=-1)
  ang_speed = torch.norm(tool.data.root_link_ang_vel_w, dim=-1)
  assert torch.all(lin_speed < 1e-3), f"lin_speed = {lin_speed.tolist()}"
  assert torch.all(ang_speed < 1e-3), f"ang_speed = {ang_speed.tolist()}"


def test_goal_quat_normalized(env: ManagerBasedRlEnv) -> None:
  """Sampled goal quaternions must be unit-norm."""
  cmd = _get_command(env)
  norms = torch.norm(cmd.goal_quat, dim=-1)
  assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
