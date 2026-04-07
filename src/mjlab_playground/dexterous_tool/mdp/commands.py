"""Sequential 6-DoF goal pose command for dexterous tool manipulation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
)

from .events import tool_axis_bounds, tool_support_height

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))
_CURRENT_FRAME_COLORS = ((1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.4, 1.0))


class ToolGoalPoseCommand(CommandTerm):
  """Samples sequential 6-DoF goal poses for tool manipulation.

  The first goal is sampled uniformly in the workspace. Subsequent goals are sampled as
  deltas from the previous goal. Success is measured by keypoint distance between the
  object's current pose and the goal pose.
  """

  cfg: ToolGoalPoseCommandCfg

  def __init__(self, cfg: ToolGoalPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.tool: Entity = env.scene[cfg.entity_name]
    self._ghost_model = None
    self._ghost_color = np.array(cfg.ghost_color, dtype=np.float32)

    # Goal state.
    self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self.goal_quat[:, 0] = 1.0  # Identity quaternion (w,x,y,z).

    # Keypoints: use the actual tool sites so rewards/debug viz match the asset.
    self._keypoint_site_ids, _ = self.tool.find_sites(
      cfg.keypoint_site_names, preserve_order=True
    )
    self._grasp_site_ids, _ = self.tool.find_sites(
      (cfg.grasp_site_name,), preserve_order=True
    )
    self._global_keypoint_site_ids = self.tool.indexing.site_ids[
      self._keypoint_site_ids
    ]
    site_body_ids = torch.as_tensor(
      env.sim.mj_model.site_bodyid,
      device=self.device,
      dtype=torch.int32,
    )[self._global_keypoint_site_ids]
    if not all(
      int(body_id) == self.tool.indexing.root_body_id for body_id in site_body_ids
    ):
      raise ValueError(
        "Tool goal keypoint sites must be attached to the tool root body."
      )
    self._support_geom_ids, _ = self.tool.find_geoms(
      cfg.support_geom_names, preserve_order=True
    )
    self._global_support_geom_ids = self.tool.indexing.geom_ids[self._support_geom_ids]
    self._support_geom_types = torch.as_tensor(
      env.sim.mj_model.geom_type,
      device=self.device,
      dtype=torch.int32,
    )[self._global_support_geom_ids]
    self._default_support_geom_size = env.sim.get_default_field("geom_size")[
      self._global_support_geom_ids
    ]
    self._default_support_geom_pos = env.sim.get_default_field("geom_pos")[
      self._global_support_geom_ids
    ]
    self._footprint_table: Entity | None = None
    self._global_footprint_site_ids: torch.Tensor | None = None
    self._footprint_site_body_id: int | None = None
    self._footprint_table_top_local_z: float | None = None
    if cfg.footprint_entity_name and cfg.footprint_site_names:
      table: Entity = env.scene[cfg.footprint_entity_name]
      self._footprint_table = table
      footprint_site_ids, _ = table.find_sites(
        cfg.footprint_site_names, preserve_order=True
      )
      global_footprint_site_ids = table.indexing.site_ids[footprint_site_ids]
      self._global_footprint_site_ids = global_footprint_site_ids
      footprint_site_body_ids = torch.as_tensor(
        env.sim.mj_model.site_bodyid,
        device=self.device,
        dtype=torch.int32,
      )[global_footprint_site_ids]
      self._footprint_site_body_id = int(footprint_site_body_ids[0])
      table_geom_ids, _ = table.find_geoms(
        (cfg.footprint_table_geom_name,), preserve_order=True
      )
      table_geom_id = int(table.indexing.geom_ids[table_geom_ids].item())
      table_geom_size = env.sim.get_default_field("geom_size")[table_geom_id]
      table_geom_pos = env.sim.get_default_field("geom_pos")[table_geom_id]
      self._footprint_table_top_local_z = float(
        table_geom_pos[2] + table_geom_size[2] + cfg.footprint_site_height_offset
      )

    # Object and goal keypoints in world frame.
    self.object_keypoints_w = torch.zeros(self.num_envs, 4, 3, device=self.device)
    self.goal_keypoints_w = torch.zeros(self.num_envs, 4, 3, device=self.device)
    self.object_initial_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

    # Stateful reward trackers.
    self.lifted_object = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.min_fingertip_dists = torch.full(
      (self.num_envs, cfg.num_fingertips), float("inf"), device=self.device
    )
    self.min_keypoint_max_dist = torch.full(
      (self.num_envs,), float("inf"), device=self.device
    )
    self.consecutive_successes = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.num_goal_resets = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Workspace bounds (relative to env origins).
    self._ws_min = torch.tensor(cfg.workspace_mins, device=self.device)
    self._ws_max = torch.tensor(cfg.workspace_maxs, device=self.device)

    # Metrics.
    self.metrics["max_keypoint_dist"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["lifted_object"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["consecutive_successes"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["num_goal_resets"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Keypoint errors: goal_keypoints - object_keypoints, flattened to (B, 12)."""
    return (self.goal_keypoints_w - self.object_keypoints_w).reshape(self.num_envs, -1)

  @property
  def grasp_pos_w(self) -> torch.Tensor:
    """Current world-frame grasp-site position. Shape: (B, 3)."""
    return self.tool.data.site_pos_w[:, self._grasp_site_ids[0]]

  def _site_pos_local(
    self, site_ids: torch.Tensor, env_ids: torch.Tensor
  ) -> torch.Tensor:
    site_pos = self._env.sim.model.site_pos
    if site_pos.ndim == 2:
      return site_pos[site_ids].unsqueeze(0).expand(len(env_ids), -1, -1)
    return site_pos[env_ids[:, None], site_ids[None, :]]

  def _geom_field_local(
    self, field: str, geom_ids: torch.Tensor, env_ids: torch.Tensor
  ) -> torch.Tensor:
    data = getattr(self._env.sim.model, field)
    if data.ndim == 2:
      return data[geom_ids].unsqueeze(0).expand(len(env_ids), -1, -1)
    return data[env_ids[:, None], geom_ids[None, :]]

  def _body_field_local(
    self, field: str, body_id: int, env_ids: torch.Tensor
  ) -> torch.Tensor:
    data = getattr(self._env.sim.model, field)
    if data.ndim == 2:
      return data[body_id].unsqueeze(0).expand(len(env_ids), -1)
    return data[env_ids, body_id]

  def _compute_keypoints(
    self,
    pos: torch.Tensor,
    quat: torch.Tensor,
    env_ids: torch.Tensor,
  ) -> torch.Tensor:
    """Compute 4 world-frame keypoints from position and quaternion.

    Args:
      pos: (B, 3) position.
      quat: (B, 4) quaternion (w,x,y,z).
      env_ids: Environment indices matching pos and quat.

    Returns:
      (B, 4, 3) world-frame keypoints.
    """
    offsets = self._site_pos_local(self._global_keypoint_site_ids, env_ids)
    rotated = quat_apply(
      quat.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4),
      offsets.reshape(-1, 3),
    ).reshape(pos.shape[0], 4, 3)
    return rotated + pos.unsqueeze(1)

  def _table_top_world_z(self, env_ids: torch.Tensor) -> torch.Tensor | None:
    if (
      self._footprint_site_body_id is None or self._footprint_table_top_local_z is None
    ):
      return None
    table_pos = self._body_field_local(
      "body_pos", self._footprint_site_body_id, env_ids
    )
    table_quat = self._body_field_local(
      "body_quat", self._footprint_site_body_id, env_ids
    )
    top_local = torch.zeros(len(env_ids), 3, device=self.device)
    top_local[:, 2] = self._footprint_table_top_local_z
    top_world = table_pos + quat_apply(table_quat, top_local)
    return top_world[:, 2]

  def _workspace_root_bounds(
    self,
    quat: torch.Tensor,
    geom_size: torch.Tensor,
    geom_pos: torch.Tensor,
    env_ids: torch.Tensor,
    ws_min: torch.Tensor,
    ws_max: torch.Tensor,
    constrain_xy_by_tool_extents: bool,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    min_z, max_z = tool_axis_bounds(
      geom_size, geom_pos, self._support_geom_types, quat, axis=2
    )
    lower = ws_min.unsqueeze(0).expand(len(env_ids), -1).clone()
    upper = ws_max.unsqueeze(0).expand(len(env_ids), -1).clone()

    if constrain_xy_by_tool_extents:
      min_x, max_x = tool_axis_bounds(
        geom_size, geom_pos, self._support_geom_types, quat, axis=0
      )
      min_y, max_y = tool_axis_bounds(
        geom_size, geom_pos, self._support_geom_types, quat, axis=1
      )
      lower[:, 0] = ws_min[0] - min_x
      upper[:, 0] = ws_max[0] - max_x
      lower[:, 1] = ws_min[1] - min_y
      upper[:, 1] = ws_max[1] - max_y

    table_top_z = self._table_top_world_z(env_ids)
    if table_top_z is None:
      lower[:, 2] = ws_min[2] - min_z
    else:
      lower[:, 2] = torch.maximum(
        ws_min[2].expand(len(env_ids)),
        table_top_z + self.cfg.goal_table_clearance - min_z,
      )
    upper[:, 2] = torch.maximum(ws_max[2].expand(len(env_ids)), lower[:, 2])
    return lower, upper

  def _sample_root_position_in_workspace(
    self,
    quat: torch.Tensor,
    geom_size: torch.Tensor,
    geom_pos: torch.Tensor,
    env_ids: torch.Tensor,
    ws_min: torch.Tensor,
    ws_max: torch.Tensor,
    origins: torch.Tensor,
    constrain_xy_by_tool_extents: bool,
  ) -> torch.Tensor:
    lo, hi = self._workspace_root_bounds(
      quat,
      geom_size,
      geom_pos,
      env_ids,
      ws_min,
      ws_max,
      constrain_xy_by_tool_extents,
    )
    sampled = sample_uniform(lo, hi, lo.shape, device=self.device)
    midpoint = 0.5 * (lo + hi)
    fallback = midpoint.clone()
    fallback[:, 2] = lo[:, 2]
    valid = lo <= hi
    return torch.where(valid, sampled, fallback) + origins

  def _clamp_root_position_to_workspace(
    self,
    pos_w: torch.Tensor,
    quat: torch.Tensor,
    geom_size: torch.Tensor,
    geom_pos: torch.Tensor,
    env_ids: torch.Tensor,
    ws_min: torch.Tensor,
    ws_max: torch.Tensor,
    origins: torch.Tensor,
    constrain_xy_by_tool_extents: bool,
  ) -> torch.Tensor:
    lo, hi = self._workspace_root_bounds(
      quat,
      geom_size,
      geom_pos,
      env_ids,
      ws_min,
      ws_max,
      constrain_xy_by_tool_extents,
    )
    pos_rel = pos_w - origins
    midpoint = 0.5 * (lo + hi)
    fallback = midpoint.clone()
    fallback[:, 2] = lo[:, 2]
    valid = lo <= hi
    clamped = torch.clamp(pos_rel, lo, hi)
    return torch.where(valid, clamped, fallback) + origins

  def _update_footprint_sites(
    self,
    env_ids: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_quat: torch.Tensor,
    geom_size: torch.Tensor,
    geom_pos: torch.Tensor,
  ) -> None:
    if self._footprint_table is None or self._global_footprint_site_ids is None:
      return
    assert self._footprint_table_top_local_z is not None
    assert self._footprint_site_body_id is not None

    min_x, max_x = tool_axis_bounds(
      geom_size, geom_pos, self._support_geom_types, obj_quat, axis=0
    )
    min_y, max_y = tool_axis_bounds(
      geom_size, geom_pos, self._support_geom_types, obj_quat, axis=1
    )
    corners_w = torch.stack(
      [
        torch.stack([obj_pos[:, 0] + min_x, obj_pos[:, 1] + min_y], dim=-1),
        torch.stack([obj_pos[:, 0] + min_x, obj_pos[:, 1] + max_y], dim=-1),
        torch.stack([obj_pos[:, 0] + max_x, obj_pos[:, 1] + min_y], dim=-1),
        torch.stack([obj_pos[:, 0] + max_x, obj_pos[:, 1] + max_y], dim=-1),
      ],
      dim=1,
    )
    corners_w = torch.cat(
      [
        corners_w,
        torch.zeros(len(env_ids), 4, 1, device=self.device),
      ],
      dim=-1,
    )
    table_pos = self._body_field_local(
      "body_pos", self._footprint_site_body_id, env_ids
    )
    table_quat = self._body_field_local(
      "body_quat", self._footprint_site_body_id, env_ids
    )
    local = quat_apply(
      quat_inv(table_quat).unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4),
      (corners_w - table_pos.unsqueeze(1)).reshape(-1, 3),
    ).reshape(len(env_ids), 4, 3)
    local[..., 2] = self._footprint_table_top_local_z

    site_pos = self._env.sim.model.site_pos
    if site_pos.ndim == 2:
      site_pos[self._global_footprint_site_ids] = local[0]
    else:
      site_pos[env_ids[:, None], self._global_footprint_site_ids[None, :]] = local

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    origins = self._env.scene.env_origins[env_ids]

    # Reset stateful trackers.
    self.lifted_object[env_ids] = False
    self.min_fingertip_dists[env_ids] = float("inf")
    self.min_keypoint_max_dist[env_ids] = float("inf")
    self.consecutive_successes[env_ids] = 0
    self.num_goal_resets[env_ids] = 0

    # Random orientation.
    yaw = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
    pitch = sample_uniform(-math.pi / 4, math.pi / 4, (n,), device=self.device)
    roll = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
    self.goal_quat[env_ids] = quat_from_euler_xyz(roll, pitch, yaw)

    current_geom_size = self._geom_field_local(
      "geom_size", self._global_support_geom_ids, env_ids
    )
    current_geom_pos = self._geom_field_local(
      "geom_pos", self._global_support_geom_ids, env_ids
    )

    # Sample first goal uniformly in workspace, accounting for full tool extents.
    self.goal_pos[env_ids] = self._sample_root_position_in_workspace(
      self.goal_quat[env_ids],
      current_geom_size,
      current_geom_pos,
      env_ids,
      self._ws_min,
      self._ws_max,
      origins,
      constrain_xy_by_tool_extents=self.cfg.goal_constrain_xy_by_tool_extents,
    )

    # Compute goal keypoints.
    self.goal_keypoints_w[env_ids] = self._compute_keypoints(
      self.goal_pos[env_ids], self.goal_quat[env_ids], env_ids
    )

    # Reset object on table.
    if self.cfg.object_pose_range is not None:
      r = self.cfg.object_pose_range
      lo = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      hi = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      obj_pos = sample_uniform(lo, hi, (n, 3), device=self.device) + origins
      roll = sample_uniform(r.roll[0], r.roll[1], (n,), device=self.device)
      pitch = sample_uniform(r.pitch[0], r.pitch[1], (n,), device=self.device)
      yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
      obj_quat = quat_from_euler_xyz(
        roll,
        pitch,
        yaw,
      )
      current_support = tool_support_height(
        current_geom_size,
        current_geom_pos,
        self._support_geom_types,
        obj_quat,
      )
      min_x, max_x = tool_axis_bounds(
        current_geom_size,
        current_geom_pos,
        self._support_geom_types,
        obj_quat,
        axis=0,
      )
      min_y, max_y = tool_axis_bounds(
        current_geom_size,
        current_geom_pos,
        self._support_geom_types,
        obj_quat,
        axis=1,
      )
      default_support = tool_support_height(
        self._default_support_geom_size.unsqueeze(0).expand(n, -1, -1),
        self._default_support_geom_pos.unsqueeze(0).expand(n, -1, -1),
        self._support_geom_types,
        obj_quat,
      )
      x_lo = torch.full((n,), r.x[0], device=self.device) - min_x
      x_hi = torch.full((n,), r.x[1], device=self.device) - max_x
      y_lo = torch.full((n,), r.y[0], device=self.device) - min_y
      y_hi = torch.full((n,), r.y[1], device=self.device) - max_y
      valid_x = x_lo <= x_hi
      valid_y = y_lo <= y_hi
      obj_pos[:, 0] = torch.where(
        valid_x,
        sample_uniform(x_lo, x_hi, (n,), device=self.device),
        0.5 * (x_lo + x_hi),
      )
      obj_pos[:, 1] = torch.where(
        valid_y,
        sample_uniform(y_lo, y_hi, (n,), device=self.device),
        0.5 * (y_lo + y_hi),
      )
      obj_pos[:, :2] += origins[:, :2]
      obj_pos[:, 2] = obj_pos[:, 2] + current_support - default_support
      self._update_footprint_sites(
        env_ids, obj_pos, obj_quat, current_geom_size, current_geom_pos
      )
      pose = torch.cat([obj_pos, obj_quat], dim=-1)
      velocity = torch.zeros(n, 6, device=self.device)
      self.object_initial_pos_w[env_ids] = obj_pos
      self.tool.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      self.tool.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  def _sample_delta_goal(self, env_ids: torch.Tensor) -> None:
    """Sample a new goal as a delta from the current goal."""
    n = len(env_ids)
    origins = self._env.scene.env_origins[env_ids]

    # Position delta.
    delta_pos = sample_uniform(
      -self.cfg.delta_position,
      self.cfg.delta_position,
      (n, 3),
      device=self.device,
    )
    new_pos = self.goal_pos[env_ids] + delta_pos
    # Clamp to workspace.
    new_pos = (
      torch.clamp(
        new_pos - origins,
        self._ws_min,
        self._ws_max,
      )
      + origins
    )
    self.goal_pos[env_ids] = new_pos

    # Rotation delta.
    max_rad = math.radians(self.cfg.delta_rotation_deg)
    delta_roll = sample_uniform(-max_rad, max_rad, (n,), device=self.device)
    delta_pitch = sample_uniform(-max_rad, max_rad, (n,), device=self.device)
    delta_yaw = sample_uniform(-max_rad, max_rad, (n,), device=self.device)
    delta_quat = quat_from_euler_xyz(delta_roll, delta_pitch, delta_yaw)
    self.goal_quat[env_ids] = quat_mul(delta_quat, self.goal_quat[env_ids])

    current_geom_size = self._geom_field_local(
      "geom_size", self._global_support_geom_ids, env_ids
    )
    current_geom_pos = self._geom_field_local(
      "geom_pos", self._global_support_geom_ids, env_ids
    )
    self.goal_pos[env_ids] = self._clamp_root_position_to_workspace(
      self.goal_pos[env_ids],
      self.goal_quat[env_ids],
      current_geom_size,
      current_geom_pos,
      env_ids,
      self._ws_min,
      self._ws_max,
      origins,
      constrain_xy_by_tool_extents=self.cfg.goal_constrain_xy_by_tool_extents,
    )

    # Recompute goal keypoints.
    self.goal_keypoints_w[env_ids] = self._compute_keypoints(
      self.goal_pos[env_ids], self.goal_quat[env_ids], env_ids
    )

    # Reset progress trackers for the new goal.
    self.min_keypoint_max_dist[env_ids] = float("inf")
    self.consecutive_successes[env_ids] = 0
    self.num_goal_resets[env_ids] += 1

  def _update_command(self) -> None:
    # Update object keypoints from the actual tool sites.
    self.object_keypoints_w = self.tool.data.site_pos_w[:, self._keypoint_site_ids]

    obj_pos = self.tool.data.root_link_pos_w

    # Track lifting.
    obj_delta_z = obj_pos[:, 2] - self.object_initial_pos_w[:, 2]
    self.lifted_object = self.lifted_object | (obj_delta_z > self.cfg.lift_threshold)

    # Compute max keypoint distance to goal.
    kp_dists = torch.norm(
      self.goal_keypoints_w - self.object_keypoints_w, dim=-1
    )  # (B, 4)
    max_kp_dist = kp_dists.max(dim=-1).values  # (B,)

    # Update min tracker.
    self.min_keypoint_max_dist = torch.minimum(self.min_keypoint_max_dist, max_kp_dist)

    # Check success: within tolerance for consecutive steps.
    at_goal = max_kp_dist < self.cfg.success_tolerance
    self.consecutive_successes = torch.where(
      at_goal,
      self.consecutive_successes + 1,
      torch.zeros_like(self.consecutive_successes),
    )

    # Resample goal for envs that reached the target.
    success_envs = (
      (self.consecutive_successes >= self.cfg.success_steps)
      .nonzero(as_tuple=False)
      .squeeze(-1)
    )
    if len(success_envs) > 0:
      self._sample_delta_goal(success_envs)

    # Update metrics.
    self.metrics["max_keypoint_dist"] = max_kp_dist
    self.metrics["lifted_object"] = self.lifted_object.float()
    self.metrics["consecutive_successes"] = self.consecutive_successes.float()
    self.metrics["num_goal_resets"] = self.num_goal_resets.float()

  def _update_metrics(self) -> None:
    pass  # Metrics updated in _update_command.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.show_goal_ghost:
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = 0.0
        tool_geom_ids = self.tool.indexing.geom_ids.cpu().numpy()
        self._ghost_model.geom_rgba[tool_geom_ids] = self._ghost_color

      free_joint_q_adr = self.tool.indexing.free_joint_q_adr.cpu().numpy()
      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq, dtype=np.float64)
        qpos[free_joint_q_adr[0:3]] = self.goal_pos[batch].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.goal_quat[batch].cpu().numpy()
        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          alpha=float(self._ghost_color[3]),
          label=f"tool_goal_ghost_{batch}",
        )

    if self.cfg.show_com_frames:
      env_ids = torch.tensor(env_indices, device=self.device, dtype=torch.int32)
      root_body_id = self.tool.indexing.root_body_id
      com_offset_b = self._body_field_local("body_ipos", root_body_id, env_ids)
      desired_com = self.goal_pos[env_ids] + quat_apply(
        self.goal_quat[env_ids], com_offset_b
      )
      desired_rotm = matrix_from_quat(self.goal_quat[env_ids]).cpu().numpy()
      current_com = self.tool.data.root_com_pos_w[env_ids].cpu().numpy()
      current_rotm = (
        matrix_from_quat(self.tool.data.root_com_quat_w[env_ids]).cpu().numpy()
      )
      desired_com = desired_com.cpu().numpy()

      for local_idx, batch in enumerate(env_indices):
        visualizer.add_frame(
          position=desired_com[local_idx],
          rotation_matrix=desired_rotm[local_idx],
          scale=self.cfg.com_frame_scale,
          axis_radius=self.cfg.com_frame_axis_radius,
          alpha=self.cfg.com_frame_alpha,
          axis_colors=_DESIRED_FRAME_COLORS,
          label=f"tool_goal_com_{batch}",
        )
        visualizer.add_frame(
          position=current_com[local_idx],
          rotation_matrix=current_rotm[local_idx],
          scale=self.cfg.com_frame_scale,
          axis_radius=self.cfg.com_frame_axis_radius,
          alpha=self.cfg.com_frame_alpha,
          axis_colors=_CURRENT_FRAME_COLORS,
          label=f"tool_current_com_{batch}",
        )

    for batch in env_indices:
      if self.cfg.show_goal_center:
        goal_pos = self.goal_pos[batch].cpu().numpy()
        visualizer.add_sphere(
          center=goal_pos,
          radius=0.03,
          color=(0.0, 1.0, 0.0, 0.3),
          label=f"goal_pos_{batch}",
        )
      if self.cfg.show_goal_keypoints:
        for i in range(4):
          kp = self.goal_keypoints_w[batch, i].cpu().numpy()
          visualizer.add_sphere(
            center=kp,
            radius=0.01,
            color=(1.0, 0.0, 0.0, 0.5),
            label=f"goal_kp_{batch}_{i}",
          )


@dataclass(kw_only=True)
class ToolGoalPoseCommandCfg(CommandTermCfg):
  entity_name: str = "tool"
  """Name of the tool entity in the scene."""

  # Workspace bounds (relative to env origin).
  workspace_mins: tuple[float, float, float] = (-0.35, -0.1, 0.15)
  workspace_maxs: tuple[float, float, float] = (0.35, 0.2, 0.52)

  # Delta goal sampling.
  delta_position: float = 0.1
  """Max position delta per axis in meters."""
  delta_rotation_deg: float = 90.0
  """Max rotation delta in degrees."""

  # Success criteria.
  success_tolerance: float = 0.075
  """Max keypoint distance for success (meters). Curriculum can narrow this."""
  success_steps: int = 10
  """Consecutive steps within tolerance to trigger goal resample."""

  # Lifting.
  lift_threshold: float = 0.15
  """Height above the reset pose to consider the object lifted (meters)."""

  num_fingertips: int = 5
  """Number of fingertips used for stateful grasp-progress tracking."""

  # Keypoints.
  keypoint_site_names: tuple[str, ...] = (
    "keypoint_0",
    "keypoint_1",
    "keypoint_2",
    "keypoint_3",
  )
  """Tool sites used for goal/reward keypoints."""
  grasp_site_name: str = "grasp_center"
  """Tool site used for grasp-distance reward and termination logic."""
  support_geom_names: tuple[str, ...] = ("handle", "head")
  """Geoms used to compute table contact support for placement."""

  footprint_entity_name: str = ""
  """Optional static entity whose sites visualize the sampled support footprint."""

  footprint_site_names: tuple[str, ...] = ()
  """Four sites updated to the sampled tabletop support rectangle corners."""

  footprint_table_geom_name: str = "table_geom"
  """Geom used to place footprint sites slightly above the tabletop."""

  footprint_site_height_offset: float = 0.005
  """Height offset above the tabletop for footprint marker sites."""

  show_goal_ghost: bool = True
  """Whether to render a transparent tool ghost at the desired goal pose."""

  ghost_color: tuple[float, float, float, float] = (0.3, 0.9, 0.4, 0.35)
  """RGBA color for the goal ghost tool mesh."""

  show_goal_center: bool = False
  """Whether to draw the goal-position sphere in addition to the ghost."""

  show_goal_keypoints: bool = False
  """Whether to draw goal keypoint spheres in addition to the ghost."""

  show_com_frames: bool = True
  """Whether to draw current and desired tool COM frames."""

  com_frame_scale: float = 0.11
  """Axis length for current/desired COM debug frames."""

  com_frame_axis_radius: float = 0.0025
  """Axis thickness for current/desired COM debug frames."""

  com_frame_alpha: float = 0.9
  """Transparency for current/desired COM debug frames."""

  goal_constrain_xy_by_tool_extents: bool = False
  """If False, goals are free-space above the table and only root x/y stays in
  workspace."""

  goal_table_clearance: float = 0.01
  """Minimum clearance above the table for the full desired tool geometry."""

  # Object pose range for reset.
  @dataclass
  class ObjectPoseRangeCfg:
    x: tuple[float, float] = (0.2, 0.4)
    y: tuple[float, float] = (-0.15, 0.15)
    z: tuple[float, float] = (0.02, 0.05)
    roll: tuple[float, float] = (0.0, 0.0)
    pitch: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg | None = field(
    default_factory=ObjectPoseRangeCfg
  )

  def build(self, env: ManagerBasedRlEnv) -> ToolGoalPoseCommand:
    return ToolGoalPoseCommand(self, env)
