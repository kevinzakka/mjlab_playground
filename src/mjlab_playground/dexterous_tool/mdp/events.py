"""Event functions for dexterous tool manipulation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_GEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX.value
_GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER.value


def _box_support_along_axis(
  half_sizes: torch.Tensor,
  axis_body: torch.Tensor,
) -> torch.Tensor:
  return torch.sum(torch.abs(axis_body) * half_sizes, dim=-1)


def _cylinder_support_along_axis(
  radius: torch.Tensor,
  half_length: torch.Tensor,
  axis_body: torch.Tensor,
) -> torch.Tensor:
  radial = torch.sqrt(axis_body[..., 0] ** 2 + axis_body[..., 1] ** 2)
  axial = torch.abs(axis_body[..., 2])
  return radius * radial + half_length * axial


def _quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
  """Convert (w, x, y, z) quaternions to rotation matrices."""
  w, x, y, z = quat.unbind(dim=-1)
  two = 2.0
  return torch.stack(
    (
      1.0 - two * (y * y + z * z),
      two * (x * y - z * w),
      two * (x * z + y * w),
      two * (x * y + z * w),
      1.0 - two * (x * x + z * z),
      two * (y * z - x * w),
      two * (x * z - y * w),
      two * (y * z + x * w),
      1.0 - two * (x * x + y * y),
    ),
    dim=-1,
  ).reshape(-1, 3, 3)


def tool_support_height(
  geom_size: torch.Tensor,
  geom_pos: torch.Tensor,
  geom_types: torch.Tensor,
  quat: torch.Tensor,
) -> torch.Tensor:
  """Maximum downward extent of the tool below its root for each env pose."""
  min_proj, _ = tool_axis_bounds(geom_size, geom_pos, geom_types, quat, axis=2)
  return -min_proj


def tool_axis_bounds(
  geom_size: torch.Tensor,
  geom_pos: torch.Tensor,
  geom_types: torch.Tensor,
  quat: torch.Tensor,
  axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Min/max projection of the tool along a world axis for each env pose."""
  rotmat = _quat_to_rotmat(quat)
  axis_in_body = rotmat[:, axis, :]
  axis_in_body = axis_in_body[:, None, :].expand_as(geom_pos)
  center_proj = torch.sum(geom_pos * axis_in_body, dim=-1)

  geom_types = geom_types.to(device=geom_size.device, dtype=torch.int32)
  is_cylinder = geom_types[None, :] == _GEOM_CYLINDER
  is_box = geom_types[None, :] == _GEOM_BOX
  is_supported = is_cylinder | is_box
  if not bool((is_cylinder[0] | is_box[0]).all()):
    unsupported = geom_types[~is_supported[0]]
    names = [mujoco.mjtGeom(int(geom_type)).name for geom_type in unsupported.cpu()]
    raise ValueError(f"Unsupported tool geom types for axis bounds: {names}")

  extent = torch.zeros_like(center_proj)
  extent = torch.where(
    is_cylinder,
    _cylinder_support_along_axis(geom_size[..., 0], geom_size[..., 1], axis_in_body),
    extent,
  )
  extent = torch.where(is_box, _box_support_along_axis(geom_size, axis_in_body), extent)

  return (center_proj - extent).min(dim=-1).values, (center_proj + extent).max(
    dim=-1
  ).values


def _write_geom_bounds(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  geom_ids: torch.Tensor,
) -> None:
  """Update geom_rbound and geom_aabb after size randomization."""
  env_ids = env_ids.to(device=env.device, dtype=torch.long)
  geom_ids = geom_ids.to(device=env.device, dtype=torch.long)
  env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
  size = env.sim.model.geom_size[env_grid, geom_grid]
  geom_types = torch.as_tensor(
    env.sim.model.geom_type, device=env.device, dtype=torch.int32
  )[geom_ids]

  s0, s1, s2 = size[..., 0], size[..., 1], size[..., 2]
  is_cylinder = geom_types[None, :] == _GEOM_CYLINDER
  is_box = geom_types[None, :] == _GEOM_BOX
  is_supported = is_cylinder | is_box
  if not bool((is_cylinder[0] | is_box[0]).all()):
    unsupported = geom_types[~is_supported[0]]
    names = [mujoco.mjtGeom(int(geom_type)).name for geom_type in unsupported.cpu()]
    raise ValueError(f"Unsupported tool geom types for geom bounds: {names}")

  rbound = torch.zeros_like(s0)
  rbound = torch.where(is_cylinder, torch.sqrt(s0 * s0 + s1 * s1), rbound)
  rbound = torch.where(is_box, torch.sqrt(s0 * s0 + s1 * s1 + s2 * s2), rbound)

  aabb_half_x = torch.zeros_like(s0)
  aabb_half_y = torch.zeros_like(s0)
  aabb_half_z = torch.zeros_like(s0)
  aabb_half_x = torch.where(is_cylinder, s0, aabb_half_x)
  aabb_half_y = torch.where(is_cylinder, s0, aabb_half_y)
  aabb_half_z = torch.where(is_cylinder, s1, aabb_half_z)
  aabb_half_x = torch.where(is_box, s0, aabb_half_x)
  aabb_half_y = torch.where(is_box, s1, aabb_half_y)
  aabb_half_z = torch.where(is_box, s2, aabb_half_z)
  aabb_half = torch.stack([aabb_half_x, aabb_half_y, aabb_half_z], dim=-1)

  env.sim.model.geom_rbound[env_grid, geom_grid] = rbound
  env.sim.model.geom_aabb[env_grid, geom_grid, 1] = aabb_half


@requires_model_fields(
  "geom_size",
  "geom_pos",
  "geom_rbound",
  "geom_aabb",
  "site_pos",
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def randomize_tool_geometry(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg | None = None,
  handle_scale_range: tuple[float, float] = (0.5, 2.0),
  head_scale_range: tuple[float, float] = (0.5, 2.0),
  handle_density: float = 500.0,
  head_density: float = 1500.0,
  handle_geom_name: str = "handle",
  head_geom_name: str = "head",
  keypoint_site_names: tuple[str, ...] = (
    "keypoint_0",
    "keypoint_1",
    "keypoint_2",
    "keypoint_3",
  ),
  grasp_site_name: str = "grasp_center",
) -> None:
  """Randomize the hammer geometry and matching inertial properties coherently."""
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("tool")
  entity: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  handle_local_ids, _ = entity.find_geoms((handle_geom_name,), preserve_order=True)
  head_local_ids, _ = entity.find_geoms((head_geom_name,), preserve_order=True)
  keypoint_local_ids, _ = entity.find_sites(keypoint_site_names, preserve_order=True)
  grasp_local_ids, _ = entity.find_sites((grasp_site_name,), preserve_order=True)

  handle_geom_id = entity.indexing.geom_ids[handle_local_ids].item()
  head_geom_id = entity.indexing.geom_ids[head_local_ids].item()
  geom_ids = torch.tensor(
    [handle_geom_id, head_geom_id], device=env.device, dtype=env_ids.dtype
  )

  keypoint_site_ids = entity.indexing.site_ids[keypoint_local_ids]
  grasp_site_id = entity.indexing.site_ids[grasp_local_ids].item()
  root_body_id = entity.indexing.root_body_id

  default_geom_size = env.sim.get_default_field("geom_size")[geom_ids]
  default_geom_pos = env.sim.get_default_field("geom_pos")[geom_ids]
  default_site_pos = env.sim.get_default_field("site_pos")[keypoint_site_ids]

  handle_default = default_geom_size[0]
  head_default = default_geom_size[1]

  n = len(env_ids)
  handle_scale = torch.empty(n, device=env.device).uniform_(*handle_scale_range)
  head_scale = torch.empty(n, device=env.device).uniform_(*head_scale_range)

  handle_size = handle_default.unsqueeze(0).repeat(n, 1)
  handle_size[:, 0] *= handle_scale
  handle_size[:, 1] *= handle_scale

  head_size = head_default.unsqueeze(0).repeat(n, 1) * head_scale.unsqueeze(-1)
  head_pos = default_geom_pos[1].unsqueeze(0).repeat(n, 1)
  head_pos[:, 2] = handle_size[:, 1] + head_size[:, 2]

  env.sim.model.geom_size[env_ids, handle_geom_id] = handle_size
  env.sim.model.geom_size[env_ids, head_geom_id] = head_size
  env.sim.model.geom_pos[env_ids, head_geom_id] = head_pos

  keypoint_pos = default_site_pos.unsqueeze(0).repeat(n, 1, 1)
  keypoint_pos[..., 0] *= handle_scale[:, None]
  keypoint_pos[..., 1] *= handle_scale[:, None]
  keypoint_pos[..., 2] *= handle_scale[:, None]
  env.sim.model.site_pos[env_ids[:, None], keypoint_site_ids[None, :]] = keypoint_pos
  env.sim.model.site_pos[env_ids, grasp_site_id] = 0.0

  _write_geom_bounds(env, env_ids, geom_ids)

  handle_radius = handle_size[:, 0]
  handle_half_length = handle_size[:, 1]
  head_half_sizes = head_size
  head_center_z = head_pos[:, 2]

  handle_volume = math.pi * handle_radius * handle_radius * (2.0 * handle_half_length)
  head_volume = (
    8.0 * head_half_sizes[:, 0] * head_half_sizes[:, 1] * head_half_sizes[:, 2]
  )
  handle_mass = handle_density * handle_volume
  head_mass = head_density * head_volume
  total_mass = handle_mass + head_mass

  com_z = (head_mass * head_center_z) / total_mass
  body_ipos = torch.stack(
    [torch.zeros_like(com_z), torch.zeros_like(com_z), com_z],
    dim=-1,
  )

  handle_len = 2.0 * handle_half_length
  handle_ixx = (handle_mass / 12.0) * (3.0 * handle_radius**2 + handle_len**2)
  handle_iyy = handle_ixx
  handle_izz = 0.5 * handle_mass * handle_radius**2

  hx, hy, hz = head_half_sizes.unbind(dim=-1)
  head_ixx = (head_mass / 3.0) * (hy**2 + hz**2)
  head_iyy = (head_mass / 3.0) * (hx**2 + hz**2)
  head_izz = (head_mass / 3.0) * (hx**2 + hy**2)

  handle_dz = -com_z
  head_dz = head_center_z - com_z
  total_ixx = (
    handle_ixx + handle_mass * handle_dz**2 + head_ixx + head_mass * head_dz**2
  )
  total_iyy = (
    handle_iyy + handle_mass * handle_dz**2 + head_iyy + head_mass * head_dz**2
  )
  total_izz = handle_izz + head_izz

  env.sim.model.body_mass[env_ids, root_body_id] = total_mass
  env.sim.model.body_ipos[env_ids, root_body_id] = body_ipos
  env.sim.model.body_inertia[env_ids, root_body_id] = torch.stack(
    [total_ixx, total_iyy, total_izz], dim=-1
  )
  body_iquat = torch.zeros(n, 4, device=env.device)
  body_iquat[:, 0] = 1.0
  env.sim.model.body_iquat[env_ids, root_body_id] = body_iquat
