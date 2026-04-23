"""Reset events for the getup task."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@functools.lru_cache(maxsize=4)
def _load_motion_frames(motion_source: str, device: str) -> dict[str, torch.Tensor]:
  path = Path(motion_source)
  if path.exists():
    motion_path = str(path)
  else:
    import wandb

    registry = motion_source if ":" in motion_source else f"{motion_source}:latest"
    artifact = wandb.Api().artifact(registry)
    motion_path = str(Path(artifact.download()) / "motion.npz")

  data = np.load(motion_path)
  return {
    "joint_pos": torch.tensor(data["joint_pos"], dtype=torch.float32, device=device),
    "joint_vel": torch.tensor(data["joint_vel"], dtype=torch.float32, device=device),
    "root_pos_w": torch.tensor(
      data["body_pos_w"][:, 0], dtype=torch.float32, device=device
    ),
    "root_quat_w": torch.tensor(
      data["body_quat_w"][:, 0], dtype=torch.float32, device=device
    ),
    "root_lin_vel_w": torch.tensor(
      data["body_lin_vel_w"][:, 0], dtype=torch.float32, device=device
    ),
    "root_ang_vel_w": torch.tensor(
      data["body_ang_vel_w"][:, 0], dtype=torch.float32, device=device
    ),
  }


def reset_fallen_or_standing(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  fall_probability: float = 0.6,
  fall_height: float = 0.5,
  velocity_range: float = 0.5,
  joint_range_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset robots to either a random fallen configuration or standing.

  With ``fall_probability``, the robot is placed at ``fall_height`` with a random
  orientation, random joint positions across the full joint range, and random root
  velocities. Otherwise it starts in the default standing pose.

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all environments.
    fall_probability: Probability of starting in a fallen configuration.
    fall_height: Height (m) to place the robot when fallen.
    velocity_range: Root velocity sampled uniformly in [-range, range].
    joint_range_scale: Scale factor in (0, 1] applied to the soft joint range
      symmetrically around the midpoint. 1.0 = full range; 0.5 = half range.
    asset_cfg: Asset configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  n = len(env_ids)
  asset: Entity = env.scene[asset_cfg.name]

  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  # Sample fall mask and store it so the action term can skip settle for standing envs.
  fall_mask = torch.rand(n, device=env.device) < fall_probability
  if "settle_mask" not in env.extras:
    env.extras["settle_mask"] = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
  env.extras["settle_mask"][env_ids] = fall_mask

  # Root state.
  root_states = default_root_state[env_ids].clone()

  # Fallen: random quaternion, fixed height, random velocities.
  random_quat = torch.randn(n, 4, device=env.device)
  random_quat = F.normalize(random_quat, dim=-1)

  fallen_positions = env.scene.env_origins[env_ids].clone()
  fallen_positions[:, 2] += fall_height

  fallen_velocities = sample_uniform(
    -velocity_range, velocity_range, (n, 6), env.device
  )

  # Standing: default state offset to env origin with a small z bump to avoid ground
  # penetration.
  standing_positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
  standing_positions[:, 2] += 0.02

  mask = fall_mask.unsqueeze(-1)
  positions = torch.where(mask, fallen_positions, standing_positions)
  orientations = torch.where(mask, random_quat, root_states[:, 3:7])
  velocities = torch.where(mask, fallen_velocities, root_states[:, 7:13])

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)

  # Joint state.
  joint_limits = soft_joint_pos_limits[env_ids]
  mid = (joint_limits[..., 0] + joint_limits[..., 1]) * 0.5
  half = (joint_limits[..., 1] - joint_limits[..., 0]) * 0.5 * joint_range_scale
  random_joint_pos = sample_uniform(mid - half, mid + half, mid.shape, env.device)

  joint_pos = torch.where(mask, random_joint_pos, default_joint_pos[env_ids].clone())
  joint_vel = torch.where(
    mask,
    sample_uniform(-velocity_range, velocity_range, joint_pos.shape, env.device),
    default_joint_vel[env_ids].clone(),
  )

  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_from_motion_or_fall(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  motion_source: str,
  motion_probability: float = 0.7,
  fall_height: float = 0.5,
  velocity_range: float = 0.5,
  joint_range_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset robots from a reference motion frame or a random fallen configuration.

  With ``motion_probability``, sample a uniform timestep from the reference motion
  (loaded from a local path or wandb registry) and set the root + joint state to
  that frame — biasing the initial-state distribution toward human-like poses.
  Otherwise, drop the robot at ``fall_height`` with random orientation, joint
  positions across the soft-limit range, and random root/joint velocities.

  Motion-reset envs skip the action settle window, since they start in a coherent
  pose the policy can act on immediately. Fall envs set ``settle_mask`` to True.

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all environments.
    motion_source: Local path to a motion ``.npz`` file, or a wandb registry name
      (e.g. ``"org/wandb-registry-Motions/name"``). Alias defaults to ``:latest``.
    motion_probability: Probability of initializing from a reference motion frame.
    fall_height: Height (m) to place the robot when falling.
    velocity_range: Root/joint velocity sampled uniformly in [-range, range] for
      the fall branch.
    joint_range_scale: Scale in (0, 1] applied to soft joint range for the fall
      branch. 1.0 = full range; 0.5 = half range.
    asset_cfg: Asset configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  n = len(env_ids)
  asset: Entity = env.scene[asset_cfg.name]

  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  motion = _load_motion_frames(motion_source, str(env.device))
  num_frames = motion["joint_pos"].shape[0]

  motion_mask = torch.rand(n, device=env.device) < motion_probability
  fall_mask = ~motion_mask

  # Fall envs go through the action settle window; motion envs act immediately.
  if "settle_mask" not in env.extras:
    env.extras["settle_mask"] = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
  env.extras["settle_mask"][env_ids] = fall_mask

  # Motion branch: uniform timestep, recenter XY to env origin.
  t = torch.randint(0, num_frames, (n,), device=env.device)
  motion_pos = motion["root_pos_w"][t].clone()
  motion_pos[:, :2] = 0.0
  motion_pos += env.scene.env_origins[env_ids]
  motion_quat = motion["root_quat_w"][t]
  motion_lin_vel = motion["root_lin_vel_w"][t]
  motion_ang_vel = motion["root_ang_vel_w"][t]
  motion_joint_pos = motion["joint_pos"][t]
  motion_joint_vel = motion["joint_vel"][t]

  # Fall branch: random orientation at fall_height, random velocities and joints.
  random_quat = F.normalize(torch.randn(n, 4, device=env.device), dim=-1)
  fall_pos = env.scene.env_origins[env_ids].clone()
  fall_pos[:, 2] += fall_height
  fall_lin_vel = sample_uniform(-velocity_range, velocity_range, (n, 3), env.device)
  fall_ang_vel = sample_uniform(-velocity_range, velocity_range, (n, 3), env.device)

  joint_limits = soft_joint_pos_limits[env_ids]
  mid = (joint_limits[..., 0] + joint_limits[..., 1]) * 0.5
  half = (joint_limits[..., 1] - joint_limits[..., 0]) * 0.5 * joint_range_scale
  fall_joint_pos = sample_uniform(mid - half, mid + half, mid.shape, env.device)
  fall_joint_vel = sample_uniform(
    -velocity_range, velocity_range, fall_joint_pos.shape, env.device
  )

  mask = motion_mask.unsqueeze(-1)
  positions = torch.where(mask, motion_pos, fall_pos)
  orientations = torch.where(mask, motion_quat, random_quat)
  lin_vel = torch.where(mask, motion_lin_vel, fall_lin_vel)
  ang_vel = torch.where(mask, motion_ang_vel, fall_ang_vel)
  joint_pos = torch.where(mask, motion_joint_pos, fall_joint_pos)
  joint_vel = torch.where(mask, motion_joint_vel, fall_joint_vel)

  # Retargeted motion may slightly exceed the robot's soft joint limits.
  joint_pos = torch.clip(joint_pos, joint_limits[..., 0], joint_limits[..., 1])

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(
    torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids
  )
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
