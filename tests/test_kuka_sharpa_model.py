"""Tests that the compiled KUKA iiwa14 + SHARPA hand MjModel matches the Python config."""

from __future__ import annotations

import re

import mujoco
import numpy as np
import pytest
from mjlab.entity.entity import Entity
from mjlab_playground.asset_zoo.robots.kuka_sharpa.kuka_sharpa_constants import (
  ARM_ARMATURE,
  ARM_EFFORT_LIMIT,
  ARM_JOINT_NAMES,
  ARM_KP,
  ARM_KV,
  HAND_ARMATURE,
  HAND_EFFORT_LIMIT,
  HAND_FRICTIONLOSS,
  HAND_JOINT_DAMPING,
  HAND_JOINT_NAMES,
  HAND_KP,
  HAND_KV,
  get_kuka_sharpa_robot_cfg,
)

ALL_JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES

# Patterns from COLLISION_CFG.
_COLLISION_PATTERNS = (re.compile(r".*_collision"), re.compile(r".*_elastomer.*"))
# Pattern from ARM_NO_COLLISION_CFG.
_ARM_DISABLED_PATTERN = re.compile(r"(base|link[12]).*_collision")


def _is_collision_geom(name: str) -> bool:
  # Use re.match (not fullmatch) to mirror mjlab's filter_exp behavior.
  return any(p.match(name) for p in _COLLISION_PATTERNS)


# Fixtures.


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return Entity(get_kuka_sharpa_robot_cfg(arm_collisions=True)).compile()


@pytest.fixture(scope="module")
def model_no_arm_collisions() -> mujoco.MjModel:
  return Entity(get_kuka_sharpa_robot_cfg(arm_collisions=False)).compile()


# Constants.


def test_actuator_count(model: mujoco.MjModel):
  assert model.nu == len(ALL_JOINT_NAMES)


# Actuator gains (kp / kv).


def test_actuator_gains(model: mujoco.MjModel):
  all_kp = {**HAND_KP, **ARM_KP}
  all_kv = {**HAND_KV, **ARM_KV}
  for name in ALL_JOINT_NAMES:
    kp = all_kp[name]
    kv = all_kv[name]
    act_id = model.actuator(name).id

    np.testing.assert_allclose(
      model.actuator_gainprm[act_id, 0], kp, rtol=1e-5, err_msg=f"{name} gainprm"
    )
    np.testing.assert_allclose(
      model.actuator_biasprm[act_id, 1], -kp, rtol=1e-5, err_msg=f"{name} biasprm[1]"
    )
    np.testing.assert_allclose(
      model.actuator_biasprm[act_id, 2], -kv, rtol=1e-5, err_msg=f"{name} biasprm[2]"
    )


# Actuator force limits.


def test_actuator_force_limits(model: mujoco.MjModel):
  all_effort = {**HAND_EFFORT_LIMIT, **ARM_EFFORT_LIMIT}
  for name in ALL_JOINT_NAMES:
    effort = all_effort[name]
    act_id = model.actuator(name).id

    # forcelimited must be on, otherwise forcerange is ignored by the simulator.
    assert model.actuator_forcelimited[act_id] == 1, f"{name} not force-limited"
    # ctrllimited must be off so setpoints can exceed joint range.
    assert model.actuator_ctrllimited[act_id] == 0, f"{name} unexpectedly ctrl-limited"

    np.testing.assert_allclose(
      model.actuator_forcerange[act_id],
      [-effort, effort],
      rtol=1e-5,
      err_msg=f"{name} forcerange",
    )


# Joint armature (reflected inertia).


def test_joint_armature(model: mujoco.MjModel):
  all_armature = {**HAND_ARMATURE, **ARM_ARMATURE}
  for name in ALL_JOINT_NAMES:
    expected = all_armature[name]
    dof_id = model.jnt_dofadr[model.joint(name).id]

    np.testing.assert_allclose(
      model.dof_armature[dof_id],
      expected,
      rtol=1e-5,
      err_msg=f"{name} armature",
    )


# Hand joint dynamics (frictionloss + viscous damping).


def test_hand_joint_dynamics(model: mujoco.MjModel):
  for name in HAND_JOINT_NAMES:
    dof_id = model.jnt_dofadr[model.joint(name).id]

    np.testing.assert_allclose(
      model.dof_frictionloss[dof_id],
      HAND_FRICTIONLOSS[name],
      rtol=1e-5,
      err_msg=f"{name} frictionloss",
    )
    np.testing.assert_allclose(
      model.dof_damping[dof_id],
      HAND_JOINT_DAMPING[name],
      rtol=1e-5,
      err_msg=f"{name} damping",
    )


# Collision geoms configured.


def test_collision_geoms_configured(model: mujoco.MjModel):
  found_collision = False
  found_elastomer = False

  for geom_id in range(model.ngeom):
    name = model.geom(geom_id).name
    if not _is_collision_geom(name):
      continue

    if "_collision" in name:
      found_collision = True
    if "_elastomer" in name:
      found_elastomer = True

    assert model.geom_condim[geom_id] == 4, f"{name} condim"
    assert model.geom_contype[geom_id] == 1, f"{name} contype"
    assert model.geom_conaffinity[geom_id] == 1, f"{name} conaffinity"
    assert model.geom_priority[geom_id] == 1, f"{name} priority"

    np.testing.assert_allclose(
      model.geom_friction[geom_id],
      [1.0, 5e-3, 5e-4],
      rtol=1e-5,
      err_msg=f"{name} friction",
    )
    np.testing.assert_allclose(
      model.geom_solref[geom_id],
      [0.01, 1.0],
      rtol=1e-5,
      err_msg=f"{name} solref",
    )

  assert found_collision, "No *_collision geoms found"
  assert found_elastomer, "No *_elastomer* geoms found"


# Non-collision geoms disabled.


def test_non_collision_geoms_disabled(model: mujoco.MjModel):
  count = 0
  for geom_id in range(model.ngeom):
    name = model.geom(geom_id).name
    if _is_collision_geom(name):
      continue
    count += 1
    assert model.geom_contype[geom_id] == 0, f"{name} contype should be 0"
    assert model.geom_conaffinity[geom_id] == 0, f"{name} conaffinity should be 0"

  assert count > 0, "No non-collision geoms found"


# arm_collisions=False variant.


def test_arm_no_collision_variant(model_no_arm_collisions: mujoco.MjModel):
  m = model_no_arm_collisions
  disabled_count = 0
  enabled_count = 0

  for geom_id in range(m.ngeom):
    name = m.geom(geom_id).name
    if not _is_collision_geom(name):
      continue

    if _ARM_DISABLED_PATTERN.match(name):
      # base/link1/link2 collision geoms should be disabled.
      disabled_count += 1
      assert m.geom_contype[geom_id] == 0, f"{name} contype should be 0"
      assert m.geom_conaffinity[geom_id] == 0, f"{name} conaffinity should be 0"
    elif re.match(r"link[3-7].*_collision", name):
      # link3+ collision geoms should remain enabled.
      enabled_count += 1
      assert m.geom_contype[geom_id] == 1, f"{name} contype should be 1"
      assert m.geom_conaffinity[geom_id] == 1, f"{name} conaffinity should be 1"

  assert disabled_count > 0, "No disabled arm collision geoms found"
  assert enabled_count > 0, "No enabled arm collision geoms found"
