"""Compute principled PD gains for the KUKA iiwa14 + SHARPA hand.

Applies motor armature to each joint before compiling, so the mass matrix
diagonal M_ii = body_inertia + armature gives the full effective inertia
seen by each actuator. We then compute:

  kp = M_ii * omega_n^2
  kv = 2 * zeta * M_ii * omega_n

where:
  M_ii    = diagonal of the joint-space mass matrix at home config
  omega_n = natural frequency (rad/s)
  zeta    = damping ratio

Usage:
  python -m mjlab_playground.asset_zoo.robots.kuka_sharpa.compute_gains
"""

import mujoco
import numpy as np

from mjlab_playground.asset_zoo.robots.kuka_sharpa.kuka_sharpa_constants import (
  ARM_ARMATURE,
  ARM_JOINT_NAMES,
  HAND_ARMATURE,
  HAND_JOINT_NAMES,
  get_kuka_sharpa_spec,
)

NATURAL_FREQ_HZ = 5.0
NATURAL_FREQ = NATURAL_FREQ_HZ * 2.0 * np.pi  # rad/s
DAMPING_RATIO = 2.0


def compute_gains() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
  """Compute kp, kv, and effective inertia for each joint."""
  spec = get_kuka_sharpa_spec()

  # Apply armature to joints before compiling so M_ii includes reflected motor inertia.
  for name, armature in {**ARM_ARMATURE, **HAND_ARMATURE}.items():
    spec.joint(name).armature = armature

  model = spec.compile()
  data = mujoco.MjData(model)

  # Set to home configuration.
  # Arm home from menagerie: joint2=0.785398, joint4=-1.5708, rest=0.
  # Hand: all zeros (open hand).
  home_qpos = {
    "joint2": 0.785398,
    "joint4": -1.5708,
  }
  for name, val in home_qpos.items():
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr = model.jnt_qposadr[jnt_id]
    data.qpos[qadr] = val

  mujoco.mj_forward(model, data)

  nv = model.nv
  mass_matrix = np.zeros((nv, nv))
  mujoco.mj_fullM(model, mass_matrix, data.qM)

  M_diag = np.diag(mass_matrix)

  all_joint_names = ARM_JOINT_NAMES + HAND_JOINT_NAMES
  kp_dict: dict[str, float] = {}
  kv_dict: dict[str, float] = {}
  inertia_dict: dict[str, float] = {}

  for name in all_joint_names:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    dof_adr = model.jnt_dofadr[jnt_id]
    m_ii = float(M_diag[dof_adr])

    kp = m_ii * NATURAL_FREQ**2
    kv = 2.0 * DAMPING_RATIO * m_ii * NATURAL_FREQ

    kp_dict[name] = kp
    kv_dict[name] = kv
    inertia_dict[name] = m_ii

  return kp_dict, kv_dict, inertia_dict


def main() -> None:
  kp_dict, kv_dict, inertia_dict = compute_gains()

  print(f"Natural frequency: {NATURAL_FREQ_HZ} Hz ({NATURAL_FREQ:.2f} rad/s)")
  print(f"Damping ratio: {DAMPING_RATIO}")
  print()

  # Print arm joints.
  print("=" * 72)
  print("ARM JOINTS (KUKA iiwa14)")
  print("=" * 72)
  print(f"{'Joint':<25} {'M_ii (kg·m²)':>14} {'kp (Nm/rad)':>14} {'kv (Nms/rad)':>14}")
  print("-" * 72)
  for name in ARM_JOINT_NAMES:
    print(
      f"{name:<25} {inertia_dict[name]:>14.6f} {kp_dict[name]:>14.4f} {kv_dict[name]:>14.4f}"
    )

  print()
  print("=" * 72)
  print("HAND JOINTS (SHARPA)")
  print("=" * 72)
  print(f"{'Joint':<25} {'M_ii (kg·m²)':>14} {'kp (Nm/rad)':>14} {'kv (Nms/rad)':>14}")
  print("-" * 72)
  for name in HAND_JOINT_NAMES:
    print(
      f"{name:<25} {inertia_dict[name]:>14.6f} {kp_dict[name]:>14.4f} {kv_dict[name]:>14.4f}"
    )

  # Print as Python dict literals for copy-paste into constants.
  print()
  print("=" * 72)
  print("COPY-PASTE INTO kuka_sharpa_constants.py:")
  print("=" * 72)
  print()
  print(
    "# Computed at home config with omega_n={:.1f} Hz, zeta={:.1f}".format(
      NATURAL_FREQ_HZ, DAMPING_RATIO
    )
  )
  print("ARM_KP: dict[str, float] = {")
  for name in ARM_JOINT_NAMES:
    print(f'  "{name}": {kp_dict[name]:.4f},')
  print("}")
  print()
  print("ARM_KV: dict[str, float] = {")
  for name in ARM_JOINT_NAMES:
    print(f'  "{name}": {kv_dict[name]:.4f},')
  print("}")
  print()
  print("HAND_KP: dict[str, float] = {")
  for name in HAND_JOINT_NAMES:
    print(f'  "{name}": {kp_dict[name]:.6f},')
  print("}")
  print()
  print("HAND_KV: dict[str, float] = {")
  for name in HAND_JOINT_NAMES:
    print(f'  "{name}": {kv_dict[name]:.6f},')
  print("}")


if __name__ == "__main__":
  main()
