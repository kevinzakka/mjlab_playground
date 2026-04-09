"""KUKA iiwa14 + SHARPA hand fused robot constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# Paths.
##

_XMLS_DIR: Path = Path(__file__).parent / "xmls"
_IIWA14_XML: Path = _XMLS_DIR / "iiwa14.xml"
_SHARPA_XML: Path = _XMLS_DIR / "left_sharpa_ha4_v2_1.xml"

assert _IIWA14_XML.exists(), f"Missing {_IIWA14_XML}"
assert _SHARPA_XML.exists(), f"Missing {_SHARPA_XML}"

##
# Joint name constants.
##

ARM_JOINT_NAMES = (
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
  "joint7",
)

HAND_JOINT_NAMES = (
  "left_thumb_CMC_FE",
  "left_thumb_CMC_AA",
  "left_thumb_MCP_FE",
  "left_thumb_MCP_AA",
  "left_thumb_IP",
  "left_index_MCP_FE",
  "left_index_MCP_AA",
  "left_index_PIP",
  "left_index_DIP",
  "left_middle_MCP_FE",
  "left_middle_MCP_AA",
  "left_middle_PIP",
  "left_middle_DIP",
  "left_ring_MCP_FE",
  "left_ring_MCP_AA",
  "left_ring_PIP",
  "left_ring_DIP",
  "left_pinky_CMC",
  "left_pinky_MCP_FE",
  "left_pinky_MCP_AA",
  "left_pinky_PIP",
  "left_pinky_DIP",
)

FINGERTIP_BODY_NAMES = (
  "left_thumb_DP",
  "left_index_DP",
  "left_middle_DP",
  "left_ring_DP",
  "left_pinky_DP",
)

# Fingertip site offset in the distal phalanx body frame (from simtoolreal).
FINGERTIP_SITE_OFFSET = (0.02, 0.002, 0.0)

# Palm center offset in the left_hand_C_MC body frame (from simtoolreal).
PALM_CENTER_OFFSET = (0.006, 0.001, 0.04)


##
# Spec composition.
##


def get_kuka_sharpa_spec() -> mujoco.MjSpec:
  """Load and compose the KUKA iiwa14 arm + SHARPA hand into a single spec."""
  arm_spec = mujoco.MjSpec.from_file(str(_IIWA14_XML))
  hand_spec = mujoco.MjSpec.from_file(str(_SHARPA_XML))

  # Find link7 in the arm spec and attach the hand there.
  link7 = arm_spec.body("link7")
  hand_frame = link7.add_frame(pos=[0, 0, 0.045])
  arm_spec.attach(hand_spec, prefix="", frame=hand_frame)

  # Add palm center site.
  palm_body = arm_spec.body("left_hand_C_MC")
  palm_body.add_site(
    name="palm_center",
    pos=PALM_CENTER_OFFSET,
    size=[0.005],
    rgba=[0, 1, 0, 0.5],
    group=4,
  )

  # Add fingertip sites.
  for body_name in FINGERTIP_BODY_NAMES:
    body = arm_spec.body(body_name)
    finger = body_name.replace("left_", "").replace("_DP", "")
    body.add_site(
      name=f"fingertip_{finger}",
      pos=FINGERTIP_SITE_OFFSET,
      size=[0.003],
      rgba=[1, 0, 0, 0.5],
      group=4,
    )

  return arm_spec


##
# Actuator config.
##

# From https://github.com/RobotLocomotion/models/blob/master/iiwa_description/README.md#rotor-inertia-and-gear-ratios-for-iiwa-14
# Motors are RoboDrive ILM series with single-stage Harmonic Drive CSG gearboxes.
# Reflected inertia = rotor_inertia * gear_ratio^2.
#
# |Axis data  | motor      | gear            | gear ratio | rotor inertia (kg m^2)|
# |-----------|-----------:|----------------:|-----------:|----------------------:|
# |Axis 1 (A1)|ILM 85x23   |CSG-32-160-2A-GR |160         |1.321e-4               |
# |Axis 2 (A2)|ILM 85x23   |CSG-32-160-2A-GR |160         |1.321e-4               |
# |Axis 3 (A3)|ILM 70x18   |CSG-32-160-2A-GR |160         |1.321e-4               |
# |Axis 4 (A4)|ILM 70x18   |CSG-32-160-2A-GR |160         |1.321e-4               |
# |Axis 5 (A5)|ILM 70x18   |CSG-32-100-2A-GR |100         |1.321e-4               |
# |Axis 6 (A6)|ILM 50x08   |CSG-20-160-2A-GR |160         |4.54e-5                |
# |Axis 7 (A7)|ILM 50x08   |CSG-20-160-2A-GR |160         |4.54e-5                |

_ARM_ROTOR_INERTIA = {
  "joint1": 1.321e-4,
  "joint2": 1.321e-4,
  "joint3": 1.321e-4,
  "joint4": 1.321e-4,
  "joint5": 1.321e-4,
  "joint6": 4.54e-5,
  "joint7": 4.54e-5,
}

_ARM_GEAR_RATIO = {
  "joint1": 160,
  "joint2": 160,
  "joint3": 160,
  "joint4": 160,
  "joint5": 100,
  "joint6": 160,
  "joint7": 160,
}

ARM_ARMATURE: dict[str, float] = {
  name: reflected_inertia(_ARM_ROTOR_INERTIA[name], _ARM_GEAR_RATIO[name])
  for name in ARM_JOINT_NAMES
}

ARM_EFFORT_LIMIT: dict[str, float] = {
  "joint1": 320.0,
  "joint2": 320.0,
  "joint3": 176.0,
  "joint4": 176.0,
  "joint5": 110.0,
  "joint6": 40.0,
  "joint7": 40.0,
}

# Computed by compute_gains.py at home config with omega_n=5.0 Hz, zeta=2.0.
# fmt: off
ARM_KP: dict[str, float] = {
  "joint1": 6611.3818,
  "joint2": 7084.8186,
  "joint3": 4674.5008,
  "joint4": 4654.9539,
  "joint5": 1315.9574,
  "joint6": 1214.9072,
  "joint7": 1149.7593,
}

ARM_KV: dict[str, float] = {
  "joint1": 841.7873,
  "joint2": 902.0671,
  "joint3": 595.1759,
  "joint4": 592.6871,
  "joint5": 167.5529,
  "joint6": 154.6868,
  "joint7": 146.3919,
}
# fmt: on

_ARM_ACTUATORS = tuple(
  BuiltinPositionActuatorCfg(
    target_names_expr=(name,),
    stiffness=ARM_KP[name],
    damping=ARM_KV[name],
    effort_limit=ARM_EFFORT_LIMIT[name],
    armature=ARM_ARMATURE[name],
  )
  for name in ARM_JOINT_NAMES
)

# Hand: kp from XML, kv resolved from dampratio=2.0 on the standalone hand model.
# Armature, frictionloss, and joint damping from SHARPA XML joint class defaults.
# fmt: off

# Joint-level damping from SHARPA XML defaults (very small, passive dissipation).
HAND_JOINT_DAMPING: dict[str, float] = {
  "left_thumb_CMC_FE": 4.20e-05,
  "left_thumb_CMC_AA": 4.20e-05,
  "left_thumb_MCP_FE": 2.38e-05,
  "left_thumb_MCP_AA": 2.38e-05,
  "left_thumb_IP": 4.06e-06,
  "left_index_MCP_FE": 2.38e-05,
  "left_index_MCP_AA": 2.38e-05,
  "left_index_PIP": 4.06e-06,
  "left_index_DIP": 1.21e-06,
  "left_middle_MCP_FE": 2.38e-05,
  "left_middle_MCP_AA": 2.38e-05,
  "left_middle_PIP": 4.06e-06,
  "left_middle_DIP": 1.21e-06,
  "left_ring_MCP_FE": 2.38e-05,
  "left_ring_MCP_AA": 2.38e-05,
  "left_ring_PIP": 4.06e-06,
  "left_ring_DIP": 1.21e-06,
  "left_pinky_CMC": 4.20e-05,
  "left_pinky_MCP_FE": 2.38e-05,
  "left_pinky_MCP_AA": 2.38e-05,
  "left_pinky_PIP": 4.06e-06,
  "left_pinky_DIP": 1.21e-06,
}

# Frictionloss from SHARPA XML joint class defaults.
HAND_FRICTIONLOSS: dict[str, float] = {
  "left_thumb_CMC_FE": 0.132,
  "left_thumb_CMC_AA": 0.132,
  "left_thumb_MCP_FE": 0.07456,
  "left_thumb_MCP_AA": 0.07456,
  "left_thumb_IP": 0.01276,
  "left_index_MCP_FE": 0.07456,
  "left_index_MCP_AA": 0.07456,
  "left_index_PIP": 0.01276,
  "left_index_DIP": 0.003787,
  "left_middle_MCP_FE": 0.07456,
  "left_middle_MCP_AA": 0.07456,
  "left_middle_PIP": 0.01276,
  "left_middle_DIP": 0.003787,
  "left_ring_MCP_FE": 0.07456,
  "left_ring_MCP_AA": 0.07456,
  "left_ring_PIP": 0.01276,
  "left_ring_DIP": 0.003787,
  "left_pinky_CMC": 0.012,
  "left_pinky_MCP_FE": 0.07456,
  "left_pinky_MCP_AA": 0.07456,
  "left_pinky_PIP": 0.01276,
  "left_pinky_DIP": 0.003787,
}

# Armature from SHARPA XML joint class defaults.
HAND_ARMATURE: dict[str, float] = {
  "left_thumb_CMC_FE": 0.0032,
  "left_thumb_CMC_AA": 0.0032,
  "left_thumb_MCP_FE": 0.00265,
  "left_thumb_MCP_AA": 0.00265,
  "left_thumb_IP": 0.0006,
  "left_index_MCP_FE": 0.00265,
  "left_index_MCP_AA": 0.00265,
  "left_index_PIP": 0.0006,
  "left_index_DIP": 0.00042,
  "left_middle_MCP_FE": 0.00265,
  "left_middle_MCP_AA": 0.00265,
  "left_middle_PIP": 0.0006,
  "left_middle_DIP": 0.00042,
  "left_ring_MCP_FE": 0.00265,
  "left_ring_MCP_AA": 0.00265,
  "left_ring_PIP": 0.0006,
  "left_ring_DIP": 0.00042,
  "left_pinky_CMC": 0.00012,
  "left_pinky_MCP_FE": 0.00265,
  "left_pinky_MCP_AA": 0.00265,
  "left_pinky_PIP": 0.0006,
  "left_pinky_DIP": 0.00042,
}

# Effort limits from SHARPA XML actuatorfrcrange joint class defaults.
HAND_EFFORT_LIMIT: dict[str, float] = {
  "left_thumb_CMC_FE": 3.3,
  "left_thumb_CMC_AA": 3.3,
  "left_thumb_MCP_FE": 1.864,
  "left_thumb_MCP_AA": 1.864,
  "left_thumb_IP": 0.638,
  "left_index_MCP_FE": 1.864,
  "left_index_MCP_AA": 1.864,
  "left_index_PIP": 0.638,
  "left_index_DIP": 0.189369,
  "left_middle_MCP_FE": 1.864,
  "left_middle_MCP_AA": 1.864,
  "left_middle_PIP": 0.638,
  "left_middle_DIP": 0.189369,
  "left_ring_MCP_FE": 1.864,
  "left_ring_MCP_AA": 1.864,
  "left_ring_PIP": 0.638,
  "left_ring_DIP": 0.189369,
  "left_pinky_CMC": 0.5285,
  "left_pinky_MCP_FE": 1.864,
  "left_pinky_MCP_AA": 1.864,
  "left_pinky_PIP": 0.638,
  "left_pinky_DIP": 0.189369,
}

HAND_KP: dict[str, float] = {
  "left_thumb_CMC_FE": 6.95,
  "left_thumb_CMC_AA": 13.2,
  "left_thumb_MCP_FE": 4.76,
  "left_thumb_MCP_AA": 6.62,
  "left_thumb_IP": 0.9,
  "left_index_MCP_FE": 4.76,
  "left_index_MCP_AA": 6.62,
  "left_index_PIP": 0.9,
  "left_index_DIP": 0.9,
  "left_middle_MCP_FE": 4.76,
  "left_middle_MCP_AA": 6.62,
  "left_middle_PIP": 0.9,
  "left_middle_DIP": 0.9,
  "left_ring_MCP_FE": 4.76,
  "left_ring_MCP_AA": 6.62,
  "left_ring_PIP": 0.9,
  "left_ring_DIP": 0.9,
  "left_pinky_CMC": 1.38,
  "left_pinky_MCP_FE": 4.76,
  "left_pinky_MCP_AA": 6.62,
  "left_pinky_PIP": 0.9,
  "left_pinky_DIP": 0.9,
}

# kv resolved with dampratio=2.0 (overdamped) instead of XML's 0.9.
HAND_KV: dict[str, float] = {
  "left_thumb_CMC_FE": 0.637043,
  "left_thumb_CMC_AA": 0.907892,
  "left_thumb_MCP_FE": 0.453223,
  "left_thumb_MCP_AA": 0.534439,
  "left_thumb_IP": 0.093064,
  "left_index_MCP_FE": 0.463598,
  "left_index_MCP_AA": 0.546678,
  "left_index_PIP": 0.094301,
  "left_index_DIP": 0.077847,
  "left_middle_MCP_FE": 0.463598,
  "left_middle_MCP_AA": 0.546678,
  "left_middle_PIP": 0.094301,
  "left_middle_DIP": 0.077847,
  "left_ring_MCP_FE": 0.463598,
  "left_ring_MCP_AA": 0.546678,
  "left_ring_PIP": 0.094301,
  "left_ring_DIP": 0.077847,
  "left_pinky_CMC": 0.061701,
  "left_pinky_MCP_FE": 0.463598,
  "left_pinky_MCP_AA": 0.546678,
  "left_pinky_PIP": 0.094301,
  "left_pinky_DIP": 0.077847,
}
# fmt: on

_HAND_ACTUATORS = tuple(
  BuiltinPositionActuatorCfg(
    target_names_expr=(name,),
    stiffness=HAND_KP[name],
    damping=HAND_KV[name],
    effort_limit=HAND_EFFORT_LIMIT[name],
    armature=HAND_ARMATURE[name],
    frictionloss=HAND_FRICTIONLOSS[name],
    viscous_damping=HAND_JOINT_DAMPING[name],
  )
  for name in HAND_JOINT_NAMES
)

##
# Keyframe.
##

# Arm reaching-ready pose above the table (hand open).
# joint1=0 so the arm reaches in +x toward the table.
# Remaining joints from SimToolReal's home configuration.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint4": -1.5708,
    "joint6": 0.5,
    "left_.*": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

COLLISION_CFG = CollisionCfg(
  geom_names_expr=(".*_collision", ".*_elastomer.*"),
  solref=(0.01, 1),
  condim={".*": 4},
  friction={".*": (1.0, 5e-3, 5e-4)},
  priority=1,
)

# Disables base and lower arm collisions (link1-2), keeping link3+ and hand.
ARM_NO_COLLISION_CFG = CollisionCfg(
  geom_names_expr=(r"(base|link[12]).*_collision",),
  contype=0,
  conaffinity=0,
  disable_other_geoms=False,
)

##
# Final config.
##

KUKA_SHARPA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=_ARM_ACTUATORS + _HAND_ACTUATORS,
)


def get_kuka_sharpa_robot_cfg(arm_collisions: bool = True) -> EntityCfg:
  """Get a fresh KUKA iiwa14 + SHARPA hand robot configuration."""
  collisions = (
    (COLLISION_CFG,) if arm_collisions else (COLLISION_CFG, ARM_NO_COLLISION_CFG)
  )
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=collisions,
    spec_fn=get_kuka_sharpa_spec,
    articulation=KUKA_SHARPA_ARTICULATION,
  )


##
# Action scales (heuristic: 0.25 * effort_limit / stiffness).
##

IIWA_ACTION_SCALE: dict[str, float] = {
  name: 0.25 * ARM_EFFORT_LIMIT[name] / ARM_KP[name] for name in ARM_JOINT_NAMES
}

SHARPA_ACTION_SCALE: dict[str, float] = {
  name: 0.25 * HAND_EFFORT_LIMIT[name] / HAND_KP[name] for name in HAND_JOINT_NAMES
}


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_kuka_sharpa_robot_cfg())
  viewer.launch(robot.spec.compile())
