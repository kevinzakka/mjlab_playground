"""Floating SHARPA hand robot constants (no arm, freejoint base)."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Re-use the same SHARPA XML from kuka_sharpa.
_SHARPA_XML: Path = (
  Path(__file__).parent / ".." / "kuka_sharpa" / "xmls" / "left_sharpa_ha4_v2_1.xml"
)
assert _SHARPA_XML.exists(), f"Missing {_SHARPA_XML}"

##
# Joint name constants.
##

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

FINGERTIP_SITE_NAMES = tuple(
  f"fingertip_{name.replace('left_', '').replace('_DP', '')}"
  for name in FINGERTIP_BODY_NAMES
)

PALM_CENTER_SITE_NAME = "palm_center"

# Fingertip site offset in the distal phalanx body frame (from simtoolreal).
FINGERTIP_SITE_OFFSET = (0.02, 0.002, 0.0)

# Palm center offset in the left_hand_C_MC body frame (from simtoolreal).
PALM_CENTER_OFFSET = (0.006, 0.001, 0.04)


##
# Spec composition.
##


def get_floating_sharpa_spec() -> mujoco.MjSpec:
  """Load the SHARPA hand with a mocap body + freejoint + weld.

  The hand root body (``left_hand_C_MC``) gets a freejoint so it is
  physically simulated.  A separate mocap body (``hand_mocap``) is added
  to the worldbody and a weld equality constraint pulls the hand toward
  the mocap pose, giving direct kinematic-style control while still
  allowing physical interaction with objects.
  """
  spec = mujoco.MjSpec.from_file(str(_SHARPA_XML))

  # The SHARPA XML root body is "left_hand_C_MC". Add a freejoint so the
  # hand is physically simulated.
  root_body = spec.body("left_hand_C_MC")
  root_body.add_freejoint(name="floating_base")

  # Mocap body: kinematically controlled target for the hand base.
  # Don't set pos here — mj_resetData uses the body pos for mocap_pos
  # but zeros the freejoint qpos, causing a gap. Both default to origin;
  # the env cfg / keyframe will position them at runtime.
  mocap_body = spec.worldbody.add_body(name="hand_mocap", mocap=True)
  mocap_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.02, 0.02, 0.02],
    rgba=[0.2, 0.8, 0.2, 0.3],
    contype=0,
    conaffinity=0,
  )

  # Weld equality: pull the hand root toward the mocap body.
  # data[0:3] = relative position (zero = coincident),
  # data[3:7] = relative quaternion (identity = same orientation).
  weld = spec.add_equality()
  weld.name = "hand_mocap_weld"
  weld.type = mujoco.mjtEq.mjEQ_WELD
  weld.objtype = mujoco.mjtObj.mjOBJ_BODY
  weld.name1 = "hand_mocap"
  weld.name2 = "left_hand_C_MC"
  weld.data[0:7] = [0, 0, 0, 1, 0, 0, 0]
  weld.solref[:] = [0.002, 1]

  # Add palm center site.
  root_body.add_site(
    name=PALM_CENTER_SITE_NAME,
    pos=PALM_CENTER_OFFSET,
    size=[0.005],
    rgba=[0, 1, 0, 0.5],
    group=4,
  )

  # Add fingertip sites.
  for body_name, site_name in zip(
    FINGERTIP_BODY_NAMES, FINGERTIP_SITE_NAMES, strict=True
  ):
    body = spec.body(body_name)
    body.add_site(
      name=site_name,
      pos=FINGERTIP_SITE_OFFSET,
      size=[0.003],
      rgba=[1, 0, 0, 0.5],
      group=4,
    )

  return spec


##
# Actuator config.
##

# All hand parameters copied from kuka_sharpa_constants.py.

# Joint-level damping from SHARPA XML defaults (very small, passive dissipation).
# fmt: off
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

# Hand floating in space, palm facing down (fingers toward +x).
# Rotation: +90° around y.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  rot=(0.7071, 0.0, 0.7071, 0.0),
  joint_pos={"left_.*": 0.0},
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

##
# Final config.
##

FLOATING_SHARPA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=_HAND_ACTUATORS,
)


def get_floating_sharpa_robot_cfg() -> EntityCfg:
  """Get a floating SHARPA hand robot configuration (freejoint base)."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(COLLISION_CFG,),
    spec_fn=get_floating_sharpa_spec,
    articulation=FLOATING_SHARPA_ARTICULATION,
  )


##
# Action scales (heuristic: 0.25 * effort_limit / stiffness).
##

SHARPA_ACTION_SCALE: dict[str, float] = {
  name: 0.25 * HAND_EFFORT_LIMIT[name] / HAND_KP[name] for name in HAND_JOINT_NAMES
}


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_floating_sharpa_robot_cfg())
  viewer.launch(robot.spec.compile())
