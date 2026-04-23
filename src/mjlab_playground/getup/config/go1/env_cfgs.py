"""Unitree Go1 getup environment configuration."""

import math

from mjlab.asset_zoo.robots import get_go1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_playground.getup import mdp
from mjlab_playground.getup.getup_env_cfg import make_getup_env_cfg

_TORSO_HEIGHT = 0.275


def unitree_go1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 getup task configuration."""
  cfg = make_getup_env_cfg()

  robot_cfg = get_go1_robot_cfg()

  robot_cfg.collisions = (
    CollisionCfg(
      geom_names_expr=(".*_collision",),
      solref=(0.01, 1),
      condim=3,
      friction=(0.6,),
      priority=1,
    ),
  )

  cfg.scene.entities = {"robot": robot_cfg}

  # Self-collision sensor (history_length matches decimation=4).
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)

  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("trunk",)
  )
  cfg.metrics["getup_success"].params["desired_height"] = _TORSO_HEIGHT

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Per-joint posture std: tight hips (prevent splay), medium thighs, looser calves.
  cfg.rewards["posture"].params["std"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }

  cfg.rewards["joint_vel_hinge"].weight = -0.05
  cfg.rewards["joint_vel_hinge"].params["threshold"] = 2 * math.pi

  cfg.viewer.body_name = "trunk"

  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("trunk",)
  )
  cfg.events["geom_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  # Scale static friction: *U(0.5, 1.5).
  cfg.events["joint_frictionloss_scale"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "ranges": (0.5, 1.5),
    },
  )
  # Scale armature: *U(0.95, 1.05).
  cfg.events["joint_armature_scale"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "ranges": (0.95, 1.05),
    },
  )

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_fallen_or_standing"].params["fall_probability"] = 1.0

  return cfg
