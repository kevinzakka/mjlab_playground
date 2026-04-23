"""Booster T1 getup environment configuration."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_playground.asset_zoo.robots.booster_t1.t1_constants import get_t1_robot_cfg
from mjlab_playground.getup import mdp
from mjlab_playground.getup.getup_env_cfg import make_getup_env_cfg

# Derived from home keyframe.
_TORSO_HEIGHT = 0.67
_WAIST_HEIGHT = 0.55


def booster_t1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 getup task configuration."""
  cfg = make_getup_env_cfg()

  # Setup.

  cfg.scene.entities = {"robot": get_t1_robot_cfg()}

  # Self-collision sensor.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)

  # Rewards.

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["joint_vel_hinge"].weight = 0.0
  cfg.rewards["joint_vel_hinge"].params["threshold"] = math.pi
  # Torso + waist height. Waist reward prevents "sitting on booty or knees" local
  # minimum where torso is high but waist (pelvis) stays near ground.
  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("Trunk",)
  )
  cfg.rewards["waist_height"] = RewardTermCfg(
    func=mdp.height_reward,
    weight=1.0,
    params={
      "desired_height": _WAIST_HEIGHT,
      "asset_cfg": SceneEntityCfg("robot", body_names=("Waist",)),
    },
  )
  # Per-joint posture std: tight hips, medium knees and ankles, loose arms and waist.
  cfg.rewards["posture"].params["std"] = {
    r".*_Hip_Roll": 0.08,
    r".*_Hip_Yaw": 0.08,
    r".*_Hip_Pitch": 0.12,
    r".*_Knee_Pitch": 0.15,
    r".*_Ankle_Pitch": 0.2,
    r".*_Ankle_Roll": 0.2,
    r"(AAHead_yaw|Head_pitch)": 0.15,
    r"(Waist|.*_Shoulder.*|.*_Elbow.*)": 0.5,
  }

  # Metrics.

  cfg.metrics["getup_success"].params["desired_height"] = _TORSO_HEIGHT

  # Events.

  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("Trunk",)
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
  cfg.events["reset_fallen_or_standing"].params["fall_height"] = 0.8

  # Curriculum.

  cfg.curriculum = {
    "action_rate_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_rate_l2",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 800 * 24, "weight": -0.05},
          {"step": 1200 * 24, "weight": -0.08},
          {"step": 1500 * 24, "weight": -0.1},
        ],
      },
    ),
    "joint_vel_hinge_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": 800 * 24, "weight": -0.01},
          {"step": 1200 * 24, "weight": -0.05},
          {"step": 1500 * 24, "weight": -0.1},
        ],
      },
    ),
  }

  # Misc.

  cfg.viewer.body_name = "Trunk"

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_fallen_or_standing"].params["fall_probability"] = 1.0

  return cfg
