# Dexterous Tool Task

## Summary

This task trains a KUKA iiwa14 arm with a SHARPA hand to pick up a hammer-like tool from a table and match a sequence of 6-DoF goal poses.

The task is defined by:

- base task config: `src/mjlab_playground/dexterous_tool/dexterous_tool_env_cfg.py`
- robot-specific config: `src/mjlab_playground/dexterous_tool/config/kuka_sharpa/env_cfgs.py`
- MDP implementation: `src/mjlab_playground/dexterous_tool/mdp/`

## Scene

The scene contains:

- `robot`: KUKA iiwa14 + SHARPA hand
- `tool`: procedural hammer with one free joint
- `table`: static box support surface
- `terrain`: flat plane with textures and materials disabled

The hammer asset has:

- one cylindrical `handle`
- one box `head`
- four keypoint sites `keypoint_0..3`
- one `grasp_center` site at the tool body origin

The table asset has:

- one box geom `table_geom`
- four optional support-footprint sites `support_corner_0..3`

## Task Definition

At reset:

- tool geometry is randomized
- a first goal pose is sampled in a workspace above the table
- the tool is placed flat on the table

During the episode:

- the policy must lift and manipulate the tool
- success is defined by keypoint alignment between the current tool pose and the goal pose
- after sustained success, a new goal is sampled as a bounded delta from the previous goal

This is therefore a sequential in-episode goal tracking task, not a single-goal-per-episode task.

## Action Space

The current action space is split into two terms:

- `arm_joint_pos`: delta joint-position action on the 7 KUKA arm actuators
- `hand_joint_pos`: delta joint-position action on the 22 SHARPA hand actuators

Current scales:

- arm: `0.0125`
- hand: `0.025`

Total action dimension: `29`

Action semantics:

- each policy output is interpreted as a delta on the previous commanded joint target
- the commanded target is updated once per policy step
- the stored target is then held constant across the MuJoCo decimation substeps

Simulation timing:

- MuJoCo timestep: `0.005`
- control decimation: `4`
- policy step: `0.02 s`

## Observations

### Actor Observations

The actor observation set is:

1. `joint_pos`
2. `joint_vel`
3. `prev_action_targets`
4. `palm_pose`
5. `fingertip_pos_rel_palm`
6. `object_orientation`
7. `keypoints_rel_palm`
8. `keypoints_rel_goal`
9. `object_scales`

### Actor Observation Semantics

`joint_pos`

- robot joint positions relative to default
- gives the policy proprioceptive state

`joint_vel`

- robot joint velocities relative to default
- gives local motion state for damping and timing

`prev_action_targets`

- previous commanded joint targets for arm and hand
- exposes the controller state used by the delta-target action term

`palm_pose`

- world-frame palm position and quaternion
- provides the global pose of the hand base for reaching and reorientation

`fingertip_pos_rel_palm`

- fingertip positions expressed relative to the palm
- describes hand shape independently of global translation

`object_orientation`

- tool root quaternion in world frame
- gives the current orientation of the tool independent of position

`keypoints_rel_palm`

- current tool keypoint positions relative to the palm
- tells the policy where the tool is with respect to the hand

`keypoints_rel_goal`

- current tool keypoints minus desired goal keypoints
- is the main pose-tracking error signal

`object_scales`

- current geometry-dependent size observation from the selected grasp geom
- exposes domain-randomized tool scale to the policy

For the KUKA+SHARPA config:

- arm joints: `7`
- hand joints: `22`
- fingertips: `5`
- keypoints: `4`

Actor observation dimension: `140`

### Critic Observations

The critic gets all actor observations plus:

1. `palm_velocity`
2. `object_velocity`
3. `closest_keypoint_max_dist`
4. `closest_fingertip_dist`
5. `lifted_object`
6. `progress`
7. `successes`
8. `reward`

### Critic Observation Semantics

`palm_velocity`

- world-frame palm linear and angular velocity
- gives privileged dynamic state of the hand base

`object_velocity`

- world-frame tool linear and angular velocity
- gives privileged dynamic state of the manipulated object

`closest_keypoint_max_dist`

- best max-keypoint error achieved for the current goal
- exposes progress memory used by the shaped keypoint reward

`closest_fingertip_dist`

- best fingertip-to-tool distances achieved so far in the episode
- exposes progress memory used by the grasp-approach reward

`lifted_object`

- binary indicator that the tool has crossed the lift threshold
- tells the critic whether the episode has entered the post-lift phase

`progress`

- log-scaled episode progress signal
- gives the critic coarse temporal context

`successes`

- log-scaled number of goal resamples in the current episode
- tells the critic how many goals have already been solved

`reward`

- scaled current reward
- provides a privileged summary of instantaneous task performance

Critic observation dimension: `162`

## Command and Goal Sampling

The command term is `ToolGoalPoseCommand`.

It maintains:

- `goal_pos`
- `goal_quat`
- `goal_keypoints_w`
- `object_keypoints_w`
- per-episode state for lift, grasp progress, and success counting

### Initial Goal Sampling

The first goal is sampled by:

- sampling a random orientation
- reading the current randomized tool geometry
- computing the tool support bounds under that orientation
- sampling a root pose inside the configured workspace with table-clearance enforcement

### Delta Goal Sampling

Subsequent goals are sampled by:

- bounded Cartesian delta
- bounded Euler-angle delta
- workspace clamping with geometry-aware table clearance

### Goal Success

Let

`d_i = ||goal_keypoint_i - object_keypoint_i||`

Then success uses

`max_i d_i < success_tolerance`

for `success_steps` consecutive control steps.

Current defaults:

- `success_tolerance = 0.075 m`
- `success_steps = 10`

## Object Reset

The tool is reset by the command term, not by a separate reset event.

Current KUKA+SHARPA object reset:

- root pose sampled over the full tabletop bounds
- orientation fixed to lie flat on the table
- yaw randomized
- support height adjusted using the current randomized geometry

The reset uses full geometry-aware support bounds so the tool remains supported by the table.

## Rewards

The reward set is:

1. `fingertip_approach`
2. `lift_object`
3. `keypoint_goal`
4. `goal_success_bonus`
5. `arm_velocity_penalty`
6. `hand_velocity_penalty`
7. `arm_dof_pos_limits`
8. `hand_dof_pos_limits`
9. `action_rate_l2`

### Reward Semantics

`fingertip_approach`

- dense delta reward on reduction in fingertip-to-`grasp_center` distance
- uses the best-so-far fingertip distances stored in the command term

`lift_object`

- dense reward on vertical lift above reset height
- one-time sparse bonus when lift exceeds `lift_threshold`

`keypoint_goal`

- dense delta reward on reduction in max keypoint error
- active only after the tool is considered lifted

`goal_success_bonus`

- sparse bonus spread across `success_steps`
- active only while lifted and within success tolerance

`arm_velocity_penalty`, `hand_velocity_penalty`

- L1 joint-velocity penalties on the selected joint subsets

`arm_dof_pos_limits`, `hand_dof_pos_limits`

- stock soft joint-limit penalties on the selected joint subsets

`action_rate_l2`

- L2 penalty on change in raw policy action

## Terminations

The termination set is:

- timeout
- object fallen
- object dropped after lift
- hand too far

### Termination Semantics

`object_fallen`

- terminates when tool root height relative to `env_origin` falls below `0.32`

`object_dropped_after_lift`

- once the object has been lifted, terminates if current tool root height falls below reset height

`hand_too_far`

- terminates when the maximum fingertip-to-`grasp_center` distance exceeds `0.45`

## Domain Randomization

The main domain randomization term is `randomize_tool_geometry`.

It samples:

- handle scale
- head scale

Current ranges:

- handle: `0.7 .. 1.25`
- head: `0.7 .. 1.25`

For each sampled tool, it updates coherently:

- `geom_size`
- `geom_pos` for the head
- keypoint site positions
- `grasp_center`
- geom bounds
- body mass
- body COM
- body inertia

The command term then uses the current randomized geometry for:

- object reset support height
- tabletop footprint bounds
- goal table-clearance checks

## Visualization

Optional command debug visualization includes:

- translucent goal ghost of the tool
- desired COM frame
- current COM frame
- optional goal-center sphere
- optional goal-keypoint spheres
- optional table support-footprint corner sites

Current KUKA+SHARPA play camera:

- anchored to the table body
- focused slightly above the tabletop workspace

## Current Design Notes

Important current semantics:

- the task uses actual tool keypoint sites, not synthetic keypoints
- the task uses free-space goals above the table, not tabletop-constrained target footprints
- grasp-distance logic is site-based and uses `grasp_center`
- the tool root frame is currently handle-centered because `alignfree` is not enabled

## File Map

- `dexterous_tool_env_cfg.py`: base task wiring
- `config/kuka_sharpa/env_cfgs.py`: robot-specific asset, workspace, camera, and subsets
- `mdp/commands.py`: goal sampling, object reset, debug visualization, state trackers
- `mdp/events.py`: geometry randomization and support-bound utilities
- `mdp/observations.py`: actor and critic observation functions
- `mdp/rewards.py`: dense and sparse reward terms
- `mdp/terminations.py`: failure conditions
- `config/kuka_sharpa/rl_cfg.py`: PPO defaults
