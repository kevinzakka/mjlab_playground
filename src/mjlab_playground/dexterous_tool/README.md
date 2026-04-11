# Dexterous Tool Task

A dexterous arm-and-hand grasps a procedural hammer from a table and tracks a
sequence of 6-DoF goal poses. The episode does not end on success — when the
agent stabilizes at the current goal, a new bounded delta-goal is sampled and
the episode continues.

| File | Role |
| --- | --- |
| `dexterous_tool_env_cfg.py` | base task wiring (rewards, obs, sim, terminations) |
| `config/kuka_sharpa/env_cfgs.py` | KUKA iiwa14 + SHARPA hand asset and tuning |
| `mdp/commands.py` | `ToolGoalPoseCommand`: goal sampling + tool reset |
| `mdp/observations.py` | observation functions |
| `mdp/rewards.py` | reward functions |
| `mdp/terminations.py` | termination functions |
| `mdp/events.py` | reset events and tool geometry randomization |

## Scene

- `robot`: KUKA iiwa14 + SHARPA hand (29 actuated joints: 7 arm + 22 hand)
- `tool`: floating-base hammer with one cylindrical handle, one box head,
  and a `grasp_center` site at the body origin
- `table`: jointless box welded to an auto-wrapped mocap parent (acts as a
  static obstacle; contributes 0 DoF)

## Action space

| Term | Joints | Type | Action dim |
| --- | --- | --- | --- |
| `arm_joint_pos` | 7 | relative joint position | 7 |
| `hand_joint_pos` | 22 | relative joint position | 22 |

Each policy output is a delta on the previous commanded joint target. Targets
are held constant across `decimation=4` MuJoCo substeps. Physics step
`0.005 s`; control step `0.02 s`.

## Observations

The actor and critic observe the same set (no privileged terms). All spatial
observations are anchored to the **palm** frame — the frame whose pose
responds directly to joint actions via forward kinematics. This is the same
"anchor to the action-controlled body" principle used by mjlab's tracking
task; for tracking the action-controlled body is the pelvis, for our task
it's the palm.

| Term | Shape | Description |
| --- | --- | --- |
| `arm_joint_pos` | (B, 7) | arm joint positions, relative to default |
| `arm_joint_vel` | (B, 7) | arm joint velocities |
| `hand_joint_pos` | (B, 22) | hand joint positions, relative to default |
| `hand_joint_vel` | (B, 22) | hand joint velocities |
| `fingertip_pos_in_palm` | (B, 15) | 5 fingertip positions in the palm's local frame |
| `tool_pose_in_palm` | (B, 9) | tool pose in the palm's local frame: `[pos(3), ori_6d(6)]` |
| `goal_pose_in_palm` | (B, 9) | goal pose in the palm's local frame: `[pos(3), ori_6d(6)]` |
| `goal_pose_in_tool` | (B, 9) | redundant: goal pose in the tool's local frame (the SE(3) tracking error) |
| `actions` | (B, 29) | previous policy action |

There is **no world-frame observation**. For a fixed-base robot, palm pose
in world is fully determined by `arm_joint_pos` via FK, so exposing it is
redundant. Removing it makes the observation set frame-coherent.

### Reference frames

All `_in_palm` observations are computed via `subtract_frame_transforms`
against the palm site:

```
T_in_palm = T_palm⁻¹ · T_target
pos_in_palm = quat_inv(palm_quat) · (target_pos − palm_pos)
ori_in_palm = quat_inv(palm_quat) · target_quat
```

The 6D rotation representation (Zhou et al. 2019) is the first two rows of
the rotation matrix derived from `ori_in_palm`. It is continuous,
singularity-free, double-cover safe (`q` and `−q` give the same observation),
and invariant to global translation and rotation of the scene.

`goal_pose_in_tool` is the same construction with the tool body as the
anchor instead of the palm. It directly encodes the SE(3) error the policy
must drive to identity. It is redundant with `goal_pose_in_palm` +
`tool_pose_in_palm` (the policy could compute it), but providing it as an
explicit feature gives a direct gradient signal at no real cost (9 dims).

Identity values when everything is aligned:

```
fingertip_pos_in_palm     → constant per-grasp configuration
tool_pose_in_palm         → constant after a stable grasp
goal_pose_in_palm         → (0, 0, 0, 1, 0, 0, 0, 1, 0) at the goal
goal_pose_in_tool         → (0, 0, 0, 1, 0, 0, 0, 1, 0) at the goal
```

## Rewards

The task signal is **one** multiplicatively-staged term that bridges
reach → lift → 6-DoF tracking. Each factor is a bounded `[0, 1]` shaping
signal; the product spans `[0, 3]`. This is the same trick mjlab's lift-cube
task uses (`reach · (1 + bring)`), recursed one level for the extra lift
phase.

```
staged_track = approach · (1 + height · (1 + airborne · track))
```

| Term | Weight | Form |
| --- | --- | --- |
| `staged_track` | +1.0 | `approach · (1 + height · (1 + airborne · track))`, range `[0, 3]` |
| `arm_posture` | +0.01 | nullspace regularization toward home pose |
| `arm_action_rate` | −0.001 | L2 on `arm_action[t] − arm_action[t−1]` |
| `hand_action_rate` | −0.0001 | L2 on `hand_action[t] − hand_action[t−1]` |
| `arm_joint_pos_limits` | −10.0 | soft-limit penalty on arm joints |
| `hand_joint_pos_limits` | −10.0 | soft-limit penalty on hand joints |
| `arm_joint_vel_hinge` | −0.001 | hinge above `\|q̇\| > 0.5 rad/s` |
| `hand_joint_vel_hinge` | −0.001 | hinge above `\|q̇\| > 0.5 rad/s` |
| `hand_table_collision` | −0.01 | max contact force in the hand–table contact sensor |

### The four factors

**`approach`** — multi-scale Gaussian on the mean fingertip→`grasp_center`
distance, `fingertip_stds = (0.4, 0.1) m`. The wide std gives the long-range
pre-grasp signal; the narrow std the close-range grasp alignment.

**`height`** — Gaussian on the env-local height *deficit*
`max(0, height_target − (obj_z − env_origin_z))` with `height_target = 0.6 m`,
`σ = 0.1 m`. Saturates at 1.0 once the tool reaches `height_target` (~22 cm
above the `0.38 m` table top, just past the command's `lift_threshold`). The
`max(0, ·)` clamp means there is no penalty for going higher. This is the
smooth signal that bridges "fingers gripping tool on table" to "tool in air"
— the only piece in the staged form that gives gradient through the lift
transition itself.

**`airborne`** — hard `{0, 1}` indicator from the `tool_table_collision`
contact sensor: `1` iff zero contacts are reported between any tool geom and
the table body this step. This is the **principled** "tool is held in the
air" signal — geometry-independent (works for any tool shape),
state-independent (no sticky flags), and impossible to exploit (you cannot
be both touching and not touching the table simultaneously). Closes the
slide-along-table exploit and the stand-the-tool-on-its-head exploit that
any purely-geometric height proxy would admit.

**`track`** — `(pos_gauss + ori_gauss) / 2` over `pos_stds = (0.3, 0.03) m`
and `ori_stds = (60°, 5°)`. Position uses L2 distance (rotation-invariant);
orientation uses `quat_error_magnitude` (frame-invariant and double-cover
safe — `q` and `−q` give identical reward). Position and orientation are
*summed*, not multiplied: they are two projections of the same SE(3) error,
not separate phases, and multiplying would punish "perfect position, wrong
orientation" too harshly.

### Multi-scale Gaussian

```
multiscale_gaussian(err, stds) = mean_{s ∈ stds} exp(−err² / s²)
```

A length-1 `stds` tuple is exactly a plain Gaussian. Length-2 averages a
wide shaping Gaussian and a narrow precision Gaussian into one bounded
reward that has meaningful gradient at every error magnitude. The
arithmetic mean keeps each factor in `[0, 1]`.

### Why staged multiplication

Per-phase behaviour with the current scales:

| Phase | `approach` | `height` | `airborne` | `track` | `staged_track` | live gradient |
| --- | --- | --- | --- | --- | --- | --- |
| Pre-grasp (fingers far) | small | ≈0 | 0 | 0 | ≈ `approach` | `approach` |
| Grasped on table | ≈1 | ramping | 0 | 0 | `1 + height` ∈ [1, 2] | `height` |
| Just lifted | ≈1 | ≈1 | 1 | small | ≈ 2 + small | `track` (now unlocked) |
| Tracking goal | ≈1 | ≈1 | 1 | →1 | →3 | `track` precision tail |

Two properties this gives us:

1. **No phase plateau.** Some factor always has a live gradient. There is
   no flat region the policy can park on.

2. **No free constants.** After grasp, `approach ≈ 1` looks like a constant,
   but it is the multiplier that unlocks the `(1 + height · …)` bonus —
   without it the downstream stages collapse. Same for `height` after lift.
   Every saturated factor is doing the gating job for the stage above it.
   This is the structural property that makes the staged form preferable to
   four additive terms with the same sub-pieces.

Dropping the tool back onto the table immediately flips `airborne` to 0 and
zeroes `track`, so no episode-level "drop termination" is needed — the
per-step penalty is sharp.

## Command (`ToolGoalPoseCommand`)

State maintained per env:

- `goal_pos`, `goal_quat`
- `lifted_object` (sticky bool, true once `obj_z > reset_z + lift_threshold`)
- `consecutive_successes`, `num_goal_resets`
- `object_initial_pos_w` (snapshot at reset)

### Goal sampling

The first goal in an episode is sampled from a workspace box above the table:
random orientation, then a position uniformly sampled with the workspace
shrunk by the tool's current oriented extents and a minimum table clearance.

After the agent stabilizes at a goal (see *Success* below), a new goal is
sampled as a bounded delta from the current goal:

- position delta: uniform in `±delta_position` per axis (default `±0.1 m`)
- rotation delta: uniform in `±delta_rotation_deg` per Euler axis (default
  `±90°`)

then clamped back into the workspace.

### Success

Decoupled tolerance, applied per env:

```
pos_err = ||goal_pos − tool_pos||
ori_err = quat_error_magnitude(goal_quat, tool_quat)   # radians

at_goal = (pos_err < pos_tolerance) ∧ (ori_err < ori_tolerance)
```

When `at_goal` holds for `success_steps` consecutive control steps, the
agent's `consecutive_successes` counter increments and a new delta-goal is
sampled. Defaults: `pos_tolerance = 2.5 cm`, `ori_tolerance = 15°`,
`success_steps = 10` (≈ 0.2 s at 50 Hz control).

### Object reset

The command writes the tool's reset state directly (no separate event):

- root pose sampled from `object_pose_range` (default: tabletop xy box, lying
  flat with random yaw)
- support height adjusted via `tool_support_height` so the lowest tool point
  rests on the table given its current randomized geometry and orientation
- linear and angular velocities zeroed

## Terminations

| Term | Condition |
| --- | --- |
| `time_out` | episode length exceeded (default `10 s`) |
| `object_fallen` | `tool_z − env_origin_z < 0.32 m` |
| `object_velocity_exceeded` | `\|\|v\|\| > 5.0 m/s` or `\|\|ω\|\| > 20 rad/s` (loose safety net for sim blow-ups; tighten via `Episode_Metrics/object_lin_speed`/`object_ang_speed`) |
| `hand_too_far` | max fingertip-to-`grasp_center` distance > `0.45 m` |
| `arm_collision` | contact force on arm bodies (`link3..link7`) above threshold |

In play mode, `object_velocity_exceeded` is removed because mouse drag in the
viewer easily clears the threshold.

## Metrics (logged via `MetricsManager`)

| Metric | Source |
| --- | --- |
| `object_lin_speed` | `\|\|tool.root_link_lin_vel_w\|\|` |
| `object_ang_speed` | `\|\|tool.root_link_ang_vel_w\|\|` |
| `pose_pos_err` | meters (from `ToolGoalPoseCommand.metrics`) |
| `pose_ori_err_deg` | degrees |
| `lifted_object` | float of the sticky bool |
| `consecutive_successes` | per-step success-counter snapshot |
| `num_goal_resets` | cumulative per episode |

## Sim and timing

| Field | Value |
| --- | --- |
| `mujoco.timestep` | 0.005 s |
| `decimation` | 4 |
| control step | 0.02 s |
| `episode_length_s` | 10 s |
| `iterations` | 10 |
| `ls_iterations` | 20 |
| `nconmax` | 100 |
| `njmax` | 500 |
