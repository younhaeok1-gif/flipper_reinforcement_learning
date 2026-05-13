# Cmd Vel Flipper Task Notes

This note captures the current structure and intent of the `cmd_vel_flipper`
task so future Codex sessions can recover the context quickly.

## Main Files

- `source/mobile_mani/mobile_mani/tasks/mobile/cmd_vel_flipper/cmd_vel_flipper_env_cfg.py`
  contains most of the task definition: scene, custom action term, observations,
  rewards, terminations, reset events, and the final env config.
- `source/mobile_mani/mobile_mani/tasks/mobile/cmd_vel_flipper/cmd_vel_flipper_env.py`
  defines `CmdVelFlipperEnv`, which owns the sampled `commanded_cmd_vel`.
- `source/mobile_mani/mobile_mani/tasks/mobile/cmd_vel_flipper/__init__.py`
  registers the Gym task as `Template-Mobile-CmdVel-Flipper-v0`.
- `source/mobile_mani/mobile_mani/tasks/mobile/cmd_vel_flipper/agents/`
  contains PPO configs for different RL backends.

## Task Concept

The robot receives an externally sampled planar velocity command:

- linear x velocity
- angular z velocity

The policy does not directly control the drive tracks. Instead, it controls only
one normalized action for the front flippers. The custom action term converts:

- sampled `cmd_vel` into left/right track wheel velocity targets
- policy action into front left/right flipper position targets

This means the learning problem is mainly: given the commanded motion and local
terrain observations, choose front flipper angles that help the robot move
smoothly and keep the flippers in useful positions.

## Command Sampling

`CmdVelFlipperEnv` creates `commanded_cmd_vel` and resamples it on reset.

Current ranges are in `CmdVelFlipperEnvCfg`:

- `command_lin_vel_range = (0.0, 0.6)`
- `command_ang_vel_range = (-0.4, 0.4)`

The helper `commanded_base_velocity(env)` returns this command and falls back to
zeros if the env does not have the buffer yet.

## Custom Action Term

`CmdVelFrontFlipperAction` has `action_dim == 1`.

Action flow:

1. Clamp and sanitize policy action to `[-1, 1]`.
2. Multiply by `angle_limit` to get the front flipper target angle.
3. Apply `joint_signs` to both front flipper joints.
4. Compute drive track velocity from `cmd_vel` plus simple P correction:
   - `lin_vel_p_gain`
   - `ang_vel_p_gain`
5. Send velocity targets to drive joints and position targets to flipper joints.

The accepted/clamped action is stored in `raw_actions`, so observations and
custom rewards can use the finite action value instead of trusting raw policy
output.

## Observations

The policy observation group concatenates:

- commanded velocity
- base linear velocity
- base angular velocity
- projected gravity
- front flipper joint positions
- compact terrain height grid
- front obstacle features
- previous accepted action

Important terrain helpers:

- `local_height_grid(...)` returns a 10-cell local height summary relative to
  support ground.
- `front_obstacle_features(...)` returns compact obstacle timing features:
  height, distance, presence, left/center/right heights, and work-zone height.

## Reward Structure

Current active rewards in `RewardsCfg`:

- `action_rate`: negative action-rate penalty using the clamped flipper action.
- `action_l2`: negative action magnitude penalty using the clamped flipper action.
- `flipper_front_terrain_alignment`: positive reward for aligning the front
  flipper vector with the steepest front terrain vector.
- `flipper_cruise_clearance`: positive reward for keeping front flipper tips near
  target clearance during cruise/idle, only while the tip rollers are not in
  contact.
- `termination`: large negative penalty on termination.

Several helper rewards are defined but not currently active:

- `velocity_tracking_exp`
- `obstacle_gated_velocity_tracking_exp`
- `excessive_flat_orientation_l2`
- `excessive_pitch_l2`
- `flipper_distal_contact_pitch_l2`
- `wheel_contact_smoothness_l2`
- `flipper_down_without_obstacle_l2`
- `base_orientation_stability_exp`
- `terrain_relative_orientation_l2`

Note: `wheel_contact_smoothness_l2` expects a `wheel_contacts` sensor, but the
current scene config does not define one. Do not enable it without adding the
sensor.

Note: a `track_wheel_contact_count_reward` experiment was removed because it
encouraged the policy to keep the flippers raised. The reward only saw wheel
contact count, so lowering the flippers could reduce short-term wheel contact
and become unintentionally discouraged.

## Front Terrain Alignment Reward

`flipper_front_terrain_alignment_exp(...)` is based on a paper-style idea:
compare the robot's front terrain direction with the flipper direction.

Implementation in this repo:

- Use `height_scanner` ray hits in front of the robot.
- Convert hit points into the robot body frame.
- Convert `support_scanner` hits into the robot body frame and fit a local
  support plane.
- Measure each front hit's signed height relative to that local support plane,
  instead of using world `z` height directly.
- For each front grid point, compute a terrain angle using
  `atan2(relative_height, local_x)`.
- Pick the front grid point with the largest terrain angle as the steepest
  terrain target.
- Build a terrain vector `[x, 0, relative_height]`.
- Build a flipper vector from robot root to the average of `ffl_roller_9` and
  `ffr_roller_9`.
- Project both vectors into the body x-z plane.
- Give an exponential reward when the angle between the two vectors is small.

The initial reward weight is deliberately modest so the policy does not ignore
other objectives and only chase terrain alignment.

## Recent Change: Cruise Clearance Became Praise

`flipper_cruise_clearance` used to be a penalty:

- function returned squared clearance error
- reward weight was negative

It has been changed to praise good behavior:

- function is now `flipper_cruise_clearance_exp`
- it returns an exponential reward close to `1.0` when the front flipper tips are
  near `target_clearance`
- it is active only when `ffl_roller_9` and `ffr_roller_9` contact force is below
  `tip_contact_force_threshold`
- it returns `0.0` when the robot is actively approaching an obstacle and should
  focus on obstacle handling instead of cruise posture
- reward weight is positive

Intent: encourage the robot when it keeps the flipper tips lightly above local
support ground during cruise/idle, instead of only punishing bad clearance.

## Distal Flipper Contact Pitch Penalty

The plain `excessive_pitch` reward only sees body orientation. It cannot tell
whether the robot pitched up because the front flippers were pressed into the
ground.

To target that failure mode more directly, the scene now includes a contact
sensor on the front flipper rollers:

- `ffl_roller_.*`
- `ffr_roller_.*`

The reward `flipper_distal_contact_pitch_l2(...)` reads this sensor and gives
larger weight to contacts closer to the tip. For the current naming scheme,
`*_roller_9` receives the largest contact weight.

It penalizes only when both are true:

- weighted distal front flipper contact force is above `contact_force_threshold`
- pitch-like body tilt is above `pitch_deadband`

The penalty also scales up with contact load, capped to avoid a huge unstable
reward spike. This is meant to discourage the policy from planting the front
flippers, especially their distal links, and lifting the chassis onto them.

## Proposed Next Experiment: Success Reset Bonus

Potential next shaping idea:

Reward the robot strongly and reset the episode when it maintains a good cruise
state for several seconds in a new environment.

Suggested success condition:

- front flipper tip clearance stays near the desired target
- base linear velocity tracks commanded linear velocity
- base angular velocity tracks commanded angular velocity
- body orientation remains reasonable
- all conditions hold continuously for `required_success_time`, for example
  `2.0` seconds

Suggested implementation shape:

1. Add a `success_time_buf` tensor to `CmdVelFlipperEnv`.
2. Reset `success_time_buf[env_ids] = 0.0` in `_reset_idx`.
3. Add a helper function in `cmd_vel_flipper_env_cfg.py` that computes whether
   each env is currently in the success state.
4. Each step, increment the buffer by `env.step_dt` while success is true, else
   reset it to zero.
5. Add a positive terminal bonus reward when
   `success_time_buf >= required_success_time`.
6. Add a success termination term so those envs reset and sample a new command
   and terrain state.

Important caution: a large terminal bonus can create shortcut behavior if it is
too large. Start with a moderate bonus, for example `10` to `30`, and keep the
dense tracking and clearance rewards active so the agent learns to maintain the
behavior instead of only chasing the terminal event.

## Absolute Path Caveat

The robot USD path in `CmdVelFlipperSceneCfg` is currently absolute:

`/home/mesylab/mobile_vel/mobile_mani/robots/mobile/tracked/tracked_v1.usd`

This is fine locally, but it is a portability risk if the repo is moved or used
on another machine.
