import torch

from dataclasses import MISSING
from math import pi

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils


def commanded_base_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the externally commanded planar velocity [linear_x, angular_z]."""
    if not hasattr(env, "commanded_cmd_vel"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.commanded_cmd_vel


def velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    lin_std: float = 0.25,
    ang_std: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of the commanded cmd_vel by the measured base velocity."""
    robot = env.scene[asset_cfg.name]
    command = commanded_base_velocity(env)
    lin_error = torch.square(command[:, 0] - robot.data.root_lin_vel_b[:, 0])
    ang_error = torch.square(command[:, 1] - robot.data.root_ang_vel_b[:, 2])
    return torch.exp(-(lin_error / lin_std**2 + ang_error / ang_std**2))


def clamped_last_action(env: ManagerBasedRLEnv, action_name: str = "front_flipper") -> torch.Tensor:
    """Return the finite, clamped action stored by the flipper action term."""
    return env.action_manager.get_term(action_name).raw_actions


def clamped_action_l2(env: ManagerBasedRLEnv, action_name: str = "front_flipper") -> torch.Tensor:
    """Penalize the action actually accepted by the flipper action term."""
    actions = env.action_manager.get_term(action_name).raw_actions
    return torch.sum(torch.square(actions), dim=1)


def clamped_action_rate_l2(env: ManagerBasedRLEnv, action_name: str = "front_flipper") -> torch.Tensor:
    """Penalize changes in the finite, clamped flipper action."""
    action_term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(action_term.raw_actions - action_term.prev_raw_actions), dim=1)


def front_flipper_tip_z_w(
    env: ManagerBasedRLEnv,
    tip_frame_cfg: SceneEntityCfg = SceneEntityCfg("flipper_tip_frames"),
) -> torch.Tensor:
    """Return the average world-z position of the two front flipper tip links."""
    tip_frames = env.scene.sensors[tip_frame_cfg.name]
    tip_pos_w = tip_frames.data.target_pos_w
    tip_z_w = torch.mean(tip_pos_w[..., 2], dim=1)
    return torch.nan_to_num(tip_z_w, nan=0.0, posinf=0.0, neginf=0.0)


def wheel_contact_smoothness_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contacts"),
    force_threshold: float = 300.0,
    delta_threshold: float = 150.0,
    variation_scale: float = 0.25,
) -> torch.Tensor:
    """Penalize large main-wheel contact force spikes and abrupt force changes."""
    sensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.net_forces_w_history
    if force_history is None or force_history.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)

    force_now = torch.linalg.norm(force_history[:, 0], dim=-1)
    force_prev = torch.linalg.norm(force_history[:, 1], dim=-1)
    force_now = torch.nan_to_num(force_now, nan=0.0, posinf=force_threshold, neginf=0.0)
    force_prev = torch.nan_to_num(force_prev, nan=0.0, posinf=force_threshold, neginf=0.0)

    max_force = torch.max(force_now, dim=1).values
    max_delta = torch.max(torch.abs(force_now - force_prev), dim=1).values
    force_spike = torch.square(torch.clamp(max_force - force_threshold, min=0.0))
    force_variation = torch.square(torch.clamp(max_delta - delta_threshold, min=0.0))
    return force_spike + variation_scale * force_variation


def flipper_down_without_obstacle_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["flipper_front_.*_joint"]),
    front_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),
    obstacle_x_range: tuple[float, float] = (0.0, 0.9),
    obstacle_y_range: tuple[float, float] = (-0.5, 0.5),
    obstacle_threshold: float = 0.06,
    command_threshold: float = 0.05,
    down_angle_deadband: float = 0.15,
) -> torch.Tensor:
    """Penalize digging the front flippers downward when there is no useful obstacle to engage."""
    robot = env.scene[asset_cfg.name]
    front_sensor = env.scene.sensors[front_sensor_cfg.name]
    support_sensor = env.scene.sensors[support_sensor_cfg.name]

    points_w = front_sensor.data.ray_hits_w
    finite = torch.isfinite(points_w).all(dim=-1)
    safe_points_w = torch.where(finite.unsqueeze(-1), points_w, robot.data.root_pos_w.unsqueeze(1))
    rel_points_w = safe_points_w - robot.data.root_pos_w.unsqueeze(1)
    num_rays = points_w.shape[1]
    points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_rays, dim=0),
        rel_points_w.reshape(-1, 3),
    ).view_as(rel_points_w)

    support_points_w = support_sensor.data.ray_hits_w
    support_finite = torch.isfinite(support_points_w).all(dim=-1)
    support_count = support_finite.sum(dim=1).clamp_min(1)
    support_ground_z = torch.where(
        support_finite,
        support_points_w[..., 2],
        torch.zeros_like(support_points_w[..., 2]),
    ).sum(dim=1) / support_count

    relative_height = torch.clamp(safe_points_w[..., 2] - support_ground_z.unsqueeze(1), min=0.0, max=1.0)
    front_mask = (
        finite
        & (points_b[..., 0] >= obstacle_x_range[0])
        & (points_b[..., 0] <= obstacle_x_range[1])
        & (points_b[..., 1] >= obstacle_y_range[0])
        & (points_b[..., 1] <= obstacle_y_range[1])
    )
    obstacle_height = torch.max(torch.where(front_mask, relative_height, torch.zeros_like(relative_height)), dim=1).values
    obstacle_active = obstacle_height > obstacle_threshold
    command_active = commanded_base_velocity(env)[:, 0] > command_threshold

    flipper_angles = robot.data.joint_pos[:, asset_cfg.joint_ids]
    downward_angle = torch.clamp(flipper_angles, min=0.0)
    downward_penalty = torch.mean(torch.square(torch.clamp(downward_angle - down_angle_deadband, min=0.0)), dim=1)
    should_not_dig = ~(obstacle_active & command_active)
    return torch.where(should_not_dig, downward_penalty, torch.zeros_like(downward_penalty))


def flipper_cruise_clearance_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),
    tip_frame_cfg: SceneEntityCfg = SceneEntityCfg("flipper_tip_frames"),
    obstacle_x_range: tuple[float, float] = (0.0, 0.9),
    obstacle_y_range: tuple[float, float] = (-0.5, 0.5),
    obstacle_threshold: float = 0.06,
    command_threshold: float = 0.05,
    target_clearance: float = 0.03,
    deadband: float = 0.02,
    std: float = 0.08,
) -> torch.Tensor:
    """Reward front flipper tips staying lightly above the local support ground during cruise/idle."""
    robot = env.scene[asset_cfg.name]
    front_sensor = env.scene.sensors[front_sensor_cfg.name]
    support_sensor = env.scene.sensors[support_sensor_cfg.name]

    points_w = front_sensor.data.ray_hits_w
    finite = torch.isfinite(points_w).all(dim=-1)
    safe_points_w = torch.where(finite.unsqueeze(-1), points_w, robot.data.root_pos_w.unsqueeze(1))
    rel_points_w = safe_points_w - robot.data.root_pos_w.unsqueeze(1)
    num_rays = points_w.shape[1]
    points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_rays, dim=0),
        rel_points_w.reshape(-1, 3),
    ).view_as(rel_points_w)

    support_points_w = support_sensor.data.ray_hits_w
    support_finite = torch.isfinite(support_points_w).all(dim=-1)
    support_count = support_finite.sum(dim=1).clamp_min(1)
    support_ground_z = torch.where(
        support_finite,
        support_points_w[..., 2],
        torch.zeros_like(support_points_w[..., 2]),
    ).sum(dim=1) / support_count

    relative_height = torch.clamp(safe_points_w[..., 2] - support_ground_z.unsqueeze(1), min=0.0, max=1.0)
    front_mask = (
        finite
        & (points_b[..., 0] >= obstacle_x_range[0])
        & (points_b[..., 0] <= obstacle_x_range[1])
        & (points_b[..., 1] >= obstacle_y_range[0])
        & (points_b[..., 1] <= obstacle_y_range[1])
    )
    obstacle_height = torch.max(torch.where(front_mask, relative_height, torch.zeros_like(relative_height)), dim=1).values
    obstacle_active = obstacle_height > obstacle_threshold
    command_active = commanded_base_velocity(env)[:, 0] > command_threshold

    tip_clearance = front_flipper_tip_z_w(env, tip_frame_cfg) - support_ground_z

    clearance_error = torch.abs(tip_clearance - target_clearance)
    clearance_error = torch.clamp(clearance_error - deadband, min=0.0)
    clearance_reward = torch.exp(-torch.square(clearance_error / std))
    cruise_or_idle = ~(obstacle_active & command_active)
    return torch.where(cruise_or_idle, clearance_reward, torch.zeros_like(clearance_reward))


def excessive_pitch_l2(
    env: ManagerBasedRLEnv,
    deadband: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing the robot up on its front flippers."""
    robot = env.scene[asset_cfg.name]
    pitch_like_tilt = torch.abs(robot.data.projected_gravity_b[:, 0])
    return torch.square(torch.clamp(pitch_like_tilt - deadband, min=0.0))


def flipper_distal_contact_pitch_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("front_flipper_contacts"),
    contact_force_threshold: float = 20.0,
    contact_force_scale: float = 150.0,
    pitch_deadband: float = 0.25,
    max_roller_index: int = 9,
) -> torch.Tensor:
    """Penalize pitching up with contact load concentrated toward the front flipper tips."""
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]

    tip_force = torch.linalg.norm(sensor.data.net_forces_w, dim=-1)
    tip_force = torch.nan_to_num(tip_force, nan=0.0, posinf=contact_force_threshold, neginf=0.0)

    distal_weights = []
    for body_name in sensor.body_names:
        try:
            roller_index = int(body_name.rsplit("_", 1)[-1])
        except ValueError:
            roller_index = max_roller_index
        distal_weights.append((roller_index + 1) / (max_roller_index + 1))
    distal_weights = torch.tensor(distal_weights, device=env.device).unsqueeze(0)
    distal_contact_load = torch.sum(tip_force * distal_weights, dim=1)

    pitch_like_tilt = torch.abs(robot.data.projected_gravity_b[:, 0])
    pitch_penalty = torch.square(torch.clamp(pitch_like_tilt - pitch_deadband, min=0.0))
    contact_active = distal_contact_load > contact_force_threshold
    contact_load = 1.0 + torch.clamp(
        (distal_contact_load - contact_force_threshold) / contact_force_scale,
        min=0.0,
        max=2.0,
    )
    return torch.where(contact_active, pitch_penalty * contact_load, torch.zeros_like(pitch_penalty))


def base_orientation_stability_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward low roll/pitch angular velocity to discourage jolting over obstacles."""
    robot = env.scene[asset_cfg.name]
    roll_pitch_rate_l2 = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)
    return torch.exp(-roll_pitch_rate_l2 / std**2)


def excessive_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    deadband: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only excessive roll/pitch tilt while allowing moderate terrain-induced inclination."""
    robot = env.scene[asset_cfg.name]
    tilt = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1)
    return torch.square(torch.clamp(tilt - deadband, min=0.0))


def local_height_grid(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("terrain_grid_scanner"),
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),
) -> torch.Tensor:
    """Return max terrain height per local grid cell, relative to support ground height."""
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    support_sensor = env.scene.sensors[support_sensor_cfg.name]

    points_w = sensor.data.ray_hits_w
    finite = torch.isfinite(points_w).all(dim=-1)
    safe_points_w = torch.where(finite.unsqueeze(-1), points_w, robot.data.root_pos_w.unsqueeze(1))
    rel_points_w = safe_points_w - robot.data.root_pos_w.unsqueeze(1)
    num_rays = points_w.shape[1]
    points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_rays, dim=0),
        rel_points_w.reshape(-1, 3),
    ).view_as(rel_points_w)

    support_points_w = support_sensor.data.ray_hits_w
    support_finite = torch.isfinite(support_points_w).all(dim=-1)
    support_count = support_finite.sum(dim=1).clamp_min(1)
    support_ground_z = torch.where(
        support_finite,
        support_points_w[..., 2],
        torch.zeros_like(support_points_w[..., 2]),
    ).sum(dim=1) / support_count

    x_bins = ((0.7, 1.0), (0.3, 0.7), (0.0, 0.3), (-0.3, 0.0), (-0.7, -0.3))
    y_bins = ((-0.5, 0.0), (0.0, 0.5))
    cell_heights = []
    for x_min, x_max in x_bins:
        for y_min, y_max in y_bins:
            cell_mask = (
                finite
                & (points_b[..., 0] >= x_min)
                & (points_b[..., 0] < x_max)
                & (points_b[..., 1] >= y_min)
                & (points_b[..., 1] < y_max)
            )
            masked_z = torch.where(cell_mask, safe_points_w[..., 2], torch.full_like(points_w[..., 2], -1.0e6))
            max_z = torch.max(masked_z, dim=1).values
            has_hit = cell_mask.any(dim=1)
            relative_height = torch.where(has_hit, max_z - support_ground_z, torch.zeros_like(max_z))
            cell_heights.append(torch.clamp(relative_height, -0.5, 1.0))

    return torch.stack(cell_heights, dim=1)


def front_obstacle_features(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),
    obstacle_x_range: tuple[float, float] = (0.0, 1.0),
    obstacle_y_range: tuple[float, float] = (-0.5, 0.5),
    work_zone_x_range: tuple[float, float] = (0.25, 0.75),
    work_zone_y_range: tuple[float, float] = (-0.45, 0.45),
    obstacle_threshold: float = 0.06,
    max_distance: float = 1.0,
) -> torch.Tensor:
    """Return compact front-terrain features for flipper timing.

    Features are [height, distance, presence, left_height, center_height, right_height, work_zone_height].
    Heights are relative to the support ground under the robot. Distance is the closest local x position
    whose relative height exceeds the obstacle threshold, or max_distance when no obstacle is detected.
    """
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    support_sensor = env.scene.sensors[support_sensor_cfg.name]

    points_w = sensor.data.ray_hits_w
    finite = torch.isfinite(points_w).all(dim=-1)
    safe_points_w = torch.where(finite.unsqueeze(-1), points_w, robot.data.root_pos_w.unsqueeze(1))
    rel_points_w = safe_points_w - robot.data.root_pos_w.unsqueeze(1)
    num_rays = points_w.shape[1]
    points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_rays, dim=0),
        rel_points_w.reshape(-1, 3),
    ).view_as(rel_points_w)

    support_points_w = support_sensor.data.ray_hits_w
    support_finite = torch.isfinite(support_points_w).all(dim=-1)
    support_count = support_finite.sum(dim=1).clamp_min(1)
    support_ground_z = torch.where(
        support_finite,
        support_points_w[..., 2],
        torch.zeros_like(support_points_w[..., 2]),
    ).sum(dim=1) / support_count

    relative_height = torch.clamp(safe_points_w[..., 2] - support_ground_z.unsqueeze(1), min=0.0, max=1.0)
    front_mask = (
        finite
        & (points_b[..., 0] >= obstacle_x_range[0])
        & (points_b[..., 0] <= obstacle_x_range[1])
        & (points_b[..., 1] >= obstacle_y_range[0])
        & (points_b[..., 1] <= obstacle_y_range[1])
    )
    obstacle_mask = front_mask & (relative_height > obstacle_threshold)

    def max_height(mask: torch.Tensor) -> torch.Tensor:
        masked_height = torch.where(mask, relative_height, torch.zeros_like(relative_height))
        return torch.max(masked_height, dim=1).values

    obstacle_height = max_height(front_mask)
    obstacle_presence = obstacle_mask.any(dim=1).float()
    masked_x = torch.where(obstacle_mask, points_b[..., 0], torch.full_like(points_b[..., 0], max_distance))
    obstacle_distance = torch.min(masked_x, dim=1).values.clamp(0.0, max_distance)

    left_mask = front_mask & (points_b[..., 1] >= 0.15)
    center_mask = front_mask & (points_b[..., 1] > -0.15) & (points_b[..., 1] < 0.15)
    right_mask = front_mask & (points_b[..., 1] <= -0.15)
    work_zone_mask = (
        finite
        & (points_b[..., 0] >= work_zone_x_range[0])
        & (points_b[..., 0] <= work_zone_x_range[1])
        & (points_b[..., 1] >= work_zone_y_range[0])
        & (points_b[..., 1] <= work_zone_y_range[1])
    )

    features = torch.stack(
        [
            obstacle_height,
            obstacle_distance,
            obstacle_presence,
            max_height(left_mask),
            max_height(center_mask),
            max_height(right_mask),
            max_height(work_zone_mask),
        ],
        dim=1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)


def obstacle_gated_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    lin_std: float = 0.25,
    ang_std: float = 0.4,
    obstacle_threshold: float = 0.08,
    obstacle_scale: float = 0.25,
) -> torch.Tensor:
    """Reward velocity tracking, but soften it when the front scanner sees a relevant obstacle."""
    tracking = velocity_tracking_exp(env, lin_std=lin_std, ang_std=ang_std)
    obstacle_features = front_obstacle_features(env)
    obstacle_height = obstacle_features[:, 0]
    near_obstacle = obstacle_height > obstacle_threshold
    return torch.where(near_obstacle, tracking * obstacle_scale, tracking)


def terrain_relative_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),
    deadband: float = 0.02,
) -> torch.Tensor:
    """Penalize body up direction error relative to the locally fitted terrain normal."""
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]

    points_w = sensor.data.ray_hits_w
    finite = torch.isfinite(points_w).all(dim=-1)
    finite_count = finite.sum(dim=1, keepdim=True).clamp_min(1)
    points_sum = torch.where(finite.unsqueeze(-1), points_w, torch.zeros_like(points_w)).sum(dim=1, keepdim=True)
    centroid = points_sum / finite_count.unsqueeze(-1)
    points_w = torch.where(finite.unsqueeze(-1), points_w, centroid)

    centered_points = points_w - points_w.mean(dim=1, keepdim=True)
    _, _, vh = torch.linalg.svd(centered_points)
    terrain_normal_w = torch.nn.functional.normalize(vh[:, -1, :], dim=1)
    terrain_normal_w = torch.where(terrain_normal_w[:, 2:3] < 0.0, -terrain_normal_w, terrain_normal_w)

    up_axis_b = torch.zeros(env.num_envs, 3, device=env.device)
    up_axis_b[:, 2] = 1.0
    robot_up_w = math_utils.quat_apply(robot.data.root_quat_w, up_axis_b)

    alignment_error = 1.0 - torch.sum(robot_up_w * terrain_normal_w, dim=1).clamp(-1.0, 1.0)
    alignment_error = torch.clamp(alignment_error - deadband, min=0.0)
    return torch.where(finite_count.squeeze(1) >= 3, alignment_error, torch.zeros_like(alignment_error))


def flipper_tip_obstacle_height_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),
    tip_frame_cfg: SceneEntityCfg = SceneEntityCfg("flipper_tip_frames"),
    obstacle_x_range: tuple[float, float] = (0.7, 1.0),
    obstacle_y_range: tuple[float, float] = (-0.5, 0.5),
    obstacle_threshold: float = 0.08,
    clearance: float = 0.03,
    top_k: int = 5,
) -> torch.Tensor:
    """Penalize front flipper tip height error against the estimated upcoming obstacle height."""
    robot = env.scene[asset_cfg.name]
    front_sensor = env.scene.sensors[front_sensor_cfg.name]
    support_sensor = env.scene.sensors[support_sensor_cfg.name]

    points_w = front_sensor.data.ray_hits_w
    rel_points_w = points_w - robot.data.root_pos_w.unsqueeze(1)
    num_rays = points_w.shape[1]
    points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_rays, dim=0),
        rel_points_w.reshape(-1, 3),
    ).view_as(rel_points_w)
    finite = torch.isfinite(points_w).all(dim=-1)
    front_mask = (
        finite
        & (points_b[..., 0] >= obstacle_x_range[0])
        & (points_b[..., 0] <= obstacle_x_range[1])
        & (points_b[..., 1] >= obstacle_y_range[0])
        & (points_b[..., 1] <= obstacle_y_range[1])
    )

    masked_front_z = torch.where(front_mask, points_w[..., 2], torch.full_like(points_w[..., 2], -1.0e6))
    top_values = torch.topk(masked_front_z, k=min(top_k, masked_front_z.shape[1]), dim=1).values
    top_valid = top_values > -1.0e5
    top_count = top_valid.sum(dim=1).clamp_min(1)
    front_ground_z = torch.where(top_valid, top_values, torch.zeros_like(top_values)).sum(dim=1) / top_count

    support_points_w = support_sensor.data.ray_hits_w
    support_finite = torch.isfinite(support_points_w).all(dim=-1)
    support_count = support_finite.sum(dim=1).clamp_min(1)
    support_ground_z = torch.where(
        support_finite,
        support_points_w[..., 2],
        torch.zeros_like(support_points_w[..., 2]),
    ).sum(dim=1) / support_count

    obstacle_height = torch.clamp(front_ground_z - support_ground_z, min=0.0)
    reward_active = (front_mask.sum(dim=1) > 0) & (obstacle_height > obstacle_threshold)

    tip_z_w = front_flipper_tip_z_w(env, tip_frame_cfg)
    target_tip_z_w = support_ground_z + obstacle_height + clearance
    tip_error = torch.abs(tip_z_w - target_tip_z_w)
    return torch.where(reward_active, tip_error, torch.zeros_like(tip_error))


@configclass
class CmdVelFlipperSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/mesylab/mobile_vel/mobile_mani/robots/mobile/tracked/tracked_v1.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        actuators={
            "drive_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    "(left_wheel|right_wheel|ffl_roller|ffr_roller)_.*_joint",
                ],
                stiffness=0.0,
                damping=10000.0,
            ),
            "front_flipper_joints": ImplicitActuatorCfg(
                joint_names_expr=["flipper_front_(left|right)_joint"],
                stiffness=50000.0,
                damping=5000.0,
            ),
        },
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=3.0,
            dynamic_friction=3.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        debug_vis=False,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
        offset=RayCasterCfg.OffsetCfg(pos=(0.8, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=GridPatternCfg(resolution=0.2, size=(1.6, 1.2)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    support_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=GridPatternCfg(resolution=0.15, size=(1.0, 0.7)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    terrain_grid_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
        offset=RayCasterCfg.OffsetCfg(pos=(0.15, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=GridPatternCfg(resolution=0.1, size=(1.7, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    flipper_tip_frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="front_left_tip",
                prim_path="{ENV_REGEX_NS}/Robot/ffl_roller_9",
            ),
            FrameTransformerCfg.FrameCfg(
                name="front_right_tip",
                prim_path="{ENV_REGEX_NS}/Robot/ffr_roller_9",
            ),
        ],
    )

    front_flipper_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(ffl_roller_.*|ffr_roller_.*)",
        history_length=3,
        debug_vis=False,
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )


@configclass
class CmdVelFrontFlipperActionCfg(ActionTermCfg):
    class_type = None
    asset_name: str = MISSING
    action_mode: str = "stepped"
    flipper_joint_names: list[str] = MISSING
    left_drive_joint_expr: str = MISSING
    right_drive_joint_expr: str = MISSING
    angle_limit: float = MISSING
    stepped_angle_limit_deg: float = 90.0
    stepped_angle_step_deg: float = 15.0
    stepped_action_deadband: float = 0.33
    joint_signs: list[float] = MISSING
    wheel_base: float = MISSING
    wheel_radius: float = MISSING
    lin_vel_p_gain: float = 0.0
    ang_vel_p_gain: float = 0.0
    max_lin_vel: float = 0.5
    max_ang_vel: float = MISSING


class CmdVelFrontFlipperAction(ActionTerm):
    """Policy controls only front flipper angle; cmd_vel drives the tracks."""

    cfg: CmdVelFrontFlipperActionCfg

    def __init__(self, cfg: CmdVelFrontFlipperActionCfg, env):
        super().__init__(cfg, env)
        self.left_drive_joint_ids, _ = self._asset.find_joints(cfg.left_drive_joint_expr)
        self.right_drive_joint_ids, _ = self._asset.find_joints(cfg.right_drive_joint_expr)

        flipper_joint_ids = []
        for joint_name in cfg.flipper_joint_names:
            joint_ids, _ = self._asset.find_joints(joint_name)
            flipper_joint_ids.extend(joint_ids)

        self._drive_joint_ids = torch.cat(
            [
                torch.tensor(self.left_drive_joint_ids, device=self.device),
                torch.tensor(self.right_drive_joint_ids, device=self.device),
            ]
        )
        self._flipper_joint_ids = torch.tensor(flipper_joint_ids, device=self.device)
        self._joint_signs = torch.tensor(cfg.joint_signs, device=self.device).unsqueeze(0)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_raw_actions = torch.zeros_like(self._raw_actions)
        self._flipper_targets = torch.zeros(self.num_envs, len(self._flipper_joint_ids), device=self.device)
        self._drive_targets = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)

        stepped_limit = cfg.stepped_angle_limit_deg * pi / 180.0
        stepped_step = cfg.stepped_angle_step_deg * pi / 180.0
        self._stepped_angle_bins = torch.arange(
            -stepped_limit,
            stepped_limit + 0.5 * stepped_step,
            stepped_step,
            device=self.device,
        )
        self._stepped_angle_index = torch.full(
            (self.num_envs,),
            len(self._stepped_angle_bins) // 2,
            device=self.device,
            dtype=torch.long,
        )

    @property
    def action_dim(self):
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def prev_raw_actions(self) -> torch.Tensor:
        return self._prev_raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._flipper_targets

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._prev_raw_actions[env_ids] = 0.0
        self._stepped_angle_index[env_ids] = len(self._stepped_angle_bins) // 2

    def process_actions(self, actions):
        self._prev_raw_actions[:] = self._raw_actions
        safe_actions = torch.nan_to_num(actions.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        safe_actions = torch.clamp(safe_actions, -1.0, 1.0)

        if self.cfg.action_mode == "continuous":
            self._raw_actions[:] = safe_actions
            flipper_angle = self._raw_actions[:, 0] * self.cfg.angle_limit
        elif self.cfg.action_mode == "stepped":
            action_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            action_step = torch.where(
                safe_actions[:, 0] > self.cfg.stepped_action_deadband,
                torch.ones_like(action_step),
                action_step,
            )
            action_step = torch.where(
                safe_actions[:, 0] < -self.cfg.stepped_action_deadband,
                -torch.ones_like(action_step),
                action_step,
            )
            self._stepped_angle_index[:] = torch.clamp(
                self._stepped_angle_index + action_step,
                0,
                len(self._stepped_angle_bins) - 1,
            )
            flipper_angle = self._stepped_angle_bins[self._stepped_angle_index]
            self._raw_actions[:, 0] = flipper_angle / self._stepped_angle_bins[-1].clamp_min(1.0e-6)
        else:
            raise ValueError(f"Unsupported action_mode: {self.cfg.action_mode}")

        self._flipper_targets[:] = flipper_angle.unsqueeze(1) * self._joint_signs

    def apply_actions(self):
        command = commanded_base_velocity(self._env)
        root_lin_vel_b = self._asset.data.root_lin_vel_b[:, 0]
        root_ang_vel_b = self._asset.data.root_ang_vel_b[:, 2]

        lin_error = command[:, 0] - root_lin_vel_b
        ang_error = command[:, 1] - root_ang_vel_b
        lin_vel = command[:, 0] + self.cfg.lin_vel_p_gain * lin_error
        ang_vel = command[:, 1] + self.cfg.ang_vel_p_gain * ang_error
        lin_vel = torch.clamp(lin_vel, 0.0, self.cfg.max_lin_vel)
        ang_vel = torch.clamp(ang_vel, -self.cfg.max_ang_vel, self.cfg.max_ang_vel)

        left_track_vel = lin_vel - (ang_vel * self.cfg.wheel_base / 2.0)
        right_track_vel = lin_vel + (ang_vel * self.cfg.wheel_base / 2.0)
        left_wheel_vel = left_track_vel / self.cfg.wheel_radius
        right_wheel_vel = right_track_vel / self.cfg.wheel_radius

        left_targets = left_wheel_vel.unsqueeze(1).repeat(1, len(self.left_drive_joint_ids))
        right_targets = right_wheel_vel.unsqueeze(1).repeat(1, len(self.right_drive_joint_ids))
        self._drive_targets[:] = torch.cat([left_targets, right_targets], dim=1)

        self._asset.set_joint_velocity_target(self._drive_targets, joint_ids=self._drive_joint_ids)
        self._asset.set_joint_position_target(self._flipper_targets, joint_ids=self._flipper_joint_ids)


CmdVelFrontFlipperActionCfg.class_type = CmdVelFrontFlipperAction


@configclass
class ActionsCfg:
    front_flipper = CmdVelFrontFlipperActionCfg(
        class_type=CmdVelFrontFlipperAction,
        asset_name="robot",
        action_mode="continuous",
        flipper_joint_names=[
            "flipper_front_left_joint",
            "flipper_front_right_joint",
        ],
        left_drive_joint_expr="(left_wheel|ffl_roller)_.*_joint",
        right_drive_joint_expr="(right_wheel|ffr_roller)_.*_joint",
        angle_limit=1.2,
        joint_signs=[1.0, 1.0],
        wheel_base=0.5,
        wheel_radius=0.1,
        lin_vel_p_gain=0.5,
        ang_vel_p_gain=0.3,
        max_lin_vel=0.5,
        max_ang_vel=0.8,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        cmd_vel = ObsTerm(func=commanded_base_velocity, scale=1.0)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, scale=1.0)
        flipper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["flipper_front_.*_joint"])},
            scale=1.0,
        )
        terrain_height_grid = ObsTerm(func=local_height_grid, scale=1.0)
        front_obstacle_features = ObsTerm(func=front_obstacle_features, scale=1.0)
        actions = ObsTerm(func=clamped_last_action, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    velocity_tracking = RewTerm(func=obstacle_gated_velocity_tracking_exp, weight=0.5)
    action_rate = RewTerm(func=clamped_action_rate_l2, weight=-0.02)
    action_l2 = RewTerm(func=clamped_action_l2, weight=-0.02)
    excessive_flat_orientation = RewTerm(func=excessive_flat_orientation_l2, weight=-1.0)
    excessive_pitch = RewTerm(func=excessive_pitch_l2, weight=-1.0)
    flipper_distal_contact_pitch = RewTerm(func=flipper_distal_contact_pitch_l2, weight=-5.0)
    flipper_cruise_clearance = RewTerm(func=flipper_cruise_clearance_exp, weight=0.5)
    termination = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})


@configclass
class EventsCfg:
    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
        mode="reset",
    )


@configclass
class CmdVelFlipperEnvCfg(ManagerBasedRLEnvCfg):
    decimation: int = 4
    episode_length_s: float = 20.0
    viewer = ViewerCfg(
        eye=(-6.0, 0.0, 5.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )

    scene: CmdVelFlipperSceneCfg = CmdVelFlipperSceneCfg(num_envs=256, env_spacing=15.0)

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    command_lin_vel_range: tuple[float, float] = (0.0, 0.6)
    command_ang_vel_range: tuple[float, float] = (-0.4, 0.4)
