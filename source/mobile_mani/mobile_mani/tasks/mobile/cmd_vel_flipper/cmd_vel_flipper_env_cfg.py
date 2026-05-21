import torch

from dataclasses import MISSING
from math import pi

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils


STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 6.0),
    border_width=20.0,
    num_rows=8,
    num_cols=4,
    curriculum=True,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.04, 0.18),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


def commanded_base_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the externally commanded planar velocity [linear_x, angular_z]."""
    if not hasattr(env, "commanded_cmd_vel"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.commanded_cmd_vel


def stairs_terrain_levels(
    env: ManagerBasedRLEnv,     #RL environment 객체 
    env_ids,                    #이번에 curriculum을 업데이트할 env index들
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),    #어떤 asset을 기준으로 볼지 설정
    success_distance: float = 3.0,      #거리보다 앞으로 가면 성공 terrain level 상승
    failure_distance: float = 0.5,      #거리보다 못가면 하락
) -> torch.Tensor:
    """Increase stair difficulty when the robot moves far enough along +x."""
    if isinstance(env_ids, slice):                  
        env_ids = torch.arange(env.num_envs, device=env.device)                 #env_ids가 slice같은 형태로 들어올 때 env index를 직접 tensor로 만든다
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)    #env_ids가 python list나 tuple이면 torch tensor로 바꿈

    robot = env.scene[asset_cfg.name]   #로봇 현재 위치, 속도, 관절 상태 등등 가져옴
    terrain = env.scene.terrain         #terrain정보를 가지고옴
    forward_distance = robot.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]    #로봇이 얼마나 이동했는지 계산
    move_up = forward_distance > success_distance       #이동 얼마 이상 했으면 move up
    move_down = forward_distance < failure_distance     #이동 얼마 이상 못했으면 move down
    move_down &= ~move_up               #move up, down이 동시에 true가 되는 상황을 막음
    terrain.update_env_origins(env_ids, move_up, move_down) #isaac lab에서 terrain level을 업데이트 함
    return torch.mean(terrain.terrain_levels.float())   #현재 전체 env의 평균 terrain level반환


def velocity_tracking_exp(
    env: ManagerBasedRLEnv,         #객체
    lin_std: float = 0.25,          #선속도의 오차를 얼마나 민감하게 볼지 정하는 값(작을수록 선속도 오차에 엄격함)
    ang_std: float = 0.4,           #각속도에 오차를 얼마나 민감하게 볼지 정하는 값
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),    #어떤 asset의 속도를 볼지 설정
) -> torch.Tensor:
    """Reward tracking of the commanded cmd_vel by the measured base velocity."""
    robot = env.scene[asset_cfg.name]       #robot의 asset을 가져옴
    command = commanded_base_velocity(env)  #env에 저장된 명령 속도 가져옴
    lin_error = torch.square(command[:, 0] - robot.data.root_lin_vel_b[:, 0])   #에러값 계산
    ang_error = torch.square(command[:, 1] - robot.data.root_ang_vel_b[:, 2])   #에러값 계산
    return torch.exp(-(lin_error / lin_std**2 + ang_error / ang_std**2))    #에러값 기반 최종 reward 계산


def clamped_last_action(env: ManagerBasedRLEnv, action_name: str = "front_flipper") -> torch.Tensor:
    """Return the finite, clamped action stored by the flipper action term."""
    return env.action_manager.get_term(action_name).raw_actions  # action term 안에 저장된 안전한 action값 반환


def clamped_action_l2(env: ManagerBasedRLEnv, action_name: str = "front_flipper") -> torch.Tensor:
    """Penalize the action actually accepted by the flipper action term."""
    actions = env.action_manager.get_term(action_name).raw_actions    # clamp/sanitize된 실제 적용 action
    return torch.sum(torch.square(actions), dim=1)                    # action 크기 제곱 패널티


def clamped_action_rate_l2(env: ManagerBasedRLEnv, action_name: str = "front_flipper") -> torch.Tensor:
    """Penalize changes in the finite, clamped flipper action."""
    action_term = env.action_manager.get_term(action_name)    # 현재 action term 가져옴
    return torch.sum(torch.square(action_term.raw_actions - action_term.prev_raw_actions), dim=1)  # action 변화량 패널티


def front_flipper_tip_z_w(      #앞쪽 플리퍼 끝부분의 world z높이를 계산하는 함수
    env: ManagerBasedRLEnv,     #객체
    tip_frame_cfg: SceneEntityCfg = SceneEntityCfg("flipper_tip_frames"),   #sensor 설정, 어떤 센서를 사용할지 정함, 지금은 flipper_tip_frames
) -> torch.Tensor:
    """Return the average world-z position of the two front flipper tip links."""
    tip_frames = env.scene.sensors[tip_frame_cfg.name]    # flipper tip frame sensor 가져옴
    tip_pos_w = tip_frames.data.target_pos_w              # tip들의 world 좌표
    tip_z_w = torch.mean(tip_pos_w[..., 2], dim=1)        # 좌우 tip의 z높이를 평균냄
    return torch.nan_to_num(tip_z_w, nan=0.0, posinf=0.0, neginf=0.0)  # 비정상값 방지


def wheel_contact_smoothness_l2(
    env: ManagerBasedRLEnv,     # 객체
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contacts"),    # 휠 contact sensor
    force_threshold: float = 300.0,       # 힘이 이 값보다 크면 spike로 봄
    delta_threshold: float = 150.0,       # 이전 step 대비 force 변화량 기준
    variation_scale: float = 0.25,        # force 변화량 penalty 비중
) -> torch.Tensor:
    """Penalize large main-wheel contact force spikes and abrupt force changes."""
    sensor = env.scene.sensors[sensor_cfg.name]        # contact sensor 가져옴
    force_history = sensor.data.net_forces_w_history   # 최근 contact force history
    if force_history is None or force_history.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)   # history가 부족하면 penalty 없음

    force_now = torch.linalg.norm(force_history[:, 0], dim=-1)    # 현재 contact force 크기
    force_prev = torch.linalg.norm(force_history[:, 1], dim=-1)   # 이전 contact force 크기
    force_now = torch.nan_to_num(force_now, nan=0.0, posinf=force_threshold, neginf=0.0)
    force_prev = torch.nan_to_num(force_prev, nan=0.0, posinf=force_threshold, neginf=0.0)

    max_force = torch.max(force_now, dim=1).values                     # env별 최대 contact force
    max_delta = torch.max(torch.abs(force_now - force_prev), dim=1).values  # env별 최대 force 변화량
    force_spike = torch.square(torch.clamp(max_force - force_threshold, min=0.0))       # 큰 충격 penalty
    force_variation = torch.square(torch.clamp(max_delta - delta_threshold, min=0.0))   # 급격한 변화 penalty
    return force_spike + variation_scale * force_variation


def flipper_down_without_obstacle_l2(
    env: ManagerBasedRLEnv,     # 객체
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["flipper_front_.*_joint"]),  # front flipper joint
    front_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),      # 앞쪽 terrain scanner
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),   # 로봇 아래 지면 scanner
    obstacle_x_range: tuple[float, float] = (0.0, 0.9),   # 장애물 확인 x범위
    obstacle_y_range: tuple[float, float] = (-0.5, 0.5),  # 장애물 확인 y범위
    obstacle_threshold: float = 0.06,     # 이 높이 이상이면 장애물로 판단
    command_threshold: float = 0.05,      # 전진 command가 이 값보다 크면 주행 중으로 판단
    down_angle_deadband: float = 0.15,    # 이 정도 아래각은 허용
) -> torch.Tensor:
    """Penalize digging the front flippers downward when there is no useful obstacle to engage."""
    robot = env.scene[asset_cfg.name]                         # robot asset
    front_sensor = env.scene.sensors[front_sensor_cfg.name]    # 앞쪽 ray sensor
    support_sensor = env.scene.sensors[support_sensor_cfg.name]  # support ground sensor

    points_w = front_sensor.data.ray_hits_w    # 앞쪽 terrain hit point(world)
    finite = torch.isfinite(points_w).all(dim=-1)  # ray hit이 유효한지 확인
    safe_points_w = torch.where(finite.unsqueeze(-1), points_w, robot.data.root_pos_w.unsqueeze(1))  # invalid hit 대체
    rel_points_w = safe_points_w - robot.data.root_pos_w.unsqueeze(1)   # robot root 기준 상대좌표
    num_rays = points_w.shape[1]    # ray 개수
    points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_rays, dim=0),
        rel_points_w.reshape(-1, 3),
    ).view_as(rel_points_w)    # world 좌표를 body 좌표로 변환

    support_points_w = support_sensor.data.ray_hits_w          # 로봇 아래쪽 ray hit
    support_finite = torch.isfinite(support_points_w).all(dim=-1)
    support_count = support_finite.sum(dim=1).clamp_min(1)     # 유효한 support ray 개수
    support_ground_z = torch.where(
        support_finite,
        support_points_w[..., 2],
        torch.zeros_like(support_points_w[..., 2]),
    ).sum(dim=1) / support_count       # 로봇 아래 평균 지면 높이

    relative_height = torch.clamp(safe_points_w[..., 2] - support_ground_z.unsqueeze(1), min=0.0, max=1.0)  # 상대 높이
    front_mask = (
        finite
        & (points_b[..., 0] >= obstacle_x_range[0])
        & (points_b[..., 0] <= obstacle_x_range[1])
        & (points_b[..., 1] >= obstacle_y_range[0])
        & (points_b[..., 1] <= obstacle_y_range[1])
    )   # 앞쪽 관심 영역 mask
    obstacle_height = torch.max(torch.where(front_mask, relative_height, torch.zeros_like(relative_height)), dim=1).values
    obstacle_active = obstacle_height > obstacle_threshold          # 앞에 유효한 장애물이 있는지
    command_active = commanded_base_velocity(env)[:, 0] > command_threshold  # 전진 중인지

    flipper_angles = robot.data.joint_pos[:, asset_cfg.joint_ids]  # front flipper joint angle
    downward_angle = torch.clamp(flipper_angles, min=0.0)          # 아래로 내린 각도만 사용
    downward_penalty = torch.mean(torch.square(torch.clamp(downward_angle - down_angle_deadband, min=0.0)), dim=1)
    should_not_dig = ~(obstacle_active & command_active)           # 장애물 없거나 전진 중이 아니면 digging 방지
    return torch.where(should_not_dig, downward_penalty, torch.zeros_like(downward_penalty))


def flipper_cruise_clearance_exp(
    env: ManagerBasedRLEnv,     # 객체
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),             # robot asset
    front_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),      # 앞쪽 지형 scanner
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),   # 로봇 아래 지면 scanner
    tip_frame_cfg: SceneEntityCfg = SceneEntityCfg("flipper_tip_frames"),     # flipper tip frame sensor
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("front_flipper_contacts"),  # front flipper contact sensor
    obstacle_x_range: tuple[float, float] = (0.0, 0.9),   # 앞 장애물 확인 x범위
    obstacle_y_range: tuple[float, float] = (-0.5, 0.5),  # 앞 장애물 확인 y범위
    obstacle_threshold: float = 0.06,     # 이 높이 이상이면 장애물로 판단
    command_threshold: float = 0.05,      # 전진 command 기준
    tip_contact_force_threshold: float = 5.0,  # tip 접촉 여부 판단 force
    target_clearance: float = 0.03,       # 목표 tip clearance
    deadband: float = 0.02,               # clearance 오차 허용 범위
    std: float = 0.08,                    # exponential reward 민감도
) -> torch.Tensor:
    """Reward front flipper tips staying lightly above ground without touching during cruise/idle."""
    robot = env.scene[asset_cfg.name]                         # robot asset 가져옴
    front_sensor = env.scene.sensors[front_sensor_cfg.name]    # 앞쪽 ray sensor
    support_sensor = env.scene.sensors[support_sensor_cfg.name]  # support ground sensor
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]  # front flipper contact sensor

    points_w = front_sensor.data.ray_hits_w    # 앞쪽 지형 hit point(world)
    finite = torch.isfinite(points_w).all(dim=-1)
    safe_points_w = torch.where(finite.unsqueeze(-1), points_w, robot.data.root_pos_w.unsqueeze(1))  # invalid hit 대체
    rel_points_w = safe_points_w - robot.data.root_pos_w.unsqueeze(1)   # robot root 기준 상대좌표
    num_rays = points_w.shape[1]
    points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_rays, dim=0),
        rel_points_w.reshape(-1, 3),
    ).view_as(rel_points_w)    # body frame 좌표

    support_points_w = support_sensor.data.ray_hits_w
    support_finite = torch.isfinite(support_points_w).all(dim=-1)
    support_count = support_finite.sum(dim=1).clamp_min(1)
    support_ground_z = torch.where(
        support_finite,
        support_points_w[..., 2],
        torch.zeros_like(support_points_w[..., 2]),
    ).sum(dim=1) / support_count    # 현재 로봇 아래 평균 지면 높이

    relative_height = torch.clamp(safe_points_w[..., 2] - support_ground_z.unsqueeze(1), min=0.0, max=1.0)  # 앞 지형 상대 높이
    front_mask = (
        finite
        & (points_b[..., 0] >= obstacle_x_range[0])
        & (points_b[..., 0] <= obstacle_x_range[1])
        & (points_b[..., 1] >= obstacle_y_range[0])
        & (points_b[..., 1] <= obstacle_y_range[1])
    )   # 앞쪽 관심 영역
    obstacle_height = torch.max(torch.where(front_mask, relative_height, torch.zeros_like(relative_height)), dim=1).values
    obstacle_active = obstacle_height > obstacle_threshold          # 앞에 장애물이 있는지
    command_active = commanded_base_velocity(env)[:, 0] > command_threshold  # 전진 중인지

    tip_clearance = front_flipper_tip_z_w(env, tip_frame_cfg) - support_ground_z  # 지면 기준 tip 높이

    clearance_error = torch.abs(tip_clearance - target_clearance)   # 목표 clearance와의 차이
    clearance_error = torch.clamp(clearance_error - deadband, min=0.0)   # deadband 안쪽은 오차 0
    clearance_reward = torch.exp(-torch.square(clearance_error / std))   # clearance가 맞을수록 1에 가까움
    cruise_or_idle = ~(obstacle_active & command_active)     # 장애물 전진 상황이 아닐 때만 활성

    tip_body_ids = [
        body_id
        for body_id, body_name in enumerate(contact_sensor.body_names)
        if body_name in ("ffl_roller_9", "ffr_roller_9")
    ]   # 앞 플리퍼 tip roller body id 찾기
    if tip_body_ids:
        tip_forces = torch.linalg.norm(contact_sensor.data.net_forces_w[:, tip_body_ids], dim=-1)  # tip 접촉 force
        tip_forces = torch.nan_to_num(tip_forces, nan=0.0, posinf=tip_contact_force_threshold, neginf=0.0)
        max_tip_force = torch.max(tip_forces, dim=1).values  # 좌우 tip 중 큰 force
        tips_not_touching = max_tip_force <= tip_contact_force_threshold  # tip이 땅에 닿지 않았는지
    else:
        tips_not_touching = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)  # body id 못 찾으면 접촉 없음으로 처리

    reward_active = cruise_or_idle & tips_not_touching   # 순항/대기 + tip 비접촉일 때만 보상
    return torch.where(reward_active, clearance_reward, torch.zeros_like(clearance_reward))


def flipper_front_terrain_alignment_exp(
    env: ManagerBasedRLEnv,     # 객체
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),             # robot asset
    front_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),      # 앞쪽 지형 scanner
    tip_frame_cfg: SceneEntityCfg = SceneEntityCfg("flipper_tip_frames"),     # flipper tip frame sensor
    x_range: tuple[float, float] = (0.7, 0.9),       # 플리퍼가 따라갈 preview band의 앞쪽 x범위
    y_range: tuple[float, float] = (-0.35, 0.35),    # 플리퍼가 따라갈 preview band의 좌우 y범위
    top_k: int = 3,                       # 올라가는 자세에서 높은 지형 대표값을 만들 때 쓸 point 개수
    bottom_k: int = 3,                    # 내려가는 자세에서 낮은 지형 대표값을 만들 때 쓸 point 개수
    front_joint_x: float = 0.237,        # base_link 기준 front flipper joint x 위치
    target_z_offset: float = 0.21,       # 지형점보다 살짝 위를 보게 해서 평지에서 바닥을 찍는 방향을 완화
    pitch_blend: float = 0.25,           # pitch가 이 값에 가까워질수록 mean 대신 top/bottom을 더 강하게 사용
    pitch_sign: float = 1.0,             # 올라가는 자세에서 projected_gravity_b[:,0] 부호가 반대면 -1.0으로 변경
    min_vector_length: float = 0.05,     # 너무 짧은 vector는 무효 처리
    std: float = 0.45,                   # 각도 오차 reward 민감도
) -> torch.Tensor:
    """앞 플리퍼 링크 방향이 일정 거리 앞 지형 preview 방향과 비슷해지도록 보상한다.

    두 vector를 robot body x-z plane에서 만든다.
    1. terrain_vec_b: front flipper joint center -> 일정 거리 앞 preview band의 대표 지형점.
    2. flipper_vec_b: front flipper joint center -> 앞쪽 flipper tip 평균 위치.

    두 vector 사이 각도가 작을수록 reward가 커진다.
    """
    robot = env.scene[asset_cfg.name]                         # robot asset 가져옴
    front_sensor = env.scene.sensors[front_sensor_cfg.name]    # 앞쪽 ray sensor
    tip_frames = env.scene.sensors[tip_frame_cfg.name]         # flipper tip frame sensor

    # 1) 앞쪽 ray hit point들을 robot body frame으로 변환한다.
    #    이렇게 해야 "로봇 기준 앞쪽 x/y 범위"를 안정적으로 고를 수 있다.
    points_w = front_sensor.data.ray_hits_w    # 앞쪽 지형 hit point(world)
    finite = torch.isfinite(points_w).all(dim=-1) #유효한 hit인지 검사하는 코드
    safe_points_w = torch.where(finite.unsqueeze(-1), points_w, robot.data.root_pos_w.unsqueeze(1))  # invalid hit 대체
    rel_points_w = safe_points_w - robot.data.root_pos_w.unsqueeze(1)   # root 기준 상대좌표
    num_rays = points_w.shape[1]
    points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_rays, dim=0),
        rel_points_w.reshape(-1, 3),
    ).view_as(rel_points_w)    # 앞쪽 hit point를 body frame으로 변환

    # 2) URDF의 front flipper joint origin을 기준점으로 둔다.
    #    좌우 joint의 x 위치가 같으므로 y는 0으로 두고 중앙 joint처럼 사용한다.
    joint_center_b = torch.zeros(env.num_envs, 3, device=env.device)  # body frame의 joint center
    joint_center_b[:, 0] = front_joint_x                             # base_link 기준 front joint x

    # 3) 일정 거리 앞 preview band만 남긴다.
    #    "가장 높은 점"을 고르면 상승에는 유리하지만, 하강에서는 낮은 지형을 target으로 잡기 어렵다.
    #    그래서 x/y 범위 안의 대표 지형점을 그대로 따라가게 해서 위/평지/아래 지형을 모두 표현한다.
    preview_mask = (
        finite
        & (points_b[..., 0] >= x_range[0])
        & (points_b[..., 0] <= x_range[1])
        & (points_b[..., 1] >= y_range[0])
        & (points_b[..., 1] <= y_range[1])
    )   # preview band에 들어온 ray만 선택

    # 4) preview band 안의 point들에서 mean/top/bottom z를 각각 만든다.
    #    평지나 자세가 중립이면 mean_z를 보고, 올라가는 자세이면 top_z를 더 보고,
    #    내려가는 자세이면 bottom_z를 더 보도록 pitch에 따라 부드럽게 섞는다.
    #    이렇게 하면 올라가는 중 계단의 움푹 들어간 낮은 point에 덜 속고,
    #    내려가는 중에는 낮은 지형을 자연스럽게 따라갈 수 있다.
    preview_count = preview_mask.sum(dim=1).clamp_min(1)                 # 유효 preview point 개수
    preview_z_b = torch.where(
        preview_mask,
        points_b[..., 2],
        torch.zeros_like(points_b[..., 2]),
    )
    mean_z_b = preview_z_b.sum(dim=1) / preview_count                    # preview band 평균 지형 높이

    masked_top_z_b = torch.where(
        preview_mask,
        points_b[..., 2],
        torch.full_like(points_b[..., 2], -1.0e6),
    )                                                                    # top-k 계산용 invalid 제거
    top_values = torch.topk(masked_top_z_b, k=min(top_k, masked_top_z_b.shape[1]), dim=1).values
    top_valid = top_values > -1.0e5                                      # 실제 top-k point인지 확인
    top_count = top_valid.sum(dim=1).clamp_min(1)
    top_z_b = torch.where(top_valid, top_values, torch.zeros_like(top_values)).sum(dim=1) / top_count

    masked_bottom_z_b = torch.where(
        preview_mask,
        points_b[..., 2],
        torch.full_like(points_b[..., 2], 1.0e6),
    )                                                                    # bottom-k 계산용 invalid 제거
    bottom_values = torch.topk(-masked_bottom_z_b, k=min(bottom_k, masked_bottom_z_b.shape[1]), dim=1).values
    bottom_values = -bottom_values                                       # 가장 낮은 z값들을 다시 원래 부호로 복원
    bottom_valid = bottom_values < 1.0e5                                 # 실제 bottom-k point인지 확인
    bottom_count = bottom_valid.sum(dim=1).clamp_min(1)
    bottom_z_b = torch.where(bottom_valid, bottom_values, torch.zeros_like(bottom_values)).sum(dim=1) / bottom_count

    pitch_signal = pitch_sign * robot.data.projected_gravity_b[:, 0]     # 올라가는 자세를 + 방향으로 맞춘 pitch 신호
    up_weight = torch.clamp(pitch_signal / pitch_blend, min=0.0, max=1.0)        # 올라갈수록 top_z 비중 증가
    down_weight = torch.clamp(-pitch_signal / pitch_blend, min=0.0, max=1.0)     # 내려갈수록 bottom_z 비중 증가
    high_blend_z_b = (1.0 - up_weight) * mean_z_b + up_weight * top_z_b          # mean -> top 부드러운 보간
    low_blend_z_b = (1.0 - down_weight) * mean_z_b + down_weight * bottom_z_b    # mean -> bottom 부드러운 보간
    representative_z_b = torch.where(pitch_signal >= 0.0, high_blend_z_b, low_blend_z_b)

    representative_point_b = torch.zeros(env.num_envs, 3, device=env.device)  # preview band 대표 지형점
    representative_point_b[:, 0] = 0.5 * (x_range[0] + x_range[1])        # target x는 preview band 중앙으로 고정
    representative_point_b[:, 2] = representative_z_b                    # pitch에 따라 섞은 대표 z를 사용
    has_target = preview_mask.any(dim=1)                                 # preview band에 유효 point가 있는지

    # 5) front joint에서 preview 대표 지형점보다 target_z_offset만큼 위를 향하는 terrain vector를 만든다.
    #    평평한 지형 point는 joint보다 낮기 때문에 그대로 쓰면 플리퍼가 바닥을 찍는 방향도 보상을 받을 수 있다.
    #    offset만 더하고 clamp는 하지 않아서, 하강 지형의 아래 방향 정보는 완전히 지우지 않는다.
    terrain_vec_b = representative_point_b - joint_center_b              # joint -> 지형 대표점
    terrain_vec_b = torch.stack(
        [
            terrain_vec_b[:, 0],
            torch.zeros_like(terrain_vec_b[:, 0]),
            terrain_vec_b[:, 2] + target_z_offset,
        ],
        dim=1,
    )   # terrain vector를 x-z plane으로 투영하면서 target 높이에 여유를 줌
    terrain_vec_b = torch.where(
        has_target.unsqueeze(1),
        terrain_vec_b,
        torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0),
    )   # target이 없으면 기본 전방 vector 사용

    # 6) front joint에서 앞 플리퍼 tip 평균 위치까지의 flipper vector를 만든다.
    #    이렇게 해야 "플리퍼 링크 자체 방향"과 "지형 target 방향"을 같은 시작점에서 비교할 수 있다.
    tip_pos_w = torch.mean(tip_frames.data.target_pos_w, dim=1)   # 좌우 flipper tip 평균 위치
    tip_rel_pos_w = tip_pos_w - robot.data.root_pos_w             # robot root -> flipper tip 위치
    tip_pos_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, tip_rel_pos_w)  # tip 위치를 body frame으로 변환
    flipper_vec_b = tip_pos_b - joint_center_b                    # front joint -> flipper tip vector
    flipper_vec_b = torch.stack(
        [
            flipper_vec_b[:, 0],
            torch.zeros_like(flipper_vec_b[:, 0]),
            flipper_vec_b[:, 2],
        ],
        dim=1,     # flipper vector도 x-z plane만 사용
    )

    # 7) 두 vector가 충분히 길 때만 방향을 비교한다.
    #    짧은 vector를 normalize하면 노이즈가 커지기 때문에 무효 처리한다.
    terrain_len = torch.linalg.norm(terrain_vec_b, dim=1)     # 지형 vector 길이
    flipper_len = torch.linalg.norm(flipper_vec_b, dim=1)     # flipper vector 길이
    valid = (terrain_len > min_vector_length) & (flipper_len > min_vector_length)  # 너무 짧으면 무효
    terrain_dir = torch.nn.functional.normalize(terrain_vec_b, dim=1)
    flipper_dir = torch.nn.functional.normalize(flipper_vec_b, dim=1)

    # 8) 두 방향의 각도 오차를 exponential reward로 변환한다.
    #    완전히 같은 방향이면 reward가 1에 가깝고, 어긋날수록 0에 가까워진다.
    alignment = torch.sum(terrain_dir * flipper_dir, dim=1).clamp(-1.0, 1.0)  # 두 방향의 cosine 유사도
    angle_error = torch.acos(alignment)    # 두 vector 사이 각도 오차
    reward = torch.exp(-torch.square(angle_error / std))  # 각도 오차가 작을수록 큰 보상
    return torch.where(valid, reward, torch.zeros_like(reward))


def flipper_support_plane_alignment_exp(
    env: ManagerBasedRLEnv,     # 객체
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),             # robot asset
    front_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),      # 앞쪽 지형 scanner
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),   # 로봇 아래 지면 scanner
    tip_frame_cfg: SceneEntityCfg = SceneEntityCfg("flipper_tip_frames"),     # flipper tip frame sensor
    x_range: tuple[float, float] = (0.65, 0.85),       # 정렬 target을 찾을 앞쪽 x범위
    y_range: tuple[float, float] = (-0.45, 0.45),      # 정렬 target을 찾을 y범위
    min_vector_length: float = 0.05,                   # 너무 짧은 vector는 무효 처리
    std: float = 0.45,                                 # 각도 오차 reward 민감도
) -> torch.Tensor:
    """support plane 기준으로 앞쪽 상승 지형을 찾고, flipper 방향과 정렬시키는 이전 구조 reward."""
    robot = env.scene[asset_cfg.name]                         # robot asset 가져옴
    front_sensor = env.scene.sensors[front_sensor_cfg.name]    # 앞쪽 ray sensor
    support_sensor = env.scene.sensors[support_sensor_cfg.name]  # 로봇 아래쪽 ray sensor
    tip_frames = env.scene.sensors[tip_frame_cfg.name]         # flipper tip frame sensor

    # 1) 앞쪽 ray hit point들을 robot body frame으로 변환한다.
    #    world 좌표 그대로 쓰면 로봇 yaw/pitch에 따라 앞쪽 범위 선택이 흔들리므로 body frame에서 처리한다.
    points_w = front_sensor.data.ray_hits_w                    # 앞쪽 지형 hit point(world)
    finite = torch.isfinite(points_w).all(dim=-1)              # 유효한 hit인지 확인
    safe_points_w = torch.where(finite.unsqueeze(-1), points_w, robot.data.root_pos_w.unsqueeze(1))  # invalid hit 대체
    rel_points_w = safe_points_w - robot.data.root_pos_w.unsqueeze(1)   # root 기준 상대좌표
    num_rays = points_w.shape[1]                               # 앞쪽 scanner ray 개수
    points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_rays, dim=0),
        rel_points_w.reshape(-1, 3),
    ).view_as(rel_points_w)                                    # 앞쪽 hit point를 body frame으로 변환

    # 2) support scanner point들로 로봇 아래 local ground plane을 추정한다.
    #    이 plane은 "현재 발밑 지면" 기준이므로, 앞쪽 계단/턱이 얼마나 솟았는지 판단하는 기준이 된다.
    support_points_w = support_sensor.data.ray_hits_w          # 로봇 아래쪽 지면 hit point(world)
    support_finite = torch.isfinite(support_points_w).all(dim=-1)  # support hit 유효성
    safe_support_points_w = torch.where(
        support_finite.unsqueeze(-1),
        support_points_w,
        robot.data.root_pos_w.unsqueeze(1),
    )                                                          # invalid support hit 대체
    support_rel_points_w = safe_support_points_w - robot.data.root_pos_w.unsqueeze(1)  # root 기준 support 좌표
    num_support_rays = support_points_w.shape[1]               # support scanner ray 개수
    support_points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_support_rays, dim=0),
        support_rel_points_w.reshape(-1, 3),
    ).view_as(support_rel_points_w)                            # support point를 body frame으로 변환
    support_count = support_finite.sum(dim=1).clamp_min(1)     # 유효 support point 개수
    support_centroid_b = torch.where(
        support_finite.unsqueeze(-1),
        support_points_b,
        torch.zeros_like(support_points_b),
    ).sum(dim=1) / support_count.unsqueeze(-1)                 # support point 중심
    centered_support_points = torch.where(
        support_finite.unsqueeze(-1),
        support_points_b - support_centroid_b.unsqueeze(1),
        torch.zeros_like(support_points_b),
    )                                                          # plane fitting을 위해 중심 기준으로 이동
    _, _, vh = torch.linalg.svd(centered_support_points)       # SVD로 support plane normal 계산
    support_normal_b = torch.nn.functional.normalize(vh[:, -1, :], dim=1)  # 가장 작은 축이 plane normal
    support_normal_b = torch.where(support_normal_b[:, 2:3] < 0.0, -support_normal_b, support_normal_b)  # normal 위쪽 정렬

    # 3) 앞쪽 point들이 support plane보다 얼마나 높은지 계산한다.
    #    바닥 point는 relative_height가 작고, 계단/턱 point는 relative_height가 커져 target으로 선택되기 쉽다.
    relative_height = torch.sum(
        (points_b - support_centroid_b.unsqueeze(1)) * support_normal_b.unsqueeze(1),
        dim=-1,
    )                                                          # support plane 기준 앞쪽 지형 높이
    front_mask = (
        finite
        & (points_b[..., 0] >= x_range[0])
        & (points_b[..., 0] <= x_range[1])
        & (points_b[..., 1] >= y_range[0])
        & (points_b[..., 1] <= y_range[1])
    )                                                          # 앞쪽 관심 영역

    # 4) support plane 기준으로 가장 가파르게 올라간 앞쪽 point를 target으로 고른다.
    #    x가 가까우면서 relative_height가 높은 point일수록 큰 angle을 갖는다.
    terrain_angle = torch.atan2(relative_height, torch.clamp(points_b[..., 0], min=1.0e-3))  # 지형 상승 각도
    terrain_angle = torch.where(front_mask, terrain_angle, torch.full_like(terrain_angle, -pi))  # 영역 밖 제외
    target_ids = torch.argmax(terrain_angle, dim=1)            # 가장 큰 상승 각도를 가진 point 선택
    env_ids = torch.arange(env.num_envs, device=env.device)    # env index
    has_target = front_mask.any(dim=1)                         # 앞쪽 영역에 유효 target이 있는지

    # 5) robot root에서 target 지형까지의 terrain vector를 만든다.
    #    이전 구조와 동일하게 root 기준 vector를 쓰고, x-z plane에서만 비교한다.
    target_x = points_b[env_ids, target_ids, 0]                # target point body x
    target_z = relative_height[env_ids, target_ids]            # support plane 기준 target 높이
    terrain_vec_b = torch.stack(
        [
            target_x,
            torch.zeros_like(target_x),
            target_z,
        ],
        dim=1,
    )                                                          # root -> 앞쪽 상승 지형 vector
    terrain_vec_b = torch.where(
        has_target.unsqueeze(1),
        terrain_vec_b,
        torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0),
    )                                                          # target이 없으면 기본 전방 vector 사용

    # 6) robot root에서 앞 플리퍼 tip 평균 위치까지의 flipper vector를 만든다.
    #    이전 구조처럼 flipper tip이 root 기준 어느 방향에 있는지를 terrain vector와 비교한다.
    tip_pos_w = torch.mean(tip_frames.data.target_pos_w, dim=1)   # 좌우 front flipper tip 평균 위치
    flipper_vec_w = tip_pos_w - robot.data.root_pos_w             # root -> flipper tip vector(world)
    flipper_vec_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, flipper_vec_w)  # body frame 변환
    flipper_vec_b = torch.stack(
        [
            flipper_vec_b[:, 0],
            torch.zeros_like(flipper_vec_b[:, 0]),
            flipper_vec_b[:, 2],
        ],
        dim=1,
    )                                                          # flipper vector도 x-z plane만 사용

    # 7) 두 vector가 충분히 길 때만 normalize해서 방향을 비교한다.
    #    너무 짧은 vector는 작은 노이즈에도 방향이 크게 튀므로 reward를 0으로 둔다.
    terrain_len = torch.linalg.norm(terrain_vec_b, dim=1)      # terrain vector 길이
    flipper_len = torch.linalg.norm(flipper_vec_b, dim=1)      # flipper vector 길이
    valid = (terrain_len > min_vector_length) & (flipper_len > min_vector_length)  # 방향 비교 가능 여부
    terrain_dir = torch.nn.functional.normalize(terrain_vec_b, dim=1)  # terrain 방향 단위벡터
    flipper_dir = torch.nn.functional.normalize(flipper_vec_b, dim=1)  # flipper 방향 단위벡터

    # 8) 두 방향의 각도 오차를 exponential reward로 변환한다.
    #    방향이 같으면 1에 가깝고, 각도가 벌어질수록 0에 가까워진다.
    alignment = torch.sum(terrain_dir * flipper_dir, dim=1).clamp(-1.0, 1.0)  # cosine 유사도
    angle_error = torch.acos(alignment)                        # 두 vector 사이 각도
    reward = torch.exp(-torch.square(angle_error / std))       # 각도 오차 기반 reward
    return torch.where(valid, reward, torch.zeros_like(reward)) # 무효한 경우 reward 0


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
    env: ManagerBasedRLEnv,     # 객체
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),               # robot asset
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("terrain_grid_scanner"),        # local height grid용 scanner
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg("support_scanner"),     # 기준 지면 높이 scanner
    top_k: int = 3,     # 각 grid cell에서 높은 ray hit 몇 개를 평균낼지
) -> torch.Tensor:
    """Return top-k average terrain height per local grid cell in the robot body frame."""
    robot = env.scene[asset_cfg.name]                    # robot asset 가져옴
    sensor = env.scene.sensors[sensor_cfg.name]          # terrain grid scanner
    support_sensor = env.scene.sensors[support_sensor_cfg.name]  # support scanner

    points_w = sensor.data.ray_hits_w       # grid scanner hit point(world)
    finite = torch.isfinite(points_w).all(dim=-1)    # 유효 hit인지
    safe_points_w = torch.where(finite.unsqueeze(-1), points_w, robot.data.root_pos_w.unsqueeze(1))  # invalid hit 대체
    rel_points_w = safe_points_w - robot.data.root_pos_w.unsqueeze(1)   # robot root 기준 상대좌표
    num_rays = points_w.shape[1]
    points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_rays, dim=0),
        rel_points_w.reshape(-1, 3),
    ).view_as(rel_points_w)    # body frame으로 변환

    support_points_w = support_sensor.data.ray_hits_w
    support_finite = torch.isfinite(support_points_w).all(dim=-1)
    safe_support_points_w = torch.where(
        support_finite.unsqueeze(-1),
        support_points_w,
        robot.data.root_pos_w.unsqueeze(1),
    )   # invalid support hit 대체
    support_rel_points_w = safe_support_points_w - robot.data.root_pos_w.unsqueeze(1)  # robot root 기준 support 좌표
    num_support_rays = support_points_w.shape[1]
    support_points_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w.repeat_interleave(num_support_rays, dim=0),
        support_rel_points_w.reshape(-1, 3),
    ).view_as(support_rel_points_w)    # support point를 body frame으로 변환
    support_count = support_finite.sum(dim=1).clamp_min(1)   # 유효 support hit 개수
    support_ground_z_b = torch.where(
        support_finite,
        support_points_b[..., 2],
        torch.zeros_like(support_points_b[..., 2]),
    ).sum(dim=1) / support_count      # body frame 기준 로봇 아래 평균 지면 높이

    x_bins = (
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0),
        (1.0, 1.1),
        (1.1, 1.2),
    )
    y_bins = ((-0.5, -0.25), (-0.25, 0.0), (0.0, 0.25), (0.25, 0.5))
    cell_heights = []
    for x_min, x_max in x_bins:       # 앞쪽 x방향 12칸
        for y_min, y_max in y_bins:   # 좌우 y방향 4칸
            cell_mask = (
                finite
                & (points_b[..., 0] >= x_min)
                & (points_b[..., 0] < x_max)
                & (points_b[..., 1] >= y_min)
                & (points_b[..., 1] < y_max)
            )   # 해당 grid cell에 들어온 ray만 선택
            masked_z_b = torch.where(cell_mask, points_b[..., 2], torch.full_like(points_b[..., 2], -1.0e6))
            top_values = torch.topk(masked_z_b, k=min(top_k, masked_z_b.shape[1]), dim=1).values  # cell 내 body z top-k
            top_valid = top_values > -1.0e5       # 실제 hit인지 확인
            top_count = top_valid.sum(dim=1).clamp_min(1)
            top_mean_z_b = torch.where(top_valid, top_values, torch.zeros_like(top_values)).sum(dim=1) / top_count
            has_hit = top_valid.any(dim=1)
            relative_height = torch.where(
                has_hit,
                top_mean_z_b - support_ground_z_b,
                torch.zeros_like(top_mean_z_b),
            )  # body frame에서 본 로봇 아래 지면 기준 상대 높이
            cell_heights.append(torch.clamp(relative_height, -0.5, 1.0))  # observation 범위 제한

    return torch.stack(cell_heights, dim=1)  # 12x4=48차원 height grid


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
        terrain_generator=STAIRS_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=10.0,
            dynamic_friction=10.0,
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
        lin_vel_p_gain=0.0,
        ang_vel_p_gain=0.0,
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
        actions = ObsTerm(func=clamped_last_action, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    action_rate = RewTerm(func=clamped_action_rate_l2, weight=-0.04)
    action_l2 = RewTerm(func=clamped_action_l2, weight=-0.02)
    flipper_front_terrain_alignment = RewTerm(func=flipper_front_terrain_alignment_exp, weight=1.0)
    # flipper_front_terrain_alignment_old = RewTerm(func=flipper_support_plane_alignment_exp, weight=0.3)
    termination = RewTerm(func=mdp.is_terminated, weight=-1.0)


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
                "x": (-4.5, -4.0),
                "y": (-0.15, 0.15),
                "z": (0.5, 0.5),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
        },
        mode="reset",
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=stairs_terrain_levels)


@configclass
class CmdVelFlipperEnvCfg(ManagerBasedRLEnvCfg):
    decimation: int = 4
    episode_length_s: float = 40.0
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
    curriculum: CurriculumCfg = CurriculumCfg()

    command_lin_vel_range: tuple[float, float] = (0.4, 0.4)
    command_ang_vel_range: tuple[float, float] = (0.0, 0.0)
