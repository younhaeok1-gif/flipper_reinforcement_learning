# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.
"""

import argparse
import sys
import copy

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default=None, help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"], help="ML framework.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="RL algorithm.")
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime
import gymnasium as gym
import skrl
from packaging import version

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
# 기본 Runner 가져오기
from skrl.utils.runner.torch import Runner 

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

import mobile_mani.tasks

logger = logging.getLogger(__name__)

# =============================================================================
# 1. 모델 클래스 정의 (에러 방지 처리 완료됨)
# =============================================================================

class MobileNavPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum", 
                 **kwargs):
        
        # 불필요한 인자 제거
        _ = kwargs.pop("network", None)
        _ = kwargs.pop("initial_log_std", None)
        _ = kwargs.pop("class", None)
        _ = kwargs.pop("output", None)

        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # ----------------------------------------------------------------
        # [설정] 이미지 크기 및 네트워크 구조
        # ----------------------------------------------------------------
        self.img_h, self.img_w = 64, 64  # 이미지 크기 (config와 맞춰주세요)
        self.img_channels = 3
        self.img_flat_size = self.img_h * self.img_w * self.img_channels # 21168

        # CNN: (84, 84, 3) -> Flatten 3136
        # CNN: (64, 64, 3) -> Flatten 1024
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ELU(),
            nn.Flatten(),
        )
        self.cnn_out_size = 1024 

        # MLP: 센서 데이터 처리
        # 입력 차원(19)은 config.py의 센서 개수에 따라 조절 필요
        self.mlp = nn.Sequential(
            nn.Linear(47, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU()
        )

        # Output Network
        self.net = nn.Sequential(
            nn.Linear(self.cnn_out_size + 128, 256), nn.ELU(),
            nn.Linear(256, self.num_actions)
        )
        
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # 1. 데이터 가져오기 (SKRL은 "states"에 데이터를 담습니다)
        states = inputs.get("states", inputs)
        
        img_data = None
        sensor_data = None

        # ----------------------------------------------------------------
        # CASE A: 딕셔너리로 들어온 경우 (Wrap=False, Preprocessor=None)
        # ----------------------------------------------------------------
        if isinstance(states, dict):
            # 1. 카메라 데이터 찾기 (Config 이름: "camera")
            if "camera" in states:
                val = states["camera"]
                # {"camera": {"image": ...}} 형태인지 {"camera": ...} 형태인지 확인
                if isinstance(val, dict) and "image" in val:
                    img_data = val["image"]
                elif isinstance(val, dict):
                    img_data = next(iter(val.values()))
                else:
                    img_data = val
            
            # 2. 센서 데이터 찾기 (Config 이름: "policy")
            if "policy" in states:
                sensor_data = states["policy"]
            else:
                # 혹시 이름이 다를 경우를 대비해 'camera'가 아닌 나머지 텐서를 찾음
                for k, v in states.items():
                    if k != "camera" and isinstance(v, torch.Tensor):
                        sensor_data = v
                        break

        # ----------------------------------------------------------------
        # CASE B: 텐서로 뭉쳐서 들어온 경우 (Wrap=True, Preprocessor=Active)
        # ----------------------------------------------------------------
        elif isinstance(states, torch.Tensor):
            # 이미지 크기만큼 앞을 자름
            if states.shape[1] > self.img_flat_size:
                # (Batch, 21168) -> (Batch, 84, 84, 3)
                img_flat = states[:, :self.img_flat_size]
                img_data = img_flat.view(-1, self.img_h, self.img_w, self.img_channels)
                
                # 나머지는 센서 데이터
                sensor_data = states[:, self.img_flat_size:]
            else:
                # 데이터가 너무 작으면 센서 데이터로만 간주
                sensor_data = states

        # ----------------------------------------------------------------
        # 데이터 처리 및 결합
        # ----------------------------------------------------------------
        
        # 1. 이미지 처리
        if img_data is not None:
            # (Batch, H, W, C) -> (Batch, C, H, W) PyTorch 순서로 변환
            if img_data.dim() == 4 and img_data.shape[-1] == 3:
                img_data = img_data.permute(0, 3, 1, 2)
            
            x_img = self.cnn(img_data.float() / 255.0)
        else:
            # 이미지가 없으면 0으로 채움 (에러 방지)
            x_img = torch.zeros((states.shape[0], self.cnn_out_size), device=self.device)

        # 2. 센서 처리
        if sensor_data is not None:
            x_sensor = self.mlp(sensor_data)
        else:
            x_sensor = torch.zeros((states.shape[0], 128), device=self.device)

        # 3. 결합
        x = torch.cat([x_img, x_sensor], dim=1)
        
        return self.net(x), self.log_std_parameter, {}

class MobileNavValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=False, **kwargs):
        
        _ = kwargs.pop("network", None)
        _ = kwargs.pop("initial_log_std", None)
        _ = kwargs.pop("class", None)
        _ = kwargs.pop("output", None)

        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

        # 이미지 설정
        self.img_h, self.img_w = 64, 64
        self.img_channels = 3
        self.img_flat_size = self.img_h * self.img_w * self.img_channels

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ELU(),
            nn.Flatten(),
        )
        self.cnn_out_size = 1024

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(47, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU()
        )

        # Value Network (Output 1)
        self.net = nn.Sequential(
            nn.Linear(self.cnn_out_size + 128, 256), nn.ELU(),
            nn.Linear(256, 1)
        )

    def compute(self, inputs, role):
        states = inputs.get("states", inputs)
        img_data = None
        sensor_data = None

        # CASE A: Dictionary
        if isinstance(states, dict):
            if "camera" in states:
                val = states["camera"]
                if isinstance(val, dict) and "image" in val: img_data = val["image"]
                elif isinstance(val, dict): img_data = next(iter(val.values()))
                else: img_data = val
            
            if "policy" in states:
                sensor_data = states["policy"]
            else:
                 for k, v in states.items():
                    if k != "camera" and isinstance(v, torch.Tensor):
                        sensor_data = v
                        break

        # CASE B: Tensor
        elif isinstance(states, torch.Tensor):
            if states.shape[1] > self.img_flat_size:
                img_flat = states[:, :self.img_flat_size]
                img_data = img_flat.view(-1, self.img_h, self.img_w, self.img_channels)
                sensor_data = states[:, self.img_flat_size:]
            else:
                sensor_data = states

        # Process
        if img_data is not None:
            if img_data.dim() == 4 and img_data.shape[-1] == 3:
                img_data = img_data.permute(0, 3, 1, 2)
            x_img = self.cnn(img_data.float() / 255.0)
        else:
            x_img = torch.zeros((states.shape[0], self.cnn_out_size), device=self.device)

        if sensor_data is not None:
            x_sensor = self.mlp(sensor_data)
        else:
            x_sensor = torch.zeros((states.shape[0], 128), device=self.device)

        x = torch.cat([x_img, x_sensor], dim=1)
        
        # Value function returns simple dict
        return self.net(x), {}
# =============================================================================
# 2. 커스텀 Runner (이게 핵심 해결책!)
# =============================================================================
class CustomRunner(Runner):
    """
    기본 Runner를 상속받아, YAML 설정을 무시하고 우리가 만든 클래스를 강제로 주입합니다.
    """
    def _generate_models(self, env, cfg):
        # cfg["models"]["policy"] 경로 접근
        cfg_models = cfg["models"] 

        # Policy 모델 생성
        policy = MobileNavPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **cfg_models["policy"]
        )
        
        # Value 모델 생성
        value = MobileNavValue(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **cfg_models["value"]
        )

        # [핵심 수정] 
        # SKRL Runner는 "agent"라는 키를 통해 모델을 찾습니다.
        # 따라서 { "agent": { ... } } 형태로 감싸서 리턴해야 합니다.
        return {
            "agent": {
                "policy": policy,
                "value": value
            }
        }
# =============================================================================
# Main
# =============================================================================

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    
    # 1. 환경 설정 적용
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training requires GPU.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # 2. 로깅 설정
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    env_cfg.log_dir = log_dir

    # 3. 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # SKRL 환경 래퍼
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    agent_cfg["models"]["policy"]["class"] = MobileNavPolicy
    agent_cfg["models"]["value"]["class"] = MobileNavValue

    agent_cfg["agent"]["state_preprocessor"] = None
    agent_cfg["agent"]["state_preprocessor_kwargs"] = None
    agent_cfg["agent"]["value_preprocessor"] = None
    agent_cfg["agent"]["value_preprocessor_kwargs"] = None

    # 4. CustomRunner 실행! (여기가 달라짐)
    # 일반 Runner 대신 우리가 만든 CustomRunner를 사용하여 모델을 직접 주입합니다.
    runner = CustomRunner(env, agent_cfg)

    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()