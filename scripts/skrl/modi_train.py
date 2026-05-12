# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.
"""

import argparse
import sys
import copy
from pathlib import Path

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

# 비디오 녹화 시 카메라 활성화
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Ensure this workspace's package source is imported before any editable install.
repo_source = Path(__file__).resolve().parents[2] / "source" / "mobile_mani"
if str(repo_source) not in sys.path:
    sys.path.insert(0, str(repo_source))

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime
import gymnasium as gym

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
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
# 1. 모델 클래스 정의 (MLP 전용)
# =============================================================================

class MobileNavPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum", 
                 initial_log_std=0.0,
                 **kwargs):
        
        _ = kwargs.pop("network", None)
        _ = kwargs.pop("class", None)
        _ = kwargs.pop("output", None)

        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        obs_size = self.num_observations
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, self.num_actions)
        )
        
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * initial_log_std)

    def compute(self, inputs, role):
        states = inputs.get("states", inputs)
        return self.net(states), self.log_std_parameter, {}

class MobileNavValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=False, **kwargs):
        
        _ = kwargs.pop("network", None)
        _ = kwargs.pop("initial_log_std", None)
        _ = kwargs.pop("class", None)
        _ = kwargs.pop("output", None)

        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

        obs_size = self.num_observations

        self.net = nn.Sequential(
            nn.Linear(obs_size, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        states = inputs.get("states", inputs)
        return self.net(states), {}

# =============================================================================
# 2. 커스텀 Runner
# =============================================================================
class CustomRunner(Runner):
    def _generate_models(self, env, cfg):
        cfg_models = cfg["models"] 

        policy = MobileNavPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **cfg_models["policy"]
        )
        
        value = MobileNavValue(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **cfg_models["value"]
        )

        return {
            "agent": {
                "policy": policy,
                "value": value
            }
        }

# =============================================================================
# Main
# =============================================================================

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    
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
    # 스크립트 위치 기준으로 로그 루트 디렉토리 설정
    base_log_dir = os.path.join(os.path.dirname(__file__), "logs", "skrl")
    
    # 실험별 서브 디렉토리 이름 (예: mobile_cmd_vel_flipper)
    exp_dir_name = agent_cfg["agent"]["experiment"]["directory"]
    
    # 개별 실행 이름 (timestamp + algorithm + custom_name)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        run_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    
    # skrl Runner 설정 업데이트
    agent_cfg["agent"]["experiment"]["directory"] = os.path.join(base_log_dir, exp_dir_name)
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
    
    # 최종 로그 디렉토리 및 필수 하위 폴더 생성
    log_dir = os.path.join(agent_cfg["agent"]["experiment"]["directory"], run_name)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "videos"), exist_ok=True)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    env_cfg.log_dir = log_dir

    # 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # [복구] 비디오 녹화 래퍼 적용
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
