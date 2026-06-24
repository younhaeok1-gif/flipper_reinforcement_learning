# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record robot pitch while crossing stairs with a trained skrl flipper policy.

The script mirrors the plotting behavior of ``record_no_flipper_pitch.py`` but
loads a skrl PPO checkpoint and uses the policy output for front-flipper control.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "skrl"
    / "logs"
    / "skrl"
    / "mobile_cmd_vel_flipper"
    / "2026-05-21_00-16-29_ppo_torch_ppo_mlp_run"
    / "checkpoints"
    / "best_agent.pt"
)

parser = argparse.ArgumentParser(description="Record trained flipper-control pitch over the stair task.")
parser.add_argument("--task", type=str, default="Template-Mobile-CmdVel-Flipper-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Use 1 for recording.")
parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to skrl model checkpoint.")
parser.add_argument("--duration", type=float, default=20.0, help="Maximum recording time in seconds. Use 0 to run until closed.")
parser.add_argument("--max_steps", type=int, default=0, help="Optional maximum simulation steps. Use 0 to disable.")
parser.add_argument("--terrain_level", type=int, default=5, help="Fixed stair terrain level. Use -1 to keep default reset level.")
parser.add_argument("--plot_interval", type=int, default=5, help="Plot update interval in simulation steps.")
parser.add_argument("--save_path", type=str, default="trained_flipper_pitch.png", help="Path to save the pitch plot PNG.")
parser.add_argument("--csv_path", type=str, default="trained_flipper_pitch.csv", help="Path to save recorded pitch samples.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch"], help="ML framework used by skrl.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO"], help="RL algorithm used by skrl.")
parser.add_argument("--follow_camera", action="store_true", default=False, help="Follow the robot with the viewer camera.")
parser.add_argument("--keep_bad_orientation_done", action="store_true", default=False, help="Keep bad-orientation termination enabled.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Hydra should only see Hydra overrides, not this script's custom arguments.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import matplotlib
import skrl
import torch
from packaging import version

if getattr(args_cli, "headless", False):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_SOURCE_PATH = Path(__file__).resolve().parents[1] / "source" / "mobile_mani"
if PROJECT_SOURCE_PATH.is_dir() and str(PROJECT_SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE_PATH))

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
import mobile_mani.tasks  # noqa: F401


SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    simulation_app.close()
    raise SystemExit(1)


def _convert_legacy_net_checkpoint(resume_path: str) -> str:
    """Convert old skrl checkpoints using ``net.*`` keys to current ``net_container.*`` keys."""
    checkpoint = torch.load(resume_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return resume_path

    converted = False
    for role in ("policy", "value"):
        state_dict = checkpoint.get(role)
        if not isinstance(state_dict, dict):
            continue
        has_legacy_net = any(key.startswith("net.") for key in state_dict.keys())
        has_current_net = any(key.startswith("net_container.") for key in state_dict.keys())
        if not has_legacy_net or has_current_net:
            continue

        checkpoint[role] = OrderedDict(
            (f"net_container.{key[4:]}" if key.startswith("net.") else key, value)
            for key, value in state_dict.items()
        )
        converted = True

    if not converted:
        return resume_path

    root, ext = os.path.splitext(resume_path)
    converted_path = f"{root}_net_container{ext}"
    torch.save(checkpoint, converted_path)
    print(f"[INFO] Converted legacy skrl checkpoint for current model keys: {converted_path}")
    return converted_path


def _base_pitch_rad(env) -> torch.Tensor:
    """Return root pitch angle in radians for every environment."""
    robot = env.scene["robot"]
    _, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    return pitch


def _set_terrain_level(env, terrain_level: int):
    """Force all envs to start from a specific curriculum terrain row."""
    if terrain_level < 0:
        return
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        print("[WARN] Terrain level override skipped because this scene has no curriculum terrain origins.")
        return

    max_level = terrain.terrain_origins.shape[0] - 1
    level = max(0, min(terrain_level, max_level))
    env_ids = torch.arange(env.num_envs, device=env.device)
    terrain.terrain_levels[env_ids] = level
    terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
    print(f"[INFO] Fixed terrain level: {level} / {max_level}")


def _save_csv(path: str, times: list[float], pitch_rad: list[float], pitch_deg: list[float], action_values: list[float]):
    """Write recorded pitch and policy action samples to a CSV file."""
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time_s", "pitch_rad", "pitch_deg", "policy_action"])
        writer.writerows(zip(times, pitch_rad, pitch_deg, action_values))


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Run the trained policy and record pitch."""
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if not args_cli.keep_bad_orientation_done:
        env_cfg.terminations.bad_orientation = None
    if not args_cli.follow_camera:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = None
        env_cfg.viewer.eye = (-6.0, -4.0, 3.0)
        env_cfg.viewer.lookat = (-1.0, 0.0, 0.6)

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    _set_terrain_level(raw_env.unwrapped, args_cli.terrain_level)

    dt = raw_env.unwrapped.step_dt
    env = SkrlVecEnvWrapper(raw_env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    resume_path = _convert_legacy_net_checkpoint(resume_path)
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()

    times: list[float] = []
    pitch_rad_values: list[float] = []
    pitch_deg_values: list[float] = []
    action_values: list[float] = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    (line,) = ax.plot([], [], linewidth=2.0, label="trained flipper policy")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Base Pitch While Crossing Stairs With Trained Flipper Control")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch (deg)")
    if args_cli.duration > 0.0:
        ax.set_xlim(0.0, args_cli.duration)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    step = 0
    while simulation_app.is_running():
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            break
        if args_cli.duration > 0.0 and step * dt >= args_cli.duration:
            break

        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)

            pitch_rad = _base_pitch_rad(raw_env.unwrapped)[0].detach().cpu().item()
            pitch_deg = pitch_rad * 180.0 / torch.pi
            action_value = actions[0, 0].detach().cpu().item()
            time_s = step * dt

            times.append(time_s)
            pitch_rad_values.append(pitch_rad)
            pitch_deg_values.append(float(pitch_deg))
            action_values.append(action_value)

            if step % args_cli.plot_interval == 0:
                line.set_data(times, pitch_deg_values)
                ax.relim()
                ax.autoscale_view()
                if args_cli.duration > 0.0:
                    ax.set_xlim(0.0, args_cli.duration)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

            if bool(terminated[0].item()) or bool(truncated[0].item()):
                break

        step += 1

    line.set_data(times, pitch_deg_values)
    ax.relim()
    ax.autoscale_view()
    if args_cli.duration > 0.0:
        ax.set_xlim(0.0, args_cli.duration)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)

    if args_cli.csv_path:
        _save_csv(args_cli.csv_path, times, pitch_rad_values, pitch_deg_values, action_values)
    if args_cli.save_path:
        fig.savefig(args_cli.save_path, dpi=180)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
