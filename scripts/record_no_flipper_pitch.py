# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record robot pitch while driving stairs without learned flipper control.

This script follows the same launch style as ``visualize_height_grid.py``. It
runs one environment, sends zero flipper actions, and records the base pitch so
the uncontrolled baseline can be compared against a trained policy later.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
from math import pi

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record no-flipper-control pitch over the stair task.")
parser.add_argument("--task", type=str, default="Template-Mobile-CmdVel-Flipper-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Use 1 for recording.")
parser.add_argument("--duration", type=float, default=20.0, help="Maximum recording time in seconds. Use 0 to run until closed.")
parser.add_argument("--max_steps", type=int, default=0, help="Optional maximum simulation steps. Use 0 to disable.")
parser.add_argument("--terrain_level", type=int, default=5, help="Fixed stair terrain level. Use -1 to keep default reset level.")
parser.add_argument("--plot_interval", type=int, default=5, help="Plot update interval in simulation steps.")
parser.add_argument("--save_path", type=str, default="no_flipper_pitch.png", help="Path to save the pitch plot PNG.")
parser.add_argument("--csv_path", type=str, default="no_flipper_pitch.csv", help="Path to save recorded pitch samples.")
parser.add_argument("--fixed_flipper_angle_deg", type=float, default=-180.0, help="Fixed front flipper target angle in degrees.")
parser.add_argument("--follow_camera", action="store_true", default=False, help="Follow the robot with the viewer camera.")
parser.add_argument("--keep_bad_orientation_done", action="store_true", default=False, help="Keep bad-orientation termination enabled.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import matplotlib
import torch

if getattr(args_cli, "headless", False):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as math_utils

import mobile_mani.tasks  # noqa: F401


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


def _save_csv(path: str, times: list[float], pitch_rad: list[float], pitch_deg: list[float]):
    """Write recorded pitch samples to a CSV file."""
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time_s", "pitch_rad", "pitch_deg"])
        writer.writerows(zip(times, pitch_rad, pitch_deg))


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    if not args_cli.keep_bad_orientation_done:
        env_cfg.terminations.bad_orientation = None
    fixed_flipper_angle_rad = args_cli.fixed_flipper_angle_deg * pi / 180.0
    env_cfg.actions.front_flipper.angle_limit = max(abs(fixed_flipper_angle_rad), 1.0e-6)
    if not args_cli.follow_camera:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = None
        env_cfg.viewer.eye = (-6.0, -4.0, 3.0)
        env_cfg.viewer.lookat = (-1.0, 0.0, 0.6)

    env = gym.make(args_cli.task, cfg=env_cfg)
    _set_terrain_level(env.unwrapped, args_cli.terrain_level)
    env.reset()

    dt = env.unwrapped.step_dt
    times: list[float] = []
    pitch_rad_values: list[float] = []
    pitch_deg_values: list[float] = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    (line,) = ax.plot([], [], linewidth=2.0, label=f"fixed flipper {args_cli.fixed_flipper_angle_deg:g} deg")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Base Pitch While Crossing Stairs With Fixed Front Flipper")
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
            # Fixed action keeps the front flipper at the requested angle while cmd_vel still drives the tracks.
            action_value = 1.0 if fixed_flipper_angle_rad >= 0.0 else -1.0
            actions = torch.full(env.action_space.shape, action_value, device=env.unwrapped.device)
            _, _, terminated, truncated, _ = env.step(actions)

            pitch_rad = _base_pitch_rad(env.unwrapped)[0].detach().cpu().item()
            pitch_deg = pitch_rad * 180.0 / torch.pi
            time_s = step * dt

            times.append(time_s)
            pitch_rad_values.append(pitch_rad)
            pitch_deg_values.append(float(pitch_deg))

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
        _save_csv(args_cli.csv_path, times, pitch_rad_values, pitch_deg_values)
    if args_cli.save_path:
        fig.savefig(args_cli.save_path, dpi=180)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
