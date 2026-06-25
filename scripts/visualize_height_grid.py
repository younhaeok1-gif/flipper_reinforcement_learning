# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize the cmd_vel flipper task's local height grid observation.

This script does not modify the environment config. It runs one environment and
plots the 24 x 8 terrain height grid computed by ``local_height_grid``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize local height grid observation for one environment.")
parser.add_argument("--task", type=str, default="Template-Mobile-CmdVel-Flipper-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Use 1 for visualization.")
parser.add_argument("--update_interval", type=int, default=5, help="Plot update interval in simulation steps.")
parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the latest heatmap PNG.")
parser.add_argument("--follow_camera", action="store_true", default=False, help="Follow the robot with the viewer camera.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import mobile_mani.tasks  # noqa: F401
from mobile_mani.tasks.mobile.cmd_vel_flipper.cmd_vel_flipper_env_cfg import local_height_grid


def _format_grid(grid: torch.Tensor) -> torch.Tensor:
    """Convert flat 192-dim grid to [x, y] image layout for plotting."""
    # local_height_grid appends values as x-major: for each x bin, iterate y bins.
    # Reshape to [x=24, y=8] so imshow's vertical axis is robot-forward x.
    return grid.reshape(24, 8)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    if not args_cli.follow_camera:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = None
        env_cfg.viewer.eye = (-6.0, -4.0, 3.0)
        env_cfg.viewer.lookat = (-1.0, 0.0, 0.6)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    image = ax.imshow(
        torch.zeros(24, 8).cpu().numpy(),
        origin="lower",
        extent=(-0.5, 0.5, 0.0, 1.2),
        aspect="auto",
        cmap="coolwarm",
        vmin=-0.3,
        vmax=0.3,
    )
    fig.colorbar(image, ax=ax, label="local height relative to support ground (m)")
    ax.set_title("24 x 8 Local Height Grid Observation")
    ax.set_xlabel("y lateral from robot body frame (m)")
    ax.set_ylabel("x forward from robot body frame (m)")
    ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax.set_yticks([0.0, 0.3, 0.6, 0.9, 1.2])
    fig.tight_layout()

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

            if step % args_cli.update_interval == 0:
                height_grid = local_height_grid(env.unwrapped)[0].detach().cpu()
                grid_image = _format_grid(height_grid)
                image.set_data(grid_image.numpy())
                ax.set_title(
                    "24 x 8 Local Height Grid Observation "
                    f"| min={height_grid.min():+.3f} m, max={height_grid.max():+.3f} m"
                )
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                if args_cli.save_path:
                    fig.savefig(args_cli.save_path, dpi=180)

        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
