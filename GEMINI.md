# Mobile Mani Project Overview

This project is a specialized template for developing mobile manipulation tasks within the **Isaac Lab** (formerly Orbit) framework, powered by **NVIDIA Isaac Sim**. It is structured as an Omniverse extension.

## Key Frameworks & Tools
- **Isaac Lab**: Core framework for robot learning and simulation.
- **NVIDIA Isaac Sim / Omniverse**: Underlying simulation platform.
- **RL Libraries**: Integrated support for `rl-games`, `rsl_rl`, `sb3`, and `skrl`.
- **Linting & Formatting**: `Ruff` and `Pyright` are used for code quality and static typing.

## Directory Structure
- `robots/`: Robot assets (URDFs) and configurations.
- `scripts/`: Training and playback scripts for different RL frameworks.
- `source/`: Core Python package containing task definitions and environment configurations.
- `source/mobile_mani/mobile_mani/tasks/`: Specific RL task implementations (Scene, Observations, Rewards, etc.).

## General Instructions for AI
- Adhere to Isaac Lab's configuration-based MDP manager pattern.
- Follow the project's coding standards (Ruff, Pyright).
- When creating new tasks or modifying existing ones, ensure compatibility with the supported RL libraries in the `scripts/` directory.
- Refer to Isaac Lab documentation for specialized simulation components (Sensors, Actuators, etc.).
