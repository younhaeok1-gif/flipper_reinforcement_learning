# RL Execution Scripts

This directory contains scripts for training and playing RL agents using various libraries.

## Supported Frameworks
- `rl_games/`: Using the `rl-games` library (optimized for GPU PPO).
- `rsl_rl/`: Using the `rsl_rl` library (common in locomotion tasks).
- `sb3/`: Using the `Stable Baselines3` library (versatile and easy to use).
- `skrl/`: Using the `skrl` library (highly modular RL library for PyTorch/JAX).

## Usage for AI
- **Consistency**: All sub-directories should maintain a consistent interface with `train.py` and `play.py`.
- **CLI Arguments**: Ensure that script arguments are aligned with the project's standard (e.g., `--task`, `--num_envs`, `--headless`).
- **Configuration Matching**: Verify that the RL library configuration files (`.yaml` or `.py`) match the environment's observation and action space.
- **Environment Registration**: Use `list_envs.py` to verify if a new task is correctly registered in the Isaac Lab environment registry before training.
