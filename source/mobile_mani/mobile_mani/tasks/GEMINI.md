# RL Tasks and Environment Definitions

This directory contains the logic for creating and configuring RL tasks in Isaac Lab. It is the heart of the environment definition.

## Task Structure
Follow the Isaac Lab MDP manager-based pattern:
1. `SceneCfg`: Scene configuration including robots, sensors, and objects.
2. `ObservationCfg`: Observation managers (e.g., joint states, LiDAR, etc.).
3. `ActionCfg`: Action managers (e.g., joint position or velocity control).
4. `RewardCfg`: Reward functions and their weights.
5. `EventCfg`: Environment events (e.g., randomization at reset).
6. `TerminationCfg`: Termination conditions (e.g., episode timeout, collision).

## Guidelines for AI
- **Modularity**: Keep configurations modular. Define common elements in base classes or reusable files.
- **Normalization**: Ensure that rewards and observations are properly normalized to stabilize training.
- **Configuration Overriding**: Use the `@@` syntax for overriding configurations in sub-classes as per Isaac Lab standards.
- **Gym Registry**: Every new task must be registered using `gym.register()` within the task's `__init__.py` to be discoverable by training scripts.
- **Configuration Validation**: Check that action and observation spaces match those in the corresponding RL agent configurations (located in `agents/` sub-directories).
