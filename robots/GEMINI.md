# Robot Assets and Configuration

This directory contains the physical assets (URDFs) and specific configurations for the robots used in the simulation.

## Sub-Directories
- `manipulator/`: URDFs and configurations for robotic arms.
- `mobile/`: URDFs and configurations for mobile platforms (e.g., tracked robots).

## Guidelines for AI
- **Asset Registration**: When adding a new robot asset (URDF), it must be properly registered in the Isaac Lab asset system to be usable in task environments.
- **URDF Consistency**: Ensure that joint names and links in the URDF match the expected names in the task's observation/action configurations.
- **Collision Models**: Always check that collision meshes are correctly defined for realistic simulation.
- **Actuator Types**: Define the appropriate actuator configurations (e.g., Implicit, IdealPD) in the Python asset configuration file.
