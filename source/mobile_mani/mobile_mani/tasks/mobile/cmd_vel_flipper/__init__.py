import gymnasium as gym


gym.register(
    id="Template-Mobile-CmdVel-Flipper-v0",
    entry_point=f"{__name__}.cmd_vel_flipper_env:CmdVelFlipperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmd_vel_flipper_env_cfg:CmdVelFlipperEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
    },
)
