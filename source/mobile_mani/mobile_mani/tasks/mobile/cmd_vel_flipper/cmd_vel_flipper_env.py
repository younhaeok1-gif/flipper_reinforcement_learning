# -*- coding: utf-8 -*-

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .cmd_vel_flipper_env_cfg import CmdVelFlipperEnvCfg


class CmdVelFlipperEnv(ManagerBasedRLEnv):
    """cmd_vel tracks drive motion while the policy learns front flipper control."""

    cfg: CmdVelFlipperEnvCfg

    def __init__(self, cfg: CmdVelFlipperEnvCfg, render_mode: str | None = None, **kwargs):
        self.commanded_cmd_vel = torch.zeros(cfg.scene.num_envs, 2, device=cfg.sim.device)
        super().__init__(cfg, render_mode, **kwargs)
        self.commanded_cmd_vel = torch.zeros(self.num_envs, 2, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._sample_cmd_vel(env_ids)

    def _sample_cmd_vel(self, env_ids: torch.Tensor):
        lin_min, lin_max = self.cfg.command_lin_vel_range
        ang_min, ang_max = self.cfg.command_ang_vel_range

        self.commanded_cmd_vel[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(lin_min, lin_max)
        self.commanded_cmd_vel[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(ang_min, ang_max)
