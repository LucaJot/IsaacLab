# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .agents.rsl_rl_ppo_cfg import *

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ur5-RL-Direct-v0",
    entry_point=f"{__name__}.ur5_rl_env:HawUr5Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_rl_env_cfg:HawUr5EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Ur5RLPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
