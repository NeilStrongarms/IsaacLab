# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka Pick Place environment.
"""

import gymnasium as gym

from . import agents
from .franka_pick_place_env import FrankaPickPlaceEnv, FrankaPickPlaceEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Pick-Place-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_pick_place:FrankaPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPickPlaceEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPickPlacePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)