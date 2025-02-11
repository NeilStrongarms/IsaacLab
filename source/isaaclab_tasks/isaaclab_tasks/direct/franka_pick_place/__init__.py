# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka Pick Place environment.
"""

import gymnasium as gym

from . import agents
# from .franka_pick_place_env import FrankaPickPlaceEnv, FrankaPickPlaceEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Pick-Place-Direct-v0",
    entry_point=f"{__name__}.franka_pick_place_env:FrankaPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_pick_place_env:FrankaPickPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPickPlacePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
