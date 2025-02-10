# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Claus

"""Launch Isaac Sim Simulator first."""

from __future__ import annotations
import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Running Franka pick place direct environment")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from franka_pick_place_env import FrankaPickPlaceEnvCfg, FrankaPickPlaceEnv

def main():
    """Main function."""
    env_cfg = FrankaPickPlaceEnvCfg()
    # env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.num_envs = 1
    # setup environment
    env = FrankaPickPlaceEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # resetting scene periodically
            if count % 2000 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                print("^" * 80)

            # joint_efforts = torch.randn(env.num_envs, env._robot.num_joints, device=env.device)
            joint_efforts = torch.zeros(env.num_envs, env._robot.num_joints, device=env.device)
            finger_efforts = torch.randn(env.num_envs, 2, device=env.device)
            joint_efforts[:,7:9] = finger_efforts
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts) # Shape is (num_envs, action_dim)
            POS = env.robot_grasp_pos
            ROT = env.robot_grasp_rot
            # print(f"Grasp Position: {POS}")  # Should be (num_envs, 3)

            POS2 = env.cube_pos
            ROT2 = env.cube_rot
            # print(f"Cube Position: {POS2}")  # Should be (num_envs, 3)


            count += 1
            
    # close the environment
    env.close()



if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
