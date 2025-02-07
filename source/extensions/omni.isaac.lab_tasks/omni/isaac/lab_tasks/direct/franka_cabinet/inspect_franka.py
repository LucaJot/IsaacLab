# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the xxx with ROS 2 integration."""

import argparse

from omni.isaac.lab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on running the cartpole RL environment."
)

parser.add_argument(
    "--num_envs", type=int, default=4, help="Number of environments to spawn."
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from franka_cabinet_env import FrankaCabinetEnvCfg, FrankaCabinetEnv
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# SETUP VARS ----------------
args_cli.pub2ros = False
args_cli.log_data = False
args_cli.num_envs = 5
args_cli.pp_setup = True
# ---------------------------


def main():
    """Main function."""

    # create environment configuration
    env_cfg = FrankaCabinetEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = FrankaCabinetEnv(cfg=env_cfg)

    # Wrap the environment in a gym wrapper
    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # Step the environment
            # take random actions
            actions = torch.rand(args_cli.num_envs, 9) * 2 - 1
            obs, rew, terminated, truncated, info = env.step(actions)
            print("-" * 10 + f" Step {count} " + "-" * 10)
            print(f"Observation: {obs}")
            print(f"Reward: {rew}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info: {info}")

            obs_policy = obs["policy"]
            print(f"Shape of observation: {obs_policy.shape}")
            print(f"Shape of reward: {rew.shape}")
            print(f"Shape of terminated: {terminated.shape}")
            print(f"Shape of truncated: {truncated.shape}")

            print("-" * 30)

            # update counter
            count += 1

    # close the environment
    env.close()

    # Shutdown ROS 2 (if initialized)
    # rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
