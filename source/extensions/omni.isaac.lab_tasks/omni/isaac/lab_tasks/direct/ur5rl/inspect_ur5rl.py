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
from ur5_rl_env import HawUr5EnvCfg, HawUr5Env
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# SETUP VARS ----------------
args_cli.num_envs = 1
# ---------------------------


def main():
    """Main function."""

    # create environment configuration
    env_cfg = HawUr5EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = HawUr5Env(cfg=env_cfg)

    # Wrap the environment in a gym wrapper
    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # Step the environment
            # take random actions
            actions = torch.rand(args_cli.num_envs, 7) * 2 - 1
            obs, rew, terminated, truncated, info = env.step(actions)
            # print("-" * 10 + f" Step {count} " + "-" * 10)

            # print("-" * 30)

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
