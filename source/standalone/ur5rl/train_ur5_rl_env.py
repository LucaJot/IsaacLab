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
    "--pub2ros",
    type=bool,
    default=False,
    help="Publish the action commands via a ros node to a forward position position controller. This will enable real robot parallel control.",
)

parser.add_argument(
    "--num_envs", type=int, default=4, help="Number of environments to spawn."
)
parser.add_argument(
    "--log_data",
    type=bool,
    default=False,
    help="Log the joint angles into the influxdb / grafana setup.",
)

parser.add_argument(
    "--pp_setup",
    type=bool,
    default="False",
    help="Spawns a container table and a cube for pick and place tasks.",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Check if --pub2ros is True
if args_cli.pub2ros and args_cli.num_envs != 1:
    print(
        "[INFO]: --pub2ros is enabled. Setting --num-envs to 1 as only one environment can be spawned when publishing to ROS."
    )
    args_cli.num_envs = 1
elif args_cli.log_data and not args_cli.num_envs == 1:
    print(
        "[INFO]: --log_data is enabled. Setting --num-envs to 1 as only one environment can be spawned when logging data."
    )
    args_cli.num_envs = 1


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from ur5_rl_env_standalone import HawUr5EnvCfg, HawUr5Env
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# ----------------- ROS -----------------
import rclpy


# Get the Ur5JointController class from the ur5_basic_control_fpc module
from ros2_humble_ws.src.ur5_parallel_control.ur5_parallel_control.ur5_basic_control_fpc import (
    Ur5JointController,
)
import threading


# Separate thread to run the ROS 2 node in parallel to the simulation
def ros_node_thread(node: Ur5JointController):
    """
    Function to spin the ROS 2 node in a separate thread.
    """
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------

# ----------- Data Analysis -------------
from influx_datalogger import InfluxDataLogger


def store_joint_positions(
    logger: InfluxDataLogger, sim: list, real: list | None, bucket: str
):
    joints = [
        "shoulder_pan_joint",  # 0
        "shoulder_lift_joint",  # -110
        "elbow_joint",  # 110
        "wrist_1_joint",  # -180
        "wrist_2_joint",  # -90
        "wrist_3_joint",  # 0
        "gripper_goalstate",  # 0
    ]
    logger.log_joint_positions(
        joint_names=joints, sim_positions=sim, real_positions=real, bucket=bucket
    )


def store_cube_positions(
    logger: InfluxDataLogger,
    cube_position_tracked: list,
    cube_position_gt: list,
    bucket: str,
):
    logger.store_cube_positions(
        cube_position_tracked=cube_position_tracked,
        cube_position_gt=cube_position_gt,
        bucket=bucket,
    )


# ---------------------------------------


def sync_sim_joints_with_real_robot(env: HawUr5Env, ur5_controller: Ur5JointController):
    """Sync the simulated robot joints with the real robot."""
    # Sync sim joints with real robot
    print("[INFO]: Waiting for joint positions from the real robot...")
    while ur5_controller.get_joint_positions() == None:
        pass
    real_joint_positions = ur5_controller.get_joint_positions()
    env.set_joint_angles_absolute(joint_angles=real_joint_positions)


def check_cube_deviation(cube_position_sim: list, cube_position_real: list):
    """
    Check the deviation between the cube positions in the simulation and real world.
    """
    # Calculate the distance between the cube positions
    deviation = np.linalg.norm(
        np.array(cube_position_sim) - np.array(cube_position_real)
    )
    if deviation > 0.1:
        return True
    return False


# SETUP VARS ----------------
args_cli.pub2ros = False
args_cli.log_data = False
args_cli.num_envs = 5
args_cli.pp_setup = True
# ---------------------------


def main():
    """Main function."""

    # create environment configuration
    env_cfg = HawUr5EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.pp_setup = True  # args_cli.pp_setup
    # setup RL environment
    env = HawUr5Env(cfg=env_cfg)
    print(env.max_episode_length)
    env.camera_rgb.reset()
    env.camera_depth.reset()

    # Wrap the environment in a gym wrapper
    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # Step the environment
            # take random actions
            actions = torch.rand(args_cli.num_envs, 7) * 2 - 1
            obs, rew, terminated, truncated, info = env.step(actions)
            print(truncated)

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
