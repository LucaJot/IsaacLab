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
from source.standalone.ur5rl.backup_files.influx_datalogger import InfluxDataLogger


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
args_cli.num_envs = 1
args_cli.pp_setup = True
# ---------------------------


def main():
    """Main function."""
    elbow_lift = -0.2
    wrist1_lift = 0.0

    ### Get run configurations
    # Check if the user wants to publish the actions to ROS2
    PUBLISH_2_ROS = args_cli.pub2ros
    # Check if the user wants to log the joint data
    LOG_DATA = args_cli.log_data

    # create environment configuration
    env_cfg = HawUr5EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.pp_setup = True  # args_cli.pp_setup
    # setup RL environment
    env = HawUr5Env(cfg=env_cfg)
    env.camera_rgb.reset()
    env.camera_depth.reset()

    if PUBLISH_2_ROS:
        # ROS 2 initialization
        rclpy.init()
        ur5_controller = Ur5JointController()
        # Start the ROS node in a separate thread
        ros_thread = threading.Thread(
            target=ros_node_thread, args=(ur5_controller,), daemon=True
        )
        ros_thread.start()

    if LOG_DATA:
        # Initialize datalogger
        logger = InfluxDataLogger(
            org="haw",
            influx_url="http://localhost:8086",
            run_info="Using ground truth in script and updating it on strong deviations. Sending elbow lift command. Same update parameters for both sim and real.",
            action_scaling=env.action_scale,
        )

    count = 0

    #! Experiment Cube tracking
    pos1 = [
        -0.0,
        -1.1612923781024378,
        1.98793363571167,
        -3.9438589254962366,
        -1.5171125570880335,
        -0.0027774016009729507,
        -1.0,
    ]

    pos2 = [
        -0.0,
        -1.1,
        1.0,
        -2.25,
        -1.5171125570880335,
        -0.0027774016009729507,
        -1.0,
    ]
    pd = [x - y for x, y in zip(pos2, pos1)]
    pd_step = [x / 4 for x in pd]

    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000000 == 0:
                count = 0
                # env.reset()
                print("-" * 80)
                print("[INFO]: Env reset.")
                if PUBLISH_2_ROS:
                    sync_sim_joints_with_real_robot(env, ur5_controller)

            #! DEBUG STATEMENT ----------------
            if count == 0:
                # Set gripper to cube hight parallel to the table
                env.set_joint_angles_absolute(joint_angles=pos1)  # type: ignore
            if count > 500 and count < 1100:
                actions = pd_step
            else:
                actions = [0.0] * 7
            #! ^^^DEBUG STATEMENT^^^ ----------
            # Get test action for the gripper
            gripper_action = -1 + count / 100 % 2
            # create a tensor for joint position targets with 7 values (6 for joints, 1 for gripper)
            actions = torch.tensor(
                [
                    actions
                    # [
                    #     0.0,
                    #     0.0,
                    #     elbow_lift,
                    #     wrist1_lift,  # wrist1,
                    #     0.0,
                    #     0.0,
                    #     -1,  # gripper_action,
                    # ]
                ]
                * env_cfg.scene.num_envs
            )

            # Control real robot and setup logging
            if PUBLISH_2_ROS:
                # Send ros actions to the real robot
                ur5_controller.set_joint_delta(actions[0, :7].numpy())  # type: ignore
                real_joint_positions = ur5_controller.get_joint_positions()
                bucket = "simrealjointdata"
            # If not publishing to ROS, set real joint positions to None and switch to sim data bucket
            elif LOG_DATA:
                real_joint_positions = None
                bucket = "simjointdata"

            # Log the joint positions
            if LOG_DATA:
                sim_joint_positions = env.get_sim_joint_positions()
                if sim_joint_positions is not None:
                    sim_joint_positions = sim_joint_positions.squeeze().tolist()
                    store_joint_positions(
                        logger=logger,
                        sim=sim_joint_positions,
                        real=real_joint_positions,
                        bucket=bucket,
                    )

            # Step the environment
            obs, rew, terminated, truncated, info = env.step(actions)

            # goal_dist = obs["distance_to_goal"].cpu().numpy().tolist()
            obs_vals: torch.Tensor = obs["policy"]  # type: ignore
            # Print cube positions
            cube_position_tracked = obs["cube_pos"].cpu().numpy().tolist()

            # print(f"Step: {count}: Cube Distances {goal_dist}\n")
            print(f"Cube pos tracked: {cube_position_tracked}")

            # Log the cube positions
            if LOG_DATA:
                if count > 500 and count < 1100:
                    cube_position_tracked = (
                        obs["info"]["cube_pos"][0].cpu().numpy().tolist()  # type: ignore
                    )
                    store_cube_positions(
                        logger=logger,
                        cube_position_tracked=cube_position_tracked,
                        cube_position_gt=[1.0, 0.1, 0.25],
                        bucket="cube_pos",
                    )

            obs_vals: torch.Tensor = obs["policy"]  # type: ignore

            elbow = obs["policy"].cpu().numpy()[0][0][2]  # type: ignore
            wrist1 = obs["policy"].cpu().numpy()[0][0][3]  # type: ignore
            # print(f"Elbow: {elbow}, Wrist1: {wrist1}")
            if elbow > 2.0:
                elbow_lift = 0
            elif wrist1 < -3.0:
                wrist1_lift = 0.0

            # update counter
            count += 1

    # close the environment
    env.close()
    if LOG_DATA:
        logger.close()

    # Shutdown ROS 2 (if initialized)
    # rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
