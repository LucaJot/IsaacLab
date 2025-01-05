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
    default=True,
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
from ur5_rl_env import HawUr5EnvCfg, HawUr5Env
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


# ---------------------------------------


def sync_sim_joints_with_real_robot(env: HawUr5Env, ur5_controller: Ur5JointController):
    """Sync the simulated robot joints with the real robot."""
    # Sync sim joints with real robot
    print("[INFO]: Waiting for joint positions from the real robot...")
    while ur5_controller.get_joint_positions() == None:
        pass
    real_joint_positions = ur5_controller.get_joint_positions()
    env.set_joint_angles_absolute(joint_angles=real_joint_positions)


class RealTimeVisualizer:
    def __init__(self, rgb_shape, depth_shape, save_to_disk=False):
        self.save_to_disk = save_to_disk
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.rgb_shape = rgb_shape
        self.depth_shape = depth_shape

        self.axes[0].set_title("RGB Image")
        self.axes[1].set_title("Depth Image")

        self.axes[0].axis("off")
        self.axes[1].axis("off")

        self.rgb_plot = self.axes[0].imshow(np.zeros(self.rgb_shape, dtype=np.uint8))
        self.depth_plot = self.axes[1].imshow(
            np.zeros(self.depth_shape, dtype=np.uint8), cmap="gray", vmin=0, vmax=255
        )

        if not save_to_disk:
            plt.ion()

    def update(self, rgb_image, depth_image, step=0):
        if step % 100 == 0:
            # Print depth statistics for debugging
            print(f"Step {step}: Depth Image Statistics")
            print(
                f"Min: {depth_image.min()}, Max: {depth_image.max()}, Mean: {depth_image.mean()}"
            )

            # Replace invalid values
            depth_image = np.nan_to_num(depth_image, nan=0.0)

            # Normalize depth image
            if depth_image.max() > depth_image.min():  # Avoid divide-by-zero
                depth_image_normalized = (
                    (np.clip(depth_image, a_min=0.0, a_max=2.0)) / (2) * 255
                ).astype(np.uint8)
            else:
                depth_image_normalized = np.zeros_like(depth_image, dtype=np.uint8)

            # Update the plots
            self.rgb_plot.set_data(rgb_image)
            self.depth_plot.set_data(depth_image_normalized)

            if self.save_to_disk:
                plt.savefig(
                    f"/home/luca/Pictures/isaacsimcameraframes/frame_{step}.png"
                )
            else:
                self.fig.canvas.draw_idle()
                plt.pause(0.1)

    def close(self):
        if not self.save_to_disk:
            plt.ioff()
        plt.close(self.fig)


def main():
    """Main function."""

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

    elbow_lift = +0.8
    count = 0

    rgb_shape = (480, 640, 3)  # Replace with the actual shape of your RGB images
    depth_shape = (480, 640)  # Replace with the actual shape of your Depth images
    visualizer = RealTimeVisualizer(rgb_shape, depth_shape, save_to_disk=True)

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

            # Get test action for the gripper
            gripper_action = -1 + count / 100 % 2
            # create a tensor for joint position targets with 7 values (6 for joints, 1 for gripper)
            actions = torch.tensor(
                [
                    [
                        0.0,
                        0.0,
                        elbow_lift,
                        0.0,
                        0.0,
                        0.0,
                        gripper_action,
                    ]
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

            rgb_images = obs["images"]["rgb"]  # type: ignore
            depth_images = obs["images"]["depth"]  # type: ignore

            # Ensure tensors are converted to numpy arrays
            if isinstance(rgb_images, torch.Tensor):
                rgb_images = rgb_images.cpu().numpy()
            if isinstance(depth_images, torch.Tensor):
                depth_images = depth_images.cpu().numpy()

            # Extract the first environment's images
            rgb_env = rgb_images[0]  # First environment's RGB image
            depth_env = depth_images[0, ..., 0]  # First environment's Depth image

            # Update the visualization
            visualizer.update(rgb_env, depth_env, step=count)

            # # Camera logging DEBUG
            # # print information from the sensors
            # print("-------------------------------")
            # print(env.scene["camera"])
            # print(
            #     "Received shape of rgb   image: ",
            #     env.scene["camera"].data.output["rgb"].shape,
            # )
            # print(
            #     "Received shape of depth image: ",
            #     env.scene["camera"].data.output["distance_to_image_plane"].shape,
            # )
            # print("-------------------------------")

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
