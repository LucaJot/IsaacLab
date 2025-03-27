# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video

args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlVecEnvWrapper,
)

import torch
import gymnasium as gym

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


from ros_to_gym import ros_to_gym
from env_utils import *


#! TESTING ONLY - REMOVE
import csv
import os
import torch
import numpy as np


def move_to_detection_test_positions_alt(
    env,
    output_file="/home/luca/isaaclab_ws/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5rl/logdir/detection_test_log.csv",
    steps_per_target=400,
):
    # Define target joint positions (only the first 6 used)
    target_positions = [
        # [-0.0, -2.26, 2.43, -2.54, -1.71, -0.0, 0.0],
        [-0.0, -2.66, 2.43, -2.54, -1.71, -0.0, -1.0],
        [-0.0, -1.26, 0.77, -2.2, -1.71, -0.0, -1.0],
        [-0.0, -0.99, 0.77, -2.0, -1.71, -0.0, -1.0],
        [-0.0, -2.66, 2.43, -2.54, -1.71, -0.0, -1.0],
        # [-0.0, -1.25, 1.65, -2.84, -1.71, -0.0, -1.0],
    ]

    # Create log file and write header
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "target_idx",
                *[f"joint_{i}" for i in range(6)],
                "cube_x",
                "cube_y",
                "cube_z",
            ]
        )

        # Reset environment and get initial observation
        actions = torch.zeros(7).unsqueeze(0)
        obs, _, _, info = env.step(actions)

        step_counter = 0
        for idx, target in enumerate(target_positions):
            target_tensor = torch.tensor(target[:6])

            for _ in range(steps_per_target):
                joint_angles = obs[0][0:6]
                cube_pos = obs[0][19:22]

                delta = target_tensor - joint_angles.cpu()
                direction = torch.sign(delta).clamp(-1.0, 1.0)

                # Action = direction for each joint, last element = 0 for gripper
                action = torch.cat((direction, torch.tensor([0.0]))).unsqueeze(0)
                if idx > 40:
                    action = torch.zeros(7).unsqueeze(0)

                obs, _, _, info = env.step(action)

                # Log data
                writer.writerow(
                    [
                        step_counter,
                        idx,
                        *obs[0][0:6].cpu().numpy(),
                        *obs[0][19:22].cpu().numpy(),
                    ]
                )
                step_counter += 1


def move_to_detection_test_positions(
    env,
    output_file="/home/luca/isaaclab_ws/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5rl/logdir/detection_test_log.csv",
    steps_per_target=400,
    stop_epsilon=0.01,  # Stop if joint error is below this [rad]
):
    import csv
    import torch

    target_positions = [
        [-0.0, -2.66, 2.43, -2.54, -1.71, -0.0],
        [-0.0, -1.26, 0.77, -2.3, -1.71, -0.0],
        [-0.0, -0.99, 0.77, -2.0, -1.71, -0.0],
        # [-0.0, -1.26, 0.77, -2.3, -1.71, -0.0],
        # [-0.0, -2.66, 2.43, -2.54, -1.71, -0.0],
    ]

    gripper_action = torch.tensor([-1.0])  # ✅ Fixed gripper value

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "target_idx",
                *[f"joint_{i}" for i in range(6)],
                "cube_x",
                "cube_y",
                "cube_z",
            ]
        )

        # Reset environment
        actions = torch.zeros(7).unsqueeze(0)
        obs, _, _, info = env.step(actions)

        step_counter = 0
        for idx, target in enumerate(target_positions):
            target_tensor = torch.tensor(target)

            for _ in range(steps_per_target):
                joint_angles = obs[0][0:6]
                cube_pos = obs[0][19:22]

                # Compute delta and stop if close enough
                delta = target_tensor - joint_angles.cpu()
                if torch.all(delta.abs() < stop_epsilon):
                    print(f"[INFO] Reached target {idx} after {step_counter} steps.")
                    break

                scaled_delta = torch.clamp(delta * 5.0, -1.0, 1.0)  # Soft gain
                action = torch.cat((scaled_delta, gripper_action)).unsqueeze(0)

                obs, _, _, info = env.step(action)

                # Log data
                writer.writerow(
                    [
                        step_counter,
                        idx,
                        *obs[0][0:6].cpu().numpy(),
                        *obs[0][19:22].cpu().numpy(),
                    ]
                )
                step_counter += 1


#! TESTING ONLY - REMOVE


def main():
    """Main function."""

    # Get init state from real hw or stored state
    use_real_hw = False
    # Pretrain the model
    pretrain = True
    # Resume the last training
    resume = True
    EXPERIMENT_NAME = "_"
    NUM_ENVS = 1

    # Set the goal state of the cube
    cube_goal_pos = [1.0, -0.1, 0.8]

    # Set rl learning configuration
    agent_cfg, log_dir, resume_path = set_learning_config()
    # Get initial state from real world or fake state
    if use_real_hw:
        # Initialize ROS2 to Gymnasium translation node
        rg_node = ros_to_gym()
        # Start spinning both ROS2 nodes
        rg_node.run_ros_nodes()
        # Get observations from the real world
        _, (
            real_joint_info,
            cube_pos,
            _,
            _,
            _,
            _,
        ) = rg_node.get_obs_from_real_world()
        # Shift cube pos to the correct height
        cube_pos[2] += 0.2
        # Store joint angles
        real_joint_angles = real_joint_info["joint_positions"]  # type: ignore
    else:
        real_joint_angles = [
            -0.15472919145692998,
            -1.8963201681720179,
            1.5,
            -2.460175625477926,
            -1.5792139212237757,
            -0.0030048529254358414,
            -1.0,
        ]

        #! Teesting
        real_joint_angles = [-0.0, -2.66, 2.43, -2.54, -1.71, -0.0, -1.0]

        cube_pos = [1.0, 0.0, 0.58]

        # Start up the digital twin

    # Set environment configuration
    env_cfg = parse_env_cfg(
        task_name="Isaac-Ur5-RL-Direct-v0",
        num_envs=NUM_ENVS,
    )
    # env_cfg.cube_init_state = cube_pos  # type: ignore
    env_cfg.arm_joints_init_state = real_joint_angles[:-1]  # type: ignore
    env_cfg.cube_init_state = cube_pos  # type: ignore

    # Create simulated environment with the real-world state
    env = gym.make(
        id="Isaac-Ur5-RL-Direct-v0",
        cfg=env_cfg,
        cube_goal_pos=cube_goal_pos,
    )

    if env.unwrapped.set_arm_init_pose(real_joint_angles):  # type: ignore
        print("Set arm init pose successful!")
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)  # type: ignore

    #! TESTING ONLY - REMOVE
    env.unwrapped.set_eval_mode()  # type: ignore
    move_to_detection_test_positions(env)
    return
    #! TESTING ONLY - REMOVE

    if pretrain:
        start_iter = 2
        env.unwrapped.set_train_mode()  # type: ignore
        for i in range(start_iter, 3):
            log_results = train_rsl_rl_agent_init(
                env, env_cfg, agent_cfg, CL=i + 1, resume=resume
            )
            print(log_results)
            resume = True

    env.unwrapped.set_eval_mode()  # type: ignore
    # Run the task in the simulator
    success, interrupt, obs, policy = run_task_in_sim(
        env,
        log_dir=log_dir,
        resume_path=resume_path,
        agent_cfg=agent_cfg,
        simulation_app=simulation_app,
    )
    # --------------------------------------------------------------------

    print(f"Success: {success}")
    print(f"Interrupt: {interrupt}")
    interrupt = True  #! Force Retrain for Debug
    # success = True  #! Force Real Robot for Debug
    return

    if success:
        print("Task solved in Sim!")
        print("Moving network control to real robot...")

        while not rg_node.goal_reached():
            obs = rg_node.step_real(
                policy,
                action_scale=env_cfg.action_scale,
            )
            print(f"Observations: {obs}")
            # TODO Interrupts catchen

    elif interrupt:
        # get interrupt state
        # env.close()
        # env = None

        arm_interrupt_state = obs[0][0:6].cpu().numpy()
        gripper_interrupt_state = obs[0][18].cpu().numpy()
        env_cfg.arm_joints_init_state = arm_interrupt_state  #! Das funktioniert nicht
        # agent_cfg.experiment_name = EXPERIMENT_NAME

        train_rsl_rl_agent(env, env_cfg, agent_cfg, resume)

    return


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
