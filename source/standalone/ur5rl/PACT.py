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


# Set headless mode
args_cli.headless = False
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
import json
import os
import torch
import numpy as np
import time
from termcolor import colored


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


def move_to_detection_test_positions_real(
    rg_node: ros_to_gym,
    env,
    output_file="/home/luca/isaaclab_ws/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5rl/logdir/detection_test_log.csv",
    steps_per_target=1500,
    stop_epsilon=0.02,  # Stop if joint error is below this [rad]
    action_scale=0.08,  # Scale for real robot
):
    import csv
    import torch

    target_positions = [
        [-0.0, -2.66, 2.43, -2.54, -1.71, -0.0],
        [-0.0, -1.26, 0.77, -2.3, -1.71, -0.0],
        [-0.0, -0.99, 0.77, -2.0, -1.71, -0.0],
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
        obs = rg_node.get_obs_from_real_world()
        actions = torch.zeros(7).unsqueeze(0)
        obs_sim, _, _, _ = env.step(actions)

        step_counter = 0
        for idx, target in enumerate(target_positions):
            target_tensor = torch.tensor(target)

            for _ in range(steps_per_target):
                joint_angles = obs[0][0][0:6]
                joint_angles_sim = obs_sim[0][0:6]
                gripper_action = obs[0][0][19]
                gripper_action_sim = obs_sim[0][19]
                cube_pos = obs[0][0][19:22]
                cube_pos_sim = obs_sim[0][19:22]

                # Compute delta and stop if close enough
                delta = target_tensor - joint_angles.cpu()
                if torch.all(delta.abs() < stop_epsilon):
                    print(f"[INFO] Reached target {idx} after {step_counter} steps.")
                    break

                scaled_delta = torch.clamp(delta * 5.0, -1.0, 1.0)  # Soft gain
                action = torch.cat((scaled_delta, gripper_action)).unsqueeze(0)
                action = torch.tanh(action)

                rg_node.step_real_with_action(action, action_scale=action_scale)
                obs = rg_node.get_obs_from_real_world()
                obs_sim, _, _, _ = env.step(action)

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


def move_to_detection_test_positions_real_and_sim(
    rg_node: ros_to_gym,
    env,
    output_file="/home/luca/isaaclab_ws/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5rl/logdir/sim2real_test_log.csv",
    steps_per_target=1500,
    stop_epsilon=0.02,  # Stop if joint error is below this [rad]
    action_scale=0.08,  # Scale for real robot
):

    target_positions = [
        [-0.0, -2.66, 2.43, -2.54, -1.71, -0.0],
        [-0.0, -1.26, 0.77, -2.3, -1.71, -0.0],
        [-0.0, -0.99, 0.77, -2.0, -1.71, -0.0],
    ]

    gripper_action = torch.tensor([-1.0])  # Fixed gripper value

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "target_idx",
                *[f"real_joint_{i}" for i in range(6)],
                "real_gripper",
                "real_cube_x",
                "real_cube_y",
                "real_cube_z",
                *[f"real_vel_{i}" for i in range(6)],
                *[f"real_torque_{i}" for i in range(6)],
                *[f"sim_joint_{i}" for i in range(6)],
                "sim_gripper",
                "sim_cube_x",
                "sim_cube_y",
                "sim_cube_z",
                *[f"sim_vel_{i}" for i in range(6)],
                *[f"sim_torque_{i}" for i in range(6)],
            ]
        )

        # Reset environment
        obs = rg_node.get_obs_from_real_world()
        actions = torch.zeros(7).unsqueeze(0)
        obs_sim, _, _, _ = env.step(actions)

        step_counter = 0
        for idx, target in enumerate(target_positions):
            target_tensor = torch.tensor(target)

            for _ in range(steps_per_target):
                # Real-world observations
                joint_angles = obs[0][0][0:6].cpu().numpy()
                gripper_state = obs[0][0][18].cpu().item()
                cube_pos = obs[0][0][19:22].cpu().numpy()
                vel = obs[0][0][6:12].cpu().numpy()
                torque = obs[0][0][12:18].cpu().numpy()

                # Simulated observations
                joint_angles_sim = obs_sim[0][0:6].cpu().numpy()
                gripper_state_sim = obs_sim[0][18].cpu().item()
                cube_pos_sim = obs_sim[0][19:22].cpu().numpy()
                vel_sim = obs_sim[0][6:12].cpu().numpy()
                torque_sim = obs_sim[0][12:18].cpu().numpy()

                # Compute delta and stop if close enough
                delta = target_tensor - torch.tensor(joint_angles)
                if torch.all(delta.abs() < stop_epsilon):
                    print(f"[INFO] Reached target {idx} after {step_counter} steps.")
                    break

                scaled_delta = torch.clamp(delta * 5.0, -1.0, 1.0)  # Soft gain
                action = torch.cat(
                    (scaled_delta, torch.tensor([gripper_state]))
                ).unsqueeze(0)
                action = torch.tanh(action)

                rg_node.step_real_with_action(action, action_scale=action_scale)
                obs = rg_node.get_obs_from_real_world()
                obs_sim, _, _, _ = env.step(action)

                writer.writerow(
                    [
                        step_counter,
                        idx,
                        *joint_angles,
                        gripper_state,
                        *cube_pos,
                        *vel,
                        *torque,
                        *joint_angles_sim,
                        gripper_state_sim,
                        *cube_pos_sim,
                        *vel_sim,
                        *torque_sim,
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
        obs, _, _, _ = env.step(actions)

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

                obs, _, _, _ = env.step(action)

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


def do_nothing(env, steps=500):
    actions = torch.zeros(7).unsqueeze(0)
    for _ in range(steps):
        start = time.time()
        obs, _, _, info = env.step(actions)
        elapsed = time.time() - start
        print(f"[DEBUG] Doing nothing once took: {(elapsed):.4f} s")


def extract_learning_metrics(log_results):
    import torch
    import statistics
    from collections import deque

    metrics = {}

    # Extract core metrics
    for key in [
        "mean_value_loss",
        "mean_surrogate_loss",
        "collection_time",
        "learn_time",
        "it",
        "tot_iter",
    ]:
        value = log_results.get(key)
        if isinstance(value, torch.Tensor):
            value = value.item()
        metrics[key] = value

    # --- Snapshot-based reward metrics (from unfinished episodes) ---
    reward_sum = log_results.get("cur_reward_sum")
    ep_lengths = log_results.get("cur_episode_length")
    if isinstance(reward_sum, torch.Tensor) and isinstance(ep_lengths, torch.Tensor):
        mask = ep_lengths > 0
        if mask.any():
            rewards = reward_sum[mask]
            lengths = ep_lengths[mask]
            metrics["mean_reward_per_episode_snapshot"] = (
                (rewards / lengths).mean().item()
            )
            metrics["mean_reward_per_step"] = (rewards.sum() / lengths.sum()).item()

    # --- RSL-style mean reward over recent completed episodes ---
    rewbuffer = log_results.get("rewbuffer")
    if isinstance(rewbuffer, (list, deque)) and len(rewbuffer) > 0:
        try:
            metrics["mean_reward_per_episode_rsl"] = statistics.mean(rewbuffer)
        except Exception as e:
            metrics["mean_reward_per_episode_rsl"] = f"Error: {e}"

    return metrics


def train_CL_agent(
    env: RslRlVecEnvWrapper,
    curriculum_thresholds,
    env_cfg,
    agent_cfg,
    start_cl=3,
    resume=False,
):

    # Plateau detection parameters
    plateau_window = 3  # how many recent rewards to track (100 episodes each return)
    plateau_tolerance = 0.01  # min improvement needed to continue

    max_iters_per_cl = 20  # Each iter runs for 100 episodes = 2000 steps per cl

    max_cl = max(curriculum_thresholds.keys())

    env.unwrapped.set_train_mode()
    current_cl = start_cl

    while current_cl <= max_cl:
        recent_rewards = []  # reset reward history per CL

        for local_iter in range(max_iters_per_cl):
            print(
                f"\n🎯 Training at Curriculum Level CL{current_cl} (Iteration {local_iter})"
            )

            # Run training iteration
            log_results = train_rsl_rl_agent_init(
                env, env_cfg, agent_cfg, CL=current_cl, resume=resume
            )

            # Extract metrics and save
            log_results_ser = extract_learning_metrics(log_results)
            log_path = f"/home/luca/isaaclab_ws/IsaacLab/source/standalone/ur5rl/pretrain_CL{current_cl}_iter{local_iter}.json"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(log_results_ser, f, indent=4)

            print(json.dumps(log_results_ser, indent=2))

            # Check reward
            mean_rew = log_results_ser.get("mean_reward_per_episode_rsl", 0.0)

            # Update recent rewards for plateau detection
            recent_rewards.append(mean_rew)
            if len(recent_rewards) > plateau_window:
                recent_rewards.pop(0)

            # Condition 1: Threshold passed
            if mean_rew >= curriculum_thresholds[current_cl]:
                print(
                    f"✅ Reward {mean_rew:.4f} passed threshold {curriculum_thresholds[current_cl]} — moving to CL{current_cl + 1}"
                )
                current_cl += 1
                break

            # Condition 2: Plateau detected
            if len(recent_rewards) == plateau_window:
                improvement = recent_rewards[-1] - recent_rewards[0]
                if improvement < plateau_tolerance:
                    print(
                        f"📉 Plateau detected (Δreward={improvement:.4f} < {plateau_tolerance}) — moving to CL{current_cl + 1}"
                    )
                    current_cl += 1
                    break

            resume = True

        else:
            print(
                f"⚠️ Max iterations reached for CL{current_cl}, moving to next CL level."
            )
            current_cl += 1


def print_boxed(message, color=["grey"], symbol="*"):
    line = symbol * (len(message) + 4)
    print(colored(line, color))
    print(colored(f"{symbol} {message} {symbol}", color))
    print(colored(line, color))


def main():
    """Main function."""

    # Get init state from real hw or stored state
    use_real_hw = True
    # Pretrain the model
    pretrain = False
    # Resume the last training
    resume = True
    start_cl = 3
    EXPERIMENT_NAME = "_"
    NUM_ENVS = 2  #  28
    ACTION_SCALE_REAL = 0.04
    RANDOMIZE = False

    # Set the goal state of the cube
    cube_goal_pos = [1.0, -0.1, 0.8]

    # Set environment configuration
    env_cfg = parse_env_cfg(
        task_name="Isaac-Ur5-RL-Direct-v0",
        num_envs=NUM_ENVS,
    )

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
        # Shift cube pos height to avoid collision
        # cube_pos[2] -= 0.2
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
        cube_pos = [0.93, 0.0, 0.57]

    # env_cfg.cube_init_state = cube_pos  # type: ignore
    env_cfg.arm_joints_init_state = real_joint_angles[:-1]  # type: ignore
    env_cfg.cube_init_state = cube_pos  # type: ignore
    if use_real_hw:
        env_cfg.action_scale = 1.0  # type: ignore to increase overall execution speed
        env_cfg.CL_state = 1  # type: ignore
        env_cfg.episode_length_s = 15  # was 90000

    # Create digital twin with the real-world state
    env = gym.make(
        id="Isaac-Ur5-RL-Direct-v0",
        cfg=env_cfg,
        cube_goal_pos=cube_goal_pos,
        randomize=RANDOMIZE,
    )

    # Set the initial pose of the arm in the simulator
    if env.unwrapped.set_arm_init_pose(real_joint_angles):  # type: ignore
        print("Set arm init pose successful!")
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)  # type: ignore

    if pretrain:
        import os
        import json

        curriculum_thresholds = {
            0: 0.0,  # unussed
            1: -0.10,  # min mean_reward_per_episode_rsl to move to CL1
            2: 5.0,  # min reward to go to CL2
            3: 11.0,  # etc.
            4: 9.0,
        }

        train_CL_agent(
            env=env,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            start_cl=start_cl,
            resume=resume,
            curriculum_thresholds=curriculum_thresholds,
        )

    # ------------------- CORE WORKFLOW INIT -----------------------------

    while True:
        env.unwrapped.set_eval_mode()  # type: ignore

        # VALIDATION RUN:
        success, interrupt, time_out, obs, policy = run_task_in_sim(
            env,
            log_dir=log_dir,
            resume_path=resume_path,
            agent_cfg=agent_cfg,
            simulation_app=simulation_app,
        )
        obs = obs.squeeze()
        obs = obs[0]

        print(f"Success: {success}")
        print(f"Interrupt: {interrupt}")
        print(f"Time out: {time_out}")
        # print(f"Obs: {obs}")
        # interrupt = True  #! Force Retrain for Debug

        # REAL-WORLD RUN:
        if success and use_real_hw:
            print_boxed("🎯 Task solved in Simulation!", color="green")
            print(colored("🤖 Moving network control to real robot...", "cyan"))
            success = False
            interrupt = False
            time_out = False

            while not (success or interrupt or time_out):
                success, interrupt, time_out, obs = rg_node.step_real(
                    policy, action_scale=ACTION_SCALE_REAL
                )
            obs = obs.squeeze()
        if success and not use_real_hw:
            print_boxed("🎯 Task solved in Simulation!", color="green")
            print(colored("🤖 Real Hardware is disabled -> Done", color="cyan"))
            break

        # STATE SPECIFIC RETRAINING:
        if interrupt:
            print_boxed("🚨 Interrupt received, stopping the robot!", color="red")
            joint_angles = obs[0:6].cpu().numpy()
            gripper = bool(obs[7].cpu().numpy())
        # GENERAL RETRAINING:
        elif time_out:
            print_boxed("⏰ Time out reached!", color="yellow")
            joint_angles = [
                -0.15472919145692998,
                -1.8963201681720179,
                1.5,
                -2.460175625477926,
                -1.5792139212237757,
                -0.0030048529254358414,
            ]
            gripper = [-1.0]

            env.unwrapped.set_gripper_action_bin(gripper)
            env.unwrapped.set_arm_init_pose(joint_angles)
            env.unwrapped.set_eval_mode()
        else:
            print_boxed("❗ Termination without event! Exiting loop...", color="red")
            break

        print_boxed("🔄 Retraining the agent...", color="blue", symbol="=")
        curriculum_thresholds = {4: 5.0}
        start_cl = 4
        train_CL_agent(
            env=env,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            start_cl=start_cl,
            resume=resume,
            curriculum_thresholds=curriculum_thresholds,
        )

    # --------------------------------------------------------------------
    return


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
