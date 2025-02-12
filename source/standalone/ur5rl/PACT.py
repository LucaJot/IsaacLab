# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import sys
import os

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

import gymnasium
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

from rsl_rl.modules import ActorCritic

from agents.rsl_rl_ppo_cfg import (
    Ur5RLPPORunnerCfg,
)

from ur5_rl_env_cfg import HawUr5EnvCfg

from ros2_humble_ws.src.ur5_parallel_control.ur5_parallel_control.ur5_basic_control_fpc import (
    Ur5JointController,
)

from ros2_humble_ws.src.ur5_parallel_control.ur5_parallel_control.realsense_obs import (
    realsense_obs_reciever,
)

import threading
import rclpy

import torch
import gymnasium as gym
from omni.isaac.lab.envs import DirectRLEnv

import numpy as np


# Separate thread to run the ROS 2 node in parallel to the simulation
def joint_controller_node_thread(node: Ur5JointController):
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


def realsense_node_thread(node: realsense_obs_reciever):
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


def get_current_joint_pos_from_real_robot(ur5_controller: Ur5JointController):
    """Sync the simulated robot joints with the real robot."""
    # Sync sim joints with real robot
    print("[INFO]: Waiting for joint positions from the real robot...")
    while ur5_controller.get_joint_positions() == None:
        pass
    real_joint_positions = ur5_controller.get_joint_positions()
    return real_joint_positions


def run_task_in_sim(
    env: gym.Env,
    log_dir: str,
    resume_path: str,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Play with RSL-RL agent."""
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)  # type: ignore

    policy = load_most_recent_model(
        env=env,
        log_dir=log_dir,
        resume_path=resume_path,
        agent_cfg=agent_cfg,
    )

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, reward, dones, info = env.step(actions)  # type: ignore
            # print_dict(info)
            if dones[0]:  # type: ignore
                if info["time_outs"][0]:  # type: ignore
                    print("Time Out!")
                    return False, False, obs, policy
                else:
                    print("Interrupt detected!")
                    last_obs = obs.clone()
                    torch.save(last_obs, os.path.join(log_dir, "last_obs.pt"))
                    return False, True, obs, policy

            if info["observations"]["goal_reached"][0]:  # type: ignore
                print("Goal Reached!")
                return True, False, obs, policy

    # close the simulator
    env.close()


def start_ros_nodes(
    ur5_controller: Ur5JointController, realsense_node: realsense_obs_reciever
):
    """Start both ROS 2 nodes using a MultiThreadedExecutor."""
    executor = rclpy.executors.MultiThreadedExecutor()

    executor.add_node(ur5_controller)
    executor.add_node(realsense_node)

    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    return executor, thread


def get_obs_from_real_world(ur5_controller, realsense_node, cube_goal_pos):
    """Get the observations from the real world."""
    # Get the current joint positions from the real robot
    real_joint_positions = ur5_controller.get_joint_observation()
    while real_joint_positions == None:
        real_joint_positions = ur5_controller.get_joint_observation()
    cube_pos, data_age, z = get_current_cube_pos_from_real_robot(realsense_node)
    cube_pos = torch.from_numpy(cube_pos).to("cuda:0")
    data_age = torch.tensor(data_age).to("cuda:0")
    z = torch.tensor(z).to("cuda:0")

    cube_pos = cube_pos[0]
    cube_goal = torch.tensor(cube_goal_pos).to("cuda:0")
    cube_distance_to_goal = torch.norm(
        cube_pos - cube_goal, dim=-1, keepdim=False
    ).unsqueeze(dim=0)

    real_joint_positions_t = torch.tensor(real_joint_positions["joint_positions"]).to(
        "cuda:0"
    )
    real_joint_velocities_t = torch.tensor(real_joint_positions["joint_velocities"]).to(
        "cuda:0"
    )
    real_joint_torques_t = torch.tensor(real_joint_positions["joint_torques"]).to(
        "cuda:0"
    )
    real_gripper_state_t = (
        torch.tensor(real_joint_positions["gripper_state"])
        .to("cuda:0")
        .unsqueeze(dim=0)
    )

    # Ensure correct shape for tensors before concatenation
    real_joint_positions_t = real_joint_positions_t.unsqueeze(0)  # (1, 6)
    real_joint_velocities_t = real_joint_velocities_t.unsqueeze(0)  # (1, 6)
    real_joint_torques_t = real_joint_torques_t.unsqueeze(0)  # (1, 6)
    real_gripper_state_t = real_gripper_state_t
    cube_pos = cube_pos.unsqueeze(0)  # (1, 3)
    cube_distance_to_goal = cube_distance_to_goal

    obs = torch.cat(
        (
            real_joint_positions_t.unsqueeze(dim=1),
            real_joint_velocities_t.unsqueeze(dim=1),
            real_joint_torques_t.unsqueeze(dim=1),
            real_gripper_state_t.unsqueeze(dim=1).unsqueeze(dim=1),
            cube_pos.unsqueeze(dim=1),
            cube_distance_to_goal.unsqueeze(dim=1).unsqueeze(dim=1),
            data_age.unsqueeze(dim=1).unsqueeze(dim=1),
            z.unsqueeze(dim=1).unsqueeze(dim=1),
        ),
        dim=-1,
    )

    obs = obs.float()
    obs = obs.squeeze(dim=1)

    return obs


def step_real(policy, ur5_controller, realsense_node, cube_goal_pos):
    """Play with RSL-RL agent in real world."""
    # Get the current joint positions from the real robot
    obs = get_obs_from_real_world(ur5_controller, realsense_node, cube_goal_pos)
    action = policy(obs)
    print(f"Action: {action}")
    print(f"Observations: {obs}")
    # Execute the action on the real robot
    #!ur5_controller.set_joint_delta(action.cpu().numpy())
    return obs


def get_current_cube_pos_from_real_robot(realsense_node: realsense_obs_reciever):
    """Sync the simulated cube position with the real cube position."""
    # Sync sim cube with real cube
    print("[INFO]: Waiting for cube positions from the real robot...")
    while realsense_node.get_cube_position() == None:
        pass
    real_cube_positions, data_age, z = realsense_node.get_cube_position()
    return real_cube_positions, data_age, z


def set_learning_config():
    # Get learning configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = Ur5RLPPORunnerCfg()

    # specify directory for logging experiments --------------------------
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    log_dir = os.path.dirname(resume_path)
    # --------------------------------------------------------------------
    return agent_cfg, log_dir, resume_path


def load_most_recent_model(
    env: gym.Env, log_dir, resume_path, agent_cfg: RslRlOnPolicyRunnerCfg
):
    """Load the most recent model from the log directory."""
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env,  # type: ignore
        agent_cfg.to_dict(),  # type: ignore
        log_dir=None,
        device=agent_cfg.device,
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device="cuda:0")

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic,
        ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.pt",
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    return policy


def goal_reached(realsense_node, goal_pos, threshold=0.05):
    """Check if the goal is reached."""
    cube_pos, _, _ = get_current_cube_pos_from_real_robot(realsense_node)
    cube_pos = cube_pos[0]
    distance = np.linalg.norm(cube_pos - goal_pos)
    return distance < threshold


def main():
    """Main function."""
    # Set the goal state of the cube
    cube_goal_pos = [1.0, -0.1, 0.8]

    # Set rl learning configuration
    agent_cfg, log_dir, resume_path = set_learning_config()

    # Start the ROS 2 nodes ----------------------------------------------
    rclpy.init()
    ur5_control = Ur5JointController()
    realsense = realsense_obs_reciever()
    start_ros_nodes(ur5_control, realsense)
    # --------------------------------------------------------------------

    # Get the current joint positions from the real robot ----------------
    real_joint_positions = get_current_joint_pos_from_real_robot(ur5_control)
    cube_pos, data_age, z = get_current_cube_pos_from_real_robot(realsense)
    # Unpack (real has no parallel envs)
    cube_pos = cube_pos[0]
    cube_pos[2] += 0.2
    data_age = data_age[0]
    print(f"Recieved Real Joint Positions: {real_joint_positions}")
    print(f"Recieved Real Cube Positions: {cube_pos}")
    print(f"Z: {z}")
    # --------------------------------------------------------------------

    # Run the task with real state in simulation -------------------------
    env_cfg = parse_env_cfg(
        task_name="Isaac-Ur5-RL-Direct-v0",
        num_envs=1,
    )
    env_cfg.cube_init_state = cube_pos  # type: ignore
    env_cfg.arm_init_state = real_joint_positions  # type: ignore

    # Create simulated environment with the real-world state
    env = gymnasium.make(
        id="Isaac-Ur5-RL-Direct-v0",
        cfg=env_cfg,
        cube_goal_pos=cube_goal_pos,
    )

    # Run the task in the simulator
    success, interrupt, obs, policy = run_task_in_sim(
        env, log_dir=log_dir, resume_path=resume_path, agent_cfg=agent_cfg
    )
    # --------------------------------------------------------------------

    print(f"Success: {success}")
    print(f"Interrupt: {interrupt}")

    if True:
        print("Task solved in Sim!")
        print("Moving network control to real robot...")
        # Create real env wrapper
        # real_env = RealUR5Env(ur5_control, realsense, cube_goal_pos)

        while not goal_reached(realsense, cube_goal_pos):
            obs = step_real(policy, ur5_control, realsense, cube_goal_pos)
            print(f"Observations: {obs}")

    if interrupt:
        # Get the interrupt joint positions from the real robot
        # Get current Cube position from realsense
        # Start training loop in SIM with this state
        pass

    return


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
