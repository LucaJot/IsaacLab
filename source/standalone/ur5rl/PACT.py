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


def get_current_cube_pos_from_real_robot(realsense_node: realsense_obs_reciever):
    """Sync the simulated cube position with the real cube position."""
    # Sync sim cube with real cube
    print("[INFO]: Waiting for cube positions from the real robot...")
    while realsense_node.get_cube_position() == None:
        pass
    real_cube_positions, data_age, z = realsense_node.get_cube_position()
    return real_cube_positions, data_age, z


def run_task_in_sim(
    cube_state: tuple[float, float, float] = (1.0, 0.0, 1.0),
    arm_state: list[float] = [
        0.0,
        -1.92,
        1.92,
        -3.14,
        -1.57,
        0.0,
    ],
):
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        task_name="Isaac-Ur5-RL-Direct-v0",
        num_envs=1,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = Ur5RLPPORunnerCfg()

    env_cfg.cube_init_state = cube_state  # type: ignore
    env_cfg.arm_init_state = arm_state  # type: ignore

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gymnasium.make(
        id="Isaac-Ur5-RL-Direct-v0",
        cfg=env_cfg,
        cube_init_state=[0.5, 0.5, 0.5],
        cube_goal_pos=[1.0, -0.1, 0.8],
    )

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

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
            obs, reward, dones, info = env.step(actions)
            # print_dict(info)
            if False:  # dones[0]:
                if info["time_outs"][0]:
                    print("Time Out!")
                    return False, False, obs
                else:
                    print("Interrupt detected!")
                    last_obs = obs.clone()
                    torch.save(last_obs, os.path.join(log_dir, "last_obs.pt"))
                    return False, True, obs

            if info["observations"]["goal_reached"][0]:
                print("Goal Reached!")
                return True, False, obs

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


def main():
    """Main function."""
    rclpy.init()
    ur5_control = Ur5JointController()
    realsense = realsense_obs_reciever()
    start_ros_nodes(ur5_control, realsense)

    # Get the current joint positions from the real robot
    real_joint_positions = get_current_joint_pos_from_real_robot(ur5_control)
    cube_pos, data_age, z = get_current_cube_pos_from_real_robot(realsense)
    # Unpack (real has no parallel envs)
    cube_pos = cube_pos[0]
    cube_pos[2] += 0.2
    data_age = data_age[0]

    print(f"Recieved Real Joint Positions: {real_joint_positions}")
    print(f"Recieved Real Cube Positions: {cube_pos}")
    print(f"Z: {z}")

    success, interrupt, obs = run_task_in_sim(
        arm_state=real_joint_positions[:-1], cube_state=tuple(cube_pos)
    )

    print(f"Success: {success}")
    print(f"Interrupt: {interrupt}")

    return 0


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
