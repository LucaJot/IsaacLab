# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

from datetime import datetime
import subprocess
import sys
import os

import argparse
import time

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

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
    env: RslRlVecEnvWrapper,
    log_dir: str,
    resume_path: str,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Play with RSL-RL agent."""

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
    cube_pos, data_age, z, pos_sensor = get_current_cube_pos_from_real_robot(
        realsense_node
    )
    cube_pos = torch.from_numpy(cube_pos).to("cuda:0")
    data_age = torch.tensor(data_age).to("cuda:0")
    z = torch.tensor(z).to("cuda:0")
    pos_sensor = torch.tensor(pos_sensor).to("cuda:0")

    cube_pos = cube_pos[0]
    pos_sensor = pos_sensor[0]
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
    pos_sensor = pos_sensor.unsqueeze(0)

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
            pos_sensor.unsqueeze(dim=1),
        ),
        dim=-1,
    )

    obs = obs.float()
    obs = obs.squeeze(dim=1)

    return obs


def step_real(policy, ur5_controller, realsense_node, cube_goal_pos, action_scale=1.0):
    """Play with RSL-RL agent in real world."""
    # Get the current joint positions from the real robot
    obs = get_obs_from_real_world(ur5_controller, realsense_node, cube_goal_pos)
    action = policy(obs)
    action = torch.tanh(action)  # (make sure it is in the range [-1, 1])
    action = action * 0.05  # * action_scale
    action = action.squeeze(dim=0)
    print(f"Action: {action}")
    print(f"Observations: {obs}")
    # Execute the action on the real robot
    ur5_controller.set_joint_delta(action.detach().cpu().numpy())
    return obs


def get_current_cube_pos_from_real_robot(realsense_node: realsense_obs_reciever):
    """Sync the simulated cube position with the real cube position."""
    # Sync sim cube with real cube
    print("[INFO]: Waiting for cube positions from the real robot...")
    while realsense_node.get_cube_position() == None:
        pass
    real_cube_positions, data_age, z, pos_sensor = realsense_node.get_cube_position()
    return real_cube_positions, data_age, z, pos_sensor


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


def train_rsl_rl_agent(env, env_cfg, agent_cfg, resume=True):
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if resume:
        # save resume path before creating a new log_dir
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint

    if resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )


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
    cube_pos, _, _, _ = get_current_cube_pos_from_real_robot(realsense_node)
    cube_pos = cube_pos[0]
    distance = np.linalg.norm(cube_pos - goal_pos)
    return distance < threshold


# Get init state from real hw or stored state
use_real_hw = False
# Resume the last training
resume = True
EXPERIMENT_NAME = "_"
NUM_ENVS = 16


def main():
    """Main function."""
    # Set the goal state of the cube
    cube_goal_pos = [1.0, -0.1, 0.8]

    # Set rl learning configuration
    agent_cfg, log_dir, resume_path = set_learning_config()

    if use_real_hw:
        # Start the ROS 2 nodes ----------------------------------------------
        rclpy.init()
        ur5_control = Ur5JointController()
        realsense = realsense_obs_reciever()
        start_ros_nodes(ur5_control, realsense)
        # --------------------------------------------------------------------

        # Get the current joint positions from the real robot ----------------
        real_joint_positions = get_current_joint_pos_from_real_robot(ur5_control)
        cube_pos, data_age, z, pos_sensor = get_current_cube_pos_from_real_robot(
            realsense
        )
        # real_cube_positions, data_age, z, pos_sensor
        # Unpack (real has no parallel envs)
        cube_pos = cube_pos[0]
        cube_pos[2] += 0.2
        data_age = data_age[0]
        print(f"Recieved Real Joint Positions: {real_joint_positions}")
        print(f"Recieved Real Cube Positions: {cube_pos}")
        print(f"Z: {z}")
    # --------------------------------------------------------------------
    else:
        real_joint_positions = [
            -0.15472919145692998,
            -1.8963201681720179,
            2.0,
            -2.160175625477926,
            -1.5792139212237757,
            -0.0030048529254358414,
            -1.0,
        ]
        cube_pos = [1.0, 0.0, 1.0]

    # Run the task with real state in simulation -------------------------
    env_cfg = parse_env_cfg(
        task_name="Isaac-Ur5-RL-Direct-v0",
        num_envs=NUM_ENVS,
    )
    env_cfg.cube_init_state = cube_pos  # type: ignore
    env_cfg.arm_init_state = real_joint_positions  # type: ignore

    # Create simulated environment with the real-world state
    env = gymnasium.make(
        id="Isaac-Ur5-RL-Direct-v0",
        cfg=env_cfg,
        cube_goal_pos=cube_goal_pos,
    )
    if env.set_arm_init_pose(real_joint_positions):
        print("Set arm init pose successful!")
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)  # type: ignore

    # Run the task in the simulator
    success, interrupt, obs, policy = run_task_in_sim(
        env, log_dir=log_dir, resume_path=resume_path, agent_cfg=agent_cfg
    )
    # --------------------------------------------------------------------

    print(f"Success: {success}")
    print(f"Interrupt: {interrupt}")
    interrupt = True  #! Force Retrain for Debug
    # success = True  #! Force Real Robot for Debug

    if success:
        print("Task solved in Sim!")
        print("Moving network control to real robot...")
        # Create real env wrapper
        # real_env = RealUR5Env(ur5_control, realsense, cube_goal_pos)

        while not goal_reached(realsense, cube_goal_pos):
            obs = step_real(
                policy,
                ur5_control,
                realsense,
                cube_goal_pos,
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
