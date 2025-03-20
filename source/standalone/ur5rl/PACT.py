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

# Get init state from real hw or stored state
use_real_hw = False
# Resume the last training
resume = False
EXPERIMENT_NAME = "_"
NUM_ENVS = 16


def main():
    """Main function."""
    # Set the goal state of the cube
    cube_goal_pos = [1.0, -0.1, 0.8]

    # Set rl learning configuration
    agent_cfg, log_dir, resume_path = set_learning_config()
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
        cube_pos = [1.0, 0.0, 1.0]

    # Run the task with real state in simulation -------------------------
    env_cfg = parse_env_cfg(
        task_name="Isaac-Ur5-RL-Direct-v0",
        num_envs=NUM_ENVS,
    )
    # env_cfg.cube_init_state = cube_pos  # type: ignore
    env_cfg.arm_joints_init_state = real_joint_angles[:-1]  # type: ignore

    # Create simulated environment with the real-world state
    env = gym.make(
        id="Isaac-Ur5-RL-Direct-v0",
        cfg=env_cfg,
        cube_goal_pos=cube_goal_pos,
    )

    if env.set_arm_init_pose(real_joint_angles):
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
