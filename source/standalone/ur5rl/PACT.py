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


# Real Environment Wrapper #! TODO FERTIGSTELLEN
class RealUR5Env(DirectRLEnv):
    """Wrapper to mimic an IsaacGym environment but fetch real sensor data."""

    def __init__(self, ur5_controller, realsense_node, cube_goal_pos):
        self.ur5_controller = ur5_controller
        self.realsense_node = realsense_node
        self.cube_goal_pos = cube_goal_pos

    def get_observations(self):
        """Fetch real-world joint states and cube position."""
        real_joint_states = get_current_joint_pos_from_real_robot(self.ur5_controller)
        cube_pos, data_age, z = get_current_cube_pos_from_real_robot(
            self.realsense_node
        )
        cube_pos = torch.from_numpy(cube_pos).to("cuda:0")
        data_age = torch.from_numpy(data_age).to("cuda:0")
        z = torch.from_numpy(z).to("cuda:0")

        cube_distance_to_goal = torch.norm(
            cube_pos - torch.tensor(self.cube_goal_pos, device="cuda:0"),
            dim=-1,
            keepdim=True,
        )

        joint_pos = torch.tensor(
            real_joint_states["joint_positions"],  # type: ignore
            device="cuda:0",
            dtype=torch.float32,
        ).unsqueeze(0)
        joint_vel = torch.tensor(
            real_joint_states["joint_velocities"],  # type: ignore
            device="cuda:0",
            dtype=torch.float32,
        ).unsqueeze(0)
        joint_torque = torch.tensor(
            real_joint_states["joint_torques"], device="cuda:0", dtype=torch.float32  # type: ignore
        ).unsqueeze(0)
        gripper_state = torch.tensor(
            real_joint_states["gripper_state"], device="cuda:0", dtype=torch.float32  # type: ignore
        ).unsqueeze(0)

        obs = torch.cat(
            (
                joint_pos.unsqueeze(1),
                joint_vel.unsqueeze(1),
                joint_torque.unsqueeze(1),
                gripper_state.unsqueeze(1),
                cube_pos.unsqueeze(1),
                cube_distance_to_goal.unsqueeze(1),
                data_age.unsqueeze(1).unsqueeze(1),
                z.unsqueeze(1),
            ),
            dim=-1,
        )

        return {"policy": obs.float().squeeze(1)}

    def reset(self):
        """Return initial observation."""
        return self.get_observations(), {}

    def step(self, action):
        """Send action to UR5 and return new state."""
        #!HW PROTECTION self.ur5_controller.set_joint_delta(action.squeeze(0).cpu().numpy())
        obs = self.get_observations()
        return obs, 0, False, False, {}


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


def run_task(
    env: gym.Env,
    real_mode: bool,
    log_dir: str,
    resume_path: str,
    agent_cfg: RslRlOnPolicyRunnerCfg,
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
                    return False, False, obs
                else:
                    print("Interrupt detected!")
                    last_obs = obs.clone()
                    torch.save(last_obs, os.path.join(log_dir, "last_obs.pt"))
                    return False, True, obs

            if info["observations"]["goal_reached"][0]:  # type: ignore
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


def get_obs_from_real_world(ur5_controller, realsense_node, cube_goal_pos):
    """Get the observations from the real world."""
    # Get the current joint positions from the real robot
    while ur5_controller.get_joint_observation() == None:
        pass
    real_joint_positions = ur5_controller.get_joint_observation()
    cube_pos, data_age, z = get_current_cube_pos_from_real_robot(realsense_node)
    cube_pos = torch.from_numpy(cube_pos).to("cuda:0")
    data_age = torch.tensor(data_age).to("cuda:0")
    z = torch.tensor(z).to("cuda:0")

    cube_pos = cube_pos.unsqueeze(dim=0)

    cube_distance_to_goal = torch.norm(cube_pos - cube_goal_pos, dim=-1, keepdim=False)

    obs = torch.cat(
        (
            real_joint_positions["joint_positions"].unsqueeze(dim=1),
            real_joint_positions["joint_velocities"].unsqueeze(dim=1),
            real_joint_positions["joint_torques"].unsqueeze(dim=1),
            real_joint_positions["gripper_state"].unsqueeze(dim=1),
            cube_pos.unsqueeze(dim=1),
            cube_distance_to_goal.unsqueeze(dim=1).unsqueeze(dim=1),
            data_age.unsqueeze(dim=1).unsqueeze(dim=1),
            z.unsqueeze(dim=1),
        ),
        dim=-1,
    )

    obs = obs.float()
    obs = obs.squeeze(dim=1)

    observations = {"policy": obs}
    return observations


def run_task_in_real(policy, ur5_controller, realsense_node, cube_goal_pos):
    """Play with RSL-RL agent in real world."""
    # Get the current joint positions from the real robot
    obs = get_obs_from_real_world(ur5_controller, realsense_node, cube_goal_pos)
    print(f"Observations: {obs}")


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


def load_policy_only():
    """Load the trained RL policy WITHOUT requiring an environment."""

    agent_cfg: Ur5RLPPORunnerCfg = Ur5RLPPORunnerCfg()

    # Find the most recent checkpoint
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    policy = torch.jit.load(resume_path)

    policy.eval()  # Put policy in inference mode

    return policy


def main():
    """Main function."""
    # Set the goal state of the cube
    cube_goal_pos = torch.tensor([1.0, -0.1, 0.8], device="cuda:0")

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
        cube_goal_pos=[1.0, -0.1, 0.8],
    )

    # Run the task in the simulator
    success, interrupt, obs = run_task(
        env,
        real_mode=False,
        log_dir=log_dir,
        resume_path=resume_path,
        agent_cfg=agent_cfg,
        arm_state=real_joint_positions[:-1],  # type: ignore
        cube_state=tuple(cube_pos),
    )
    # --------------------------------------------------------------------

    print(f"Success: {success}")
    print(f"Interrupt: {interrupt}")

    if True:
        print("Task solved in Sim!")
        print("Moving network control to real robot...")
        # Create real env wrapper
        # real_env = RealUR5Env(ur5_control, realsense, cube_goal_pos)

        policy = load_policy_only()
        print("Policy loaded!")
        action = policy(obs)
        print(f"Action: {action}")

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
