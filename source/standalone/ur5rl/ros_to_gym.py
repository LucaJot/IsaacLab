import time
import rclpy
import threading
import torch

from ros2_humble_ws.src.ur5_parallel_control.ur5_parallel_control.ur5_basic_control_fpc import (
    Ur5JointController,
)

from ros2_humble_ws.src.ur5_parallel_control.ur5_parallel_control.realsense_obs import (
    realsense_obs_reciever,
)

import numpy as np
from termcolor import colored


class ros_to_gym:
    def __init__(self):

        self.ur5_control: Ur5JointController = None
        self.realsense: realsense_obs_reciever = None

        self.cube_goal_pos = [1.0, -0.1, 0.8]  #! Remove and replace with pickup?
        self.real_cube_positions = None
        self.data_age = None
        self.z = None
        self.pos_sensor = None

    def run_ros_nodes(self):
        """Start both ROS 2 nodes using a MultiThreadedExecutor."""
        rclpy.init()
        self.ur5_control = Ur5JointController()
        self.realsense = realsense_obs_reciever()

        executor = rclpy.executors.MultiThreadedExecutor()

        executor.add_node(self.ur5_control)
        executor.add_node(self.realsense)

        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()

        return executor, thread

    def close(self):
        """Close the ROS 2 nodes."""
        self.ur5_control.destroy_node()
        self.realsense.destroy_node()
        rclpy.shutdown

    def get_current_joint_pos_from_real_robot(self):
        """Sync the simulated robot joints with the real robot."""
        # Sync sim joints with real robot
        print("[INFO]: Waiting for joint positions from the real robot...")
        while self.ur5_control.get_joint_positions() == None:
            pass
        real_joint_positions = self.ur5_control.get_joint_positions()
        return real_joint_positions

    def get_current_cube_pos_from_real_robot(self):
        """Sync the simulated cube position with the real cube position."""
        # Sync sim cube with real cube
        while self.realsense.get_cube_position() == None:
            print("[INFO]: Waiting for cube positions from the real robot...")
            pass
        real_cube_positions, data_age, z, pos_sensor = (
            self.realsense.get_cube_position()
        )
        return real_cube_positions, data_age, z, pos_sensor

    def get_obs_from_real_world(self):
        """Get the observations from the real world.
        return: obs (torch.Tensor): The observations from the real world as a tensor for the NN.
                obs_readable (tuple): The observations from the real world as a tuple for easy access.
                Contains the following:
                - real_joint_positions (dict): The joint positions of the real robot.
                - cube_pos (torch.Tensor): The position of the cube in the world frame.
                - cube_distance_to_goal (torch.Tensor): The distance of the cube to the goal position.
                - data_age (torch.Tensor): The age of the data from the cube.
                - z (torch.Tensor): The depth value of the cube's centroid.
                - pos_sensor (torch.Tensor): The cube's position on the sensor


        """
        # # mesure time to get obs
        # start_time = time.time()

        # Get the current joint positions from the real robot
        while self.ur5_control.get_joint_observation() == None:
            time.sleep(0.01)
        real_joint_info = self.ur5_control.get_joint_observation()

        cube_pos_cpu, data_age_cpu, z_cpu, pos_sensor_cpu = (
            self.get_current_cube_pos_from_real_robot()
        )
        cube_pos_cpu[0][2] += 0.0  #! Add offset to z axis
        cube_pos = torch.from_numpy(cube_pos_cpu).to("cuda:0")
        data_age = torch.tensor(data_age_cpu).to("cuda:0")
        z = torch.tensor(z_cpu).to("cuda:0")
        pos_sensor = torch.tensor(pos_sensor_cpu).to("cuda:0")

        cube_pos = cube_pos[0]
        pos_sensor = pos_sensor[0]
        cube_goal = torch.tensor(self.cube_goal_pos).to("cuda:0")
        cube_distance_to_goal = torch.norm(
            cube_pos - cube_goal, dim=-1, keepdim=False
        ).unsqueeze(dim=0)

        real_joint_positions_t = torch.tensor(real_joint_info["joint_positions"]).to(
            "cuda:0"
        )
        # real_joint_velocities_t = torch.tensor(real_joint_info["joint_velocities"]).to(
        #     "cuda:0"
        # )
        real_joint_torques_t = torch.tensor(real_joint_info["joint_torques"]).to(
            "cuda:0"
        )
        real_gripper_state_t = (
            torch.tensor(real_joint_info["gripper_state"]).to("cuda:0").unsqueeze(dim=0)
        )

        # Ensure correct shape for tensors before concatenation
        real_joint_positions_t = real_joint_positions_t.unsqueeze(0)  # (1, 6)
        # real_joint_velocities_t = real_joint_velocities_t.unsqueeze(0)  # (1, 6)
        real_joint_torques_t = real_joint_torques_t.unsqueeze(0)  # (1, 6)
        real_gripper_state_t = real_gripper_state_t
        cube_pos = cube_pos.unsqueeze(0)  # (1, 3)
        pos_sensor = pos_sensor.unsqueeze(0)

        obs = torch.cat(
            (
                real_joint_positions_t.unsqueeze(dim=1),
                # real_joint_velocities_t.unsqueeze(dim=1),
                real_joint_torques_t.unsqueeze(dim=1),
                real_gripper_state_t.unsqueeze(dim=1).unsqueeze(
                    dim=1
                ),  # Gripper open or clsed
                cube_pos.unsqueeze(dim=1),  # Where is the cube on the world frame
                # cube_distance_to_goal.unsqueeze(dim=1).unsqueeze(
                #     dim=1
                # ),  # How far is the cube from the goalpossition
                data_age.unsqueeze(dim=1).unsqueeze(
                    dim=1
                ),  # Age off the cubes last seen position
                z.unsqueeze(dim=1).unsqueeze(
                    dim=1
                ),  # Depth value of the cubes centroid
                pos_sensor.unsqueeze(dim=1),  # Cube position on the sensor plane
            ),
            dim=-1,
        )

        obs = obs.float()
        # obs = obs.squeeze(dim=1)

        # # time to get obs
        # elapsed = time.time() - start_time

        # print(f"Time to get obs: {elapsed:.4f} seconds")

        return obs, (
            real_joint_info,
            cube_pos_cpu[0],
            cube_distance_to_goal.cpu().item(),
            data_age_cpu[0],
            z_cpu[0],
            pos_sensor_cpu[0],
        )

    def check_grasp_success(self, obs):
        """Check if the grasp is successful."""
        # Check if the cube is in the gripper
        gripper_state = obs[0].squeeze().squeeze()[12].item()  # state == 1 is closed
        distance_cube_cam = obs[0].squeeze().squeeze()[17].item()
        pos_sensor_x = obs[0].squeeze().squeeze()[18].item()
        pos_sensor_y = obs[0].squeeze().squeeze()[19].item()

        if gripper_state == 1 and distance_cube_cam < 0.22 and distance_cube_cam > 0.15:
            # Check if the cube is in the gripper
            if pos_sensor_x > 0.2 and pos_sensor_x < 0.8 and pos_sensor_y > 0.2:
                return True

    def step_real(self, policy, action_scale=1.0):
        """Play with RSL-RL agent in real world."""
        success = False
        interrupt = False
        time_out = False
        # Get the current joint positions from the real robot
        obs, _ = self.get_obs_from_real_world()
        action = policy(obs)
        action = torch.tanh(action)  # (make sure it is in the range [-1, 1])
        action = action * action_scale
        action = action.squeeze(dim=0).squeeze(dim=0)
        action[5] = 0.0  # wrist3 lock
        # Execute the action on the real robot
        self.ur5_control.set_joint_delta(action.detach().cpu().numpy())  # type: ignore

        obs, _ = self.get_obs_from_real_world()

        if self.check_grasp_success(obs):
            print(colored("✅ [SUCCESS]: Grasp successful!", "green"))
            success = True
            return (success, interrupt, time_out, obs)
        # Check if the first 3 torques are > 95 or the last 3 torques are > 20
        torques = np.abs(obs[0].squeeze().squeeze()[6:12].cpu().numpy())
        if np.any(torques > 150) or np.any(torques[3:] > 30):
            print(colored(f"⚠️  [WARNING]: Torques too high! Values: {torques}", "red"))
            print(
                colored(
                    "🛑 [ACTION]: Stopping the robot due to excessive torque.", "red"
                )
            )
            interrupt = True
            return (success, interrupt, time_out, obs)
        if time_out:
            print(colored("⏰ [TIMEOUT]: Maximum time exceeded.", "yellow"))
            return (success, interrupt, time_out, obs)
        dist_cube_cam = obs[0].squeeze().squeeze()[17].item()
        if dist_cube_cam < 0.12 and dist_cube_cam > 0.0:
            print(
                colored(
                    f"📏 [DISTANCE WARNING]: Cube too close to camera ({dist_cube_cam:.3f} m)!",
                    "magenta",
                )
            )
            print(
                colored(
                    "🛑 [ACTION]: Interrupt triggered to prevent unsafe grasp.", "red"
                )
            )
            interrupt = True
            return (success, interrupt, time_out, obs)

        return False, False, False, obs  # No event detected

        # Check if the goal is reached

    def step_real_with_action(self, action, action_scale=0.08):
        """Apply action command to the real world."""
        interrupt = False
        action = torch.tanh(action)  # (make sure it is in the range [-1, 1])
        action = action * action_scale
        action = action.squeeze(dim=0)
        # print(f"Action: {action}")
        # Execute the action on the real robot
        self.ur5_control.set_joint_delta(action.detach().cpu().numpy())  # type: ignore
        obs, _ = self.get_obs_from_real_world()
        torques = np.abs(obs[0].squeeze().squeeze()[6:12].cpu().numpy())
        # Check if the first 3 torques are > 95 or the last 3 torques are > 20
        if np.any(torques > 90) or np.any(torques[3:] > 20):
            print(f"Torques: {torques}")
            print("[INFO]: Torques are too high, stopping the robot.")
            interrupt = True
        return (obs, interrupt)

    def goal_reached(self, threshold=0.05):
        """Check if the goal is reached."""
        cube_pos, _, _, _ = self.get_current_cube_pos_from_real_robot()
        cube_pos = cube_pos[0]
        distance = np.linalg.norm(cube_pos - self.cube_goal_pos)
        return distance < threshold
