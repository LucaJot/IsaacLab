from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import (
    GroundPlaneCfg,
    spawn_ground_plane,
    spawn_from_usd,
    UsdFileCfg,
)
from omni.isaac.lab.sim.spawners.shapes import spawn_cuboid, CuboidCfg
from omni.isaac.lab.assets import (
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import CameraCfg, Camera

# from omni.isaac.dynamic_control import _dynamic_control
from pxr import Usd, Gf

# dc = _dynamic_control.acquire_dynamic_control_interface()

import numpy as np
from numpy import float64
from scipy.spatial.transform import Rotation as R

from cube_detector import CubeDetector

from omni.isaac.lab.managers import EventTermCfg as EventTerm
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.utils.noise.noise_cfg import (
    GaussianNoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
)

from ur5_rl_env_cfg import HawUr5EnvCfg

# init pos close to the cube
# [-0.1, -1.00, 1.5, -3.30, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# init pos distant from the cube
# [0.0, -1.92, 1.92, -3.14, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class HawUr5Env(DirectRLEnv):
    cfg: HawUr5EnvCfg

    def __init__(
        self,
        cfg: HawUr5EnvCfg,
        render_mode: str | None = None,
        cube_goal_pos: list = [1.0, -0.1, 0.8],
        arm_init_state: list = [],
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_dof_idx, _ = self.robot.find_joints(self.cfg.arm_dof_name)
        self._gripper_dof_idx, _ = self.robot.find_joints(self.cfg.gripper_dof_name)
        self.haw_ur5_dof_idx, _ = self.robot.find_joints(self.cfg.haw_ur5_dof_name)
        self.action_scale = self.cfg.action_scale

        self.robot.data.default_joint_pos = self.cfg.joint_init_state.repeat(
            self.scene.num_envs, 1
        )

        # Holds the current joint positions and velocities
        self.live_joint_pos: torch.Tensor = self.robot.data.joint_pos
        self.live_joint_vel: torch.Tensor = self.robot.data.joint_vel
        self.live_joint_torque: torch.Tensor = self.robot.data.applied_torque
        self.torque_limit = self.cfg.torque_limit
        self.torque_limit_exeeded: torch.Tensor = torch.zeros(
            self.scene.num_envs, device=self.device, dtype=torch.bool
        )

        self.jointpos_script_GT: torch.Tensor = self.live_joint_pos[:, :].clone()

        self.action_dim = len(self._arm_dof_idx) + len(self._gripper_dof_idx)

        self.gripper_action_bin: torch.Tensor | None = None

        # Cube detection
        self.cubedetector = CubeDetector(num_envs=self.scene.num_envs)
        # Convert the cube goal position to a tensor
        self.cube_goal_pos = torch.FloatTensor(cube_goal_pos).to(self.device)
        # Expand the cube goal position to match the number of environments
        self.cube_goal_pos = self.cube_goal_pos.expand(cfg.scene.num_envs, -1)

        self.goal_reached = torch.zeros(self.scene.num_envs, device=self.device)
        self.data_age = torch.zeros(self.scene.num_envs, device=self.device)
        self.cube_distance_to_goal = torch.ones(self.scene.num_envs, device=self.device)
        self.dist_cube_cam = torch.zeros(self.scene.num_envs, device=self.device)
        self.mean_dist_cam_cube = 0

        # Yolo model for cube detection
        # self.yolov11 = YOLO("yolo11s.pt")

    def get_joint_pos(self):
        return self.live_joint_pos

    def _setup_scene(self):
        # add Articulation
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Check if the pick and place setup is enabled
        if self.cfg.pp_setup:
            # self.cubes = RigidObject(cfg=self.cfg.cube_rigid_obj_cfg)

            # add container table
            spawn_from_usd(
                prim_path="/World/envs/env_.*/container",
                cfg=self.cfg.container_cfg,
                translation=(1.0, 0.0, 0.0),  # usual:(0.8, 0.0, 0.0),
            )
            spawn_cuboid(
                prim_path="/World/envs/env_.*/Cube",
                cfg=self.cfg.cuboid_cfg,
                translation=self.cfg.cube_init_state,
            )

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion to scene
        self.scene.articulations["ur5"] = self.robot

        # self.scene.rigid_objects["cube"] = self.cubes
        # return the scene information
        self.camera_rgb = Camera(cfg=self.cfg.camera_rgb_cfg)
        self.scene.sensors["camera_rgb"] = self.camera_rgb
        self.camera_depth = Camera(cfg=self.cfg.camera_depth_cfg)
        self.scene.sensors["camera_depth"] = self.camera_depth

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _gripper_action_to_joint_targets(
        self, gripper_action: torch.Tensor
    ) -> torch.Tensor:
        """_summary_
        Convert gripper action [-1,1] to the corresponding angles of all gripper joints.

        Args:
            gripper_action (torch.Tensor): single value between -1 and 1 (e.g. tensor([0.], device='cuda:0'))

        Returns:
            torch.Tensor: _description_
        """
        # Convert each gripper action to the corresponding 6 gripper joint positions (min max 36 = joint limit)
        # gripper_action = gripper_action * 0 if gripper_action < 0 else 1
        self.gripper_action_bin = torch.where(
            gripper_action > 0,
            torch.tensor(1.0, device="cuda:0"),
            torch.tensor(-1.0, device="cuda:0"),
        )

        gripper_joint_targets = torch.stack(
            [
                35 * self.gripper_action_bin,  # "left_outer_knuckle_joint"
                -35 * self.gripper_action_bin,  # "left_inner_finger_joint"
                -35 * self.gripper_action_bin,  # "left_inner_knuckle_joint"
                -35 * self.gripper_action_bin,  # "right_inner_knuckle_joint"
                35 * self.gripper_action_bin,  # "right_outer_knuckle_joint"
                35 * self.gripper_action_bin,  # "right_inner_finger_joint"
            ],
            dim=1,
        )  # Shape: (num_envs, 6)
        return gripper_joint_targets

    def _check_drift(self):
        """
        Check if the joint positions in the script ground truth deviate too much from actual joints in the simulation.
        If the deviation is too high, update the GT.
        """
        # Get current joint positions from the scripts GT
        current_main_joint_positions = self.jointpos_script_GT[
            :, : len(self._arm_dof_idx)
        ]
        # Get current joint positions from the simulation
        current_main_joint_positions_sim = self.live_joint_pos[
            :, : len(self._arm_dof_idx)
        ]
        # Check if the sim joints deviate too much from the script ground truth joints
        if not torch.allclose(
            current_main_joint_positions, current_main_joint_positions_sim, atol=1e-2
        ):
            if self.cfg.verbose_logging:
                print(
                    f"[INFO]: Joint position GT in script deviates too much from the simulation\nUpdate GT"
                )
            self.jointpos_script_GT = current_main_joint_positions_sim.clone()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Get actions
        # Separate the main joint actions (first 6) and the gripper action (last one)
        main_joint_deltas = actions[:, :6]
        gripper_action = actions[:, 6]  # Shape: (num_envs)

        # Check if the sim joints deviate too much from the script ground truth joints
        self._check_drift()

        # Get current joint positions from the scripts GT
        current_main_joint_positions = self.jointpos_script_GT[
            :, : len(self._arm_dof_idx)
        ]  # self.live_joint_pos[:, : len(self._arm_dof_idx)]

        # Apply actions
        # Scale the main joint actions
        main_joint_deltas = self.cfg.action_scale * main_joint_deltas
        # Convert normalized joint action to radian deltas
        main_joint_deltas = self.cfg.stepsize * main_joint_deltas

        # Add radian deltas to current joint positions
        main_joint_targets = torch.add(current_main_joint_positions, main_joint_deltas)

        gripper_joint_targets = self._gripper_action_to_joint_targets(gripper_action)

        # Concatenate the main joint actions with the gripper joint positions and set it as new GT
        self.jointpos_script_GT = torch.cat(
            (main_joint_targets, gripper_joint_targets), dim=1
        )

        # Assign calculated joint target to self.actions
        self.actions = self.jointpos_script_GT

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.actions, joint_ids=self.haw_ur5_dof_idx
        )

    def _get_observations(self) -> dict:
        rgb = self.camera_rgb.data.output["rgb"]
        depth = self.camera_depth.data.output["distance_to_camera"]

        # Extract the cubes position from the rgb and depth images an convert it to a tensor

        cube_pos, cube_pos_w, data_age, dist_cube_cam = (
            self.cubedetector.get_cube_positions(
                rgb_images=rgb.cpu().numpy(),
                depth_images=depth.squeeze(-1).cpu().numpy(),
                rgb_camera_poses=self.camera_rgb.data.pos_w.cpu().numpy(),
                rgb_camera_quats=self.camera_rgb.data.quat_w_world.cpu().numpy(),
                camera_intrinsics_matrices_k=self.camera_rgb.data.intrinsic_matrices.cpu().numpy(),
                base_link_poses=self.scene.articulations["ur5"]
                .data.root_pos_w.cpu()
                .numpy(),
                CAMERA_RGB_2_D_OFFSET=-25,
            )
        )
        cube_pos = torch.from_numpy(cube_pos).to(self.device)
        cube_pos_w = torch.from_numpy(cube_pos_w).to(self.device)
        self.data_age = torch.tensor(data_age, device=self.device)
        self.dist_cube_cam = torch.tensor(dist_cube_cam, device=self.device)

        # Compute distance cube position to goal position
        self.cube_distance_to_goal = torch.norm(
            cube_pos_w - self.cube_goal_pos, dim=-1, keepdim=False
        )

        # print(f"Mean distance camera to cube: {self.dist_cube_cam}")

        # Obs of shape [n_envs, 1, 27])
        obs = torch.cat(
            (
                self.live_joint_pos[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.live_joint_vel[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.live_joint_torque[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.live_joint_pos[:, : len(self._gripper_dof_idx)].unsqueeze(dim=1),
                self.live_joint_vel[:, : len(self._gripper_dof_idx)].unsqueeze(dim=1),
                cube_pos.unsqueeze(dim=1),
                self.cube_distance_to_goal.unsqueeze(dim=1).unsqueeze(dim=1),
                self.data_age.unsqueeze(dim=1).unsqueeze(dim=1),
                self.dist_cube_cam.unsqueeze(dim=1).unsqueeze(dim=1),
            ),
            dim=-1,
        )

        obs = obs.float()
        obs = obs.squeeze(dim=1)

        observations = {"policy": obs, "goal_reached": self.goal_reached}
        return observations

    def get_sim_joint_positions(self) -> torch.Tensor | None:
        """_summary_
        Get the joint positions from the simulation.

        return: torch.Tensor: Joint positions of the robot in the simulation
                or None if the joint positions are not available.
        """
        arm_joint_pos = self.live_joint_pos[:, : len(self._arm_dof_idx)]
        gripper_goalpos = self.gripper_action_bin
        if gripper_goalpos != None and arm_joint_pos != None:
            gripper_goalpos = gripper_goalpos.unsqueeze(1)
            T_all_joint_pos = torch.cat((arm_joint_pos, gripper_goalpos), dim=1)
            return T_all_joint_pos
        return None

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Get torque
        # print(f"Live Torque: {self.live_joint_torque[:, : len(self._arm_dof_idx)]}")
        # Check if any torque in each environment exceeds the threshold
        torque_limit_exceeded = torch.any(
            torch.abs(self.live_joint_torque[:, : len(self._arm_dof_idx)])
            > self.torque_limit,
            dim=1,
        )

        # Provide a grace period for high torques when resetting
        pardon = torch.where(
            self.episode_length_buf < 5, torch.tensor(1), torch.tensor(0)
        )

        self.torque_limit_exeeded = torch.logical_and(
            torque_limit_exceeded == 1, pardon == 0
        )

        # position reached
        self.goal_reached = torch.where(
            self.cube_distance_to_goal.squeeze() < 0.05,
            torch.tensor(1, device=self.device),
            torch.tensor(0, device=self.device),
        )

        # Resolves the issue of the goal_reached tensor becoming a scalar when the number of environments is 1
        if self.cfg.scene.num_envs == 1:
            self.goal_reached = self.goal_reached.unsqueeze(0)
        reset_terminated = self.goal_reached | self.torque_limit_exeeded
        return reset_terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore

        # General resetting tasks (timers etc.)
        super()._reset_idx(env_ids)  # type: ignore

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        # Domain randomization (TODO make sure states are safe)
        joint_pos += torch.rand_like(joint_pos) * 0.1

        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])

        self.live_joint_pos[env_ids] = joint_pos
        self.live_joint_vel[env_ids] = joint_vel
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.data_age[env_ids] = 0
        self.goal_reached[env_ids] = 0

        # # Reset Cube Pos
        # if self.cfg.pp_setup:
        #     for id in env_ids:
        #         cube = self.scene.stage.GetPrimAtPath(f"/World/envs/env_{id}/Cube")
        #         pass
        # self.cubes
        # cube_rootstate = self.cube_object.data.default_root_state.clone()
        # self.cube_object.write_root_pose_to_sim(cube_rootstate[:, :7])
        # self.cube_object.write_root_velocity_to_sim(cube_rootstate[:, 7:])

    def set_joint_angles_absolute(self, joint_angles: list[float64]) -> bool:
        try:
            # Set arm joint angles from list
            T_arm_angles = torch.tensor(joint_angles[:6], device=self.device)
            T_arm_angles = T_arm_angles.unsqueeze(1)
            # Set gripper joint angles from list
            T_arm_angles = torch.transpose(T_arm_angles, 0, 1)

            default_velocities = self.robot.data.default_joint_vel

            # self.joint_pos = T_angles
            # self.joint_vel = default_velocities
            print(f"Setting joint angles to: {T_arm_angles}")
            print(f"Shape of joint angles: {T_arm_angles.shape}")
            self.robot.write_joint_state_to_sim(T_arm_angles, default_velocities[:, :6], self._arm_dof_idx, None)  # type: ignore
            return True
        except Exception as e:
            print(f"Error setting joint angles: {e}")
            return False

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.alive_reward_scaling,
            self.reset_terminated,
            self.live_joint_pos[:, : len(self._arm_dof_idx)],
            self.live_joint_vel[:, : len(self._arm_dof_idx)],
            self.live_joint_torque[:, : len(self._arm_dof_idx)],
            self.live_joint_pos[:, : len(self._gripper_dof_idx)],
            self.live_joint_vel[:, : len(self._gripper_dof_idx)],
            self.cfg.vel_penalty_scaling,
            self.cfg.torque_penalty_scaling,
            self.torque_limit_exeeded,
            self.cfg.torque_limit_exeeded_penalty_scaling,
            self.data_age,
            self.cfg.cube_out_of_sight_penalty_scaling,
            self.cube_distance_to_goal,
            self.cfg.distance_cube_to_goal_penalty_scaling,
            self.goal_reached,
            self.cfg.goal_reached_scaling,
            self.dist_cube_cam,
            self.cfg.dist_cube_cam_penalty_scaling,
        )
        # if self.torque_limit_exeeded.any():
        #     print(f"Torque limit exeeded: {self.torque_limit_exeeded}")
        # print all shapes of the inputs with their names
        # print(
        #     f"Shapes: \n"
        #     f"reset_terminated: {self.reset_terminated.shape}\n"
        #     f"arm_joint_pos: {self.live_joint_pos[:, : len(self._arm_dof_idx)].shape}\n"
        #     f"arm_joint_vel: {self.live_joint_vel[:, : len(self._arm_dof_idx)].shape}\n"
        #     f"arm_joint_torque: {self.live_joint_torque[:, : len(self._arm_dof_idx)].shape}\n"
        #     f"gripper_joint_pos: {self.live_joint_pos[:, : len(self._gripper_dof_idx)].shape}\n"
        #     f"gripper_joint_vel: {self.live_joint_vel[:, : len(self._gripper_dof_idx)].shape}\n"
        #     f"vel_penalty_scaling: {self.cfg.vel_penalty_scaling}\n"
        #     f"torque_penalty_scaling: {self.cfg.torque_penalty_scaling}\n"
        #     f"data_age: {self.data_age.shape}\n"
        #     f"cube_out_of_sight_penalty_scaling: {self.cfg.cube_out_of_sight_penalty_scaling}\n"
        #     f"distance_cube_to_goal_pos: {self.cube_distance_to_goal.shape}\n"
        #     f"distance_cube_to_goal_penalty_scaling: {self.cfg.distance_cube_to_goal_penalty_scaling}\n"
        #     f"goal_reached: {self.goal_reached.shape}\n"
        #     f"goal_reached_scaling: {self.cfg.goal_reached_scaling}\n"
        # )
        # print(f"Total Reward shape in function: {total_reward.shape}")
        return total_reward


@torch.jit.script
def compute_rewards(
    aliverewardscale: float,
    reset_terminated: torch.Tensor,
    arm_joint_pos: torch.Tensor,
    arm_joint_vel: torch.Tensor,
    arm_joint_torque: torch.Tensor,
    gripper_joint_pos: torch.Tensor,
    gripper_joint_vel: torch.Tensor,
    vel_penalty_scaling: float,
    torque_penalty_scaling: float,
    torque_limit_exeeded: torch.Tensor,
    torque_limit_exeeded_penalty_scaling: float,
    data_age: torch.Tensor,
    cube_out_of_sight_penalty_scaling: float,
    distance_cube_to_goal_pos: torch.Tensor,
    distance_cube_to_goal_penalty_scaling: float,
    goal_reached: torch.Tensor,
    goal_reached_scaling: float,
    dist_cube_cam: torch.Tensor,
    dist_cube_cam_penalty_scaling: float,
) -> torch.Tensor:

    penalty_alive = aliverewardscale * (1.0 - reset_terminated.float())
    penalty_vel = vel_penalty_scaling * torch.sum(torch.abs(arm_joint_vel), dim=-1)
    penalty_cube_out_of_sight = cube_out_of_sight_penalty_scaling * data_age
    penalty_distance_cube_to_goal = (
        distance_cube_to_goal_penalty_scaling * distance_cube_to_goal_pos
    )
    torque_penalty = torque_penalty_scaling * torch.sum(
        torch.abs(arm_joint_torque), dim=-1
    )

    torque_limit_exeeded_penalty = (
        torque_limit_exeeded_penalty_scaling * torque_limit_exeeded
    )

    goal_reached_reward = goal_reached_scaling * goal_reached
    dist_cube_cam_penalty = torch.where(
        dist_cube_cam > 0.3,
        dist_cube_cam_penalty_scaling * dist_cube_cam,
        torch.tensor(0.0, dtype=dist_cube_cam.dtype, device=dist_cube_cam.device),
    )

    reward = (
        penalty_alive
        + penalty_vel
        + penalty_cube_out_of_sight
        + penalty_distance_cube_to_goal
        + torque_limit_exeeded_penalty
        + goal_reached_reward
        + torque_penalty
        + dist_cube_cam_penalty
    )
    return reward
