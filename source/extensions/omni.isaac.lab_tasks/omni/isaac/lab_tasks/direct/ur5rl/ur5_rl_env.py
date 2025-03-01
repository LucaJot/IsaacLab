from __future__ import annotations

import json
import math
import os
import warnings
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
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_dof_idx, _ = self.robot.find_joints(self.cfg.arm_dof_name)
        self._gripper_dof_idx, _ = self.robot.find_joints(self.cfg.gripper_dof_name)
        self.haw_ur5_dof_idx, _ = self.robot.find_joints(self.cfg.haw_ur5_dof_name)
        self.action_scale = self.cfg.action_scale
        joint_init_state = torch.cat(
            (
                torch.tensor(self.cfg.arm_joints_init_state, device="cuda:0"),
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device="cuda:0"),
            ),
            dim=0,
        )

        self.robot.data.default_joint_pos = joint_init_state.repeat(
            self.scene.num_envs, 1
        )

        self.randomize = True
        self.joint_randomize_level = 0.5
        self.cube_randomize_level = 0.2
        self.container_randomize_level = 0.2

        # Statistics for rewards
        self.total_penalty_alive: torch.Tensor = torch.zeros(
            self.scene.num_envs, device=self.device
        )
        self.total_penalty_vel: torch.Tensor = torch.zeros(
            self.scene.num_envs, device=self.device
        )
        self.total_penalty_cube_out_of_sight: torch.Tensor = torch.zeros(
            self.scene.num_envs, device=self.device
        )
        self.total_penalty_distance_cube_to_goal: torch.Tensor = torch.zeros(
            self.scene.num_envs, device=self.device
        )
        self.total_torque_limit_exeeded_penalty: torch.Tensor = torch.zeros(
            self.scene.num_envs, device=self.device
        )
        self.total_goal_reached_reward: torch.Tensor = torch.zeros(
            self.scene.num_envs, device=self.device
        )
        self.total_torque_penalty: torch.Tensor = torch.zeros(
            self.scene.num_envs, device=self.device
        )
        self.cube_approach_reward: torch.Tensor = torch.zeros(
            self.scene.num_envs, device=self.device
        )

        self.statistics = [
            self.total_penalty_alive,
            self.total_penalty_vel,
            self.total_penalty_cube_out_of_sight,
            self.total_penalty_distance_cube_to_goal,
            self.total_torque_limit_exeeded_penalty,
            self.total_goal_reached_reward,
            self.total_torque_penalty,
            self.cube_approach_reward,
        ]

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

        self.gripper_action_bin: torch.Tensor = torch.zeros(
            self.scene.num_envs, device=self.device, dtype=torch.float32
        )
        self.gripper_locked = torch.zeros(
            self.scene.num_envs, device=self.device, dtype=torch.bool
        )
        self.gripper_steps = torch.zeros(
            self.scene.num_envs, device=self.device, dtype=torch.float32
        )

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
        self.dist_cube_cam_minimal = torch.zeros(
            self.scene.num_envs, device=self.device
        )
        self.mean_dist_cam_cube = 0

        # Yolo model for cube detection
        # self.yolov11 = YOLO("yolo11s.pt")

        #! LOGGING
        self.LOG_ENV_DETAILS = True
        self.log_dir = "/home/luca/isaaclab_ws/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5rl/logdir"
        self.episode_data = {
            "pos_sensor_x": [],
            "pos_sensor_y": [],
            "dist_cube_cam": [],
            "reward_approach": [],
            "penalty_torque": [],
            "mean_torque": [],
        }

        self.DEBUG_GRIPPER = True

    def set_eval_mode(self):
        self.randomize = False

    def set_arm_init_pose(self, joint_angles: list[float64]) -> bool:

        if len(joint_angles) != 6:
            warnings.warn(
                f"[WARNING] Expected 6 joint angles, got {len(joint_angles)}",
                UserWarning,
            )
            return False
        else:
            joint_init_state = torch.cat(
                (
                    torch.tensor(joint_angles, device="cuda:0"),
                    torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device="cuda:0"),
                ),
                dim=0,
            )

            self.robot.data.default_joint_pos = joint_init_state.repeat(
                self.scene.num_envs, 1
            )
            return True

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
            container_pos = (1.0, 0.0, 0.0)
            cube_pos = self.cfg.cube_init_state

            # add container table
            spawn_from_usd(
                prim_path="/World/envs/env_.*/container",
                cfg=self.cfg.container_cfg,
                translation=container_pos,  # usual:(0.8, 0.0, 0.0),
            )
            spawn_cuboid(
                prim_path="/World/envs/env_.*/Cube",
                cfg=self.cfg.cuboid_cfg,
                translation=cube_pos,
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
        """Converts gripper action [-1,1] into actual joint positions while respecting locking logic."""

        # Step 1: Update `gripper_action_bin` only for unlocked grippers
        self.gripper_action_bin = torch.where(
            ~self.gripper_locked,  # Only update where gripper is unlocked
            torch.where(
                gripper_action > 0,
                torch.tensor(1.0, device="cuda:0"),
                torch.tensor(-1.0, device="cuda:0"),
            ),
            self.gripper_action_bin,  # Keep previous value for locked grippers
        )

        # Step 2: Lock the gripper if action bin has been updated
        self.gripper_locked = torch.where(
            ~self.gripper_locked,  # If gripper is unlocked
            torch.tensor(True, device="cuda:0"),  # Lock it
            self.gripper_locked,  # Keep it locked if already locked
        )

        # Step 3: Gradually update `gripper_steps` towards `gripper_action_bin`
        step_size = 0.01
        self.gripper_steps = torch.where(
            self.gripper_locked,  # If gripper is locked
            self.gripper_steps
            + step_size * self.gripper_action_bin,  # Increment/decrement towards target
            self.gripper_steps,  # Keep unchanged if unlocked
        )

        # Ensure no faulty values are present
        self.gripper_steps = torch.clamp(self.gripper_steps, -1.0, 1.0)

        # Step 4: Unlock gripper once `gripper_steps` reaches `gripper_action_bin`
        reached_target = torch.isclose(
            self.gripper_steps, self.gripper_action_bin, atol=0.005
        )
        self.gripper_locked = torch.where(
            reached_target,  # Unlock when target is reached
            torch.tensor(False, device="cuda:0"),
            self.gripper_locked,
        )

        # Step 5: Convert `gripper_steps` into joint targets
        gripper_joint_targets = torch.stack(
            [
                35 * self.gripper_steps,  # "left_outer_knuckle_joint"
                -35 * self.gripper_steps,  # "left_inner_finger_joint"
                -35 * self.gripper_steps,  # "left_inner_knuckle_joint"
                -35 * self.gripper_steps,  # "right_inner_knuckle_joint"
                35 * self.gripper_steps,  # "right_outer_knuckle_joint"
                35 * self.gripper_steps,  # "right_inner_finger_joint"
            ],
            dim=1,
        )  # Shape: (num_envs, 6)

        # print(
        #     f"Env0 Debug\nGripperAction: {gripper_action[0]}\nGripperSteps: {self.gripper_steps[0]}\nGripperLocked: {self.gripper_locked[0]}\nGripperActionBin: {self.gripper_action_bin[0]}\nGripperJointTargets: {gripper_joint_targets[0]}\nReached Target: {reached_target[0]} \n"
        # )
        # print(self.gripper_steps.device, self.gripper_action_bin.device)
        # print(self.gripper_steps.dtype, self.gripper_action_bin.dtype)
        # print("Difference:", (self.gripper_steps[0] - self.gripper_action_bin[0]))

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
            # if self.cfg.verbose_logging:
            #     print(
            #         f"[INFO]: Joint position GT in script deviates too much from the simulation\nUpdate GT"
            #     )
            self.jointpos_script_GT = current_main_joint_positions_sim.clone()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Normalize the actions -1 and 1
        actions = torch.tanh(actions)
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

        self.dist_cube_cam_minimal = torch.where(
            self.dist_cube_cam > 0,
            torch.minimum(self.dist_cube_cam, self.dist_cube_cam_minimal),
            self.dist_cube_cam_minimal,
        )
        rgb = self.camera_rgb.data.output["rgb"]
        depth = self.camera_depth.data.output["distance_to_camera"]

        # Extract the cubes position from the rgb and depth images an convert it to a tensor

        cube_pos, cube_pos_w, data_age, dist_cube_cam, pos_sensor = (
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
        pos_sensor = torch.from_numpy(pos_sensor).to(self.device)

        # If env has been reset, set dist_prev to -1
        self.dist_cube_cam_minimal = torch.where(
            self.episode_length_buf == 0, 99.0, self.dist_cube_cam_minimal
        )

        # Compute distance cube position to goal position
        self.cube_distance_to_goal = torch.linalg.vector_norm(
            cube_pos_w - self.cube_goal_pos, dim=-1, keepdim=False
        )

        # print(f"Mean distance camera to cube: {self.dist_cube_cam}")
        # Obs of shape [n_envs, 1, 27])
        obs = torch.cat(
            (
                self.live_joint_pos[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.live_joint_vel[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.live_joint_torque[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.gripper_action_bin.unsqueeze(dim=1).unsqueeze(dim=1),
                cube_pos.unsqueeze(dim=1),
                self.cube_distance_to_goal.unsqueeze(dim=1).unsqueeze(dim=1),
                self.data_age.unsqueeze(dim=1).unsqueeze(dim=1),
                self.dist_cube_cam.unsqueeze(dim=1).unsqueeze(dim=1),
                pos_sensor.unsqueeze(dim=1),
            ),
            dim=-1,
        )

        obs = obs.float()
        obs = obs.squeeze(dim=1)

        # print(
        #     f"Env0 obs Debug\nDistCubeCam:{self.dist_cube_cam[0]}\nDistCubeCamMinimal: {self.dist_cube_cam_minimal[0]}\nCubePos:{cube_pos[0]}\nCubePosW:{cube_pos_w[0]}\nCubeDistToGoal:{self.cube_distance_to_goal[0]}\nDataAge:{self.data_age[0]}\nPosSensor:{pos_sensor[0]}\n\n"
        # )
        #! LOGGING
        # ✅ Save only for the first environment (Env0)
        if self.LOG_ENV_DETAILS:
            self.episode_data["pos_sensor_x"].append(float(pos_sensor[0][0].cpu()))
            self.episode_data["pos_sensor_y"].append(float(pos_sensor[0][1].cpu()))
            self.episode_data["dist_cube_cam"].append(
                float(self.dist_cube_cam[0].cpu())
            )
            mean_torque = torch.mean(
                torch.abs(self.live_joint_torque[:, : len(self._arm_dof_idx)])
            )
            self.episode_data["mean_torque"].append(float(mean_torque.cpu()))

        if torch.isnan(obs).any():
            warnings.warn("[WARNING] NaN detected in observations!", UserWarning)
            print(f"[DEBUG] NaN found in observation: {obs}\nReplacing with 0.0")
            obs = torch.where(
                torch.isnan(obs),
                torch.tensor(0.0, dtype=obs.dtype, device=obs.device),
                obs,
            )

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

    def _randomize_object_positions(self, env_id):
        """Randomizes the positions of the container and cube for a given environment."""

        # Randomize container position
        container_pos = (
            1.0
            + np.random.uniform(
                -self.container_randomize_level, self.container_randomize_level
            ),
            0.0
            + np.random.uniform(
                -self.container_randomize_level, self.container_randomize_level
            ),
            0.0,  # Z stays fixed as it sits on the ground
        )

        # Randomize cube position
        cube_pos = (
            self.cfg.cube_init_state[0]
            + np.random.uniform(-self.cube_randomize_level, self.cube_randomize_level),
            self.cfg.cube_init_state[1]
            + np.random.uniform(-self.cube_randomize_level, self.cube_randomize_level),
            self.cfg.cube_init_state[2],  # Keep cube at correct height
        )

        # Apply the new positions in Isaac Sim using USD API
        container_prim = self.scene.stage.GetPrimAtPath(
            f"/World/envs/env_{env_id}/container"
        )
        if container_prim.IsValid():
            container_xform = Usd.Prim(container_prim).GetAttribute("xformOp:translate")
            if container_xform:
                container_xform.Set(Gf.Vec3d(*container_pos))

        cube_prim = self.scene.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Cube")
        if cube_prim.IsValid():
            cube_xform = Usd.Prim(cube_prim).GetAttribute("xformOp:translate")
            if cube_xform:
                cube_xform.Set(Gf.Vec3d(*cube_pos))

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore

        # General resetting tasks (timers etc.)
        super()._reset_idx(env_ids)  # type: ignore

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        # Domain randomization (TODO make sure states are safe)
        if self.randomize:
            randomness = (
                torch.rand_like(joint_pos) * 2 - 1
            ) * self.joint_randomize_level
            joint_pos += randomness

        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])

        self.live_joint_pos[env_ids] = joint_pos
        self.live_joint_vel[env_ids] = joint_vel
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.data_age[env_ids] = 0.0
        self.cubedetector.reset_data_age(env_ids)  # type: ignore
        self.goal_reached[env_ids] = 0.0
        self.dist_cube_cam_minimal[env_ids] = 99.0

        # if self.randomize:
        #     for env_id in env_ids:
        #         self._randomize_object_positions(env_id)

        # Reset statistics
        if (
            not any(stat is None for stat in self.statistics)
            and self.cfg.verbose_logging
        ):
            if 0 in env_ids:  # type: ignore
                env_id = 0
                print("-" * 20)
                print(f"Resetting environment {env_id}")
                print(f"Statistics")
                print(f"Total penalty alive: {self.total_penalty_alive[env_id]}")
                print(f"Total penalty vel: {self.total_penalty_vel[env_id]}")
                print(
                    f"Total penalty cube out of sight: {self.total_penalty_cube_out_of_sight[env_id]}"
                )
                print(
                    f"Total penalty distance cube to goal: {self.total_penalty_distance_cube_to_goal[env_id]}"
                )
                print(
                    f"Total torque limit exeeded penalty: {self.total_torque_limit_exeeded_penalty[env_id]}"
                )
                print(
                    f"Total goal reached reward: {self.total_goal_reached_reward[env_id]}"
                )
                print(f"Total torque penalty: {self.total_torque_penalty[env_id]}")
                print(f"Cube approach reward: {self.cube_approach_reward[env_id]}")
                print("-" * 20)

                self.total_penalty_alive[env_ids] = 0  # type: ignore
                self.total_penalty_vel[env_ids] = 0  # type: ignore
                self.total_penalty_cube_out_of_sight[env_ids] = 0  # type: ignore
                self.total_penalty_distance_cube_to_goal[env_ids] = 0  # type: ignore
                self.total_torque_limit_exeeded_penalty[env_ids] = 0  # type: ignore
                self.total_goal_reached_reward[env_ids] = 0  # type: ignore
                self.total_torque_penalty[env_ids] = 0  # type: ignore
                self.cube_approach_reward[env_ids] = 0  # type: ignore

        # # Reset Cube Pos
        # if self.cfg.pp_setup:
        #     for id in env_ids:
        #         cube = self.scene.stage.GetPrimAtPath(f"/World/envs/env_{id}/Cube")
        #         pass
        # self.cubes
        # cube_rootstate = self.cube_object.data.default_root_state.clone()
        # self.cube_object.write_root_pose_to_sim(cube_rootstate[:, :7])
        # self.cube_object.write_root_velocity_to_sim(cube_rootstate[:, 7:])

        #! LOGGING
        if self.LOG_ENV_DETAILS:
            if 0 in env_ids:
                episode_id = len(
                    os.listdir(self.log_dir)
                )  # Unique filename for each episode
                with open(f"{self.log_dir}/episode_{episode_id}.json", "w") as f:
                    json.dump(self.episode_data, f)

                # ✅ Reset the storage for next episode
                self.episode_data = {
                    "pos_sensor_x": [],
                    "pos_sensor_y": [],
                    "dist_cube_cam": [],
                    "reward_approach": [],
                    "penalty_torque": [],
                    "mean_torque": [],
                }

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
        rewards = compute_rewards(
            self.cfg.alive_reward_scaling,
            self.reset_terminated,
            self.live_joint_pos[:, : len(self._arm_dof_idx)],
            self.live_joint_vel[:, : len(self._arm_dof_idx)],
            self.live_joint_torque[:, : len(self._arm_dof_idx)],
            self.gripper_action_bin,
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
            self.dist_cube_cam_minimal,
            self.cfg.approach_reward,
        )

        # self.total_penalty_alive += rewards[0]
        # self.total_penalty_vel += rewards[1]
        # self.total_penalty_cube_out_of_sight += rewards[2]
        # self.total_penalty_distance_cube_to_goal += rewards[3]
        # self.total_torque_limit_exeeded_penalty += rewards[4]
        # self.total_goal_reached_reward += rewards[5]
        self.total_torque_penalty += rewards[6]
        self.cube_approach_reward += rewards[7]

        if self.LOG_ENV_DETAILS:
            self.episode_data["reward_approach"].append(float(rewards[7][0].cpu()))
            # self.episode_data["penalty_torque"].append(float(rewards[6][0].cpu()))

        # total_reward = torch.sum(rewards, dim=0)

        total_reward = torch.sum(torch.stack([rewards[7], rewards[6]]), dim=0)

        if torch.isnan(total_reward).any():
            warnings.warn("[WARNING] NaN detected in rewards!", UserWarning)
            print(f"[DEBUG] NaN found in rewards: {total_reward}\nReplacing with 0.0")
            total_reward = torch.where(
                torch.isnan(total_reward),
                torch.tensor(0.0, dtype=total_reward.dtype, device=total_reward.device),
                total_reward,
            )

        return total_reward


@torch.jit.script
def compute_rewards(
    aliverewardscale: float,
    reset_terminated: torch.Tensor,
    arm_joint_pos: torch.Tensor,
    arm_joint_vel: torch.Tensor,
    arm_joint_torque: torch.Tensor,
    gripper_action_bin: torch.Tensor,
    vel_penalty_scaling: float,
    torque_penalty_scaling: float,
    torque_limit_exceeded: torch.Tensor,
    torque_limit_exceeded_penalty_scaling: float,
    data_age: torch.Tensor,
    cube_out_of_sight_penalty_scaling: float,
    distance_cube_to_goal_pos: torch.Tensor,
    distance_cube_to_goal_penalty_scaling: float,
    goal_reached: torch.Tensor,
    goal_reached_scaling: float,
    dist_cube_cam: torch.Tensor,
    dist_cube_cam_minimal: torch.Tensor,
    approach_reward_scaling: float,
) -> torch.Tensor:

    penalty_alive = aliverewardscale * (1.0 - reset_terminated.float())
    penalty_vel = vel_penalty_scaling * torch.sum(torch.abs(arm_joint_vel), dim=-1)
    penalty_cube_out_of_sight = cube_out_of_sight_penalty_scaling * torch.where(
        data_age > 0,
        torch.tensor(1.0, dtype=data_age.dtype, device=data_age.device),
        torch.tensor(0.0, dtype=data_age.dtype, device=data_age.device),
    )
    penalty_distance_cube_to_goal = (
        distance_cube_to_goal_penalty_scaling * distance_cube_to_goal_pos
    )

    penalty_free_limits = torch.tensor(
        [105.0, 105.0, 105.0, 20.0, 20.0, 20.0], device="cuda:0"
    )
    remaining_torque = torch.tensor([45.0, 45.0, 45.0, 8.0, 8.0, 8.0], device="cuda:0")
    torques_abs = torch.abs(arm_joint_torque)
    # calculate how much the torque exceeds the limit
    torque_limit_exceedamount = torch.relu(torques_abs - penalty_free_limits)
    # Get the percentage of the torque limit to breaking limit
    exceeded_percentage = torch.clip(
        torch.div(torque_limit_exceedamount, remaining_torque), min=0.0, max=1.0
    )
    torque_penalty = torque_penalty_scaling * exceeded_percentage

    total_torque_penalty = torch.sum(torque_penalty, dim=-1)

    torque_limit_exeeded_penalty = (
        torque_limit_exceeded_penalty_scaling * torque_limit_exceeded
    )

    goal_reached_reward = goal_reached_scaling * goal_reached

    # Option 1 for approach reward: Minimal distance to cube is stored and improvment rewarded
    # improvement = dist_cube_cam_minimal - dist_cube_cam
    # improvement_reward = torch.clamp(improvement, min=0.0) * approach_reward_scaling
    # approach_reward = torch.where(
    #     dist_cube_cam > 0,  # only give reward if the cube is in sight
    #     improvement_reward,  # Reward inversely proportional to distance
    #     torch.tensor(
    #         0.0, dtype=dist_cube_cam.dtype, device=dist_cube_cam.device
    #     ),  # No reward if cube is not in sight
    # )

    # Option 2 for approach reward: Exponential decay of reward with distance
    k = 5
    approach_reward = torch.where(
        dist_cube_cam > 0.0,
        approach_reward_scaling * torch.exp(-k * dist_cube_cam),
        torch.tensor(0.0, dtype=dist_cube_cam.dtype, device=dist_cube_cam.device),
    )

    return torch.stack(
        [
            penalty_alive,
            penalty_vel,
            penalty_cube_out_of_sight,
            penalty_distance_cube_to_goal,
            torque_limit_exeeded_penalty,
            goal_reached_reward,
            total_torque_penalty,
            approach_reward,
        ]
    )
