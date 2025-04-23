from __future__ import annotations

import json
import math
import os
import warnings
import torch
from collections.abc import Sequence
import time

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
from omni.usd import get_context
from omni.isaac.lab.sim.spawners import RigidObjectSpawnerCfg
from omni.isaac.lab.sim.spawners.shapes import spawn_cuboid, CuboidCfg
from omni.isaac.lab.assets import (
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import CameraCfg, Camera, ContactSensor
from omni.isaac.sensor import _sensor

# from omni.isaac.dynamic_control import _dynamic_control
from pxr import Usd, Gf, UsdPhysics
from pxr import UsdPhysics, PhysxSchema
from omni.physx.scripts import utils as physx_utils


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
from pxr import UsdPhysics
from omni.isaac.core.utils.prims import get_prim_at_path


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
        randomize: bool = True,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.CL_state = -1

        self.cube_z_old: torch.Tensor = None
        self.cube_z_new: torch.Tensor = None

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

        self.randomize = randomize
        self.joint_randomize_level = 0.4
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
        self.pickup_reward: torch.Tensor = torch.zeros(
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
            self.pickup_reward,
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

        self.init_root_pos = None
        self.init_root_quat = None

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
        self.mean_torque = torch.zeros(1, device=self.device, dtype=torch.float32)

        self.grasp_success = torch.zeros(self.scene.num_envs, device=self.device)
        self.partial_grasp = torch.zeros(self.scene.num_envs, device=self.device)
        self.container_contact = torch.zeros(
            self.scene.num_envs, device=self.device, dtype=torch.bool
        )

        # Yolo model for cube detection
        # self.yolov11 = YOLO("yolo11s.pt")

        #! LOGGING
        self.LOG_ENV_DETAILS = False
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

    def set_gripper_action_bin(self, gripper_action_bin: list) -> bool:
        """Set the gripper action bin.

        Args:
            gripper_action_bin (list): The gripper action bin to set.

        Returns:
            bool: True if the gripper action bin was set successfully, False otherwise.
        """

        if len(gripper_action_bin) != self.scene.num_envs:
            warnings.warn(
                f"[WARNING] Expected {self.scene.num_envs} gripper actions, got {len(gripper_action_bin)}",
                UserWarning,
            )
            return False

        gripper_action = torch.tensor(gripper_action_bin, device="cuda:0")
        self.gripper_action_bin = gripper_action
        return True

    def set_eval_mode(self):
        self.randomize = False

        self.cfg.events.robot_joint_stiffness_and_damping.params = {
            "asset_cfg": SceneEntityCfg(
                name="ur5",
                joint_names=".*",
                joint_ids=slice(None, None, None),
                fixed_tendon_names=None,
                fixed_tendon_ids=slice(None, None, None),
                body_names=None,
                body_ids=slice(None, None, None),
                object_collection_names=None,
                object_collection_ids=slice(None, None, None),
                preserve_order=False,
            ),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }
        self.cfg.events.robot_physics_material.params = {
            "asset_cfg": SceneEntityCfg(
                name="ur5",
                joint_names=None,
                joint_ids=slice(None, None, None),
                fixed_tendon_names=None,
                fixed_tendon_ids=slice(None, None, None),
                body_names=".*",
                body_ids=slice(None, None, None),
                object_collection_names=None,
                object_collection_ids=slice(None, None, None),
                preserve_order=False,
            ),
            "static_friction_range": (0.7, 0.7),
            "dynamic_friction_range": (0.5, 0.5),
            "restitution_range": (0.3, 0.3),
            "num_buckets": 250,
        }

    def set_train_mode(self):
        self.randomize = True

        self.cfg.events.robot_joint_stiffness_and_damping.params = {
            "asset_cfg": SceneEntityCfg(
                name="ur5",
                joint_names=".*",
                joint_ids=slice(None, None, None),
                fixed_tendon_names=None,
                fixed_tendon_ids=slice(None, None, None),
                body_names=None,
                body_ids=slice(None, None, None),
                object_collection_names=None,
                object_collection_ids=slice(None, None, None),
                preserve_order=False,
            ),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        }
        self.cfg.events.robot_physics_material.params = {
            "asset_cfg": SceneEntityCfg(
                name="ur5",
                joint_names=None,
                joint_ids=slice(None, None, None),
                fixed_tendon_names=None,
                fixed_tendon_ids=slice(None, None, None),
                body_names=".*",
                body_ids=slice(None, None, None),
                object_collection_names=None,
                object_collection_ids=slice(None, None, None),
                preserve_order=False,
            ),
            "static_friction_range": (0.4, 0.9),
            "dynamic_friction_range": (0.2, 0.6),
            "restitution_range": (0.0, 0.7),
            "num_buckets": 250,
        }

    def set_CL_state(self, state: int):
        self.CL_state = state

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
            self.container = spawn_from_usd(
                prim_path="/World/envs/env_.*/container",
                cfg=self.cfg.container_cfg,
                translation=container_pos,  # usual:(0.8, 0.0, 0.0),
            )
            # self.cube = spawn_cuboid(
            #     prim_path="/World/envs/env_.*/Cube",
            #     cfg=self.cfg.cuboid_cfg,
            #     translation=cube_pos,
            # )

            self.cube = spawn_from_usd(
                prim_path="/World/envs/env_.*/Cube",
                cfg=self.cfg.cube_usd_cfg,
                translation=cube_pos,
            )

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion to scene
        self.scene.articulations["ur5"] = self.robot

        # self.scene.rigid_objects["cube"] = self.cube
        # return the scene information
        self.camera_rgb = Camera(cfg=self.cfg.camera_rgb_cfg)
        self.scene.sensors["camera_rgb"] = self.camera_rgb
        self.camera_depth = Camera(cfg=self.cfg.camera_depth_cfg)
        self.scene.sensors["camera_depth"] = self.camera_depth

        self.contact_l = ContactSensor(cfg=self.cfg.contact_cfg_l)
        self.scene.sensors["contact_l"] = self.contact_l
        self.contact_r = ContactSensor(cfg=self.cfg.contact_cfg_r)
        self.scene.sensors["contact_r"] = self.contact_r
        self.contact_c = ContactSensor(cfg=self.cfg.contact_cfg_c)
        self.scene.sensors["contact_c"] = self.contact_c
        self.contact_t = ContactSensor(cfg=self.cfg.contact_cfg_t)
        self.scene.sensors["contact_t"] = self.contact_t

        self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()

        # self.cs = ContactSensor(cfg=self.cfg.contact_sensor)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        )  # intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _gripper_action_to_joint_targets(
        self, gripper_action: torch.Tensor
    ) -> torch.Tensor:
        """Converts gripper action [-1,1] into actual joint positions while respecting locking logic."""

        gripper_lim = 1.0  # Gripper limits
        # Step 1: Update `gripper_action_bin` only for unlocked grippers
        self.gripper_action_bin = torch.where(
            ~self.gripper_locked,  # Only update where gripper is unlocked
            torch.where(
                gripper_action > 0,
                torch.tensor(gripper_lim, device="cuda:0"),
                torch.tensor(-gripper_lim, device="cuda:0"),
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
        step_size = 0.05
        self.gripper_steps = torch.where(
            self.gripper_locked,  # If gripper is locked
            self.gripper_steps
            + step_size * self.gripper_action_bin,  # Increment/decrement towards target
            self.gripper_steps,  # Keep unchanged if unlocked
        )

        # Ensure no faulty values are present
        self.gripper_steps = torch.clamp(self.gripper_steps, -gripper_lim, gripper_lim)

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
        angle_lim = 0.61  # was 35
        gripper_joint_targets = torch.stack(
            [
                angle_lim * self.gripper_steps,  # "left_outer_knuckle_joint"
                -angle_lim * self.gripper_steps,  # "left_inner_finger_joint"
                -angle_lim * self.gripper_steps,  # "left_inner_knuckle_joint"
                -angle_lim * self.gripper_steps,  # "right_inner_knuckle_joint"
                angle_lim * self.gripper_steps,  # "right_outer_knuckle_joint"
                angle_lim * self.gripper_steps,  # "right_inner_finger_joint"
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
            current_main_joint_positions, current_main_joint_positions_sim, atol=0.1
        ):
            # if self.cfg.verbose_logging:
            #     print(
            #         f"[INFO]: Joint position GT in script deviates too much from the simulation\nUpdate GT"
            #     )
            self.jointpos_script_GT = current_main_joint_positions_sim.clone()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Normalize the actions -1 and 1
        actions_capped = torch.tanh(actions)
        actions_capped[:, 5] = 0.0
        # Separate the main joint actions (first 6) and the gripper action (last one)
        main_joint_deltas = actions_capped[:, :6]
        gripper_action = actions_capped[:, 6]  # Shape: (num_envs)

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
        self.jointpos_script_GT[:, 5] = 0.0
        self.actions = self.jointpos_script_GT
        # self.actions[:, 5] = 0.0

        # print(
        #     f"target: {self.gripper_action_bin[0]},\nsteps: {self.gripper_steps[0]}, \nlocked: {self.gripper_locked[0]}"
        # )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.actions, joint_ids=self.haw_ur5_dof_idx
        )

    def check_grasp_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Checks if the robot gripper in each environment successfully grasps the cube.

        Returns:
            torch.Tensor: A boolean tensor of shape (num_envs,) indicating grasp success per environment.
        """
        success_flags = []
        partial_grasp_flags = []

        self.contact_l._update_outdated_buffers()
        left_contact = self.contact_l.compute_first_contact(self.cfg.f_update).squeeze(
            -1
        )
        self.contact_r._update_outdated_buffers()
        right_contact = self.contact_r.compute_first_contact(self.cfg.f_update).squeeze(
            -1
        )
        self.contact_c._update_outdated_buffers()
        cube_contact = self.contact_c.compute_first_contact(self.cfg.f_update).squeeze(
            -1
        )
        self.contact_t._update_outdated_buffers()
        container_contact = self.contact_t.compute_first_contact(
            self.cfg.f_update
        ).squeeze(-1)
        self.container_contact = container_contact

        # Grasp success
        grasp_success = (
            left_contact
            & right_contact
            & cube_contact
            & ~container_contact
            & (self.dist_cube_cam > 0.16)
            & (self.data_age < 3.0)
        )

        partial_grasp_success = (
            left_contact
            & cube_contact
            & ~right_contact
            & ~container_contact
            & (self.dist_cube_cam > 0.16)
            & (self.data_age < 3.0)
        ) | (
            right_contact
            & cube_contact
            & ~left_contact
            & ~container_contact
            & (self.dist_cube_cam > 0.16)
            & (self.data_age < 3.0)
        )

        if torch.any(grasp_success):
            grasp_idx = torch.where(grasp_success)[0]
            print(f"Grasp success in envs: {grasp_idx}")

        if torch.any(partial_grasp_success):
            grasp_idx = torch.where(partial_grasp_success)[0]
            print(f"Partial grasp in envs: {grasp_idx}")

        grasp_success = grasp_success.squeeze(-1)
        partial_grasp_success = partial_grasp_success.squeeze(-1)

        return (grasp_success, partial_grasp_success)

    def _get_observations(self) -> dict:

        self.grasp_success, self.partial_grasp = self.check_grasp_success()

        if self.grasp_success.any():
            pass
        # print(f"Grasp success: {self.grasp_success}")

        self.dist_cube_cam_minimal = torch.where(
            self.dist_cube_cam > 0,
            torch.minimum(self.dist_cube_cam, self.dist_cube_cam_minimal),
            self.dist_cube_cam_minimal,
        )
        rgb = self.camera_rgb.data.output["rgb"]
        depth = self.camera_depth.data.output["distance_to_camera"]

        self.cube_z_old = (
            self.cube_z_new.clone() if self.cube_z_old is not None else None  # type: ignore
        )
        #! Measure potential bottleneck
        # start = time.time()

        cam_poses = self.camera_rgb.data.pos_w - self.scene.env_origins

        # Extract the cubes position from the rgb and depth images an convert it to a tensor
        cube_pos, cube_pos_w, data_age, dist_cube_cam, pos_sensor = (
            self.cubedetector.get_cube_positions(
                rgb_images=rgb.cpu().numpy(),
                depth_images=depth.squeeze(-1).cpu().numpy(),
                rgb_camera_poses=cam_poses.cpu().numpy(),
                rgb_camera_quats=self.camera_rgb.data.quat_w_world.cpu().numpy(),
                camera_intrinsics_matrices_k=self.camera_rgb.data.intrinsic_matrices.cpu().numpy(),
                base_link_poses=self.scene.articulations["ur5"]
                .data.root_pos_w.cpu()
                .numpy(),
                CAMERA_RGB_2_D_OFFSET=0,
            )
        )
        cube_pos = torch.from_numpy(cube_pos).to(self.device)
        cube_pos_w = torch.from_numpy(cube_pos_w).to(self.device)
        self.data_age = torch.tensor(data_age, device=self.device)
        self.dist_cube_cam = torch.tensor(dist_cube_cam, device=self.device)
        pos_sensor = torch.from_numpy(pos_sensor).to(self.device)

        # elapsed = time.time() - start
        # print(f"[DEBUG] Cube detection took: {(elapsed):.4f} s")
        #! Measure potential bottleneck

        self.cube_z_new = cube_pos[:, 2]

        # If on startup, set cube z pos old to new
        if self.cube_z_old is None:
            self.cube_z_old = self.cube_z_new.clone()
        # If env has been reset, set cube z pos old to new
        self.cube_z_old = torch.where(
            self.episode_length_buf == 0, cube_pos[:, 2], self.cube_z_old
        )

        # If env has been reset, set dist_prev to -1
        self.dist_cube_cam_minimal = torch.where(
            self.episode_length_buf == 0, 99.0, self.dist_cube_cam_minimal
        )

        # Compute distance cube position to goal position
        self.cube_distance_to_goal = torch.linalg.vector_norm(
            cube_pos_w - self.cube_goal_pos, dim=-1, keepdim=False
        )

        self.data_age = torch.clip(input=self.data_age, min=0.0, max=10.0)

        # print(f"Mean distance camera to cube: {self.dist_cube_cam}")
        # Obs of shape [n_envs, 1, 27])
        obs = torch.cat(
            (
                self.live_joint_pos[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                # self.live_joint_vel[:, : len(self._arm_dof_idx)].unsqueeze(
                #     dim=1
                # ),  #! remove when needed
                self.live_joint_torque[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.gripper_steps.unsqueeze(dim=1).unsqueeze(dim=1),
                cube_pos_w.unsqueeze(dim=1),
                # self.cube_distance_to_goal.unsqueeze(dim=1).unsqueeze(
                #     dim=1
                # ),  #! Not informative
                self.data_age.unsqueeze(dim=1).unsqueeze(dim=1),
                self.dist_cube_cam.unsqueeze(dim=1).unsqueeze(dim=1),
                pos_sensor.unsqueeze(dim=1),
            ),
            dim=-1,
        )

        obs = obs.float()
        obs = obs.squeeze(dim=1)

        # print(
        #     f"Env0 obs Debug\nDistCubeCam:{self.dist_cube_cam[0]}\nCubePosW:{cube_pos_w[0]}\nDataAge:{self.data_age[0]}\nPosSensor:{pos_sensor[0]}\n\n"
        # )
        #! LOGGING
        # ✅ Save only for the first environment (Env0)
        if self.LOG_ENV_DETAILS:
            self.episode_data["pos_sensor_x"].append(float(pos_sensor[0][0].cpu()))
            self.episode_data["pos_sensor_y"].append(float(pos_sensor[0][1].cpu()))
            self.episode_data["dist_cube_cam"].append(
                float(self.dist_cube_cam[0].cpu())
            )
            self.episode_data["mean_torque"].append(float(self.mean_torque.cpu()))

        if torch.isnan(obs).any():
            warnings.warn("[WARNING] NaN detected in observations!", UserWarning)
            print(f"[DEBUG] NaN found in observation: {obs}\nReplacing with 0.0")
            obs = torch.where(
                torch.isnan(obs),
                torch.tensor(0.0, dtype=obs.dtype, device=obs.device),
                obs,
            )

        # Calculate mean torque
        mean_torque = torch.mean(
            torch.abs(self.live_joint_torque[:, : len(self._arm_dof_idx)])
        )
        if mean_torque != 0:
            self.mean_torque = (self.mean_torque * 99 + mean_torque) / 100

        observations = {
            "policy": obs,
            "grasp_success": self.grasp_success,
            "mean_torque": self.mean_torque,
            "torque_limit_exeeded": self.torque_limit_exeeded,
        }
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
        if self.CL_state < 0 or self.CL_state > 3:
            torque_limit_exceeded = torch.any(
                torch.abs(self.live_joint_torque[:, : len(self._arm_dof_idx)])
                > self.torque_limit,
                dim=1,
            )

            # Provide a grace period for high torques when resetting
            pardon = torch.where(
                self.episode_length_buf < 10, torch.tensor(1), torch.tensor(0)
            )

            self.torque_limit_exeeded = torch.logical_and(
                torque_limit_exceeded == 1, pardon == 0
            )
        # Dont reset before torque penalty is applied
        else:
            self.torque_limit_exeeded = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        if torch.any(self.torque_limit_exeeded):
            idx = torch.where(self.torque_limit_exeeded)[0]
            print(f"Torque limit exeeded in envs: {idx}")

        # Resolves the issue of the goal_reached tensor becoming a scalar when the number of environments is 1
        if self.cfg.scene.num_envs == 1:
            self.grasp_success = self.grasp_success.unsqueeze(0)
        reset_terminated = self.grasp_success | self.torque_limit_exeeded
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

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore

        # Assuming only during init all episode lengths are 0
        if torch.sum(self.episode_length_buf) == 0:
            # Save the initial root position and quaternion
            self.init_root_pos = self.robot.data.root_pos_w.clone()
            self.init_root_quat = self.robot.data.root_quat_w.clone()

        # General resetting tasks (timers etc.)
        super()._reset_idx(env_ids)  # type: ignore

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        # Domain randomization
        if self.randomize:
            randomness = (
                torch.rand_like(joint_pos) * 2 - 1
            ) * self.joint_randomize_level
            joint_pos += randomness

            if (self.init_root_pos is not None) and (self.init_root_quat is not None):
                root_pos = self.init_root_pos[
                    env_ids
                ].clone()  # shape [len(env_ids), 3]
                root_quat = self.init_root_quat[
                    env_ids
                ].clone()  # shape [len(env_ids), 4]

                # Example: random XY offset up to ±0.2
                rand_x = torch.rand(len(env_ids), device=self.device) * 0.2 - 0.1
                rand_y = torch.rand(len(env_ids), device=self.device) * 0.2 - 0.1
                # We'll keep Z the same
                root_pos[:, 0] += rand_x
                root_pos[:, 1] += rand_y

                root_pose = torch.cat((root_pos, root_quat), dim=1)
                self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)

        # joint_pos[env_ids, 5] = 0.0

        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])

        self.live_joint_pos[env_ids] = joint_pos
        self.live_joint_vel[env_ids] = joint_vel
        self.robot.write_joint_state_to_sim(
            position=joint_pos, velocity=joint_vel, joint_ids=None, env_ids=env_ids
        )
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
                print(
                    f"Using {self.CL_state} as CL state, adding R[0 - {self.CL_state}]"
                )
                print(f"[0]: Total pickup reward: {self.pickup_reward[env_id]}")
                print(
                    f"[1]: Cube out of sight penalty: {self.total_penalty_cube_out_of_sight[env_id]}"
                )
                print(f"[2]: Cube approach reward: {self.cube_approach_reward[env_id]}")
                print(f"[3]: Total torque penalty: {self.total_torque_penalty[env_id]}")
                print(
                    f"[4]: Total torque limit exeeded penalty: {self.total_torque_limit_exeeded_penalty[env_id]}"
                )
                print("-" * 20)

                self.total_penalty_alive[env_ids] = 0  # type: ignore
                self.total_penalty_vel[env_ids] = 0  # type: ignore
                self.total_penalty_cube_out_of_sight[env_ids] = 0  # type: ignore
                self.total_penalty_distance_cube_to_goal[env_ids] = 0  # type: ignore
                self.total_torque_limit_exeeded_penalty[env_ids] = 0  # type: ignore
                self.pickup_reward[env_ids] = 0  # type: ignore
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
        # print(f"Delta Cube Z: {delta_cube_z}")

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
            self.cube_z_new,
            self.grasp_success,
            self.cfg.pickup_reward_scaling,
            self.partial_grasp,
            self.cfg.partial_grasp_reward_scaling,
            self.container_contact,
            self.cfg.container_contact_penalty_scaling,
        )

        # self.total_penalty_alive += rewards[0]
        # self.total_penalty_vel += rewards[1]
        self.total_penalty_cube_out_of_sight += rewards[2]
        # self.total_penalty_distance_cube_to_goal += rewards[3]
        self.total_torque_limit_exeeded_penalty += rewards[4]
        self.total_torque_penalty += rewards[5]
        self.cube_approach_reward += rewards[6]
        self.pickup_reward += rewards[7]

        if self.LOG_ENV_DETAILS:
            self.episode_data["reward_approach"].append(float(rewards[7][0].cpu()))
            # self.episode_data["penalty_torque"].append(float(rewards[6][0].cpu()))

        # total_reward = torch.sum(rewards, dim=0)

        all_rewards = [
            rewards[7],  # pickup reward
            rewards[2],  #! cube out of sight penalty
            rewards[6],  # approach reward
            rewards[5],  # torque penalty
            rewards[4],  # torque limit exceeded penalty
            rewards[3],  #! distance cube to goal penalty
            rewards[1],  #! velocity penalty
            rewards[0],  #! alive penalty
        ]
        # if (rewards[7] > 0).any():
        # print(f"[DEBUG] Positive pickup reward(s): {rewards[7]}")

        if self.CL_state > -1:
            total_reward = torch.sum(
                torch.stack(all_rewards[: (self.CL_state + 1)]), dim=0
            )
        else:
            total_reward = torch.sum(
                torch.stack([rewards[7], rewards[6], rewards[4]]), dim=0
            )

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
    cube_z: torch.Tensor,
    grasp_success: torch.Tensor,
    pickup_reward_scaling: float,
    partial_grasp: torch.Tensor,
    partial_grasp_reward_scaling: float,
    container_contact: torch.Tensor,
    container_contact_penalty_scaling: float,
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
    pickup_reward = torch.where(
        (grasp_success == True) & (dist_cube_cam > 0.16),
        torch.tensor(1.0, dtype=cube_z.dtype, device=cube_z.device),
        torch.tensor(0.0, dtype=cube_z.dtype, device=cube_z.device),
    )

    # partial_grasp_reward = torch.where(
    #     (partial_grasp == True) & (dist_cube_cam > 0.16) & (grasp_success == False),
    #     torch.tensor(1.0, dtype=cube_z.dtype, device=cube_z.device),
    #     torch.tensor(0.0, dtype=cube_z.dtype, device=cube_z.device),
    # )

    # partial_grasp_reward = partial_grasp_reward * partial_grasp_reward_scaling

    pickup_reward = pickup_reward * pickup_reward_scaling

    # open_gripper_incentive = torch.where(
    #     (dist_cube_cam > 0.22) & (dist_cube_cam < 0.4) & (gripper_action_bin > 0),
    #     torch.tensor(-0.005, dtype=dist_cube_cam.dtype, device=dist_cube_cam.device),
    #     torch.tensor(0.0, dtype=dist_cube_cam.dtype, device=dist_cube_cam.device),
    # )

    # close_gripper_incentive = torch.where(
    #     (dist_cube_cam > 0.18) & (dist_cube_cam < 0.22) & (gripper_action_bin > 0),
    #     torch.tensor(0.01, dtype=dist_cube_cam.dtype, device=dist_cube_cam.device),
    #     torch.tensor(0.0, dtype=dist_cube_cam.dtype, device=dist_cube_cam.device),
    # )

    # # Container contact penalty
    # container_contact_penalty_t = torch.where(
    #     container_contact,
    #     torch.tensor(1.0, device=container_contact.device),
    #     torch.tensor(0.0, device=container_contact.device),
    # )
    # container_contact_penalty = (
    #     container_contact_penalty_scaling * container_contact_penalty_t
    # )

    # pickup_reward += (
    #     open_gripper_incentive + close_gripper_incentive + partial_grasp_reward
    # )

    # pickup_reward -= container_contact_penalty

    # Exponential decay of reward with distance
    # dist_cube_cam = torch.where(
    #     (dist_cube_cam > 0.0) & (dist_cube_cam < 0.2),
    #     torch.tensor(0.2, dtype=dist_cube_cam.dtype, device=dist_cube_cam.device),
    #     dist_cube_cam,
    # )
    k = 5
    approach_reward = torch.where(
        (dist_cube_cam > 0.2) & (data_age < 3.0),
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
            total_torque_penalty,
            approach_reward,
            pickup_reward,
        ]
    )
