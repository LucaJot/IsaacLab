from __future__ import annotations

import math
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import (
    GroundPlaneCfg,
    spawn_ground_plane,
    spawn_from_usd,
)
from omni.isaac.lab.sim.spawners.shapes import spawn_cuboid, CuboidCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import CameraCfg, Camera
import numpy as np
from numpy import float64
import cv2
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

from cube_detector import CubeDetector


@configclass
class HawUr5EnvCfg(DirectRLEnvCfg):
    # env
    num_actions = 7
    f_update = 120
    num_observations = 27
    num_states = 5
    reward_scale_example = 1.0
    decimation = 2
    action_scale = 1.0
    v_cm = 35  # cm/s
    stepsize = v_cm * (1 / f_update) / 44  # Max angle delta per update
    pp_setup = True

    episode_length_s = 120
    observation_space = 7
    action_space = 3

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    # Objects

    # Static Object Container Table
    container_cfg = sim_utils.UsdFileCfg(
        usd_path="omniverse://localhost/MyAssets/Objects/Container.usd",
    )
    # Rigid Object Cube

    cube_cfg = sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
        ),
    )

    # cube_cfg = sim_utils.UsdFileCfg(
    #     usd_path="omniverse://localhost/MyAssets/Objects/Cube.usd",
    # )

    cube_rigid_obj = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=cube_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.0, 1.0),
        ),
    )

    # Camera
    camera_rgb_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/ur5/onrobot_rg6_model/onrobot_rg6_base_link/rgb_camera",  # onrobot_rg6_model/onrobot_rg6_base_link/camera",
        update_period=0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,  # 0.188 für Realsense D435
            focus_distance=30.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.055, -0.03, 0.025), rot=(0.71, 0.0, 0.0, 0.71), convention="ros"
        ),
    )

    # Camera
    camera_depth_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/ur5/onrobot_rg6_model/onrobot_rg6_base_link/depth_camera",  # onrobot_rg6_model/onrobot_rg6_base_link/camera",
        update_period=0,
        height=720,
        width=1280,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=30.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.055, 0.0, 0.025), rot=(0.71, 0.0, 0.0, 0.71), convention="ros"
        ),
    )

    # Gripper parameters

    # robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/MyAssets/haw_ur5_assembled/haw_u5_with_gripper.usd"
        ),
        prim_path="/World/envs/env_.*/ur5",
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
            ),
        },
    )

    arm_dof_name = [
        "shoulder_pan_joint",  # 0
        "shoulder_lift_joint",  # -110
        "elbow_joint",  # 110
        "wrist_1_joint",  # -180
        "wrist_2_joint",  # -90
        "wrist_3_joint",  # 0
    ]
    gripper_dof_name = [
        "left_outer_knuckle_joint",
        "left_inner_finger_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_inner_finger_joint",
    ]

    haw_ur5_dof_name = arm_dof_name + gripper_dof_name

    action_dim = len(arm_dof_name) + len(gripper_dof_name)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        env_spacing=10, replicate_physics=True
    )

    # reset conditions
    # ...

    # reward scales
    # ...


class HawUr5Env(DirectRLEnv):
    cfg: HawUr5EnvCfg

    def __init__(self, cfg: HawUr5EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_dof_idx, _ = self.robot.find_joints(self.cfg.arm_dof_name)
        self._gripper_dof_idx, _ = self.robot.find_joints(self.cfg.gripper_dof_name)
        self.haw_ur5_dof_idx, _ = self.robot.find_joints(self.cfg.haw_ur5_dof_name)
        self.action_scale = self.cfg.action_scale

        # Holds the current joint positions and velocities
        self.live_joint_pos: torch.Tensor = self.robot.data.joint_pos
        self.live_joint_vel: torch.Tensor = self.robot.data.joint_vel

        self.jointpos_script_GT: torch.Tensor = self.live_joint_pos[:, :].clone()

        self.action_dim = len(self._arm_dof_idx) + len(self._gripper_dof_idx)

        self.gripper_action_bin: torch.Tensor | None = None

        # Cube detection
        self.cubedetector = CubeDetector()

        # Yolo model for cube detection
        # self.yolov11 = YOLO("yolo11s.pt")

    def get_joint_pos(self):
        return self.live_joint_pos

    def _setup_scene(self):
        # add Articulation
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # self.cubes = []

        # Check if the pick and place setup is enabled
        if self.cfg.pp_setup:
            # add container table
            spawn_from_usd(
                prim_path="/World/envs/env_.*/container",
                cfg=self.cfg.container_cfg,
                translation=(0.8, 0.0, 0.0),
            )

            self.cubes = spawn_cuboid(
                prim_path="/World/envs/env_.*/Cube",
                cfg=self.cfg.cube_cfg,
                translation=(1.0, 0.0, 1.0),
            )

            # TODO uncomment if spawn cuboid is not working properly
            # # Spawn cube with individual randomization for each environment
            # for env_idx in range(self.scene.num_envs):
            #     env_prim_path = f"/World/envs/env_{env_idx}/cube"
            #     random_translation = (
            #         0.5 + np.random.uniform(-0.1, 0.5),
            #         0.0 + np.random.uniform(-0.2, 0.2),
            #         1.0,
            #     )
            #     # TODO REMOVE - spawning cube in front for testing
            #     random_translation = (
            #         1,
            #         0.0,
            #         1.0,
            #     )

            #     # spawn_from_usd(
            #     #     prim_path=env_prim_path,
            #     #     cfg=self.cfg.cube_cfg,
            #     #     translation=random_translation,
            #     # )

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion to scene
        self.scene.articulations["ur5"] = self.robot
        # return the scene information
        self.camera_rgb = Camera(cfg=self.cfg.camera_rgb_cfg)
        self.scene.sensors["camera_rgb"] = self.camera_rgb
        self.camera_depth = Camera(cfg=self.cfg.camera_depth_cfg)
        self.scene.sensors["camera_depth"] = self.camera_depth
        # self.scene.rigid_objects["cube"] = RigidObject(self.cfg.cube_rigid_obj)

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
            current_main_joint_positions, current_main_joint_positions_sim, atol=1e-3
        ):
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

    def deproject_pixel_to_point(self, cx, cy, fx, fy, pixel, z):
        """
        Deprojects pixel coordinates and depth to a 3D point relative to the same camera.

        :param intrin: A dictionary representing the camera intrinsics.
                    Example:
                    {
                        'fx': float,        # Focal length in x
                        'fy': float,        # Focal length in y
                        'cx': float,       # Principal point x
                        'cy': float,       # Principal point y
                    }
        :param pixel: Tuple or list of 2 floats representing the pixel coordinates (x, y).
        :param depth: Float representing the depth at the given pixel.
        :return: List of 3 floats representing the 3D point in space.
        """

        # Calculate normalized coordinates
        x = (pixel[0] - cx) / fx
        y = (pixel[1] - cy) / fy

        # Compute 3D point
        point = [z, -z * x, z * y]
        return point

    def transform_frame_cam2world(self, camera_pos_w, camera_q_w, point_cam_rf):
        """
        Transforms a point from the camera frame to the world frame.

        Args:
            camera_pos_w (np.ndarray): Position of the camera in the world frame.
            camera_q_w (np.ndarray): Quaternion of the camera in the world frame.
            point_cam_rf (np.ndarray): Point in the camera frame.

        Returns:
            np.ndarray: Point in the world frame.
        """
        # Create a Rotation object from the quaternion
        rotation = R.from_quat(
            [camera_q_w[1], camera_q_w[2], camera_q_w[3], camera_q_w[0]]
        )  # Scipy expects [x, y, z, w]

        # Apply rotation and translation
        p_world = rotation.apply(point_cam_rf) + camera_pos_w  # was +
        return p_world

    def get_cube_positions(self, rgb_image: torch.Tensor, depth_image: torch.Tensor):
        """
        Extract positions of red cubes in the camera frame for all environments.

        Args:
            rgb_image (torch.Tensor): RGB image of shape (n, 480, 640, 3).
            depth_image (torch.Tensor): Depth image of shape (n, 480, 640, 1).

        Returns:
            list: A list of arrays containing the positions of red cubes in each environment.
        """
        CAMERA_RGB_2_D_OFFSET = -75
        rgb_images_np = rgb_image.cpu().numpy()
        depth_images_np = depth_image.squeeze(-1).cpu().numpy()

        # Clip and normalize to a 1m range
        depth_images_np = (np.clip(depth_images_np, a_min=0.0, a_max=1.0)) / (1)

        # Get the camera poses relative to world frame
        rgb_poses = self.camera_rgb.data.pos_w.cpu().numpy()
        rgb_poses_q = self.camera_rgb.data.quat_w_world.cpu().numpy()
        rgb_intrinsic_matrices = self.camera_rgb.data.intrinsic_matrices.cpu().numpy()

        robo_rootpose = self.scene.articulations["ur5"].data.root_pos_w.cpu().numpy()
        cube_positions = []
        cube_positions_w = []

        # Make the camera pose relative to the robot base link
        rel_rgb_poses = rgb_poses - robo_rootpose

        # Iterate over the images of all environments
        for env_idx in range(rgb_image.shape[0]):
            rgb_image_np = rgb_images_np[env_idx]
            depth_image_np = depth_images_np[env_idx]
            rgb_intrinsic_matrix = rgb_intrinsic_matrices[env_idx]

            # Get the envs camera poses from base link
            rgb_pose = rel_rgb_poses[env_idx]
            rgb_pose_q = rgb_poses_q[env_idx]
            # Make pose relative to base link (z-axis offset)
            # rgb_pose[2] -= 0.35

            hsv = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2HSV)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])

            red_mask = cv2.inRange(hsv, lower_red, upper_red)

            # Find contours or the largest connected component (assuming one red cube per env)
            contours, _ = cv2.findContours(
                red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # If nothing is found, append -1 coordinates to the list
            if len(contours) == 0:
                cube_positions.append([-1, -1, -1])
                cube_positions_w.append([-1, -1, -1])
            else:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Shift the contour to the left  to compensate for the offset between the rgb and depth image
                largest_contour[:, 0, 0] += CAMERA_RGB_2_D_OFFSET  # type: ignore
                # Get the moments of the largest contour
                M = cv2.moments(largest_contour)

                # Check for zero division and small contours
                if M["m00"] == 0 or cv2.contourArea(largest_contour) < 1000:
                    cube_positions.append([-1, -1, -1])
                    cube_positions_w.append([-1, -1, -1])
                    continue

                # Get the pixel centroid of the largest contour
                cx_px = int(M["m10"] / M["m00"])
                cy_px = int(M["m01"] / M["m00"])

                print(f"Centroid [px]: {cx_px}/1200, {cy_px}/720")

                # Get depth value at the centroid
                z = depth_image_np[cy_px, cx_px]

                # Calculate the actual 3D position of the cube relative to the camera sensor
                #     [fx  0 cx]
                # K = [ 0 fy cy]
                #     [ 0  0  1]
                cube_pos_camera_rf = self.deproject_pixel_to_point(
                    fx=rgb_intrinsic_matrix[0, 0],
                    fy=rgb_intrinsic_matrix[1, 1],
                    cx=rgb_intrinsic_matrix[0, 2],
                    cy=rgb_intrinsic_matrix[1, 2],
                    pixel=(cx_px, cy_px),
                    z=z,
                )
                # Convert the cube position from camera to world frame
                cube_pos_w = self.transform_frame_cam2world(
                    camera_pos_w=rgb_pose,
                    camera_q_w=rgb_pose_q,
                    point_cam_rf=cube_pos_camera_rf,
                )
                cube_positions_w.append(cube_pos_w)

                # Normalize thee centroid
                cx = cx_px / rgb_image_np.shape[1]
                cy = cy_px / rgb_image_np.shape[0]

                cube_positions.append(cube_pos_camera_rf)

                # Store image with contour drawn -----------------------------------

                # # Convert the depth to an 8-bit range
                # depth_vis = (depth_image_np * 255).astype(np.uint8)
                # # Convert single channel depth to 3-channel BGR (for contour drawing)
                # depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                # # Draw the contour of the rgb to the depth image to viz the offset
                # cv2.drawContours(depth_vis_bgr, [largest_contour], -1, (0, 255, 0), 3)

                # cv2.imwrite(
                #     f"/home/luca/Pictures/isaacsimcameraframes/maskframe.png",
                #     depth_vis_bgr,
                # )

                # cv2.drawContours(rgb_image_np, contours, -1, (0, 255, 0), 3)
                # cv2.imwrite(
                #     f"/home/luca/Pictures/isaacsimcameraframes/maskframe.png",
                #     rgb_image_np,
                # )

                # --------------------------------------------------------------------

        return np.array(cube_positions), np.array(cube_positions_w)

    def _get_observations(self) -> dict:
        rgb = self.camera_rgb.data.output["rgb"]
        depth = self.camera_depth.data.output["distance_to_camera"]

        # Extract the cubes position from the rgb and depth images an convert it to a tensor

        cube_pos, cube_pos_w = self.cubedetector.get_cube_positions(
            rgb_images=rgb.cpu().numpy(),
            depth_images=depth.squeeze(-1).cpu().numpy(),
            rgb_camera_poses=self.camera_rgb.data.pos_w.cpu().numpy(),
            rgb_camera_quats=self.camera_rgb.data.quat_w_world.cpu().numpy(),
            camera_intrinsics_matrices_k=self.camera_rgb.data.intrinsic_matrices.cpu().numpy(),
            base_link_poses=self.scene.articulations["ur5"]
            .data.root_pos_w.cpu()
            .numpy(),
        )
        cube_pos = torch.from_numpy(cube_pos).to(self.device)
        cube_pos_w = torch.from_numpy(cube_pos_w).to(self.device)

        print(f"Cube world pos: {cube_pos_w}")
        print(f"Cube pos camera ref: {cube_pos}")

        # Obs of shape [n_envs, 1, 27])
        obs = torch.cat(
            (
                self.live_joint_pos[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.live_joint_vel[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.live_joint_pos[:, : len(self._gripper_dof_idx)].unsqueeze(dim=1),
                self.live_joint_vel[:, : len(self._gripper_dof_idx)].unsqueeze(dim=1),
                cube_pos.unsqueeze(dim=1),
            ),
            dim=-1,
        )
        # debug_shape = obs.shape

        observations = {"policy": obs, "images": {"rgb": rgb, "depth": depth}}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros(1, device=self.device)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_of_bounds = torch.zeros(1, device=self.device)
        time_out = torch.zeros(1, device=self.device)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore

        # General resetting tasks (timers etc.)
        super()._reset_idx(env_ids)  # type: ignore

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        # Domain randomization (TODO make sure states are safe)
        joint_pos += torch.rand_like(joint_pos) * 0.1

        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]

        self.live_joint_pos[env_ids] = joint_pos
        self.live_joint_vel[env_ids] = joint_vel
        # self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

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


"""
Gripper steering function info

        def gripper_steer(
    action: float, stepsize: float, current_joints: list[float]
) -> torch.Tensor:
    Steer the individual gripper joints.
       This function translates a single action
       between -1 and 1 to the gripper joint position targets.
       value to the gripper joint position targets.

    Args:
        action (float): Action to steer the gripper.

    Returns:
        torch.Tensor: Gripper joint position targets.

    # create joint position targets
    gripper_joint_pos = torch.tensor(
        [
            36 * action,  # "left_outer_knuckle_joint",
            -36 * action,  # "left_inner_finger_joint",
            -36 * action,  # "left_inner_knuckle_joint",
            -36 * action,  # "right_inner_knuckle_joint",
            36 * action,  # "right_outer_knuckle_joint",
            36 * action,  # "right_inner_finger_joint",
        ]
    )
    return gripper_joint_pos
        """
