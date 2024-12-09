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
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import CameraCfg, Camera
import numpy as np
from numpy import float64


@configclass
class HawUr5EnvCfg(DirectRLEnvCfg):
    # env
    num_actions = 7
    f_update = 120
    num_observations = 7
    num_states = 5
    reward_scale_example = 1.0
    decimation = 2
    action_scale = 1.0
    v_cm = 35  # cm/s
    stepsize = v_cm * (1 / f_update) / 44  # Max angle delta per update
    pp_setup = True

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Objects

    # Static Object Container Table
    container_cfg = sim_utils.UsdFileCfg(
        usd_path="omniverse://localhost/MyAssets/Objects/Container.usd",
    )
    # Rigid Object Cube
    cube_cfg = sim_utils.UsdFileCfg(
        usd_path="omniverse://localhost/MyAssets/Objects/Cube.usd",
    )
    # Camera
    camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/ur5/onrobot_rg6_model/onrobot_rg6_base_link/camera",  # onrobot_rg6_model/onrobot_rg6_base_link/camera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.015), rot=(0.71, 0.0, 0.0, 0.71), convention="ros"
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
        env_spacing=2.5, replicate_physics=True
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

    def get_joint_pos(self):
        return self.live_joint_pos

    def _setup_scene(self):
        # add Articulation
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Check if the pick and place setup is enabled
        if self.cfg.pp_setup:
            # add container table
            spawn_from_usd(
                prim_path="/World/envs/env_.*/container",
                cfg=self.cfg.container_cfg,
                translation=(0.8, 0.0, 0.0),
            )
            # Spawn cube with individual randomization for each environment
            for env_idx in range(self.scene.num_envs):
                env_prim_path = f"/World/envs/env_{env_idx}/cube"
                random_translation = (
                    0.5 + np.random.uniform(-0.1, 0.5),
                    0.0 + np.random.uniform(-0.2, 0.2),
                    1.0,
                )
                spawn_from_usd(
                    prim_path=env_prim_path,
                    cfg=self.cfg.cube_cfg,
                    translation=random_translation,
                )

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion to scene
        self.scene.articulations["ur5"] = self.robot
        # return the scene information
        self.camera = Camera(cfg=self.cfg.camera_cfg)
        self.scene.sensors["camera"] = self.camera

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

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.live_joint_pos[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.live_joint_vel[:, : len(self._arm_dof_idx)].unsqueeze(dim=1),
                self.live_joint_pos[:, : len(self._gripper_dof_idx)].unsqueeze(dim=1),
                self.live_joint_vel[:, : len(self._gripper_dof_idx)].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
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
