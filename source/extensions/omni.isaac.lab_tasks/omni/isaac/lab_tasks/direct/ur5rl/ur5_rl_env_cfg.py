from __future__ import annotations

import math
from omni.isaac.lab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
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
from omni.isaac.lab.sim.spawners.shapes import CuboidCfg
from omni.isaac.lab.assets import (
    RigidObjectCfg,
)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import CameraCfg, Camera, ContactSensorCfg

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


@configclass
class EventCfg:
    """Configuration for randomization."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5", body_names=".*"),
            "static_friction_range": (0.4, 0.9),
            "dynamic_friction_range": (0.2, 0.6),
            "restitution_range": (0.0, 0.7),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5", joint_names=".*"),
            # Narrow the stiffness distribution from (0.75, 1.5) → (0.9, 1.1)
            "stiffness_distribution_params": (0.9, 1.1),
            # Narrow the damping distribution from (0.3, 3.0) → (0.8, 1.2)
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class HawUr5EnvCfg(DirectRLEnvCfg):

    verbose_logging = True

    # env
    action_space = 7
    f_update = 120
    observation_space = 27
    state_space = 0
    episode_length_s = 10  # was 15

    arm_joints_init_state: list[float] = [0.0, -1.92, 2.3, -3.14, -1.57, 0.0]

    cube_init_state: tuple[float, float, float] = (1.0, 0.0, 0.57)

    alive_reward_scaling = +0.01
    terminated_penalty_scaling = 0.0
    vel_penalty_scaling = -0.00
    torque_penalty_scaling = -0.0006
    torque_limit_exeeded_penalty_scaling = -0.5
    cube_out_of_sight_penalty_scaling = -0.0003
    distance_cube_to_goal_penalty_scaling = -0.01
    goal_reached_scaling = 10.0
    approach_reward = 0.03
    pickup_reward_scaling = 5.0  # 0.03  #! was 0.2
    partial_grasp_reward_scaling = 0.01
    container_contact_penalty_scaling = 0.005
    torque_limit = 3000  # was 500.0

    decimation = 2
    action_scale = 0.5
    v_cm = 35  # cm/s
    stepsize = v_cm * (1 / f_update) / 44  # Max angle delta per update
    pp_setup = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / f_update,
        render_interval=decimation,
    )

    # Objects

    # Static Object Container Table
    container_cfg = sim_utils.UsdFileCfg(
        usd_path="omniverse://localhost/MyAssets/Objects/Container.usd",
    )

    cube_usd_cfg = sim_utils.UsdFileCfg(
        usd_path="omniverse://localhost/MyAssets/Objects/Cube.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # kinematic_enabled=False),
    )

    # Camera
    camera_rgb_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/ur5/onrobot_rg6_model/onrobot_rg6_base_link/rgb_camera",  # onrobot_rg6_model/onrobot_rg6_base_link/camera",
        update_period=0,
        height=120,
        width=212,
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
        height=120,
        width=212,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=30.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.055, -0.03, 0.025), rot=(0.71, 0.0, 0.0, 0.71), convention="ros"
        ),
    )

    contact_cfg_l = ContactSensorCfg(
        prim_path="/World/envs/env_.*/ur5/onrobot_rg6_model/left_inner_finger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_air_time=True,
    )
    contact_cfg_r = ContactSensorCfg(
        prim_path="/World/envs/env_.*/ur5/onrobot_rg6_model/right_inner_finger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_air_time=True,
    )
    contact_cfg_t = ContactSensorCfg(
        prim_path="/World/envs/env_.*/container/Container",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_air_time=True,
    )
    contact_cfg_c = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Cube/Cube",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_air_time=True,
    )

    # Gripper parameters

    cuboid_cfg = sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, kinematic_enabled=False
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
        ),
    )

    cube_rigid_obj_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,  # Let it fall if needed
                kinematic_enabled=False,  # Must be False if you want dynamic
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.0, 1.0),  # Put it somewhere
        ),
        debug_vis=True,
    )

    # contact_sensor = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/ur5/onrobot_rg6_model/left_inner_finger/Contact_Sensor_automatic",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    # )

    # robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/MyAssets/haw_ur5_assembled/haw_u5_with_gripper.usd",
            activate_contact_sensors=True,
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

    # events for domain randomization
    events: EventCfg = EventCfg()
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = (
        NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.004, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0002, operation="abs"),
        )
    )
