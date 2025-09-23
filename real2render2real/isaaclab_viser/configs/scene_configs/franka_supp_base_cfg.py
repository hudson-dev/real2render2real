# import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
# from isaaclab.sim.spawners.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg, MultiTiledCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


@configclass
class FrankaBaseCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    
    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/table2/table2_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    retain_accelerations=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, -0.057)),
    )

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # viewport_camera = MultiTiledCameraCfg( # INCLUDE THIS IN ALL CUSTOM CONFIGS TO LINK WITH A VISER VIEWPORT
    viewport_camera = CameraCfg(
        prim_path="/World/Viewport", 
        # MultiTiledCameraCfg results in prims at /World/envs/env_.*/Viewport0 and /World/envs/env_.*/Viewport1 if cams_per_env = 2
        # (For batched rendering of multiple cameras per environment)
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            # intrinsic_matrix = [600, 0, 1280/2, 0, 600, 720/2, 0, 0, 1],
            intrinsic_matrix = [278.09599369*2.5, 0, 1920/2, 0, 278.095993693*2.5, 1080/2, 0, 0, 1], # for 480x270
            # intrinsic_matrix = [278.09599369*1.666666667, 0, 800/2, 0, 278.09599369*1.666666667, 450/2, 0, 0, 1], # for 800x450
            # intrinsic_matrix = [295, 0, 480/2, 0, 295, 270/2, 0, 0, 1],
            height=1080,
            width=1920,
            # height=450,
            # width=800,
            # height = 720,
            # width = 1280,
            clipping_range=(0.01, 20),
            ),
        # cams_per_env = 1,
        )
    
    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            # color=(1.0 , 1.0, 1.0), 
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.75)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            # Using a uniform color without a texture provides diffuse lighting.
            color=(0.75, 0.75, 0.75), 
            # Lowering intensity can further reduce the harshness of reflections.
            intensity=1200.0,  # Adjusted intensity for potentially softer lighting
            # Ensure no texture file is specified to maintain uniform diffuse lighting.
            # texture_file=f"{data_dir}assets/skyboxes/rogland_clear_night_4k.exr",
            # texture_file=f"{data_dir}/assets/skyboxes/12_9_2024_BWW.jpg",
            # texture_file=f"{data_dir}/assets/skyboxes/12_14_2024_BWW.jpg",
            # texture_file=f"{data_dir}/assets/skyboxes/12_9_2024_02_36.jpg",
            # texture_file=f"{data_dir}/assets/skyboxes/f2d76ce6-7f2f-42db-8a59-ac6f8cf685d6.png",
            # texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            # texture_file=f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
            ),
    )

# https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_viewport/navigation.html#dolly