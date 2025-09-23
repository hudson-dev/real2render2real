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
from real2render2real.isaaclab_viser.configs.articulation_configs.yumi_cfg_diffIK import (
    YUMI_CFG
)

@configclass
class YumiBaseCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""
    #wall
    # wall = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Wall",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{data_dir}/assets/wall/wall.usd",
    #         scale=(2, 1, 2),
    #         ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.4, -2, 0.2), rot=(1.0 ,  0, 0, 0)),
    # )
    
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
            ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, 0.0)),
    )

    # robot
    robot: ArticulationCfg = YUMI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    viewport_camera = MultiTiledCameraCfg( # INCLUDE THIS IN ALL CUSTOM CONFIGS TO LINK WITH A VISER VIEWPORT
        prim_path="{ENV_REGEX_NS}/Viewport", 
        # MultiTiledCameraCfg results in prims at /World/envs/env_.*/Viewport0 and /World/envs/env_.*/Viewport1 if cams_per_env = 2
        # (For batched rendering of multiple cameras per environment)
        # height=270,
        # width=480,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            # intrinsic_matrix = [278.09599369, 0, 480/2, 0, 278.09599369, 270/2, 0, 0, 1], # for 480x270
            intrinsic_matrix = [278.09599369*3, 0, 1280/2, 0, 278.09599369*3, 720/2, 0, 0, 1], # for 1280x720
            height=720,
            width=1280,
            clipping_range=(0.01, 20),
            ),
        cams_per_env = 2,
        )

    # wrist_camera = TiledCameraCfg( # Wrist camera example
    #     prim_path="{ENV_REGEX_NS}/Robot/gripper_l_base/wrist_cam",
    #     height=240,
    #     width=320,
    #     data_types=["rgb", "depth"],
    #     spawn=sim_utils.PinholeCameraCfg(),
    #     offset=CameraCfg.OffsetCfg(pos=(0.0, 0.05, 0.0), rot=(1, 0, 0, 0), convention="ros"),
    #     )

  # lights/skybox
    dome_light2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light2",
        spawn=sim_utils.CylinderLightCfg(
            intensity=1003.0,
            radius=1.0,
            ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.46, -0.64, 1.0)),
    )
    
    dome_light3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light3",
        spawn=sim_utils.CylinderLightCfg(
            intensity=1003.0,
            radius=1.0,
            ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.46, 0.4, 1.0)),
    )
    
    
    
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{data_dir}/assets/skyboxes/12_9_2024_BWW.jpg",
            ),
    )
    