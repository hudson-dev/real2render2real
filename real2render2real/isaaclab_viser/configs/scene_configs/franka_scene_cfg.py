# import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.utils import configclass

from pathlib import Path
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

##
# Pre-defined configs
##
from real2render2real.isaaclab_viser.configs.scene_configs.franka_base_cfg import FrankaBaseCfg
@configclass
class FrankaCoffeeMakerCfg(FrankaBaseCfg):
    """Design the scene with sensors on the robot."""
    coffee_maker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/coffee_maker",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/coffee_maker/coffee_maker.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.43, 0.11, 0.093-0.057)),
    )
    
    mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mug",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/mug/mug.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),

        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.43, 0.11, 0.09-0.057)),

    )

