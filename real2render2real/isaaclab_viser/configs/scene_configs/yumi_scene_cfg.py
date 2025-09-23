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
from real2render2real.isaaclab_viser.configs.scene_configs.yumi_base_cfg import YumiBaseCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

@configclass
class YumiPickTigerCfg(YumiBaseCfg):
    """Design the scene with sensors on the robot."""
    
    tiger = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tiger",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/tiger/tiger_new.usd",
            # scale=(0.9, 0.9, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=16,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled = False,
            ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.00, 0.06)),
    )

@configclass
class YumiCoffeeMakerCfg(YumiBaseCfg):
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.39, 0.1, 0.093)),
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

        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.39, 0.1, 0.09)),

    )


@configclass
class YumiLedLightCfg(YumiBaseCfg):
    """Design the scene with sensors on the robot."""
    led_light2_subpart_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/led_light2_subpart_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/led_light/led_light2_subpart_0/led_light2_subpart_0.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.145)),
    )
    
    led_light2_subpart_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/led_light2_subpart_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/led_light/led_light2_subpart_1/led_light2_subpart_1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.145)),
    )

    led_light2_subpart_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/led_light2_subpart_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/led_light/led_light2_subpart_2/led_light2_subpart_2.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.145)),
    )
    
    led_light2_subpart_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/led_light2_subpart_3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/led_light/led_light2_subpart_3/led_light2_subpart_3.usd",

            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.145)),
    )
    


@configclass
class YumiFaucetCfg(YumiBaseCfg):
    """Design the scene with sensors on the robot."""
    faucet3_subpart_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/faucet3_subpart_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/faucet3/faucet3_subpart_0/faucet3_subpart_0.usd",
            # usd_path=f"{data_dir}/assets/object_scans/faucet_test/best.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.145)),
    )
    
    faucet3_subpart_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/faucet3_subpart_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/faucet3/faucet3_subpart_1/faucet3_subpart_1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.145)),
    )

@configclass
class YumiDrawerOpenCfg(YumiBaseCfg):
    """Design the scene with sensors on the robot."""
    drawer_base = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/drawer_base",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/drawer/drawer_base/drawer_base.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.059)),
    )
    
    drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/drawer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/drawer/drawer/drawer.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.059)),
    )

   

@configclass
class YumiCardboardPickupCfg(YumiBaseCfg):
    """Design the scene with sensors on the robot."""
    cardboard_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cardboard_box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/cardboard_box/cardboard_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.47, 0.0, 0.058)),
    )
   
   
@configclass
class YumiTigerPickR2R2RCfg(YumiBaseCfg):
    """Design the scene with sensors on the robot."""
    tiger = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tiger",
        spawn=sim_utils.UsdFileCfg(
            # TODO: ensure asset path is the way you expect
            usd_path=f"{data_dir}/assets/object_scans/tiger/tiger_new.usd", #f"{data_dir}/assets/object_scans/tiger_sugar/tiger.usd",
            scale=(1.1, 1.1, 1.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled = False,
            ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, 0.1, 0.039)),
    )
   