# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ABB YuMi IRB14000 arm.

The following configuration parameters are available:

* :obj:`YUMI_CFG`: The Sawyer arm with a custom deformable+compliant gripper
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

##
# Configuration
##

YUMI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{data_dir}/yumi_description/urdf/yumi.urdf",
        usd_dir=f"{data_dir}/yumi/",
        make_instanceable=True,
        fix_base=True,
        force_usd_conversion=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "yumi_joint_1_r": 1.21442839,
            "yumi_joint_2_r": -1.03205606,
            "yumi_joint_7_r": -1.10072738,
            "yumi_joint_3_r": 0.2987352 - 0.2,
            "yumi_joint_4_r": -1.85257716,
            "yumi_joint_5_r": 1.25363652,
            "yumi_joint_6_r": -2.42181893,
            "yumi_joint_1_l": -1.24839656,
            "yumi_joint_2_l": -1.09802876,
            "yumi_joint_7_l": 1.06634394,
            "yumi_joint_3_l": 0.31386161 - 0.2,
            "yumi_joint_4_l": 1.90125141,
            "yumi_joint_5_l": 1.3205139,
            "yumi_joint_6_l": 2.43563939,
            "gripper_r_joint": 0,
            "gripper_l_joint": 0,
        },
    ),
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["yumi_joint_[0-7]_r"],
            velocity_limit=2.0,
            stiffness=100000.0,
            damping=8.0,
            friction=10.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["yumi_joint_[0-7]_l"],
            velocity_limit=2.0,
            stiffness=100000.0,
            damping=8.0,
            friction=10.0,
        ),
    },
)
"""Configuration of ABB YuMi IRB14000 arm."""
