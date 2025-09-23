# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ABB YuMi IRB14000 arm.

The following configuration parameters are available:

* :obj:`YUMI_CFG`: The Sawyer arm with a custom deformable+compliant gripper
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

##
# Configuration
##

YUMI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{data_dir}/yumi_compliant_gripper/yumi.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=1,
            solver_velocity_iteration_count=1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=1, solver_velocity_iteration_count=1
        ),
        activate_contact_sensors=False,
        # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.001, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "yumi_joint_1_r": 1.21442839,
            "yumi_joint_2_r": -1.03205606,
            "yumi_joint_7_r": -1.10072738,
            "yumi_joint_3_r": 0.2987352 - 0.5,
            "yumi_joint_4_r": -1.85257716,
            "yumi_joint_5_r": 1.25363652,
            "yumi_joint_6_r": -2.42181893,
            "yumi_joint_1_l": -1.24839656,
            "yumi_joint_2_l": -1.09802876,
            "yumi_joint_7_l": 1.06634394,
            "yumi_joint_3_l": 0.31386161 - 0.5,
            "yumi_joint_4_l": 1.90125141,
            "yumi_joint_5_l": 1.3205139,
            "yumi_joint_6_l": 2.43563939,
            "gripper_r_joint": 0.024,
            "gripper_l_joint": 0.024,
            "gripper_r_joint_m": 0.024,
            "gripper_l_joint_m": 0.024,
        },
    ),
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["yumi_joint_[0-3]_r", "yumi_joint_7_r"],
            # velocity_limit=4.3, # 3.3
            stiffness=4.2e6,
            damping=20.0,
            friction=3.0,
        ),
        "right_wrist": ImplicitActuatorCfg(
            joint_names_expr=["yumi_joint_[4-6]_r"],
            # velocity_limit=5.5,
            stiffness=2.2e6,
            damping=0.0,
            friction=0.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["yumi_joint_[0-3]_l", "yumi_joint_7_l"],
            # velocity_limit=4.3, # 3.3
            stiffness=4.2e6,
            damping=20.0,
            friction=3.0,
        ),
        "left_wrist": ImplicitActuatorCfg(
            joint_names_expr=["yumi_joint_[4-6]_l"],
            # velocity_limit=5.5,
            stiffness=2.2e6,
            damping=0.0,
            friction=0.0,
        ),
        "yumi_gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_r_joint", "gripper_r_joint_m", "gripper_l_joint", "gripper_l_joint_m"],
            # effort_limit=10000.,
            velocity_limit=4.4,
            stiffness=3e4,
            damping=1e1,
        ),
    },
)
"""Configuration of ABB YuMi IRB14000 arm."""