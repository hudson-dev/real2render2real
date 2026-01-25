# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Yam arm.

The following configuration parameters are available:

* :obj:`YAM_CFG`: The Yam arm
"""
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

##
# Configuration
##

YAM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{data_dir}/yam/yam_new/yam_new.usd",
        # usd_path=f"{data_dir}/yam/modified_i2rt_yam/modified_i2rt_yam.usd",
        usd_path=f"{data_dir}/yam/modified_i2rt_yam_fix_limits/modified_i2rt_yam_fix_limits.usd",
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
        # TODO: figure out yam joints and tune initial yam joint positions
        joint_pos={
            # "joint1": math.radians(4.1),
            # "joint2": math.radians(6.8),
            # "joint3": math.radians(24.1),
            # "joint4": math.radians(10.0),
            # "joint5": math.radians(0.0),
            # "joint6": math.radians(0.0),
            # "gripper": 0.045,
            # "gripper_mirror": -0.045,
            "joint1": math.radians(0),
            "joint2": math.radians(0.0),
            "joint3": math.radians(0.0),
            "joint4": math.radians(0.0),
            "joint5": math.radians(0.0),
            "joint6": math.radians(0.0),
            "gripper": 0.045,
            "gripper_mirror": -0.045,
        },
    ),
    # TODO: figure out yam actuators
    actuators={
        # "right_arm": ImplicitActuatorCfg(
        #     joint_names_expr=["yumi_joint_[0-3]_r", "yumi_joint_7_r"],
        #     # velocity_limit=4.3, # 3.3
        #     stiffness=4.2e6,
        #     damping=20.0,
        #     friction=3.0,
        # ),
        # "right_wrist": ImplicitActuatorCfg(
        #     joint_names_expr=["yumi_joint_[4-6]_r"],
        #     # velocity_limit=5.5,
        #     stiffness=2.2e6,
        #     damping=0.0,
        #     friction=0.0,
        # ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-5]"],
            # velocity_limit=4.3, # 3.3
            stiffness=4.2e6,
            damping=20.0,
            friction=3.0,
        ),
        "left_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint6"],
            velocity_limit=1.0,
            stiffness=2.2e6,
            damping=1e6,
            friction=0.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper", "gripper_mirror"],
            # effort_limit=10000.,
            velocity_limit=4.4,
            stiffness=3e4,
            damping=1e1,
        ),
        # "yumi_gripper": ImplicitActuatorCfg(
        #     joint_names_expr=["gripper_r_joint", "gripper_r_joint_m", "gripper_l_joint", "gripper_l_joint_m"],
        #     # effort_limit=10000.,
        #     velocity_limit=4.4,
        #     stiffness=3e4,
        #     damping=1e1,
        # ),
    },
)
"""Configuration of the Yam."""