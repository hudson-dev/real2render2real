# JAX by default pre-allocates 75%, which will cause OOM
# This line needs to go above any jax import.
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
os.environ['JAX_LOG_COMPILES'] = '0'

import logging
logging.getLogger("jax").setLevel(logging.WARNING)  # or logging.INFO, logging.ERROR etc.

import jax
import jax.numpy as jnp
import jaxlie
from jaxmp import JaxKinTree, BatchedRobotFactors
from jaxmp.extras.urdf_loader import load_urdf
from jaxmp.extras.solve_ik import solve_ik_batched
from typing import Literal, Optional
from pathlib import Path
import os
import sys
import numpy as onp
import time
from loguru import logger
import viser
from dataclasses import dataclass

@dataclass
class TransformHandle:
    """Data class to store transform handles."""
    frame: viser.FrameHandle
    control: Optional[viser.TransformControlsHandle] = None

class JaxMPBatchedController:
    
    def __init__(
        self, 
        urdf_path: str,
        num_envs: int,
        num_ees: int,
        target_names: list[str], # Left to right
        home_pose: onp.ndarray = None,
        pos_weight: float = 100.0,
        rot_weight: float = 5.0,
        rest_weight: float = 0.01,
        limit_weight: float = 100.0,
        setup_viser: bool = False,
        device: Literal["cpu", "gpu"] = "gpu",
        ):
        jax.config.update("jax_platform_name", device)
        
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        self.setup_viser = setup_viser
        self.num_envs = num_envs
        self.urdf = load_urdf(None, urdf_path)
        self.kin = JaxKinTree.from_urdf(self.urdf)
        self.num_ees = num_ees
        
        
        if home_pose is None:
            import pdb; pdb.set_trace()
            # TODO: Default to half of limits
            self.rest_pose = jnp.array()
        else:
            self.rest_pose = jnp.array(home_pose[:self.kin.num_actuated_joints]) # assume final joints are actuated the same
            
        # self.rest_pose = jnp.array(list(self.YUMI_REST_POSE.values()))
        self.JointVar = BatchedRobotFactors.get_var_class(self.kin, self.rest_pose, self.num_envs)
        
        if self.setup_viser:
            self.server = viser.ViserServer()
        
        # Store weights for IK
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.rest_weight = rest_weight
        self.limit_weight = limit_weight
        
        self.tf_size_handle = 0.2
        
        # Target transform handle names HARDCODED BAD
        if self.num_ees == 2:
            self.target_names = [target_names[0], target_names[1]] # left_dummy_joint, right_dummy_joint for yumi
        elif self.num_ees == 1:
            self.target_names = [target_names[0]]
        else:
            raise ValueError(f"num_ees must be 1 or 2, got {self.num_ees}")
        
        self.joints = self.rest_pose
        
        if self.setup_viser:
            self._setup_visualization()
            self._setup_transform_handles()
        
        self.base_pose = jaxlie.SE3.identity()
        if self.setup_viser:
            self.base_frame.position = onp.array(self.base_pose.translation())
            self.base_frame.wxyz = onp.array(self.base_pose.rotation().wxyz)
        
        if self.setup_viser:
            self.env = 0
            
            # Env selector
            self.env_selector = self.server.gui.add_dropdown(
                "Environment Selector",
                [str(i) for i in range(self.num_envs)],
                initial_value='0'
            )
        
        # Initialize solver parameters
        self.solver_type = "conjugate_gradient"
        self.smooth = False
        self.manipulability_weight = 0.0
        self.has_jitted = False
        
        self._setup_dof_controls()
        
        self.base_mask, self.target_mask = self.get_freeze_masks()
        self.ConstrainedSE3Var = BatchedRobotFactors.get_constrained_se3(self.base_mask)
    
    def _setup_visualization(self):
        """Setup basic visualization elements."""
        # Add base frame and robot URDF
        self.base_frame = self.server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis = viser.extras.ViserUrdf(
            self.server, 
            self.urdf, 
            root_node_name="/base"
        )
        self.urdf_vis.update_cfg(self.rest_pose)
        
        # Add ground grid
        self.server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)
        
    def _setup_dof_controls(self):
        """Setup controls for freezing degrees of freedom."""
        self.base_dof_handles = []
        self.target_dof_handles = []
        
        for dof in ["x", "y", "z", "rx", "ry", "rz"]:
            self.base_dof_handles.append(True)
            
        for dof in ["x", "y", "z", "rx", "ry", "rz"]:
            self.target_dof_handles.append(True)
                
    def get_freeze_masks(self):
        """Get DoF freeze masks for base and targets."""
        base_mask = jnp.array([h for h in self.base_dof_handles]).astype(jnp.float32)
        target_mask = jnp.array([h for h in self.target_dof_handles]).astype(jnp.float32)
        return base_mask, target_mask
    
    
    def _setup_transform_handles(self):
        """Setup transform handles for end effectors for each environment."""
        
        # for env in range(self.num_envs):
        if self.num_ees == 2:
            self.transform_handles = {
                'left': TransformHandle(
                    frame=self.server.scene.add_frame(
                    f"tf_left_env",
                    axes_length=0.5 * self.tf_size_handle,
                    axes_radius=0.01 * self.tf_size_handle,
                    origin_radius=0.1 * self.tf_size_handle,
                ),
                # control=self.server.scene.add_transform_controls(
                #     f"target_left_env{env}",
                #     scale=self.tf_size_handle.value
                # )
            ),
            'right': TransformHandle(
                frame=self.server.scene.add_frame(
                    f"tf_right_env",
                    axes_length=0.5 * self.tf_size_handle,
                    axes_radius=0.01 * self.tf_size_handle,
                    origin_radius=0.1 * self.tf_size_handle,
                ),
                # control=self.server.scene.add_transform_controls(
                #     f"target_right_env{env}",
                #     scale=self.tf_size_handle.value
                # )
            ),
            'left_real': TransformHandle(
                frame=self.server.scene.add_frame(
                    f"tf_left_real_env",
                    axes_length=0.5 * self.tf_size_handle,
                    axes_radius=0.01 * self.tf_size_handle,
                    origin_radius=0.1 * self.tf_size_handle,
                ),
                # control=self.server.scene.add_transform_controls(
                #     f"target_left_env{env}",
                #     scale=self.tf_size_handle.value
                # )
            ),
            'right_real': TransformHandle(
                frame=self.server.scene.add_frame(
                    f"tf_right_real_env",
                    axes_length=0.5 * self.tf_size_handle,
                    axes_radius=0.01 * self.tf_size_handle,
                    origin_radius=0.1 * self.tf_size_handle,
                ),
                # control=self.server.scene.add_transform_controls(
                #     f"target_left_env{env}",
                        # scale=self.tf_size_handle.value
                    # )
                )
            }   
        elif self.num_ees == 1:
            self.transform_handles = {
                'ee': TransformHandle(
                    frame=self.server.scene.add_frame(
                    f"tf_left_env",
                    axes_length=0.5 * self.tf_size_handle,
                    axes_radius=0.01 * self.tf_size_handle,
                    origin_radius=0.1 * self.tf_size_handle,
                    )
                )
            }
        else:
            raise ValueError(f"num_ees must be 1 or 2, got {self.num_ees}")
        
       # Initialize positions
        base_pose = jnp.array(
            self.base_frame.wxyz.tolist() + self.base_frame.position.tolist()
        )
        
        # for env in range(self.num_batches):
        #     for target_frame_handle, target_name in zip(
        #         list(self.transform_handles[env].values()), self.target_names
        #     ):
        #         target_joint_idx = self.kin.joint_names.index(target_name)
        #         T_target_world = jaxlie.SE3(base_pose) @ jaxlie.SE3(
        #             self.kin.forward_kinematics(self.joints)[target_joint_idx]
        #         )

        #         target_frame_handle.control.position = onp.array(T_target_world.translation())
        #         target_frame_handle.control.wxyz = onp.array(T_target_world.rotation().wxyz)
                
        #         # Offset each environment's handles slightly to make them visible
        #         curr_pos = onp.array(target_frame_handle.control.position)
        #         target_frame_handle.control.position = curr_pos + onp.array([0.0005 * env, 0, 0])

        # Update transform handles when size changes
        # @self.tf_size_handle.on_update
        # def update_tf_size(_):
        #     for env in range(self.num_envs):
        #         for handle in self.transform_handles[env].values():
        #             if handle.control:
        #                 handle.control.scale = self.tf_size_handle
        #             handle.frame.axes_length = 0.5 * self.tf_size_handle
        #             handle.frame.axes_radius = 0.01 * self.tf_size_handle
        #             handle.frame.origin_radius = 0.1 * self.tf_size_handle
               
               
    def update_visualization(self):
        """Update visualization with current state."""
        # Update base frame
        self.base_frame.position = onp.array(self.base_pose.translation())
        self.base_frame.wxyz = onp.array(self.base_pose.rotation().wxyz)
        
        # Update robot configuration
        self.urdf_vis.update_cfg(onp.array(self.joints))
        
        # Update end-effector frames
        if self.num_ees == 2:
            target_joint_indices = {
                'left_real': self.kin.joint_names.index(self.target_names[0]),
                'right_real': self.kin.joint_names.index(self.target_names[1])
            }
        elif self.num_ees == 1:
            target_joint_indices = {
                'ee': self.kin.joint_names.index(self.target_names[0])
            }
        else:
            raise ValueError(f"num_ees must be 1 or 2, got {self.num_ees}")
        
        # Only update the currently selected environment
        
        current_env = int(self.env_selector.value)
        
        for side, idx in target_joint_indices.items():
            T_target_world = self.base_pose @ jaxlie.SE3(
                self.kin.forward_kinematics(self.joints)[idx]
            )
            self.transform_handles[side].frame.position = onp.array(T_target_world.translation())
            self.transform_handles[side].frame.wxyz = onp.array(T_target_world.rotation().wxyz)
            

    def reset(self):
        self.initial_poses = jnp.tile(self.rest_pose[None, :], (self.num_envs, 1))
        self.joints_all = self.initial_poses
        
    def compute_ik(self, target_poses: onp.ndarray):
        """
        target_poses: (n_envs, n_targets, 7) for xyz_wxyz
        """
        
        target_indices = jnp.array(
            [
                self.kin.joint_names.index(self.target_names[i]) for i in range(self.num_ees)
            ]
        )
        if self.setup_viser:
            if self.num_ees == 2:
                self.transform_handles['left'].frame.position = target_poses[self.env, 0, :3]
                self.transform_handles['left'].frame.wxyz = target_poses[self.env, 0, 3:]
                
                self.transform_handles['right'].frame.position = target_poses[self.env, 1, :3]
                self.transform_handles['right'].frame.wxyz = target_poses[self.env, 1, 3:]
            elif self.num_ees == 1:
                self.transform_handles['ee'].frame.position = target_poses[self.env, 0, :3]
                self.transform_handles['ee'].frame.wxyz = target_poses[self.env, 0, 3:]
            else:
                raise ValueError(f"num_ees must be 1 or 2, got {self.num_ees}")
                
        # Solve IK
        if not self.has_jitted:
            start_time = time.time()
            
        temp_pos = target_poses[:, :, :3]
        temp_quat = target_poses[:, :, 3:]
        target_poses = jnp.concatenate([temp_quat, temp_pos], axis=-1)
        
        target_se3s = jaxlie.SE3(jnp.array(target_poses))
        
        ik_weight = jnp.array([self.pos_weight] * 3 + [self.rot_weight] * 3)
        ik_weight = ik_weight * self.target_mask
        
        if self.smooth:
            if hasattr(self, "joints_all"):
                self.initial_poses = self.joints_all
            else:
                self.initial_poses = jnp.tile(self.rest_pose[None, :], (self.num_envs, 1))
            joint_vel_weight = self.limit_weight
        else:
            if hasattr(self, "joints_all"):
                self.initial_poses = self.joints_all
            else:
                self.initial_poses = jnp.tile(self.rest_pose[None, :], (self.num_envs, 1))
            joint_vel_weight = 0.0
            
        start_time = time.time()
        self.base_pose, self.joints_all = solve_ik_batched(
            self.kin,
            target_se3s,
            target_indices,
            self.initial_poses,  # Now batched
            self.JointVar,
            ik_weight,
            ConstrainedSE3Var=self.ConstrainedSE3Var,
            rest_weight=self.rest_weight,
            limit_weight=self.limit_weight,
            joint_vel_weight=joint_vel_weight,
            use_manipulability=(self.manipulability_weight > 0),
            manipulability_weight=self.manipulability_weight,
            solver_type=self.solver_type,
            num_batches=self.num_envs,
        )
        
        # Update timing
        jax.block_until_ready((self.base_pose, self.joints_all))
        if not self.has_jitted:
            self.timing_handle = (time.time() - start_time) * 1000
            logger.info("JIT compile + running took {} ms.", self.timing_handle)
            self.has_jitted = True
        else:
            self.solve_time = (time.time() - start_time) * 1000
            
        if self.setup_viser:
            current_env = int(self.env_selector.value)
            self.joints = self.joints_all[current_env]
            self.update_visualization()
        
        return self.joints_all.copy()