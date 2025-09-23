from xi.isaaclab_viser.base import IsaacLabViser

import torch
import numpy as np

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from xi.utils.math import reorient_quaternion_batch

import time

from collections import deque
import os
from pathlib import Path

# For Domain Randomization
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# from isaaclab.core.utils.semantics import add_update_semantics, get_semantics
# from omni.isaac.nucleus import get_assets_root_path

# print(f"[INFO]: Using assets from {ISAAC_NUCLEUS_DIR}")

# assets_root_path = get_assets_root_path()
# textures = [
#     assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/aggregate_exposed_diff.jpg",
#     assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_diff.jpg",
#     assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_multi_R_rough_G_ao.jpg",
#     assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/rough_gravel_rough.jpg",
# ]

class PushStapler(IsaacLabViser):
    def __init__(self, *args, **kwargs):
        self.debug_marker_vis = False
        super().__init__(*args, **kwargs)
        
    def _setup_viser_gui(self):
        super()._setup_viser_gui()
        # Add object frame to viser
        self.rigid_objects_viser_frame = []

        name = 'stapler'
        rigid_object = self.scene.articulations['stapler']
        self.rigid_objects_viser_frame.append(
            self.viser_server.scene.add_frame(
                name, 
                position = rigid_object.data.default_root_state[self.env][:3].cpu().detach().numpy(), 
                wxyz = rigid_object.data.default_root_state[self.env, 3:7].cpu().detach().numpy(),
                axes_length = 0.05,
                axes_radius = 0.003,
                )
            )
    
    def run_simulator(self):
        """Runs the simulation loop."""
        
        # Create controller
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)
    
        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
        sim_time = 0.0
        count = 0
        success_envs = None
        
        gripper_close = False
        ee_goal_offsets = [
        [-0.01, 0.06, 0.25, 0, -1, 0, 0],
        [-0.01, 0.06, 0.15, 0, -1, 0, 0],
        [-0.01, 0.06, 0.05, 0, -1, 0, 0],
        ]
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["gripper_l_base", "gripper_r_base"])
        robot_entity_cfg.resolve(self.scene)
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        
        # Visualization Markers
        if self.debug_marker_vis:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
            goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
                
        # Simulate physics
        while self.simulation_app.is_running():
            # Randomize every step (slows down sim step quite a bit)
            # self.randomize_lighting()
            # self.randomize_robot_texture()
            
            sim_start_time = time.time()
            # reset
            if self.client is None:
                while self.client is None:
                    self.client = self.viser_server.get_clients()[0] if len(self.viser_server.get_clients()) > 0 else None # Get the first client
                    time.sleep(0.1)
                    
            render_start_time = time.time()
            self.render_wrapped_impl()
            self.render_time_ms.value = (time.time() - render_start_time) * 1e3
            # Update the viser handle with the RGB image

            if count % 220 == 0:
                self.randomize_lighting()
                self.randomize_viewaug()
                self.randomize_skybox_rotation()
                # self.randomize_robot_texture()
                if success_envs is not None:
                    print(f"[INFO]: Success Envs: {success_envs}")
                    if getattr(self, 'data_logger', None) is not None:
                        self.data_logger.redir_data(success_envs)
                # reset counters
                sim_time = 0.0
                count = 0
                # reset the scene entities
                for index, robot in enumerate([self.scene.articulations['robot']]):
                    ik_commands = torch.zeros(self.scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
                    # root state
                    root_state = robot.data.default_root_state.clone()
                    root_state[:, :3] += self.scene.env_origins
                    robot.write_root_state_to_sim(root_state)
                    # # set joint positions
                    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                    robot.write_joint_state_to_sim(joint_pos, joint_vel)
                    
                for rigid_object in [self.scene.articulations['stapler']]:
                    root_state = rigid_object.data.default_root_state.clone()
                    root_state[:, :3] += self.scene.env_origins
                    random_xyz = torch.torch.randn_like(self.scene.env_origins) * 0.05
                    random_xyz = torch.where(random_xyz > 0.05, 0.05, random_xyz)
                    random_xyz[:, 2] = 0.0
                    root_state[:, :3] += random_xyz
                    # random_wxyz = torch.randn_like(root_state[:, 3:7]) * 0.05
                    # root_state[:, 3:7] = random_wxyz
                    
                    rigid_object.write_root_state_to_sim(root_state)
                self.scene.reset()
                print("[INFO]: Resetting state...")
            if count == 10:
                print("[INFO]: Beginning Push Simulation...")
            if count > 10:
                current_goal_idx = 0
                
                if count > 90 and count <= 150:
                    current_goal_idx = 1
                    
                elif count > 150:
                    current_goal_idx = 2
                                    
                # Manually set the root state of the rigid object to ignore physics
                if current_goal_idx != 4:
                    for rigid_object in [self.scene.articulations['stapler']]:
                        root_state = rigid_object.data.root_state_w
                        rigid_object.write_root_state_to_sim(root_state)
                
                for idx, robot in enumerate([self.scene.articulations['robot']]):
                    
                    # Inverse Kinematics Solver
                    obj_xyzs = rigid_object.data.root_state_w[:, :3] - self.scene.env_origins
                    obj_wxyzs = rigid_object.data.root_state_w[:, 3:7]
                    obj_new_wxyzs = reorient_quaternion_batch(obj_wxyzs.cpu().detach().numpy())
                                        
                    ik_commands = torch.zeros(self.scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
                    ik_commands[:] = torch.tensor(ee_goal_offsets[current_goal_idx])
                    ik_commands[:, :3] += obj_xyzs
                    ik_commands[:, 3:7] = torch.tensor(obj_new_wxyzs)
                    diff_ik_controller.set_command(ik_commands)
                    jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
                    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                    root_pose_w = robot.data.root_state_w[:, 0:7]
                    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                    # compute frame in root frame
                    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                    
                    # Right EE calculations (only for data saving)
                    ee_pose_w_r = robot.data.body_state_w[:, robot_entity_cfg.body_ids[1], 0:7]
                    # compute frame in root frame
                    ee_pos_b_r, ee_quat_b_r = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w_r[:, 0:3], ee_pose_w_r[:, 3:7])
                    
                    # compute the joint commands
                    joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
                    
                    # gripper override
                    joint_pos_des[:,-4:-2] = 0.0
                    
                    # apply actions
                    robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
                    robot.write_data_to_sim()
                    
                    if getattr(self, 'data_logger', None) is not None:
                        joint_names = [*robot.data.joint_names[:-3], (robot.data.joint_names[-2])]
                        robot_data = {
                            "joint_names": joint_names,
                            "joint_angles": torch.cat((robot.data.joint_pos[:, :-3], robot.data.joint_pos[:, -2].unsqueeze(-1)), dim=1).clone().cpu().detach().numpy(),
                            "ee_pos": torch.cat([ee_pos_b, ee_quat_b, ee_pos_b_r, ee_quat_b_r], dim=1).cpu().detach().numpy(),
                            # Left EE xyz, wxyz, Right EE xyz, wxyz
                        }
                        self.data_logger.save_data(self.camera_manager.buffers, robot_data, count-31, self.output_dir)
                        stats = self.data_logger.get_stats()
                        self.images_per_second.value = stats['images_per_second']
                        self.successful_envs.value = stats['total_successful_envs']
                
                for idx, rigid_object in enumerate([self.scene.articulations['stapler']]):
                    self.rigid_objects_viser_frame[idx].position = rigid_object.data.root_state_w[self.env][:3].cpu().detach().numpy() - self.scene.env_origins.cpu().numpy()[self.env]
                    self.rigid_objects_viser_frame[idx].wxyz = rigid_object.data.root_state_w[self.env][3:7].cpu().detach().numpy()
            
            # Apply randomization to robot starting pose and rigid object pose
            else: # count < 50
                for idx, robot in enumerate([self.scene.articulations['robot']]):

                    # generate random joint positions
                    joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.015
                    joint_pos_target = joint_pos_target.clamp_(
                        robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
                    )

                    # apply action to the robot
                    robot.set_joint_position_target(joint_pos_target)
                    # write data to sim
                    robot.write_data_to_sim()
                
                for idx, rigid_object in enumerate([self.scene.articulations['stapler']]):
                    self.rigid_objects_viser_frame[idx].position = rigid_object.data.root_state_w[self.env][:3].cpu().detach().numpy() - self.scene.env_origins.cpu().numpy()[self.env]
                    # self.rigid_objects_viser_frame[idx].wxyz = rigid_object.data.root_state_w[self.env][3:7].cpu().detach().numpy()
                    
            self.sim.step(render=True)
            # update sim-time
            sim_time += sim_dt
            count += 1
            # update buffers
            for name in self.urdf_vis.keys():
                robot = self.scene.articulations[name]
                robot.update(sim_dt)
                if self.init_viser:
                    joint_dict = {robot.data.joint_names[i]: robot.data.joint_pos[self.env][i].item() for i in range(len(robot.data.joint_pos[0]))} 
                    self.urdf_vis[name].update_cfg(joint_dict)
                    if self.debug_marker_vis and 'robot' in name:
                        # obtain quantities from simulation
                        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                        # update marker positions
                        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                        goal_marker.visualize(ik_commands[:, 0:3] + self.scene.env_origins, ik_commands[:, 3:7])
            if self.init_viser:
                self.isaac_viewport_camera.update(0, force_recompute=True)
            self.sim_step_time_ms.value = (time.time() - sim_start_time) * 1e3
            
