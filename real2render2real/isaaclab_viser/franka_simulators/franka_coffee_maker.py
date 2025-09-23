from dataclasses import dataclass
from pathlib import Path
import json
import torch
import numpy as onp
from typing import Dict, List, Optional, Tuple

from real2render2real.isaaclab_viser.base import IsaacLabViser
from real2render2real.isaaclab_viser.controllers.jaxmp_diff_ik_controller import JaxMPBatchedController
import real2render2real.utils.transforms as tf
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import subtract_frame_transforms
import time

from collections import deque
from copy import deepcopy

from trajgen.traj_interp import traj_interp_batch, generate_directional_starts
from trajgen.traj_resampling import generate_uniform_control_points_batch, generate_uniform_control_points

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
outputs_dir = os.path.join(dir_path, "../../../outputs")

@dataclass
class ManipulationConfig:
    """Configuration for manipulation phases"""
    setup_phase_steps: int = 6  # End of initial setup, begin trajectory
    grasp_phase_steps: int = 38  # Close Gripper at step
    release_offset_steps: int = 20  # Steps in trajectory after release
    resampled_part_deltas_length: int = None
    ee_retracts: Dict[str, float] = None
    resampled: bool = False
    
    def __post_init__(self):
        self.ee_retracts = {
            'start': -0.095,
            'grasp': -0.005,
            'release': -0.08
        }
    
    @property
    def release_phase_steps(self) -> int:
        """Calculate when to release based on trajectory length"""
        def calculate(resampled: bool = False) -> int:
            return (self.grasp_phase_steps + 
                    (self.resampled_part_deltas_length if resampled else self.part_deltas_length))
        return calculate
    
    @property
    def total_steps(self) -> int:
        """Calculate total steps in manipulation sequence"""
        def calculate(resampled: bool = False) -> int:
            return (self.release_phase_steps(resampled) + 
                    self.release_offset_steps)
        return calculate
    
    def set_part_deltas_length(self, length: int):
        """Set the length of part deltas trajectory"""
        self.part_deltas_length = length
        
    def set_resampled_part_deltas_length(self, length: int):
        """Set the length of part deltas trajectory"""
        self.resampled_part_deltas_length = length
        self.resampled = True

class ManipulationStateMachine:
    """Handles state transitions and actions for manipulation sequence"""
    
    def __init__(self, config: ManipulationConfig):
        self.config = config
        self.gripper_closed = False
        self.ee_goal_offset = [0.0, 0.0, 0.0, 0, -1, 0, 0]
        self.ee_rand_goal_offset = [0.0, 0.0]
        
    def update(self, count: int):
        """Update state machine based on current count"""
        self.gripper_closed = False
        
        if count <= self.config.setup_phase_steps:
            return
            
        # Handle height transitions
        if count > self.config.setup_phase_steps:
            self.ee_goal_offset[2] = self._interpolate_height(
                count,
                self.config.setup_phase_steps + 14,
                self.config.grasp_phase_steps - 3,
                self.config.ee_retracts['start'],
                self.config.ee_retracts['grasp']
            )
            # Apply exponential decay to make ee_rand_goal_offset approach zero faster than linear
            self.ee_rand_goal_offset[0] *= (0.95 ** (1.0 / (count - self.config.setup_phase_steps)))
            self.ee_rand_goal_offset[1] *= (0.95 ** (1.0 / (count - self.config.setup_phase_steps)))
            
            self.ee_goal_offset[0] = self.ee_goal_offset[2] * self.ee_rand_goal_offset[0]/self.config.ee_retracts['start']
            self.ee_goal_offset[1] = self.ee_goal_offset[2] * self.ee_rand_goal_offset[1]/self.config.ee_retracts['start']
            
            
        if count > self.config.grasp_phase_steps - 4:
            self.gripper_closed = True
            
        if count > self.config.release_phase_steps(self.config.resampled) + 3:
            self.ee_goal_offset[:2] = [0.0, 0.0]
            self.ee_goal_offset[2] = self._interpolate_height(
                count,
                self.config.release_phase_steps(self.config.resampled) + 5,
                self.config.total_steps(self.config.resampled),
                self.config.ee_retracts['grasp'],
                self.config.ee_retracts['release']
            )
            
        if count > self.config.release_phase_steps(self.config.resampled):
            self.gripper_closed = False
            
    def _interpolate_height(self, count: int, start_count: int, 
                          end_count: int, start_height: float, 
                          end_height: float) -> float:
        """Smoothly interpolate end-effector height"""
        if count <= start_count:
            return start_height
        elif count >= end_count:
            return end_height
        
        t = (count - start_count) / (end_count - start_count)
        t = t * t * (3 - 2 * t)  # Smoothstep interpolation
        return start_height + t * (end_height - start_height)
    
    def _randomize_ee_pose_offset(self):
        """Randomize end-effector pose offset"""
        self.ee_rand_goal_offset = ((torch.rand((2,))*2-1)*0.08).tolist()

class CoffeeMaker(IsaacLabViser):
    def __init__(self, *args, **kwargs):
        self.debug_marker_vis = False
        # TODO: Change to your own path
        self.dig_config_path = Path(f'{outputs_dir}/coffee_maker/dig/2025-03-18_154136/config.yml')
        self.ns_output_dir = self.dig_config_path.parent.parent.parent
        super().__init__(*args, **kwargs)
        
        self.load_track_data()
        self.state_machine = ManipulationStateMachine(ManipulationConfig())
        self.state_machine.config.set_part_deltas_length(self.part_deltas.shape[0])
        self.grasp_perturb = None
        self.render_wrist_cameras = False
        self.grasped_obj_loc_augment = True
        self.run_simulator()
    
    def run_simulator(self):
        """Main simulation loop"""

        self.robot_entity_cfg = SceneEntityCfg(
            "robot", 
            joint_names=[".*"], 
            body_names=["panda_hand"]
        )
        self.robot_entity_cfg.resolve(self.scene)
        
        
        self.controller = JaxMPBatchedController(
            urdf_path=self.urdf_path['robot'], 
            num_envs=self.scene.num_envs,
            num_ees=1,
            target_names=["panda_hand_joint"],
            home_pose=self.scene.articulations['robot'].data.default_joint_pos[0].cpu().detach().numpy(),
        )
        
        
        
        if self.debug_marker_vis:
            self._setup_debug_markers()
            
        count = 0
        sim_dt = self.sim.get_physics_dt()
        self.success_envs = None
        self.grasp_idx = 0
        
        while self.simulation_app.is_running() and self.successful_envs.value < 1000:
            sim_start_time = time.time()
            
            self._handle_client_connection()
            self._update_rendering(sim_start_time)
            
            # Reset if needed
            if count % self.state_machine.config.total_steps(self.state_machine.config.resampled) == 0:
                self._handle_reset()
                count = 0

            # Update state machine
            self.state_machine.update(count)
            
            # Handle object and robot states
            self._update_object_states(count)
            if count > self.state_machine.config.setup_phase_steps:
                self._handle_manipulation(count)
                self._log_data(count)
            else:
                self._handle_setup_phase(count)
            
            self._update_sim_stats(sim_start_time, sim_dt)
            count += 1

    def load_track_data(self):
        """Load trajectory and grasp data"""
        track_data_path = self.ns_output_dir / "track/keyframes.txt"
        dpt_json = self.dig_config_path.parent / 'dataparser_transforms.json'
        
        # Load scale and transforms
        dpt = json.loads(dpt_json.read_text())
        self._scale = dpt["scale"]
        
        # Load trajectory data
        data = json.loads(track_data_path.read_text())
        self.part_deltas = torch.tensor(data["part_deltas"]).cuda()
        self.part_deltas = self.part_deltas[::2,:,:]
        self.T_objreg_objinit = torch.tensor(data["T_objreg_objinit"]).cuda()
        self.T_world_objinit = torch.tensor(data["T_world_objinit"]).cuda()
        
        # Load grasp data
        self._load_grasp_data()
        
    def _load_grasp_data(self):
        """Load grasp data from rigid state directories"""
        rigid_state_dirs = sorted(
            [p for p in self.ns_output_dir.glob("state_rigid_*") if p.is_dir()],
            key=lambda x: x.stem.split('_')[3]
        )
        
        self.grasped_idxs = []
        self.grasp_data = []
        
        for idx, rigid_state_dir in enumerate(rigid_state_dirs):
            grasp_files = list(rigid_state_dir.glob("grasps.txt"))
            if grasp_files:
                self.grasped_idxs.append(idx)
                self.grasp_data.append(json.loads(grasp_files[0].read_text()))
                
        if not self.grasped_idxs:
            raise ValueError("No grasp data found in rigid state directories")
        if len(self.grasped_idxs) > 2:
            raise ValueError("More than two simultaneously grasped parts found")
    
    def _setup_viser_gui(self):
        """Setup viser GUI elements"""
        super()._setup_viser_gui()
        # Add object frame to viser
        self.rigid_objects_viser_frame = []
        for name, rigid_object in self.scene.rigid_objects.items():
            self.rigid_objects_viser_frame.append(
                self.viser_server.scene.add_frame(
                    name, 
                    position = rigid_object.data.default_root_state[self.env][:3].cpu().detach().numpy(), 
                    wxyz = rigid_object.data.default_root_state[self.env][3:7].cpu().detach().numpy(),
                    axes_length = 0.05,
                    axes_radius = 0.003,
                )
            )

    def _setup_viser_scene(self):
        """Setup viser scene elements"""
        super()._setup_viser_scene()
        self.tf_size_handle = 0.2
        self.transform_handles = {
            'ee': self.viser_server.scene.add_frame(
                f"tf_ee_env",
                axes_length=0.5 * self.tf_size_handle,
                axes_radius=0.01 * self.tf_size_handle,
                origin_radius=0.1 * self.tf_size_handle,
            ),
            # 'right': self.viser_server.scene.add_frame(
            #     f"tf_right_env",
            #     axes_length=0.5 * self.tf_size_handle,
            #     axes_radius=0.01 * self.tf_size_handle,
            #     origin_radius=0.1 * self.tf_size_handle,
            # ),
        }
        
    def wrist_cam_tf(self):
        ee2wrist_l = tf.SE3.from_rotation_and_translation(tf.SO3(torch.tensor([0, 0, 0, 1])), torch.tensor([0, 0.05, -0.075]))
        ee2wrist_r = tf.SE3.from_rotation_and_translation(tf.SO3(torch.tensor([1, 0, 0, 0])), torch.tensor([0, -0.05, -0.075]))
        ee_2wrist_rot = tf.SE3.from_rotation_and_translation(tf.SO3(torch.tensor([0.9848078, -0.1736482, 0, 0])), torch.tensor([0, 0, 0]))
        c2w_l = tf.SE3(torch.cat([self.ee_pose_w_left[:, 3:7], self.ee_pose_w_left[:, :3]], dim=-1)) @ ee2wrist_l @ ee_2wrist_rot
        c2w_r = tf.SE3(torch.cat([self.ee_pose_w_right[:, 3:7], self.ee_pose_w_right[:, :3]], dim=-1)) @ ee2wrist_r @ ee_2wrist_rot
        c2w_l_tensor = c2w_l.parameters().to(self.ee_pose_w_left.device) # wxyz_xyz
        c2w_r_tensor = c2w_r.parameters().to(self.ee_pose_w_right.device) # wxyz_xyz
        
        return c2w_l_tensor, c2w_r_tensor
    
    def render_wrapped_impl(self):
        if self.client is not None and self.use_viewport:
            if getattr(self.isaac_viewport_camera.cfg, "cams_per_env", None) is not None: # Handle batched tiled renderer for multiple cameras per environment
                repeat_n = self.scene_config.num_envs * self.isaac_viewport_camera.cfg.cams_per_env
            else:
                repeat_n = self.scene_config.num_envs
            xyz = torch.tensor(self.client.camera.position).unsqueeze(0).repeat(repeat_n, 1)
            xyz = torch.add(xyz, self.scene.env_origins.cpu().repeat_interleave(repeat_n//self.scene_config.num_envs, dim=0))
            wxyz = torch.tensor(self.client.camera.wxyz).unsqueeze(0).repeat(repeat_n, 1)
            
            self.isaac_viewport_camera.set_world_poses(xyz, wxyz, convention="ros")
            single_cam_ids = [self.isaac_viewport_camera.cfg.cams_per_env*i for i in list(range(self.scene_config.num_envs))]
            cam_out = {}
            for key in self.isaac_viewport_camera.data.output.keys():
                cam_out[key] = self.isaac_viewport_camera.data.output[key][single_cam_ids]
            self.camera_manager.buffers['camera_0'].append(cam_out)
        if not self.use_viewport and len(self.camera_manager.frustums) > 0: # Batched Rendering for MultiTiledCameraCfg
            if getattr(self.isaac_viewport_camera.cfg, "cams_per_env", None) is not None:
                repeat_n = self.scene_config.num_envs * self.isaac_viewport_camera.cfg.cams_per_env
                if len(self.camera_manager.frustums) > self.isaac_viewport_camera.cfg.cams_per_env:
                    raise ValueError(f"Using batched rendering. Not allowed to set a number of frustums exceeding config setting cams_per_env of {self.isaac_viewport_camera.cfg.cams_per_env}")
                else:
                    if hasattr(self, 'ee_pose_w_left') and hasattr(self, 'ee_pose_w_right') and self.render_wrist_cameras: # Wrist camera pose updating
                        c2w_l, c2w_r = self.wrist_cam_tf()
                        
                        if self.isaac_viewport_camera.cfg.cams_per_env > 2: # Brittle + not generalizable, hardcoded to handle 2 gripper cam + 1 ego view
                            if len(self.camera_manager.frustums) == self.isaac_viewport_camera.cfg.cams_per_env:
                                ego_cam_id = 2
                                xyz = torch.tensor(self.camera_manager.frustums[ego_cam_id].position)
                                wxyz = torch.tensor(self.camera_manager.frustums[ego_cam_id].wxyz)
                                c2w_ego = torch.cat([wxyz, xyz], dim=-1).repeat(self.scene_config.num_envs, 1).to(c2w_l.device)
                                c2w_ego[:, 4:] = torch.add(c2w_ego[:, 4:], self.scene.env_origins)
                                interleaved = torch.stack([c2w_l, c2w_r, c2w_ego], dim=1).reshape(-1, 7)
                            else:
                                interleaved = torch.stack([c2w_l, c2w_r, c2w_r], dim=1).reshape(-1, 7)
                        else:
                            interleaved = torch.stack([c2w_l, c2w_r], dim=1).reshape(-1, 7)
                        xyzs = interleaved[:, 4:] 
                        wxyzs = interleaved[:, :4] 
                        
                        self.isaac_viewport_camera.set_world_poses(xyzs, wxyzs, convention="ros")
                        indices = [i * self.isaac_viewport_camera.cfg.cams_per_env + j for i in range(self.scene_config.num_envs) for j in range(len(self.camera_manager.frustums))]
                        cam_out = {}
                        for key in self.isaac_viewport_camera.data.output.keys():
                            cam_out[key] = self.isaac_viewport_camera.data.output[key][indices]

                        for idx, frustum in enumerate(self.camera_manager.frustums):
                            frustum_data = {}
                            for key in cam_out.keys():
                                frustum_data[key] = cam_out[key][idx::len(self.camera_manager.frustums)]
                            buffer_key = frustum.name[1:]
                            if buffer_key not in self.camera_manager.buffers.keys():
                                self.camera_manager.buffers[buffer_key] = deque(maxlen=1)
                            self.camera_manager.buffers[buffer_key].append(deepcopy(frustum_data)) # TODO: Check if removing deepcopy breaks things

                    else:
                        xyzs = []
                        wxyzs = []
                        for camera_frustum in self.camera_manager.frustums:
                            xyzs.append(camera_frustum.position)
                            wxyzs.append(camera_frustum.wxyz)
                        for i in range(self.isaac_viewport_camera.cfg.cams_per_env-len(xyzs)): # Fill up with shape[0]==cams_per_env since a pose must be given every set_world_pose call for every camera
                            xyzs.append(xyzs[-1])
                            wxyzs.append(wxyzs[-1])
                        xyzs = torch.tensor(onp.array(xyzs)).repeat(self.scene_config.num_envs, 1)
                        wxyzs = torch.tensor(onp.array(wxyzs)).repeat(self.scene_config.num_envs, 1)
                        xyzs = torch.add(xyzs, self.scene.env_origins.cpu().repeat_interleave(repeat_n//self.scene_config.num_envs, dim=0))
                        self.isaac_viewport_camera.set_world_poses(xyzs, wxyzs, convention="ros")
                        indices = [i * self.isaac_viewport_camera.cfg.cams_per_env + j for i in range(self.scene_config.num_envs) for j in range(len(self.camera_manager.frustums))]
                        cam_out = {}
                        for key in self.isaac_viewport_camera.data.output.keys():
                            cam_out[key] = self.isaac_viewport_camera.data.output[key][indices]
                        
                        for idx, frustum in enumerate(self.camera_manager.frustums):
                            frustum_data = {}
                            for key in cam_out.keys():
                                frustum_data[key] = cam_out[key][idx::len(self.camera_manager.frustums)]
                            buffer_key = frustum.name[1:]
                            if buffer_key not in self.camera_manager.buffers.keys():
                                self.camera_manager.buffers[buffer_key] = deque(maxlen=1)
                            self.camera_manager.buffers[buffer_key].append(deepcopy(frustum_data)) # TODO: Check if removing deepcopy breaks things
                            # frustum.image = self.camera_manager.buffers[buffer_key][0]["rgb"][self.env].clone().cpu().detach().numpy()
        if self.init_viser and self.client is not None:
            if len(self.camera_manager.buffers[self.camera_manager.render_cam]) > 0:
                self.isaac_viewport_viser_handle.image = self.camera_manager.buffers[self.camera_manager.render_cam][0]["rgb"][self.env].clone().cpu().detach().numpy()
        
        self.sim.render()
        return

    def _setup_debug_markers(self):
        """Setup visualization markers for debugging"""
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_current")
        )
        self.goal_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_goal")
        )

    def _handle_client_connection(self):
        """Handle client connection setup"""
        if self.client is None:
            while self.client is None:
                self.client = (self.viser_server.get_clients()[0] 
                             if len(self.viser_server.get_clients()) > 0 
                             else None)
                time.sleep(0.1)

    def _update_rendering(self, sim_start_time: float):
        """Update rendering and timing"""
        render_start_time = time.time()
        
        if hasattr(self, 'robot'):
            self._update_ee_poses()
            
        self.render_wrapped_impl()
        self.render_time_ms.value = (time.time() - render_start_time) * 1e3

    def _update_ee_poses(self):
        """Update end effector poses"""
        self.ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        # self.ee_pose_w_right = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[1], 0:7]

    def randomize_grasp_rot(self):
        """Add small perturbation to grasp rotation"""
        grasp_perturb_x = torch.randn((1,)) * 0.05
        grasp_perturb_y = torch.randn((1,)) * 0.05
        
        # Create perturbation transform around x-axis
        perturb_tf_x = tf.SE3.from_rotation(
            tf.SO3.from_x_radians(grasp_perturb_x.to(self.scene.env_origins.device))
        )
        perturb_tf_y = tf.SE3.from_rotation(
            tf.SO3.from_y_radians(grasp_perturb_y.to(self.scene.env_origins.device))
        )
        self.grasp_perturb = perturb_tf_x @ perturb_tf_y
    
    def _handle_reset(self):
        """Handle simulation reset and environment randomization"""
        if self.success_envs is not None:
            print(f"[INFO]: Success Envs: {self.success_envs}")
            if hasattr(self, 'data_logger'):
                self.data_logger.redir_data(self.success_envs)
                
        self._reset_robot_state()
        self._reset_object_state()
        self._update_object_states(0)
        self.scene.reset()
        self.controller.reset()
        
        self.state_machine._randomize_ee_pose_offset()
        
        self.randomize_skybox()
        # TODO: figure out how ppl should be installing mdls
        self.randomize_table()
        
        self.randomize_lighting()
        self.randomize_viewaug()
        self.randomize_grasp_rot()
        self.randomize_skybox_rotation()
        
        self.grasp_idx = torch.randint(len(self.grasp_data[0]), (1,)).item() # Randomly select a saved grasp from the grasp data
        print("[INFO]: Resetting state...")
        self.success_envs = torch.ones((self.scene.num_envs,), device=self.scene.env_origins.device, dtype=bool)
        
    def _reset_robot_state(self):
        """Reset robot to initial state"""
        for robot in self.scene.articulations.values():
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += self.scene.env_origins
            robot.write_root_state_to_sim(root_state)
            
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

    def _reset_object_state(self):
        """Reset objects with randomization"""
        random_xyz = (torch.rand_like(self.scene.env_origins) * 2 - 1) * 0.06
        random_xyz[:, 2] = 0.0
        
        z_rot = torch.rand((self.scene.num_envs,)) * 0.35 + onp.pi/2 - 0.1 # 0.35 #* 0.8 + onp.pi/2 - 0.4
        
        random_z_rot = tf.SO3.from_z_radians(
            torch.tensor(z_rot, device=self.scene.env_origins.device)
        )
            
        self.parts_init_state = {}
        for group_idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
            base_transform = (tf.SE3(self.T_world_objinit[0:1]) @ 
                            tf.SE3(self.T_objreg_objinit[0:1]))
            if self.grasped_obj_loc_augment and group_idx in self.grasped_idxs: # Object init randomization + Object trajectory interpolation
                new_starts = generate_directional_starts(self.part_deltas[:, group_idx], self.scene.num_envs, magnitude=0.1, direction_weight=0.7, perp_variation=0.10)
                
                self.new_trajs = traj_interp_batch(
                    traj=self.part_deltas[:, group_idx].cpu().detach().numpy(),
                    new_starts=new_starts.cpu().detach().numpy(),
                    proportion=0.6
                )

                self.new_trajs = generate_uniform_control_points_batch(
                    self.new_trajs,
                    proportion=1.1, # 10% more control points than original
                    tension=0.1      
                )

                self.state_machine.config.set_resampled_part_deltas_length(self.new_trajs.shape[1])
                                
                self.new_starts = torch.tensor(self.new_trajs[:, 0]).to(self.T_world_objinit.device) 


            current_transform = (tf.SE3(wxyz_xyz=self.T_world_objinit[group_idx]) @ 
                            tf.SE3(wxyz_xyz=self.T_objreg_objinit[group_idx]))
            p2o_transform = base_transform.inverse() @ current_transform
            p2o_7vec = tf.SE3(
                p2o_transform.wxyz_xyz[0].unsqueeze(0).repeat(
                    self.scene.num_envs, 1
                ) 
            )
            p2o_7vec.wxyz_xyz[:, 4:] /= self._scale
            
            part_inits = tf.SE3.from_rotation(random_z_rot) @ p2o_7vec
            objects_root_state = rigid_object.data.default_root_state.clone()
            objects_root_state[:, :3] += (self.scene.env_origins + 
                                        part_inits.wxyz_xyz[:, 4:] + random_xyz)
            objects_root_state[:, 3:7] = part_inits.wxyz_xyz[:, :4]
            
            rigid_object.write_root_state_to_sim(objects_root_state)
            self.parts_init_state[
                list(self.scene.rigid_objects.keys())[group_idx]
            ] = rigid_object.data.root_state_w.clone()[:, :7]

    def _update_object_states(self, count: int):
        """Update object states based on trajectory"""
        for group_idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
            objects_root_state = torch.zeros_like(rigid_object.data.root_state_w)
            xyz_wxyz_init = self.parts_init_state[
                list(self.scene.rigid_objects.keys())[group_idx]
            ]
            wxyz_xyz_init = torch.cat(
                [xyz_wxyz_init[:, 3:7], xyz_wxyz_init[:, :3]], dim=-1
            )
            obj_init = tf.SE3(wxyz_xyz_init)
            
            # Apply appropriate delta based on current phase
            if (count >= self.state_machine.config.grasp_phase_steps and 
                count < self.state_machine.config.release_phase_steps(self.state_machine.config.resampled)):
                if self.grasped_obj_loc_augment and group_idx in self.grasped_idxs:
                    traj = torch.tensor(self.new_trajs[:, count - self.state_machine.config.grasp_phase_steps], device = self.part_deltas.device)
                    traj[:, 4:] /= self._scale
                    part_delta = tf.SE3(torch.tensor(traj))
                else:
                    traj = self.part_deltas[min(count - self.state_machine.config.grasp_phase_steps, self.part_deltas.shape[0]-1), group_idx].unsqueeze(0).repeat(self.scene.num_envs, 1)
                    traj[:, 4:] /= self._scale
                    part_delta = tf.SE3(traj)
            elif count >= self.state_machine.config.release_phase_steps(self.state_machine.config.resampled):
                if self.grasped_obj_loc_augment and group_idx in self.grasped_idxs:
                    traj = torch.tensor(self.new_trajs[:, -1], device = self.part_deltas.device)
                    traj[:, 4:] /= self._scale
                    part_delta = tf.SE3(traj)
                else:
                    traj = self.part_deltas[-1, group_idx].unsqueeze(0).repeat(self.scene.num_envs, 1)
                    traj[:, 4:] /= self._scale
                    part_delta = tf.SE3(traj)
            else:
                if self.grasped_obj_loc_augment and group_idx in self.grasped_idxs:
                    traj = torch.tensor(self.new_trajs[:, 0], device = self.part_deltas.device)
                    traj[:, 4:] /= self._scale
                    part_delta = tf.SE3(traj)
                else:
                    traj = self.part_deltas[0, group_idx].unsqueeze(0).repeat(self.scene.num_envs, 1)
                    traj[:, 4:] /= self._scale
                    part_delta = tf.SE3(traj)
            
            obj_state = obj_init @ part_delta
            objects_root_state[:, :3] = obj_state.wxyz_xyz[:, 4:]
            objects_root_state[:, 3:7] = obj_state.wxyz_xyz[:, :4]
            
            rigid_object.write_root_state_to_sim(objects_root_state)
            self._update_object_visualization(rigid_object, group_idx)

    def _update_object_visualization(self, rigid_object, idx: int):
        """Update object visualization in viser"""
        if not self.init_viser:
            return
            
        self.rigid_objects_viser_frame[idx].position = (
            rigid_object.data.root_state_w[self.env][:3].cpu().detach().numpy() - 
            self.scene.env_origins.cpu().numpy()[self.env]
        )
        self.rigid_objects_viser_frame[idx].wxyz = (
            rigid_object.data.root_state_w[self.env][3:7].cpu().detach().numpy()
        )

    def _handle_manipulation(
        self, 
        count: int,
    ):
        """Handle manipulation phase of simulation"""
        # Check success conditions at appropriate time
        # if count == self.state_machine.config.release_phase_steps - 6:
            # self.success_envs = self._check_success_conditions()
            
        # Update robot state
        for robot in self.scene.articulations.values():
            self.robot = robot
            joint_pos_des = self._update_robot_manipulation(
                count
            )
            
        # Update visualization and simulation
        self.sim.step(render=False)
        # Check if z-position is ever too low for either arm
        left_z_pos = self.ee_pose_w[:, 2]
        # right_z_pos = self.ee_pose_w_right[:, 2]
        z_pos_error_mask = (left_z_pos < 0.025) #| (right_z_pos < 0.025)
        if self.success_envs is None:
            self.success_envs = ~z_pos_error_mask
        else:
            self.success_envs = self.success_envs & ~z_pos_error_mask
        
        # Special handling for grasp phase (override joint angles instead of applying command to controller)
        if (count >= self.state_machine.config.grasp_phase_steps and 
            count < self.state_machine.config.release_phase_steps(self.state_machine.config.resampled)):
            self._handle_grasp_phase(joint_pos_des)
            self._target_state_error()
            self._joint_pos_limit_check()
        elif count == self.state_machine.config.grasp_phase_steps - 1:
            self._target_state_error()
    
    def _update_robot_manipulation(
        self,
        count: int,
    ):
        """Update robot state during manipulation"""
        rigid_object = self.scene.rigid_objects[
            list(self.scene.rigid_objects.keys())[self.grasped_idxs[0]]
        ]
        rigid_object.update(self.sim.get_physics_dt())
        
        # if count == self.state_machine.config.setup_phase_steps + 1:
            # TODO: This is scuffed, check why it's not working or for now we can set all to right hand
            # root_state_b = rigid_object.data.root_state_w[:, :3] + self.scene.env_origins
            # self.left_hand_envs = root_state_b[:, 1] > 0.8
            # self.left_hand_envs = torch.zeros((self.scene.num_envs,), dtype=bool, device=self.scene.env_origins.device)
        
        # Calculate target poses
        target_poses = self._calculate_target_poses(count,
            rigid_object
        )
        
        # Update transform handles
        self._update_transform_handles(target_poses)
        
        # Calculate and apply joint positions
        joint_pos_des = self._ik_wrapped(target_poses)
        
        # Apply actions at appropriate times
        if self._should_apply_actions(count):
            self.robot.set_joint_position_target(
                joint_pos_des, 
                joint_ids=self.robot_entity_cfg.joint_ids
            )
            self.robot.write_data_to_sim()
        return joint_pos_des

    def _target_state_error(
        self,
        ):
        # # Calculate error between target and current poses
        target_poses_tensor = torch.tensor(self.target_poses, device=self.ee_pose_w.device)

        target = target_poses_tensor[:,0,:] # Shape: (num_envs, 7)
        current = torch.cat([self.ee_pose_w[:,:3] - self.scene.env_origins, self.ee_pose_w[:,3:7]], dim=-1)

        # Calculate position and orientation errors separately
        pos_error_threshold = 0.02  # Position error threshold in meters
        quat_error_threshold = 0.45  # Quaternion error threshold in radians

        # Position errors (Euclidean distance)
        pos_error = (target[:,:3] - current[:,:3]).norm(dim=-1)
        quat_error = 2 * torch.arccos(torch.abs(torch.sum(target[:,3:7] * current[:,3:7], dim=-1)).clamp(-1, 1))
        

        self.target_state_errors = torch.stack([
            pos_error, quat_error,
        ], dim=-1)

        # Create separate masks for each type of error
        pos_error_mask = (pos_error > pos_error_threshold)
        quat_error_mask = (quat_error > quat_error_threshold)
        
        error_mask = pos_error_mask | quat_error_mask

        # Mark environments as failed if error exceeds threshold
        if self.success_envs is None:
            self.success_envs = ~error_mask
        else:
            self.success_envs = self.success_envs & ~error_mask
        if error_mask.any().item():
            exceeded_envs = torch.where(error_mask)[0]
            print(f"Target state error exceeded in environments: {exceeded_envs.cpu().numpy().tolist()}. Marking environments as failed.")
            
    def _joint_pos_limit_check(self):
        """Check if joint positions are beyond limits and mark environments as failed if any limits are exceeded"""
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

        joint_pos_limits = self.robot.data.joint_pos_limits[:, self.robot_entity_cfg.joint_ids]
        
        # Check if joint positions exceed lower or upper limits
        joint_pos_below_limit = joint_pos < joint_pos_limits[..., 0]  # Check lower bounds
        joint_pos_above_limit = joint_pos > joint_pos_limits[..., 1]  # Check upper bounds
        
        # Combine masks for any joint exceeding limits
        joint_limit_exceeded = torch.any(joint_pos_below_limit | joint_pos_above_limit, dim=1)

        # Mark environments as failed if any joint limits are exceeded
        if self.success_envs is None:
            self.success_envs = ~joint_limit_exceeded
        else:
            self.success_envs = self.success_envs & ~joint_limit_exceeded
            
        if joint_limit_exceeded.any().item():
            exceeded_envs = torch.where(joint_limit_exceeded)[0]
            print(f"URDF Joint limits exceeded in environments: {exceeded_envs.cpu().numpy().tolist()}. Marking environments as failed.")
    
    def _calculate_target_poses(
        self, 
        count: int,
        rigid_object,
    ) -> onp.ndarray:
        """Calculate target poses for robot end effectors"""
        obj_xyzs = rigid_object.data.root_state_w[:, :3] - self.scene.env_origins
        obj_wxyzs = rigid_object.data.root_state_w[:, 3:7]
        
        # Get current end effector poses
        ee_pos_b, ee_quat_b = self._get_ee_poses()
        # ee_pos_b_right, ee_quat_b_right = self._get_ee_poses(
        #     'right'
        # )
        if count == self.state_machine.config.setup_phase_steps + 1:
            self.first_ee_pos_b = ee_pos_b.clone()
            # self.first_ee_pos_b_right = ee_pos_b_right.clone()
        
        obj_tf = tf.SE3(torch.cat([obj_wxyzs, obj_xyzs], dim=-1))
        grasp_tf = tf.SE3(torch.cat([torch.tensor(self.grasp_data[0][self.grasp_idx]['orientation']), torch.tensor(self.grasp_data[0][self.grasp_idx]['position'])], dim=-1).to(self.scene.env_origins.device).unsqueeze(0))
        
        grasp_90_z_tf = tf.SE3.from_rotation(tf.SO3.from_z_radians(torch.tensor(onp.pi/2).to(self.scene.env_origins.device)))
        grasp_tcp_tf = tf.SE3.from_rotation_and_translation(tf.SO3(torch.tensor([1, 0, 0, 0]).unsqueeze(0).to(self.scene.env_origins.device)), torch.tensor([0, 0, -0.1]).unsqueeze(0).to(self.scene.env_origins.device))
        
        goal_offset_tf = tf.SE3.from_rotation_and_translation(tf.SO3(torch.tensor([1, 0, 0, 0]).unsqueeze(0).to(self.scene.env_origins.device)), torch.tensor(self.state_machine.ee_goal_offset[:3]).unsqueeze(0).to(self.scene.env_origins.device))
        
        if self.grasp_perturb is not None:
            if (count >= self.state_machine.config.grasp_phase_steps and 
                count < self.state_machine.config.release_phase_steps(self.state_machine.config.resampled)):
                grasp_perturb_x = torch.randn((1,)) * 0.003
                grasp_perturb_y = torch.randn((1,)) * 0.003
                
                # Create perturbation transform around x-axis
                perturb_tf_x = tf.SE3.from_rotation(
                    tf.SO3.from_x_radians(grasp_perturb_x.to(self.scene.env_origins.device))
                )
                perturb_tf_y = tf.SE3.from_rotation(
                    tf.SO3.from_y_radians(grasp_perturb_y.to(self.scene.env_origins.device))
                )

                perturbed_grasp_tf = grasp_tf @ self.grasp_perturb @ perturb_tf_x @ perturb_tf_y # Per timestep grasp perturbation
            else:
                perturbed_grasp_tf = grasp_tf @ self.grasp_perturb
            ee_target = obj_tf @ perturbed_grasp_tf @ goal_offset_tf @ grasp_90_z_tf @ grasp_tcp_tf
        else:
            ee_target = obj_tf @ grasp_tf @ goal_offset_tf @ grasp_90_z_tf @ grasp_tcp_tf
            
            
        command_xyzs = ee_target.wxyz_xyz[:, 4:]
        command_wxyzs = ee_target.wxyz_xyz[:, :4]
        
        # Create target poses array
        target_poses = onp.zeros((self.scene.num_envs, 1, 7))
        target_poses[:,0,:] = torch.cat([command_xyzs, command_wxyzs], dim=1).cpu().detach().numpy()
        # target_poses[:,0,:] = torch.where(
        #     # self.left_hand_envs.unsqueeze(0).T.repeat(1,7),
        #     torch.cat([command_xyzs, command_wxyzs], dim=1),
        #     torch.cat([ee_pos_b, ee_quat_b], dim=1)
        # ).cpu().detach().numpy()
        # target_poses[:,1,:] = torch.where(
        #     ~self.left_hand_envs.unsqueeze(0).T.repeat(1,7),
        #     torch.cat([command_xyzs, command_wxyzs], dim=1),
        #     torch.cat([ee_pos_b_right, ee_quat_b_right], dim=1)
        # ).cpu().detach().numpy()
        
        self.target_poses = target_poses
        self._update_ee_poses()
        
        return target_poses

    def _get_ee_poses(
        self, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get end effector poses in base frame"""
        ee_pose = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose = self.robot.data.root_state_w[:, 0:7]
        
        return subtract_frame_transforms(
            root_pose[:, 0:3],
            root_pose[:, 3:7],
            ee_pose[:, 0:3],
            ee_pose[:, 3:7]
        )

    def _update_transform_handles(self, target_poses: onp.ndarray):
        """Update transform handles visualization"""
        self.transform_handles['ee'].position = target_poses[self.env,0,:3]
        self.transform_handles['ee'].wxyz = target_poses[self.env,0,3:7]

    def _ik_wrapped(
        self,
        target_poses: onp.ndarray,
    ) -> torch.Tensor:
        """Calculate desired joint positions"""
        joints_jmp = self.controller.compute_ik(target_poses)
        joints = onp.array(joints_jmp)
        
        # Map joints from JaxMP to IsaacLab convention
        # jaxmp2isaaclab_joint_mapping = [7, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6]
        # joint_desired = onp.array(joints[:, jaxmp2isaaclab_joint_mapping], dtype=onp.float32)
        
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        joint_pos_des = torch.zeros_like(joint_pos, device=self.robot.device)
        joint_pos_des[:, 0:7] = torch.tensor(joints[:, :-1])
        
        # Handle gripper positions
        self._set_gripper_positions(joint_pos_des)
        
        return joint_pos_des

    def _set_gripper_positions(self, joint_pos_des: torch.Tensor):
        """Set gripper joint positions based on state"""
        if self.state_machine.gripper_closed:
            joint_pos_des[:,-2:] = self.grasp_data[0][self.grasp_idx]['width']
        else:
            joint_pos_des[:,-2:] = 0.04

    def _should_apply_actions(self, count: int) -> bool:
        """Determine if actions should be applied based on current count"""
        setup_complete = count > self.state_machine.config.setup_phase_steps # + 7
        pre_grasp = count < self.state_machine.config.grasp_phase_steps
        post_release = count >= self.state_machine.config.release_phase_steps(self.state_machine.config.resampled)
        return (setup_complete and pre_grasp) or post_release

    def _handle_grasp_phase(self, joint_pos_des: torch.Tensor):
        """Handle special requirements during grasp phase"""
        # joint_pos_des = self.robot.data.joint_pos_des[:, self.robot_entity_cfg.joint_ids]
        joint_vel = torch.zeros_like(
            self.robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids]
        ).to(self.robot.device)
        self.robot.write_joint_state_to_sim(joint_pos_des, joint_vel)

    def _handle_setup_phase(self, count: int):
        """Handle setup phase of simulation"""
        if count < 3: # self.state_machine.config.setup_phase_steps - 10:
            self._random_perturb_config()
        
        for idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
            rigid_object.update(self.sim.get_physics_dt())
            self._update_object_visualization(rigid_object, idx)
            
        self.sim.step(render=False)

    def _random_perturb_config(self):
        """Apply random perturbations during setup"""
        for robot in self.scene.articulations.values():
            self.robot = robot
            joint_pos_target = (self.robot.data.default_joint_pos + 
                              torch.randn_like(self.robot.data.joint_pos) * 0.015)
            joint_pos_target = joint_pos_target.clamp_(
                self.robot.data.soft_joint_pos_limits[..., 0],
                self.robot.data.soft_joint_pos_limits[..., 1]
            )
            joint_pos_target[:,-4:] = 0.04  # Override Gripper to Open
            self.robot.set_joint_position_target(joint_pos_target)
            self.robot.write_data_to_sim()
            
        for group_idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
            objects_root_state = torch.zeros_like(rigid_object.data.root_state_w)
            objects_root_state[:, :7] = self.parts_init_state[
                list(self.scene.rigid_objects.keys())[group_idx]
            ]
            rigid_object.write_root_state_to_sim(objects_root_state)

    def _update_sim_stats(self, sim_start_time: float, sim_dt: float):
        """Update simulation statistics and visualization"""
        if self.init_viser:
            for name in self.urdf_vis.keys():
                self.robot = self.scene.articulations[name]
                self.robot.update(sim_dt)
                
                joint_dict = {
                    self.robot.data.joint_names[i]: 
                    self.robot.data.joint_pos[self.env][i].item() 
                    for i in range(len(self.robot.data.joint_pos[0]))
                }
                self.urdf_vis[name].update_cfg(joint_dict)
                
                if self.debug_marker_vis:
                    self._update_debug_markers()
                    
            self.isaac_viewport_camera.update(0, force_recompute=True)
            
        self.sim_step_time_ms.value = (time.time() - sim_start_time) * 1e3
    
    def _log_data(self, count: int):
        """Log data during manipulation"""
        if not hasattr(self, 'data_logger'):
            return
        
        joint_names = [*self.robot.data.joint_names[:-2], (self.robot.data.joint_names[-2])]
        ee_pos_b, ee_quat_b = self._get_ee_poses()
        
        robot_data = {
            "joint_names": joint_names,
            "joint_angles": torch.cat((
                self.robot.data.joint_pos[:, :-2], 
                self.robot.data.joint_pos[:, -2].unsqueeze(-1) # rescale gripper proprioception to log closed as 1 and open as 0 (continuous values as opposed to binary action)
            ), dim=1).clone().cpu().detach().numpy(),
            "ee_pos": torch.cat([
                ee_pos_b, ee_quat_b, 
            ], dim=1).cpu().detach().numpy(),
            "gripper_binary_cmd": torch.cat([torch.tensor(self.state_machine.gripper_closed).repeat(self.scene.num_envs).to(self.robot.device)], dim=-1).cpu().detach().numpy()[:,None]
        }
        
        self.data_logger.save_data(
            self.camera_manager.buffers,
            robot_data, 
            count - self.state_machine.config.setup_phase_steps - 1,
            self.output_dir
        )
        
        stats = self.data_logger.get_stats()
        self.save_time_ms.value = int(stats["save_time"]*1e3)
        self.images_per_second.value = stats['images_per_second']
        self.successful_envs.value = stats['total_successful_envs']
    
    def _update_debug_markers(self):
        """Update debug visualization markers"""
        # if hasattr(self, 'left_hand_envs'):
        #     ee_pose_w_agg = torch.where(
        #         self.left_hand_envs.unsqueeze(0).T.repeat(1,7),
        #         self.ee_pose_w_left,
        #         self.ee_pose_w_right
        #     )
        # else:
        if hasattr(self, 'ee_pose_w'):
            ee_pose_w_agg = self.ee_pose_w
                
            ik_commands = torch.zeros(
                self.scene.num_envs, 7, device=self.robot.device
            )
            
            self.ee_marker.visualize(
                ee_pose_w_agg[:, 0:3],
                ee_pose_w_agg[:, 3:7]
            )
            self.goal_marker.visualize(
                ik_commands[:, 0:3] + self.scene.env_origins,
                ik_commands[:, 3:7]
            )
