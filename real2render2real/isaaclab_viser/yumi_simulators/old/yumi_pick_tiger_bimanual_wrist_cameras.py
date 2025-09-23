from xi.isaaclab_viser.base import IsaacLabViser

import torch

from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, apply_delta_pose
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

from xi.utils.math import reorient_quaternion_batch, wrap_to_pi, slerp_with_clip
import time
from collections import deque
from copy import deepcopy
import numpy as np

from collections import deque
from copy import deepcopy

class PickTiger(IsaacLabViser):
    def __init__(self, *args, **kwargs):
        self.debug_marker_vis = False
        super().__init__(*args, **kwargs)
        
    def _setup_viser_gui(self):
        super()._setup_viser_gui()
        # Add object frame to viser
        self.rigid_objects_viser_frame = []
        for name, rigid_object in self.scene.rigid_objects.items():
            self.rigid_objects_viser_frame.append(
                self.viser_server.scene.add_frame(
                    name, 
                    position = rigid_object.data.default_root_state[self.env][:3].cpu().detach().numpy(), 
                    wxyz = rigid_object.data.default_root_state[self.env, 3:7].cpu().detach().numpy(),
                    axes_length = 0.05,
                    axes_radius = 0.003,
                    )
                )
    
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
                    import pdb; pdb.set_trace()
                    
                    xyzs = []
                    wxyzs = []
                    for camera_frustum in self.camera_manager.frustums:
                        xyzs.append(camera_frustum.position)
                        wxyzs.append(camera_frustum.wxyz)
                    for i in range(self.isaac_viewport_camera.cfg.cams_per_env-len(xyzs)): # Fill up with shape[0]==cams_per_env since a pose must be given every set_world_pose call for every camera
                        xyzs.append(xyzs[-1])
                        wxyzs.append(wxyzs[-1])
                    xyzs = torch.tensor(np.array(xyzs)).repeat(self.scene_config.num_envs, 1)
                    wxyzs = torch.tensor(np.array(wxyzs)).repeat(self.scene_config.num_envs, 1)
                    xyzs = torch.add(xyzs, self.scene.env_origins.cpu().repeat_interleave(repeat_n//self.scene_config.num_envs, dim=0))
                    self.isaac_viewport_camera.set_world_poses(xyzs, wxyzs, convention="ros")
                    indices = [i * self.isaac_viewport_camera.cfg.cams_per_env + j for i in range(self.scene_config.num_envs) for j in range(len(self.camera_manager.frustums))]
                    cam_out = {}
                    for key in self.isaac_viewport_camera.data.output.keys():
                        cam_out[key] = self.isaac_viewport_camera.data.output[key][indices]
                    
                    for idx, frustum in enumerate(self.camera_manager.frustums):
                        # frustum_indices = indices[idx::len(self.camera_manager.frustums)]
                        frustum_data = {}
                        for key in cam_out.keys():
                            frustum_data[key] = cam_out[key][idx::len(self.camera_manager.frustums)]
                        buffer_key = frustum.name[1:]
                        if buffer_key not in self.camera_manager.buffers.keys():
                            self.camera_manager.buffers[buffer_key] = deque(maxlen=1)
                        self.camera_manager.buffers[buffer_key].append(deepcopy(frustum_data)) # TODO: Check if removing deepcopy breaks things
                        frustum.image = self.camera_manager.buffers[buffer_key][0]["rgb"][self.env].clone().cpu().detach().numpy()
        if self.init_viser and self.client is not None:
            if len(self.camera_manager.buffers[self.camera_manager.render_cam]) > 0:
                self.isaac_viewport_viser_handle.image = self.camera_manager.buffers[self.camera_manager.render_cam][0]["rgb"][self.env].clone().cpu().detach().numpy()
        return
    
    def run_simulator(self):
        """Runs the simulation loop."""
        
        # Create controller
        controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
        controller = DifferentialIKController(controller_cfg, num_envs=self.scene.num_envs, device=self.sim.device)
    
        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
        print(f"[INFO]: Simulation timestep: {sim_dt}")
        sim_time = 0.0
        count = 0
        success_envs = None
        
        gripper_close = False
        
        ee_goal_offset = [0.0, 0.0, 0.0, 0, -1, 0, 0]
        
        state_machine_keyframe = [
            40, # End of tiger drop, begin trajectory
            110, # Close Gripper
            160, # Open Gripper
            166 # End of trajectory
        ]
        
        def get_ee_height(count, start_count, end_count, start_height, end_height):
            """
            Smoothly interpolate end-effector height based on simulation count.
            
            Args:
                count (int): Current simulation count
                
            Returns:
                float: Interpolated height value
            """
            
            # Before movement starts
            if count <= start_count:
                return start_height
            # After movement ends
            elif count >= end_count:
                return end_height
            # During movement - smooth interpolation
            else:
                # Normalize time between 0 and 1
                t = (count - start_count) / (end_count - start_count)
                # Use smooth step function for easing
                t = t * t * (3 - 2 * t)  # Smoothstep interpolation
                return start_height + t * (end_height - start_height)
    
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["left_dummy_point", "right_dummy_point"])
        robot_entity_cfg.resolve(self.scene)
        ee_jacobi_idx_left = robot_entity_cfg.body_ids[0] - 1
        ee_jacobi_idx_right = robot_entity_cfg.body_ids[1] - 1
        
        # Visualization Markers
        if self.debug_marker_vis:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
            goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
                
        # Simulate physics
        while self.simulation_app.is_running() and self.successful_envs.value < 1000:
            sim_start_time = time.time()

            if self.client is None:
                while self.client is None:
                    self.client = self.viser_server.get_clients()[0] if len(self.viser_server.get_clients()) > 0 else None # Get the first client
                    time.sleep(0.1)
            
            render_start_time = time.time()
            self.render_wrapped_impl()
            self.render_time_ms.value = (time.time() - render_start_time) * 1e3
            # Update the viser handle with the RGB image

            if count % state_machine_keyframe[-1] == 0:
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
                for index, robot in enumerate(self.scene.articulations.values()):
                    ik_commands = torch.zeros(self.scene.num_envs, 7, device=robot.device)
                    # root state
                    root_state = robot.data.default_root_state.clone()
                    root_state[:, :3] += self.scene.env_origins
                    robot.write_root_state_to_sim(root_state)
                    # set joint positions
                    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                    robot.write_joint_state_to_sim(joint_pos, joint_vel)
                    
                for rigid_object in self.scene.rigid_objects.values():
                    root_state = rigid_object.data.default_root_state.clone()
                    root_state[:, :3] += self.scene.env_origins
                    random_xyz = torch.torch.randn_like(self.scene.env_origins) * 0.1
                    # random_xyz = torch.where(random_xyz > 0.07, 0.07, random_xyz)
                    random_xyz[:, 2] = 0.0
                    root_state[:, :3] += random_xyz
                    random_wxyz = torch.randn_like(root_state[:, 3:7]) * 0.05
                    root_state[:, 3:7] = random_wxyz
                    
                    rigid_object.write_root_state_to_sim(root_state)
                self.scene.reset()
                print("[INFO]: Resetting state...")
            if count == state_machine_keyframe[0]:
                print("[INFO]: Beginning Pick Simulation...")

            if count > state_machine_keyframe[0]:
                gripper_close = False
                
                ee_goal_offset[2] = get_ee_height(count, state_machine_keyframe[0]+10, state_machine_keyframe[1]-3, 0.18, 0.005)
                
                if count < state_machine_keyframe[1]-10:
                    for idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
                        rigid_object.update(sim_dt)
                        self.rigid_objects_viser_frame[idx].position = rigid_object.data.root_state_w[self.env][:3].cpu().detach().numpy() - self.scene.env_origins.cpu().numpy()[self.env]
                        self.rigid_objects_viser_frame[idx].wxyz = rigid_object.data.root_state_w[self.env][3:7].cpu().detach().numpy()
                    
                if count > state_machine_keyframe[1]:
                    gripper_close = True
                if count > state_machine_keyframe[1]+6:
                    ee_goal_offset[2] = get_ee_height(count, state_machine_keyframe[1]+7, state_machine_keyframe[2], 0.005, 0.15)
                if count > state_machine_keyframe[2]:
                    gripper_close = False
                
                if count == state_machine_keyframe[2]-6:
                    # Find tigers above height threshold and close to gripper pose
                    for rigid_object in self.scene.rigid_objects.values():
                        rigid_object.update(sim_dt)
                        root_state = rigid_object.data.root_state_w
                        ee_pos_w_left = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7][:,0:3] # Per env robot ee_left to world
                        ee_pos_w_right = robot.data.body_state_w[:, robot_entity_cfg.body_ids[1], 0:7][:,0:3] # Per env robot ee_left to world
                        ee_pos_w_agg = torch.where(self.left_hand_envs.unsqueeze(0).T.repeat(1,3), ee_pos_w_left, ee_pos_w_right)
                        close_to_gripper_pos = (ee_pos_w_agg - root_state[:,0:3]).norm(dim=-1) < 0.2
                        above_height_thresh = root_state[:,2] > 0.05
                        success_envs = above_height_thresh & close_to_gripper_pos

                for idx, robot in enumerate(self.scene.articulations.values()):
                    
                    # Inverse Kinematics Solver
                    obj_xyzs = rigid_object.data.root_state_w[:, :3] - self.scene.env_origins
                    if count == state_machine_keyframe[0]+1:
                        self.left_hand_envs = obj_xyzs[:,1] > 0
                    obj_wxyzs = rigid_object.data.root_state_w[:, 3:7]
                    command_wxyzs = reorient_quaternion_batch(obj_wxyzs.cpu().detach().numpy()) # Enforce gripper Z-axis down but adopt the object's Z-axis as grasp axis (privileged information)

                    ik_commands = torch.zeros(self.scene.num_envs, 7, device=robot.device)
                    ik_commands[:] = torch.tensor(ee_goal_offset)
                    ik_commands[:, :3] += obj_xyzs
                    ik_commands[:, 3:7] = torch.tensor(command_wxyzs)
                    
                    ee_pose_w_left = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7] # Per env robot ee_left to world
                    ee_pose_w_right = robot.data.body_state_w[:, robot_entity_cfg.body_ids[1], 0:7] # Per env robot ee_right to world
                    root_pose_w = robot.data.root_state_w[:, 0:7] # Per env robot base frames to world
                    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                    
                    # compute frame in root frame
                    ee_pos_b_left, ee_quat_b_left = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w_left[:, 0:3], ee_pose_w_left[:, 3:7])
                    ee_pos_b_right, ee_quat_b_right = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7])
                    
                    # Solve for which environment has tiger on left vs right side
                    ee_pos_b_agg = torch.where(self.left_hand_envs.unsqueeze(0).T.repeat(1,3), ee_pos_b_left, ee_pos_b_right)
                    ee_quat_b_agg = torch.where(self.left_hand_envs.unsqueeze(0).T.repeat(1,4), ee_quat_b_left, ee_quat_b_right)
                    
                    # Calculations for relative control mode
                    pos_error = ik_commands[:, :3] - ee_pos_b_agg
                    pos_magnitude = torch.norm(pos_error, dim=1, keepdim=True)
                    pos_vec_norm = pos_error / pos_magnitude
                    pos_command = pos_vec_norm * torch.min(pos_magnitude, torch.tensor(0.01).to(pos_magnitude))

                    q_delta = slerp_with_clip(
                        ee_quat_b_agg,
                        ik_commands[:,3:7],
                        0.3,
                        torch.pi / 16
                    )

                    rot_command = wrap_to_pi(torch.stack(euler_xyz_from_quat(q_delta)).T)
                    
                    ik_commands_relative = torch.cat((pos_command, rot_command), dim=1)
                    
                    controller.set_command(ik_commands_relative, ee_pos_b_agg, ee_quat_b_agg)
                    
                    controller.ee_quat_des = q_delta

                    jacobian_left = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx_left, :, robot_entity_cfg.joint_ids]
                    jacobian_right = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx_right, :, robot_entity_cfg.joint_ids]
                    jacobian_agg = torch.where(self.left_hand_envs.unsqueeze(0).unsqueeze(0).T.repeat(1,6,18), jacobian_left, jacobian_right)

                    # compute the joint commands
                    joint_pos_des = controller.compute(ee_pos_b_agg, ee_quat_b_agg, jacobian_agg, joint_pos)
                    
                    # gripper override
                    if gripper_close:
                        joint_pos_des[self.left_hand_envs,-4:-2] = 0.0
                        joint_pos_des[~self.left_hand_envs,-2:] = 0.0
                    else:
                        joint_pos_des[self.left_hand_envs,-4:-2] = 0.024
                        joint_pos_des[~self.left_hand_envs,-2:] = 0.024
                    
                    # apply actions
                    if count > state_machine_keyframe[0] + 7:
                        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
                        robot.write_data_to_sim()
                    
                    if getattr(self, 'data_logger', None) is not None:
                        joint_names = [*robot.data.joint_names[:-3], (robot.data.joint_names[-2])]
                        robot_data = {
                            "joint_names": joint_names,
                            "joint_angles": torch.cat((robot.data.joint_pos[:, :-3], robot.data.joint_pos[:, -2].unsqueeze(-1)), dim=1).clone().cpu().detach().numpy(),
                            # Final Two Indices: Left Gripper, Right Gripper
                            "ee_pos": torch.cat([ee_pos_b_left, ee_quat_b_left, ee_pos_b_right, ee_quat_b_right], dim=1).cpu().detach().numpy(),
                            # Left EE xyz, wxyz, Right EE xyz, wxyz
                            "gripper_binary_cmd": torch.cat([
                                torch.bitwise_and(torch.tensor(gripper_close).repeat(self.scene.num_envs).to(self.left_hand_envs.device), self.left_hand_envs).unsqueeze(0).T, 
                                torch.bitwise_and(torch.tensor(gripper_close).repeat(self.scene.num_envs).to(self.left_hand_envs.device), ~self.left_hand_envs).unsqueeze(0).T
                                ], dim = -1).cpu().detach().numpy()
                        } # Left Gripper is Closed, Right Gripper is Closed
                        self.data_logger.save_data(self.camera_manager.buffers, robot_data, count-state_machine_keyframe[0]-1, self.output_dir)
                        stats = self.data_logger.get_stats()
                        self.images_per_second.value = stats['images_per_second']
                        self.successful_envs.value = stats['total_successful_envs']
                
                for idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
                    self.rigid_objects_viser_frame[idx].position = rigid_object.data.root_state_w[self.env][:3].cpu().detach().numpy() - self.scene.env_origins.cpu().numpy()[self.env]
                    self.rigid_objects_viser_frame[idx].wxyz = rigid_object.data.root_state_w[self.env][3:7].cpu().detach().numpy()
                
                self.sim.step(render=True)
            # Apply randomization to robot starting pose and rigid object pose
            else:
                if count < state_machine_keyframe[0] - 10:
                    for idx, robot in enumerate(self.scene.articulations.values()):

                        # random perturb joint positions
                        joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.015
                        joint_pos_target = joint_pos_target.clamp_(
                            robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
                        )
                        # apply action to the robot
                        joint_pos_target[:,-4:] = 0.024 # Override Gripper to Open
                        robot.set_joint_position_target(joint_pos_target)
                        # write data to sim
                        robot.write_data_to_sim()

                for idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
                    rigid_object.update(sim_dt)
                    self.rigid_objects_viser_frame[idx].position = rigid_object.data.root_state_w[self.env][:3].cpu().detach().numpy() - self.scene.env_origins.cpu().numpy()[self.env]
                    self.rigid_objects_viser_frame[idx].wxyz = rigid_object.data.root_state_w[self.env][3:7].cpu().detach().numpy()
                if count > state_machine_keyframe[0]-5:
                    self.sim.step(render=True)
                else:
                    self.sim.step(render=False)
            
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
                    if self.debug_marker_vis:
                        # obtain quantities from simulation
                        ee_pose_w_left = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                        ee_pose_w_right = robot.data.body_state_w[:, robot_entity_cfg.body_ids[1], 0:7]
                        if hasattr(self, 'left_hand_envs'):
                            ee_pose_w_agg = torch.where(self.left_hand_envs.unsqueeze(0).T.repeat(1,7), ee_pose_w_left, ee_pose_w_right)
                        else:
                            ee_pose_w_agg = ee_pose_w_left
                        # update marker positions
                        ee_marker.visualize(ee_pose_w_agg[:, 0:3], ee_pose_w_agg[:, 3:7])
                        goal_marker.visualize(ik_commands[:, 0:3] + self.scene.env_origins, ik_commands[:, 3:7])
            if self.init_viser:
                self.isaac_viewport_camera.update(0, force_recompute=True)
            self.sim_step_time_ms.value = (time.time() - sim_start_time) * 1e3
            
