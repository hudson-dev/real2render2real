from xi.isaaclab_viser.base import IsaacLabViser

import torch

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, apply_delta_pose
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

import time
from xi.isaaclab_viser.controllers.diffusion_policy import YuMiDiffusionPolicyController
import numpy as np


class PickTigerEvalDP(IsaacLabViser):
    def __init__(self, *args, **kwargs):
        self.ckpt_path = kwargs.pop('ckpt_path')
        self.ckpt_id = kwargs.pop('ckpt_id')
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
    
    def run_simulator(self):
        """Runs the simulation loop."""
        controller = YuMiDiffusionPolicyController(ckpt_path=self.ckpt_path, ckpt_id=self.ckpt_id)
        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
        print(f"[INFO]: Simulation timestep: {sim_dt}")
        sim_time = 0.0
        count = 0
        success_envs = None
        
        gripper_close = False
        
        state_machine_keyframe = [
            40, # End of tiger drop, begin trajectory
            206 # End of trajectory
        ]
        
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["left_dummy_point", "right_dummy_point"])
        # robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["gripper_l_base", "gripper_r_base"])
        robot_entity_cfg.resolve(self.scene)
        
        # Simulate physics
        while self.simulation_app.is_running(): # and self.successful_envs.value < 10000:
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

                if success_envs is not None:
                    print(f"[INFO]: Success Envs: {success_envs}")
                    if getattr(self, 'data_logger', None) is not None:
                        self.data_logger.redir_data(success_envs)
                
                # reset counters
                sim_time = 0.0
                count = 0
                # reset the scene entities
                for index, robot in enumerate(self.scene.articulations.values()):
                    # ik_commands = torch.zeros(self.scene.num_envs, 7, device=robot.device)
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
                    random_xyz = torch.torch.randn_like(self.scene.env_origins) * 0.065
                    random_xyz = torch.where(random_xyz > 0.07, 0.07, random_xyz)
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
                
                controller.update_observation_buffers(self.camera_manager.buffers)
                # TODO: Update Buffers and run DP inference according to receding horizon or temporal ensembling

                for idx, robot in enumerate(self.scene.articulations.values()):
                    
                    ee_pose_w_left = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7] # Per env robot ee_left to world
                    root_pose_w = robot.data.root_state_w[:, 0:7] # Per env robot base frames to world
                    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                    # compute frame in root frame
                    ee_pos_b_left, ee_quat_b_left = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w_left[:, 0:3], ee_pose_w_left[:, 3:7])
    
                    # Right EE calculations (only for data saving)
                    ee_pose_w_r = robot.data.body_state_w[:, robot_entity_cfg.body_ids[1], 0:7]
                    # compute frame in root frame
                    ee_pos_b_right, ee_quat_b_right = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w_r[:, 0:3], ee_pose_w_r[:, 3:7])
                    
                    controller.setup_scene(ee_pos_b_left[0].cpu().numpy(), ee_pos_b_right[0].cpu().numpy())
                    #single environment eval only
                    controller.update_curr_proprio(
                        np.array([*ee_pos_b_left[0].cpu(), *ee_quat_b_left[0].cpu()]), # left xyz, wxyz
                        np.array([*ee_pos_b_right[0].cpu(), *ee_quat_b_right[0].cpu()]), # right xyz, wxyz
                        (joint_pos[0,-4:-2].mean().item() < 0.02),
                        (joint_pos[0,-2:].mean().item() < 0.02)
                        )
                    
                    # compute the joint commands TODO: jaxmp-ify
                    controller.compute()
                    
                    joint_pos_des = torch.zeros_like(joint_pos)

                    jaxmp2isaaclab_joint_mapping = [7, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6]
                    joint_desired = np.array([controller.joints[i] for i in jaxmp2isaaclab_joint_mapping], dtype=np.float32)
                    # import pdb; pdb.set_trace()
                    joint_pos_des[0, 0:14] = torch.tensor(joint_desired) # [N, num_joints]
                    
                    gripper_close = controller.gripper_cmd_L
                    
                    # gripper pos cmd
                    if gripper_close:
                        joint_pos_des[:,-4:-2] = 0.0
                    else:
                        joint_pos_des[:,-4:-2] = 0.024
                    
                    # apply actions after jaxmp 
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
                            "gripper_binary_cmd": torch.tensor([gripper_close, False]).numpy() # Left Gripper Closed, Right Gripper Closed
                        }
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
                for idx, robot in enumerate(self.scene.articulations.values()):

                    # random perturb joint positions
                    joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.015
                    joint_pos_target = joint_pos_target.clamp_(
                        robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
                    )
                    # apply action to the robot
                    joint_pos_target[:,-4:-2] = 0.024 # Override Gripper to Open
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
                    
            if self.init_viser:
                self.isaac_viewport_camera.update(0, force_recompute=True)
            self.sim_step_time_ms.value = (time.time() - sim_start_time) * 1e3
            
