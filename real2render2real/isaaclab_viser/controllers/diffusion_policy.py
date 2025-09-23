from yumi_realtime.base import YuMiBaseInterface
from loguru import logger
import numpy as onp
from dp_gs.policy.diffusion_wrapper import DiffusionWrapper
from collections import deque
from scipy.spatial.transform import Rotation
import torch
import time
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def rot_mat_to_rot_6d(rot_mat : onp.ndarray) -> onp.ndarray: 
    """
    Convert a rotation matrix to 6d representation
    rot_mat: N, 3, 3

    return: N, 6
    """
    rot_6d = rot_mat[:, :2, :] # N, 2, 3
    return rot_6d.reshape(-1, 6) # N, 6

def gram_schmidt(vectors : onp.ndarray) -> onp.ndarray: 
    """
    Apply Gram-Schmidt process to a set of vectors
    vectors are indexed by rows 

    vectors: batchsize, N, D 

    return: batchsize, N, D
    """
    if len(vectors.shape) == 2:
        vectors = vectors[None]
    
    basis = onp.zeros_like(vectors)
    basis[:, 0] = vectors[:, 0] / onp.linalg.norm(vectors[:, 0], axis=-1, keepdims=True)
    for i in range(1, vectors.shape[1]):
        v = vectors[:, i]
        for j in range(i):
            v -= onp.sum(v * basis[:, j], axis=-1, keepdims=True) * basis[:, j]
        basis[:, i] = v / onp.linalg.norm(v, axis=-1, keepdims=True)
    return basis

def rot_6d_to_rot_mat(rot_6d : onp.ndarray) -> onp.ndarray:
    """
    Convert a 6d representation to rotation matrix
    rot_6d: N, 6

    return: N, 3, 3
    """
    rot_6d = rot_6d.reshape(-1, 2, 3)
    # assert the first two vectors are orthogonal
    if not onp.allclose(onp.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1), 0):
        rot_6d = gram_schmidt(rot_6d)

    rot_mat = onp.zeros((rot_6d.shape[0], 3, 3))
    rot_mat[:, :2, :] = rot_6d
    rot_mat[:, 2, :] = onp.cross(rot_6d[:, 0], rot_6d[:, 1])
    return rot_mat

def rot_6d_to_quat(rot_6d : onp.ndarray) -> onp.ndarray:
    """
    Convert 6d representation to quaternion
    rot_6d: N, 6
    """
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    return Rotation.from_matrix(rot_mat).as_quat(scalar_first=True)

def action_10d_to_8d(action : onp.ndarray) -> onp.ndarray:
    """
    Convert a 10d action to a 8d action
    - 3d translation, 6d rotation, 1d gripper
    to - 3d translation, 4d euler angles, 1d gripper
    """
    return onp.concatenate([action[:3], rot_6d_to_quat(action[3:-1]).squeeze(), action[-1:]], axis=-1)

class YuMiDiffusionPolicyController(YuMiBaseInterface):
    """YuMi controller for diffusion policy control."""
    def __init__(self, ckpt_path: str = None, ckpt_id: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._interactive_handles = False
        
        for side in ['left', 'right']:
            target_handle = self.transform_handles[side]
            target_handle.control.visible = False
        
        assert ckpt_path is not None, "Diffusion Policy checkpoint path must be provided."
        
        # Setup Diffusion Policy module and weights
        self.model = DiffusionWrapper(model_ckpt_folder=ckpt_path, ckpt_id=ckpt_id, device='cuda')
        
        # ROS Camera Observation Subscriber
        self.height = None
        self.width = None
        
        self.proprio_buffer = deque([],maxlen=self.model.model.obs_horizon)

        self.observation_buffers = {}
        self.camera_buffers = {}
        self.camera_topics = ["camera_1", "camera_0"]
        
        # Initialize a deque for each camera
        self.main_camera = "camera_1"
        for idx, topic in enumerate(self.camera_topics):
            camera_name = topic
            self.observation_buffers[camera_name] = deque([], maxlen=self.model.model.obs_horizon)
        
        self.cur_proprio = None
        
        self.cartesian_pose_L = None
        self.cartesian_pose_R = None
        
        # Control mode
        self.control_mode = 'receding_horizon_control'
        # self.control_mode = 'temporal_ensemble'
        
        if self.control_mode == 'receding_horizon_control':
            self.max_len = 7
            self.action_queue = deque([],maxlen=self.max_len)
        elif self.control_mode == 'temporal_ensemble':
            self.action_queue = deque([],maxlen=self.model.model.action_horizon)
        self.prev_action = deque([],maxlen=self.model.model.obs_horizon)
                
        logger.info("Diffusion Policy controller initialized")

        self.gripper_thres = 0.5

        with self.server.gui.add_folder("State"):
            self.right_gripper_signal = self.server.gui.add_number("Right gripper pred.: ", 0.0, disabled=True)
            self.left_gripper_signal = self.server.gui.add_number("Left gripper pred.: ", 0.0, disabled=True)
        self.breakpoint_btn = self.server.gui.add_button("Breakpoint at Inference")
        
        self.breakpt_next_inference = False

        @self.breakpoint_btn.on_click
        def _(_) -> None:
            self.breakpt_next_inference = True

    def update_observation_buffers(self, image_buffer):
        assert set(image_buffer.keys()) == set(self.camera_topics)
        for camera_name, obs_ in image_buffer.items():
            obs = onp.transpose(obs_[0]['rgb'].squeeze(0).cpu().numpy(), (2, 0, 1))  # C, H, W
            obs = onp.array(obs, dtype=onp.float32) / 255.0 # uint8 to float32
            self.observation_buffers[camera_name].append(obs)
            while len(self.observation_buffers[camera_name]) < self.model.model.obs_horizon:
                self.observation_buffers[camera_name].append(obs)
        
    def compute(self):
        """Diffusion Policy controller inference step."""
        i = 0
        step = 0
        self.last_action = None

        assert len(self.proprio_buffer) == self.model.model.obs_horizon
        # Stack all camera observations
        all_camera_obs = []
        for camera_name in self.observation_buffers.keys():  # Sort to ensure consistent order
            assert len(self.observation_buffers[camera_name]) == self.model.model.obs_horizon
            cam_obs = onp.array(self.observation_buffers[camera_name])
            all_camera_obs.append(cam_obs)
        
        # Stack along the N_C dimension
        stacked_obs = onp.stack(all_camera_obs, axis=1)  # [T, N_C, C, H, W]
        
        self._update_proprio_queue_viz()
        input = {
        "observation": torch.from_numpy(stacked_obs).unsqueeze(0),  # [B, T, N_C, C, H, W]
        "proprio": torch.from_numpy(onp.array(self.proprio_buffer)).unsqueeze(0) # [B, T, D] 
            }
        
        start = time.time()
        step += 1

        if self.last_action is not None:
            # check gripper state of last action 
            target_left_gripper = self.last_action[9] < self.gripper_thres
            target_right_gripper = self.last_action[19] < self.gripper_thres

            current_left_gripper = input["proprio"][0, -1, 9] < self.gripper_thres
            current_right_gripper = input["proprio"][0, -1, 19] < self.gripper_thres
            
            # print("current_left_gripper: ", current_left_gripper)

            # delta pose 
            target_proprio_left = self.last_action[:3]
            current_proprio_left = input["proprio"][0, -1, :3].numpy()
        
            # calculate lag 
            lag = onp.linalg.norm(target_proprio_left - current_proprio_left)
            print("lag: ", lag)

            # if they are in disgreemnt, gripper control with last action 
            # if target_left_gripper != current_left_gripper or target_right_gripper != current_right_gripper:
            if target_left_gripper != current_left_gripper or target_right_gripper != current_right_gripper or lag > 0.005:
                print("blocking with last action")
                # import pdb; pdb.set_trace()
                self._yumi_control(self.last_action)
                return

        # receding horizon control
        if self.control_mode == 'receding_horizon_control':
            if len(self.action_queue) > 0:
                # self._update_action_queue_viz()
                action = self.action_queue.popleft()
                self._yumi_control(action)
                return
        # end of receding horizon control

        action_prediction = self.model(input) # Denoise action prediction from obs and proprio...

        if self.breakpt_next_inference:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.plot_predictions(input["proprio"], action_prediction, timestamp)
            obs_copy = copy.deepcopy(stacked_obs)
            self.plot_stacked_obs(copy.deepcopy(obs_copy), timestamp)
            import pdb; pdb.set_trace()

        self.action_prediction = action_prediction
        self._update_action_queue_viz()
        print("\nprediction called\n")

        action_L = action_prediction[0,:,:10]
        action_R = action_prediction[0,:,10:]
        
        action = onp.concatenate([action_L, action_R], axis=-1)
        
        # # temporal emsemble start
        if self.control_mode == 'temporal_ensemble':
            # import pdb; pdb.set_trace()
            new_actions = deque(action[:self.model.model.action_horizon])
            self.action_queue.append(new_actions)
            if self.model.model.pred_left_only:
                actions_current_timestep = onp.empty((len(self.action_queue), self.model.model.action_dim*2))
            else:
                actions_current_timestep = onp.empty((len(self.action_queue), self.model.model.action_dim))
            
            k = 0.2
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()
            exp_weights = onp.exp(k * onp.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()
            action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            self.temporal_ensemble_action = action
            
        # receding horizon # check the receding horizon block as well
        if self.control_mode == 'receding_horizon_control':
            if len(self.action_queue) == 0: 
                self.action_queue = deque([a for a in action[:self.max_len]])
            action = self.action_queue.popleft()
        
        # update yumi action 
        self._yumi_control(action)
            # self._update_action_queue_viz()
    
    def _yumi_control(self, action):
        # YuMi action update
        ######################################################################
        print("action update called")
        self.last_action = action
        l_act = action_10d_to_8d(action[:10])
        r_act = action_10d_to_8d(action[10:])
        l_xyz, l_wxyz, l_gripper_cmd = l_act[:3], l_act[3:-1], l_act[-1]
        r_xyz, r_wxyz, r_gripper_cmd = r_act[:3], r_act[3:-1], r_act[-1]
        print("left xyz: ", l_xyz)
        print("left gripper: ", l_gripper_cmd)

        self.left_gripper_signal.value = l_gripper_cmd * 1e3
        self.right_gripper_signal.value = r_gripper_cmd * 1e3

        self.update_target_pose(
        side='left',
        position=l_xyz,
        wxyz=l_wxyz,
        )
        self.gripper_cmd_L = bool(l_gripper_cmd>self.gripper_thres)
        
        # self.update_target_pose(
        # side='right',
        # position=r_xyz,
        # wxyz=r_wxyz,
        # )
        # self.gripper_cmd_R=bool(r_gripper_cmd<self.gripper_thres)
        ######################################################################
        self.solve_ik()
        self.update_visualization()
        
    def update_target_pose(self, side: str, position: onp.ndarray, wxyz: onp.ndarray):
        """Update target pose and gripper state for a given arm.
        
        Args:
            side: Either 'left' or 'right'
            position: 3D position array [x, y, z]
            wxyz: Quaternion array [w, x, y, z]
            gripper_state: True for close, False for open, or float for fine position control (meters) [0, 0.025]
            enable: Whether to update the target or snap to current position
        """
        if side not in ['left', 'right']:
            raise ValueError(f"Invalid side {side}, must be 'left' or 'right'")
            
        # Get relevant handles
        target_handle = self.transform_handles[side]
        
        # Update target position
        target_handle.frame.position = position
        target_handle.frame.wxyz = wxyz
        if target_handle.control:  # Update transform controls if they exist
            if not self._interactive_handles:
                target_handle.control.visible = False
            target_handle.control.position = position
            target_handle.control.wxyz = wxyz
                
    def update_curr_proprio(self, left_xyz_quat, right_xyz_quat, gripper_L_pos, gripper_R_pos):
        l_xyz = onp.array(left_xyz_quat[0:3])
        l_wxyz = onp.array(left_xyz_quat[3:7])
        r_xyz = onp.array(right_xyz_quat[0:3])
        r_wxyz = onp.array(right_xyz_quat[3:7])
        
        l_q = Rotation.from_quat(l_wxyz, scalar_first=True)
        l_rot = l_q.as_matrix()
        l_rot_6d = onp.squeeze(rot_mat_to_rot_6d(l_rot[None]), axis=0)# [N, 6]
        r_q = Rotation.from_quat(r_wxyz, scalar_first=True)
        r_rot = r_q.as_matrix()
        r_rot_6d = onp.squeeze(rot_mat_to_rot_6d(r_rot[None]), axis=0) # [N, 6]
        
        self.cur_proprio = onp.concatenate([l_xyz, l_rot_6d, onp.array([int(gripper_L_pos)/10000]), r_xyz, r_rot_6d, onp.array([int(gripper_R_pos)/10000])], axis=-1, dtype=onp.float32)
        assert self.cur_proprio.shape == (20,)

        self.proprio_buffer.append(self.cur_proprio)
        while len(self.proprio_buffer) < self.model.model.obs_horizon:
                self.proprio_buffer.append(self.cur_proprio)

    def setup_scene(self, ee_pos_b_left, ee_pos_b_right):
        
        self.action_queue_viz_L = self.server.scene.add_point_cloud(
            name = "action_queue_L", 
            points = onp.array([[*ee_pos_b_left]]), 
            colors = onp.array([[1.0, 0.0, 0.0]]), 
            point_size=0.002,
            point_shape='circle'
            )
        self.action_queue_viz_R = self.server.scene.add_point_cloud(
            name = "action_queue_R", 
            points = onp.array([[*ee_pos_b_right]]), 
            colors = onp.array([[1.0, 0.0, 0.0]]), 
            point_size=0.002,
            point_shape='circle'
            )
        self.proprio_queue_viz_L = self.server.scene.add_point_cloud(
            name = "proprio_queue_L", 
            points = onp.array([[*ee_pos_b_left]]), 
            colors = onp.array([[1.0, 0.0, 1.0]]), 
            point_size=0.003,
            )
        self.proprio_queue_viz_R = self.server.scene.add_point_cloud(
            name = "proprio_queue_R", 
            points = onp.array([[*ee_pos_b_right]]), 
            colors = onp.array([[1.0, 0.0, 1.0]]), 
            point_size=0.003,
            )
            
    def _update_action_queue_viz(self):
        action_queue_L = self.action_prediction[0,:,:3]
        action_queue_R = self.action_prediction[0,:,10:13]
        self.action_queue_viz_L.points = action_queue_L
        color_order = onp.linspace(0.0, 1.0, len(action_queue_L))
        self.action_queue_viz_L.colors = onp.array([onp.array([c, 0.0, 0.0]) for c in color_order])
        self.action_queue_viz_R.points = action_queue_R
        self.action_queue_viz_R.colors = onp.array([onp.array([c, 0.0, 0.0]) for c in color_order])
    
    def _update_proprio_queue_viz(self):
        if len(self.proprio_buffer) > 0:
            proprio_queue_L = onp.array([a[:3] for a in onp.array(self.proprio_buffer)])
            proprio_queue_R = onp.array([a[10:13] for a in onp.array(self.proprio_buffer)])
            self.proprio_queue_viz_L.points = proprio_queue_L
            color_order = onp.linspace(0.0, 1.0, len(proprio_queue_L))
            self.proprio_queue_viz_L.colors = onp.array([onp.array([c, 0.0, 1.0]) for c in color_order])
            self.proprio_queue_viz_R.points = proprio_queue_R
            self.proprio_queue_viz_R.colors = onp.array([onp.array([c, 0.0, 1.0]) for c in color_order])

    def plot_predictions(self, input_proprio, action_prediction, timestamp):
        """
        Plot proprio history and predicted actions with meaningful labels.
        Args:
            input_proprio: input proprio data [B, T, D]
            action_prediction: predicted actions [B, H, D] where H is horizon
        """
        
        labels = [
        # Left arm (0-9)
        'X_Left', 'Y_Left', 'Z_Left',
        'Rot1_Left', 'Rot2_Left', 'Rot3_Left', 'Rot4_Left', 'Rot5_Left', 'Rot6_Left',
        'Grip_Left',
        # Right arm (10-19)
        'X_Right', 'Y_Right', 'Z_Right',
        'Rot1_Right', 'Rot2_Right', 'Rot3_Right', 'Rot4_Right', 'Rot5_Right', 'Rot6_Right',
        'Grip_Right'
        ]
        T = input_proprio.shape[1]  # Length of proprio history
        D = input_proprio.shape[2]  # Dimension of proprio/action
        H = action_prediction.shape[1]  # Prediction horizon
                
        fig, axes = plt.subplots(5, 4, figsize=(20, 20))
        plt.suptitle(f'Proprio History and Predictions - {timestamp}')
        
        for i in range(D):
            ax = axes[i//4, i%4]
            
            # Plot proprio history
            ax.plot(range(T), 
                    input_proprio[0, :, i].numpy(), 
                    label='proprio', 
                    color='green')
            
            # Plot action predictions
            ax.plot(range(T-1, T+H-1),
                    action_prediction[0, :, i],
                    label='pred', 
                    color='red')
            ax.set_title(labels[i])
            ax.legend()
            ax.grid(True) 
        
        plt.tight_layout()

        os.makedirs(f'debug/{timestamp}', exist_ok=True)
        save_path = f'debug/{timestamp}/prediction_plot.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f'Saved prediction plot to {save_path}')


    def plot_stacked_obs(self, stacked_obs, timestamp):
        """
        Plot stacked observations from multiple cameras across time steps and save with timestamp.
        Args:
            stacked_obs: numpy array of shape [T, N_C, C, H, W]
                T: number of timesteps
                N_C: number of cameras
                C: channels (3 for RGB)
                H, W: height and width
            base_path: base name for the saved file (without extension)
        """
        
        save_path = f'debug/{timestamp}/stacked_obs.png'
        os.makedirs(f'debug/{timestamp}', exist_ok=True)
        
        T, N_C, C, H, W = stacked_obs.shape
        fig, axes = plt.subplots(T, N_C, figsize=(4*N_C, 4*T))
        
        if T == 1 and N_C == 1:
            axes = onp.array([[axes]])
        elif T == 1:
            axes = axes.reshape(1, -1)
        elif N_C == 1:
            axes = axes.reshape(-1, 1)
        
        plt.suptitle(f'Observation Stack - {timestamp}', y=1.02)
        
        for t in range(T):
            for nc in range(N_C):
                img = stacked_obs[t, nc].transpose(1, 2, 0)
                axes[t, nc].imshow(img)
                axes[t, nc].axis('off')
                axes[t, nc].set_title(f'Time {t}, Camera {nc}')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight') 
        plt.close()
        print(f'Saved plot to {save_path}')

       
# def main(
#     ckpt_path: str = "/home/xi/checkpoints/250116_1948",
#     ckpt_id: int = 599,

#     ): 
    
#     yumi_interface = YuMiDiffusionPolicyController(ckpt_path, ckpt_id)
#     yumi_interface.run()
    
# if __name__ == "__main__":
#     tyro.cli(main)
