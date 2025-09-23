import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional
import tyro
import h5py
import viser
import viser.extras
from jaxmp.extras.urdf_loader import load_urdf
from PIL import Image
import time
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")


class TrajectoryViewer:
    def __init__(self, data_dir: str, urdf_path: str = None):
        self.data_dir = Path(data_dir)
        self.current_idx = 0
        
        # Check if this is a trajectory directory or a parent directory containing multiple trajectories
        if self._is_trajectory_directory(self.data_dir):
            self.trajectories = [self.data_dir]
            self.current_trajectory = self.data_dir
        else:
            # Scan for trajectory directories
            self.trajectories = self._find_trajectory_directories(self.data_dir)
            if not self.trajectories:
                raise ValueError(f"No trajectory directories found in {self.data_dir}")
            self.current_trajectory = self.trajectories[0]
            
        # Load the first trajectory
        self._load_trajectory(self.current_trajectory)
        
        # Set up viser
        self.viser_server = viser.ViserServer()
        self.urdf = load_urdf(None, Path(urdf_path))
        
        self.render_viewport_depth = False
        self._setup_viser_scene()
        self._setup_viser_gui()
    
    def _is_trajectory_directory(self, directory: Path) -> bool:
        """Check if the given directory contains a trajectory (has camera dirs and robot_data)."""
        return any('camera' in d.name for d in directory.iterdir() if d.is_dir()) and (directory / "robot_data").exists()
    
    def _find_trajectory_directories(self, parent_dir: Path) -> List[Path]:
        """Find all trajectory directories under the parent directory."""
        trajectory_dirs = []
        for item in parent_dir.iterdir():
            if item.is_dir() and self._is_trajectory_directory(item):
                trajectory_dirs.append(item)
        return sorted(trajectory_dirs)
    
    def _load_trajectory(self, trajectory_path: Path):
        """Load data for a specific trajectory."""
        self.current_trajectory = trajectory_path
        self.camera_dirs = []
        for dirpath, dirnames, filenames in os.walk(trajectory_path):
            for dirname in dirnames:
                if 'camera' in dirname:
                    self.camera_dirs.append(dirpath+'/'+dirname+'/rgb')
                    
        self.robot_data_dir = trajectory_path / "robot_data"
        self.joint_names, self.joint_angles, self.ee_poses, self.gripper_binary_cmd = self._load_action_file()
        
        self.images = self._get_image_files()
        self.total_frames = self.images.shape[1]
        
        # Reset GUI elements if they exist
        if hasattr(self, 'slider_handle'):
            self.slider_handle.value = 0
            self.slider_handle.max = self.total_frames - 1
            
            # Update images
            for idx, img_handle in enumerate(self.viser_img_handles):
                img_handle.image = self.images[idx, 0]
    
    def _setup_viser_scene(self):
        self.base_frame = self.viser_server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis = viser.extras.ViserUrdf(
            self.viser_server,
            self.urdf, 
            root_node_name="/base"
        )
        self.tf_left_frame = self.viser_server.scene.add_frame(
                    "tf_left",
                    axes_length=0.5 * 0.2,
                    axes_radius=0.01 * 0.2,
                    origin_radius=0.1 * 0.2,
                )
    
        self.tf_right_frame = self.viser_server.scene.add_frame(
                        "tf_right",
                        axes_length=0.5 * 0.2,
                        axes_radius=0.01 * 0.2,
                        origin_radius=0.1 * 0.2,
                    )
        self.tf_left_frame.position = self.ee_poses[0, :3]
        self.tf_left_frame.wxyz = self.ee_poses[0, 3:7]
        self.tf_right_frame.position = self.ee_poses[0, 7:10]
        self.tf_right_frame.wxyz = self.ee_poses[0, 10:]
        joint_pos_dict = dict(zip(self.joint_names, self.joint_angles[0]))
        self.urdf_vis.update_cfg(joint_pos_dict)
    
    def _setup_viser_gui(self):
        with self.viser_server.gui.add_folder("Trajectory Selection"):
            # Only add trajectory selector if we have multiple trajectories
            if len(self.trajectories) > 1:
                trajectory_names = [str(p.name) for p in self.trajectories]
                self.trajectory_selector = self.viser_server.gui.add_dropdown(
                    "Select Trajectory",
                    options=trajectory_names,
                    initial_value=trajectory_names[0]
                )
                
                # Add navigation buttons for trajectories
                with self.viser_server.gui.add_folder("Navigation"):
                    self.prev_traj_button = self.viser_server.gui.add_button(
                        label="Previous Trajectory", 
                        icon=viser.Icon.CHEVRON_LEFT
                    )
                    self.next_traj_button = self.viser_server.gui.add_button(
                        label="Next Trajectory", 
                        icon=viser.Icon.CHEVRON_RIGHT
                    )
                
                @self.trajectory_selector.on_update
                def _(_) -> None:
                    selected_idx = trajectory_names.index(self.trajectory_selector.value)
                    self._load_trajectory(self.trajectories[selected_idx])
                    
                @self.prev_traj_button.on_click
                def _(_) -> None:
                    current_idx = trajectory_names.index(self.trajectory_selector.value)
                    # Wrap around to last trajectory if at the beginning
                    new_idx = (current_idx - 1) % len(trajectory_names)
                    self.trajectory_selector.value = trajectory_names[new_idx]
                    
                @self.next_traj_button.on_click
                def _(_) -> None:
                    current_idx = trajectory_names.index(self.trajectory_selector.value)
                    # Wrap around to first trajectory if at the end
                    new_idx = (current_idx + 1) % len(trajectory_names)
                    self.trajectory_selector.value = trajectory_names[new_idx]
        
        with self.viser_server.gui.add_folder("Playback Controls"):
            self.play_button = self.viser_server.gui.add_button(label="Play", icon=viser.Icon.PLAYER_PLAY_FILLED)
            self.pause_button = self.viser_server.gui.add_button(label="Pause", icon=viser.Icon.PLAYER_PAUSE_FILLED, visible=False)
            self.next_button = self.viser_server.gui.add_button(label="Step Forward", icon=viser.Icon.ARROW_BIG_RIGHT_FILLED)
            self.prev_button = self.viser_server.gui.add_button(label="Step Back", icon=viser.Icon.ARROW_BIG_LEFT_FILLED)
            
        self.slider_handle = self.viser_server.gui.add_slider(
            "Data Entry Index", min=0, max=self.joint_angles.shape[0]-1, step=1, initial_value=0
        )
        
        self.viser_img_handles = []
        with self.viser_server.gui.add_folder("Observation"):
            for idx, image in enumerate(self.images):
                self.viser_img_handles.append(self.viser_server.gui.add_image(
                    image=image[0],
                    label=f'Camera {idx}'
                ))
                
        with self.viser_server.gui.add_folder("State"):
            self.right_gripper_signal = self.viser_server.gui.add_number("Right Gripper Signal (mm): ", 0.0, disabled=True)
            self.left_gripper_signal = self.viser_server.gui.add_number("Left Gripper Signal (mm): ", 0.0, disabled=True)
            self.right_ee_z_height = self.viser_server.gui.add_number("Right Gripper Z-Height (mm): ", 0.0, disabled=True)
            self.left_ee_z_height = self.viser_server.gui.add_number("Left Gripper Z-Height (mm): ", 0.0, disabled=True)
            
        with self.viser_server.gui.add_folder("Action"):
            self.right_gripper_action = self.viser_server.gui.add_number("Right Gripper Action (bin): ", 0.0, disabled=True)
            self.left_gripper_action = self.viser_server.gui.add_number("Left Gripper Action (bin): ", 0.0, disabled=True)
        
        # Add a new folder for video export functionality
        with self.viser_server.gui.add_folder("Export Video"):
            self.save_video_button = self.viser_server.gui.add_button(
                label="Save Cameras to MP4", 
                icon=viser.Icon.VIDEO
            )
            self.video_status = self.viser_server.gui.add_text("Status", "Status: Ready")
            
        @self.next_button.on_click
        def _(_) -> None:
            self.slider_handle.value += 1
            
        @self.prev_button.on_click
        def _(_) -> None:
            self.slider_handle.value -= 1
            
        @self.play_button.on_click
        def _(_) -> None:
            self.play_button.visible = False
            self.pause_button.visible = True
            
        @self.pause_button.on_click
        def _(_) -> None:
            self.play_button.visible = True
            self.pause_button.visible = False
        
        @self.save_video_button.on_click
        def _(_) -> None:
            self.save_cameras_to_mp4()
            
        @self.slider_handle.on_update
        def _(_) -> None:
            self.tf_left_frame.position = self.ee_poses[self.slider_handle.value, :3]
            self.tf_left_frame.wxyz = self.ee_poses[self.slider_handle.value, 3:7]
            self.tf_right_frame.position = self.ee_poses[self.slider_handle.value, 7:10]
            self.tf_right_frame.wxyz = self.ee_poses[self.slider_handle.value, 10:]
            joint_pos_dict = dict(zip(self.joint_names, self.joint_angles[self.slider_handle.value]))
            self.left_gripper_signal.value = float(joint_pos_dict['gripper_l_joint']) * 1e3
            self.right_gripper_signal.value = float(joint_pos_dict['gripper_r_joint']) * 1e3
            
            self.left_ee_z_height.value = float(self.ee_poses[self.slider_handle.value, 2] * 1e3)
            self.right_ee_z_height.value = float(self.ee_poses[self.slider_handle.value, 9] * 1e3)

            self.right_gripper_action.value = int(self.gripper_binary_cmd[self.slider_handle.value, 1])
            self.left_gripper_action.value = int(self.gripper_binary_cmd[self.slider_handle.value, 0])
            self.urdf_vis.update_cfg(joint_pos_dict)
            for idx, img_handle in enumerate(self.viser_img_handles):
                img_handle.image = self.images[idx, self.slider_handle.value]
    
    def _load_action_file(self) -> List[str]:
        with h5py.File(f'{self.robot_data_dir}/robot_data.h5', 'r') as f:
            if 'joint_angles' in f:
                joint_angles = f['joint_angles'][:]
            if 'ee_poses' in f:
                ee_poses = f['ee_poses'][:]
            if 'gripper_binary_cmd' in f:
                gripper_binary_cmd = f['gripper_binary_cmd'][:]
            else:
                gripper_binary_cmd = None
        joint_names_file = self.robot_data_dir / "joint_names.txt"
        if not joint_names_file.exists():
            print("Warning: No joint names file found")
        with open(joint_names_file, 'r') as f:
            joint_names = [line.strip() for line in f.readlines()]

        return joint_names, joint_angles, ee_poses, gripper_binary_cmd
    
    def _get_image_files(self) -> List[str]:
        image_batch = []
        for dirn in self.camera_dirs:
            dir_img = []
            images = sorted([f for f in os.listdir(dirn) if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
            for image in images:
                dir_img.append(np.array(Image.open(dirn+'/'+image)))
            image_batch.append(np.array(dir_img))
        image_batch = np.array(image_batch)
        return image_batch  # [N_C, B, H, W, C]
    
    def run(self):
        while True:
            if self.pause_button.visible:
                self.slider_handle.value = (self.slider_handle.value + 1) % self.total_frames
            time.sleep(0.1)

    def save_cameras_to_mp4(self):
        """Save camera frames to MP4 videos."""
        try:
            self.video_status.value = "Status: Saving videos..."
            
            # Create timestamp for the filename
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            env_name = self.current_trajectory.name
            
            # Make sure we have images to save
            if self.images.shape[0] == 0 or self.images.shape[1] == 0:
                self.video_status.value = "Status: No images to save!"
                return
            
            # Only process camera_0 and camera_1 if they exist
            for camera_idx in range(min(2, self.images.shape[0])):
                # Get image dimensions and prepare video writer
                frames = self.images[camera_idx]
                height, width, channels = frames[0].shape
                
                # Create output filename
                output_file = f"{env_name}_camera_{camera_idx}.mp4"
                output_path = os.path.join(self.current_trajectory, output_file)
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
                video_writer = cv2.VideoWriter(output_path, fourcc, 24.0, (width, height))
                
                # Write frames to video
                for frame in frames:
                    # OpenCV uses BGR format, but our frames are likely RGB
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr_frame)
                
                # Release video writer
                video_writer.release()
                
                print(f"Saved video to {output_path}")
            
            self.video_status.value = f"Status: Saved videos to {self.current_trajectory}"
        except Exception as e:
            self.video_status.value = f"Status: Error - {str(e)}"
            print(f"Error saving videos: {e}")
            
def main(
    data_dir: str = '/home/yujustin/xi/output_data/franka_coffee_maker/supplemental/hd_renders/successes',
    urdf_path: str = f'{data_dir}/yumi_description/urdf/yumi.urdf'
):

    viewer = TrajectoryViewer(data_dir, urdf_path)
    viewer.run()


if __name__ == "__main__":
    tyro.cli(main)