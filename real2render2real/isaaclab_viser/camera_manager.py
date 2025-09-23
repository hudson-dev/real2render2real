import os
from collections import deque
import h5py
import isaaclab.sim as sim_utils
import numpy as np
import viser
from viser.transforms import SO3
from isaaclab.sensors import MultiTiledCameraCfg

class CameraManager:
    def __init__(self, viser_server, scene):
        self.viser_server = viser_server
        self.scene = scene
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.camera_pose_dir = os.path.join(dir_path, "../../data/camera_poses")
        # TODO: set this to the camera poses file you desire
        self.camera_poses_file = os.path.join(self.camera_pose_dir, "camera_poses_calib_031425.h5")
        # self.camera_poses_file = os.path.join(self.camera_pose_dir, "franka_droid_extr3_.h5")
        print(f"[INFO]: Camera poses file: {self.camera_poses_file}")
        self.used_ids = set()
        self.frustums = []
        self.buffers = {}
        self.render_cam = "camera_0"
        self.camera_selector = None
        if type(list(self.scene.sensors.values())[0].cfg) == MultiTiledCameraCfg:
            self.max_cams_per_env = list(self.scene.sensors.values())[0].cfg.cams_per_env
        else:
            self.max_cams_per_env = 1
        self.load_camera_poses()
        self.last_clicked_frustum = None  
    
    def next_lowest_available_id(self):
        """Find the lowest non-negative integer not in self.used_ids."""
        i = 0
        while i in self.used_ids:
            i += 1
        return i

    def setup_gui(self, viewport_folder, controls_folder):
        """Setup camera-related GUI elements"""
        # Setup camera selector if we have cameras
        if len(self.frustums) > 0:
            with viewport_folder:
                self.camera_selector = self.viser_server.gui.add_dropdown(
                    "Camera to View",
                    [f"camera_{i}" for i in range(len(self.frustums))],
                    initial_value='camera_0'
                )
                
                @self.camera_selector.on_update
                def _(_) -> None:
                    self.render_cam = self.camera_selector.value

        # Add camera controls
        self.controls_folder = controls_folder
        with self.controls_folder:
            self.add_camera_button = self.viser_server.gui.add_button(
                label="Add Current View",
                icon=viser.Icon.CAMERA
            )

        # Initialize buffer for render cam
        self.buffers[self.render_cam] = deque(maxlen=1)
        
        # Initialize buffers for all cameras
        for frustum in self.frustums:
            self.buffers[frustum.name[1:]] = deque(maxlen=1)
            
        if len(self.frustums) == self.max_cams_per_env:
                self.add_camera_button.disabled = True
        elif len(self.frustums) > self.max_cams_per_env:
            raise ValueError("More frustums in camera_poses.h5 file than allowed by the MultiTiledCameraCfg")
                
    def handle_add_camera(self, client):
        """Handle adding a new camera from client view"""
        if client is None:
            raise ValueError("No client found.")

        try:
            self.save_camera_pose(client.camera.position, client.camera.wxyz)
            frustum = self.create_frustum(client.camera.position, client.camera.wxyz)
            
            # Update camera selector
            if self.camera_selector is not None:
                self.camera_selector.remove()
            
            self.camera_selector = self.viser_server.gui.add_dropdown(
                "Camera to View",
                [f"camera_{i}" for i in range(len(self.frustums))],
                initial_value=f"camera_0"
            )
            
            @self.camera_selector.on_update
            def _(_) -> None:
                self.render_cam = self.camera_selector.value
                
            # Initialize buffer for new camera
            if frustum.name[1:] not in self.buffers:
                self.buffers[frustum.name[1:]] = deque(maxlen=1)
            
            if len(self.frustums) == self.max_cams_per_env:
                self.add_camera_button.disabled = True
        except Exception as e:
            print(f"Error in handle add camera: {e}")    
            
        return frustum
    
    def get_camera_params(self):
        """Get camera parameters from scene config"""
        if not isinstance(list(self.scene.sensors.values())[0].cfg.spawn, sim_utils.PinholeCameraCfg):
            return None
        sensor = list(self.scene.sensors.values())[0].cfg.spawn
        return {
            'horizontal_aperture': sensor.horizontal_aperture,
            'focal_length': sensor.focal_length,
            'width': list(self.scene.sensors.values())[0].cfg.width,
            'height': list(self.scene.sensors.values())[0].cfg.height
        }
        
    def create_frustum(self, position, wxyz, camera_params=None):
        """Create a new camera frustum with click handling"""
        if camera_params is None:
            camera_params = self.get_camera_params()
        if camera_params is None:
            print(f"[ERROR]: Unsupported camera configuration")
            return None
        
        camera_id = self.next_lowest_available_id()
        self.used_ids.add(camera_id)
        
        frustum = self.viser_server.scene.add_camera_frustum(
            f"/camera_{camera_id}",
            fov=2 * np.arctan((camera_params['horizontal_aperture']/2) / camera_params['focal_length']),
            aspect=camera_params['width'] / camera_params['height'],
            position=position,
            wxyz=wxyz,
            scale=0.02,
        )
        
        # Add click callback
        frustum.on_click(self.create_click_handler(frustum))

        self.frustums.append(frustum)
        self.buffers[frustum.name[1:]] = deque(maxlen=1)
        return frustum
    
    def update_camera_selector(self):
        """Update camera selector dropdown after camera changes"""
        if len(self.frustums) > 0:
            if self.camera_selector is None:
                self.camera_selector = self.viser_server.gui.add_dropdown(
                    "Camera to View",
                    [self.frustums[i].name[1:] for i in range(len(self.frustums))],
                    initial_value='camera_0'
                )
            else:
                self.camera_selector.remove()
                self.camera_selector = self.viser_server.gui.add_dropdown(
                    "Camera to View",
                    [self.frustums[i].name[1:] for i in range(len(self.frustums))],
                    initial_value=self.frustums[0].name[1:]
                )
            self.render_cam = self.frustums[0].name[1:]
            @self.camera_selector.on_update
            def _(_) -> None:
                self.render_cam = self.camera_selector.value
                
    def load_camera_poses(self):
        """Load saved camera poses from file"""
        if not os.path.exists(self.camera_pose_dir):
            os.makedirs(self.camera_pose_dir)
        # camera_poses_file = os.path.join(self.camera_pose_dir, "camera_poses.h5")
        if not os.path.exists(self.camera_poses_file):
            print("[INFO]: No camera poses found.")
            return
            
        with h5py.File(self.camera_poses_file, 'r') as f:
            if 'poses' not in f:
                return
            camera_poses = f['poses'][:]
            print(f"[INFO]: Found {len(camera_poses)} saved camera poses")
            for pose in camera_poses:
                self.create_frustum(pose[:3], pose[3:])
                
    def save_camera_pose(self, position, wxyz):
        """Save new camera pose to file"""
        if not os.path.exists(self.camera_pose_dir):
            os.makedirs(self.camera_pose_dir)
        pose = np.array([*position, *wxyz])
        self.camera_poses_file = os.path.join(self.camera_pose_dir, "camera_poses.h5")
        
        with h5py.File(self.camera_poses_file, 'a') as f:
            if 'poses' not in f:
                f.create_dataset('poses', data=pose[np.newaxis, :],
                               maxshape=(None, 7), chunks=True)
            else:
                dset = f['poses']
                current_size = dset.shape[0]
                dset.resize(current_size + 1, axis=0)
                dset[current_size] = pose

    def remove_camera(self, frustum):
        """Remove a camera and free its ID."""
        try:
            camera_idx = self.frustums.index(frustum)
            # Extract the camera ID from the name, assuming "/camera_X"
            camera_id_str = frustum.name.split("_")[-1]
            camera_id = int(camera_id_str)
            
            # Remove from scene and internal tracking
            frustum.remove()
            self.frustums.pop(camera_idx)

            # Mark this ID as free
            if camera_id in self.used_ids:
                self.used_ids.remove(camera_id)

            # Update H5 file (removing the corresponding pose)
            with h5py.File(self.camera_poses_file, 'a') as f:
                if 'poses' in f:
                    poses = f['poses'][:]
                    new_poses = np.delete(poses, camera_idx, axis=0)
                    del f['poses']
                    if len(new_poses) > 0:
                        f.create_dataset('poses', data=new_poses, maxshape=(None, 7), chunks=True)

            # Remove the corresponding buffer
            camera_name = f"camera_{camera_id}"
            if camera_name in self.buffers:
                del self.buffers[camera_name]

            # Update the dropdown
            self.update_camera_selector()
            
            if len(self.frustums) < self.max_cams_per_env:
                self.add_camera_button.disabled = False
            
            for button in self.delete_buttons.values():
                button.remove()
            del self.delete_buttons
            
        except Exception as e:
            print(f"Error removing camera: {e}")
            
    def create_click_handler(self, frustum):
        """Create a click handler for a frustum"""
        def click_handler(_):
            try:
                
                is_same_frustum = (frustum == self.last_clicked_frustum)
                self.last_clicked_frustum = frustum
                if is_same_frustum:
                    for button in self.delete_buttons.values():
                        button.remove()
                    del self.delete_buttons
                    for button in self.save_buttons.values():
                        button.remove()
                    del self.save_buttons
                    for vec in self.input_vector_xyz.values():
                        vec.remove()
                    del self.input_vector_xyz
                    for vec in self.input_vector_rpy.values():
                        vec.remove()
                    del self.input_vector_rpy
                    self.tf_gizmo.remove()
                    del self.tf_gizmo
                    self.last_clicked_frustum = None
                    return

                # If we haven't created a camera edit panel, do it now
                if getattr(self, 'controls_folder', None) is None:
                    self.controls_folder = self.viser_server.gui.add_folder("Controls")

                # If we haven't set up a dictionary to track delete buttons, create it now
                if not hasattr(self, 'delete_buttons'):
                    self.delete_buttons = {}
                else:
                    for button in self.delete_buttons.values():
                        button.remove()
                    del self.delete_buttons
                    self.delete_buttons = {}
                
                if not hasattr(self, 'save_buttons'):
                    self.save_buttons = {}
                else:
                    for button in self.save_buttons.values():
                        button.remove()
                    del self.save_buttons
                    self.save_buttons = {}
                    
                if not hasattr(self, 'input_vector_xyz'):
                    self.input_vector_xyz = {}
                else:
                    for vec in self.input_vector_xyz.values():
                        vec.remove()
                    del self.input_vector_xyz
                    self.input_vector_xyz = {}
                
                if not hasattr(self, 'input_vector_rpy'):
                    self.input_vector_rpy = {}
                else:
                    for vec in self.input_vector_rpy.values():
                        vec.remove()
                    del self.input_vector_rpy
                    self.input_vector_rpy = {}
                    
                if not hasattr(self, 'tf_gizmo'):
                    self.tf_gizmo = self.viser_server.scene.add_transform_controls(
                        "camera_control",
                        scale = 0.1,
                        position = frustum.position,
                        wxyz = frustum.wxyz,
                    )
                    @self.tf_gizmo.on_update
                    def _(_):
                        frustum.position = self.tf_gizmo.position
                        frustum.wxyz = self.tf_gizmo.wxyz
                        self.input_vector_xyz[frustum.name].value = self.tf_gizmo.position.tolist()
                        self.input_vector_rpy[frustum.name].value = [
                            float(SO3(self.tf_gizmo.wxyz).compute_roll_radians()*180/np.pi),
                            float(SO3(self.tf_gizmo.wxyz).compute_pitch_radians()*180/np.pi),
                            float(SO3(self.tf_gizmo.wxyz).compute_yaw_radians()*180/np.pi)
                            ]

                else:
                    self.tf_gizmo.position = frustum.position
                    self.tf_gizmo.wxyz = frustum.wxyz
                        
                # Check if a delete button already exists for this frustum
                if frustum.name in self.delete_buttons:
                    # Delete button already exists, do nothing
                    return
                
                with self.controls_folder:
                    delete_button = self.viser_server.gui.add_button(
                        f'Delete {frustum.name[1:].replace("_", " ").capitalize()}',
                        color="red",
                        icon=viser.Icon.TRASH
                    )
                    save_button = self.viser_server.gui.add_button(
                        f'Update {frustum.name[1:].replace("_", " ").capitalize()}',
                        icon=viser.Icon.DEVICE_FLOPPY,
                        disabled=True
                    )
                    vector_input_xyz = self.viser_server.gui.add_vector3(
                        "XYZ",
                        initial_value = frustum.position,
                        step = 0.1
                    )
                    
                    vector_input_rpy = self.viser_server.gui.add_vector3(
                        "RPY",
                        initial_value = (
                            SO3(frustum.wxyz).compute_roll_radians()*180/np.pi,
                            SO3(frustum.wxyz).compute_pitch_radians()*180/np.pi,
                            SO3(frustum.wxyz).compute_yaw_radians()*180/np.pi
                        ),
                        step = 15
                    )

                    self.delete_buttons[frustum.name] = delete_button
                    self.save_buttons[frustum.name] = save_button
                    self.input_vector_xyz[frustum.name] = vector_input_xyz
                    self.input_vector_rpy[frustum.name] = vector_input_rpy
                    
                    @vector_input_xyz.on_update
                    def _(_):
                        frustum.position = self.input_vector_xyz[frustum.name].value
                        save_button.disabled = False
                    @vector_input_rpy.on_update
                    def _(_):
                        frustum.wxyz = SO3.from_rpy_radians(
                            self.input_vector_rpy[frustum.name].value[0]*np.pi/180,
                            self.input_vector_rpy[frustum.name].value[1]*np.pi/180,
                            self.input_vector_rpy[frustum.name].value[2]*np.pi/180,
                            ).wxyz
                        save_button.disabled = False
                    
                    @delete_button.on_click
                    def _(_):
                        self.remove_camera(frustum)
                        # Remove the delete button reference
                        if frustum.name in self.delete_buttons:
                            """
                            Traceback (most recent call last):
                            File "/home/yujustin/anaconda3/envs/env_isaaclab/lib/python3.10/concurrent/futures/thread.py", line 58, in run
                                result = self.fn(*self.args, **self.kwargs)
                            File "/home/yujustin/xi/xi/isaaclab_viser/camera_manager.py", line 381, in _
                                if frustum.name in self.delete_buttons:
                            AttributeError: 'CameraManager' object has no attribute 'delete_buttons
                            
                            Hypothesis: If loaded from file delete buttons are not created properly?
                            """
                            del self.delete_buttons[frustum.name]
                        # Remove the camera edit panel as well
                        self.camera_edit_panel.remove()
                        self.camera_edit_panel = None
                        
                    @save_button.on_click
                    def _(_):
                        # camera_idx = self.frustums.index(frustum)
                        camera_idx = int(frustum.name.split("_")[-1])
                        
                        with h5py.File(self.camera_poses_file, 'a') as f:
                            if 'poses' in f:
                                poses = f['poses'][:]
                                poses[camera_idx] = np.array([*frustum.position, *frustum.wxyz])
                                del f['poses']
                                f.create_dataset('poses', data=poses, maxshape=(None, 7), chunks=True)
                        save_button.disabled = True
                        
            except Exception as e:
                print(f"Error in click callback: {e}")
        
        return click_handler
