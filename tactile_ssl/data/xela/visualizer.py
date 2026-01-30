from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import einops
import numpy as onp
import trimesh
import tyro
import viser
import viser.transforms as tf
import yourdfpy
from omegaconf import OmegaConf
from tqdm import tqdm

# from tactile_ssl.data.reskin.visualizer2d import plot_magnetic_heatmap
from tactile_ssl.data.xela_tactile import XelaSSLDataset
from tactile_ssl.data.xela.utils import (
    XELA_FLATTEN_ORDER,
    compute_xela_normalization,
    create_sensor_image,
    pad_xela_sample,
)


class ViserUrdf:
    """Helper for rendering URDFs in Viser."""

    def __init__(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        urdf_path: Path,
        scale: float = 1.0,
        root_node_name: str = "/",
        mesh_color_override: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        assert root_node_name.startswith("/")
        assert len(root_node_name) == 1 or not root_node_name.endswith("/")

        urdf = yourdfpy.URDF.load(
            urdf_path,
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent),
        )
        assert isinstance(urdf, yourdfpy.URDF)

        self._target = target
        self._urdf = urdf
        self._scale = scale
        self._root_node_name = root_node_name

        # Add coordinate frame for each joint.
        self._joint_frames: List[viser.SceneNodeHandle] = []
        self.base_joint_frame = self._target.add_frame(
            self._root_node_name,
            show_axes=False,
            axes_length=0.1,
            axes_radius=0.01,
        )
        for joint in self._urdf.joint_map.values():
            assert isinstance(joint, yourdfpy.Joint)
            joint_name = _viser_name_from_frame(self._urdf, joint.child, self._root_node_name)
            frame = self._target.add_frame(joint_name, show_axes=False)
            self._joint_frames.append(frame)

        # self.sensor_frames = []
        # for sensor_id in range(368):
        #     self.sensor_frames.append(
        #         self._target.add_frame(f"sensor_{sensor_id}", show_axes=True, axes_length=0.01, axes_radius=0.00075)
        #     )

        # Add the URDF's meshes/geometry to viser.
        self.xela_link_names = []
        self.xela_image_handle_names = []
        self.xela_image_handles = []
        self.xela_extents = []
        for link_name, mesh in urdf.scene.geometry.items():
            assert isinstance(mesh, trimesh.Trimesh)
            T_parent_child = urdf.get_transform(link_name, urdf.scene.graph.transforms.parents[link_name])
            name = _viser_name_from_frame(urdf, link_name, root_node_name)
            # Scale the mesh. (this will mutate it)
            mesh.apply_scale(self._scale)
            if mesh_color_override is None:
                target.add_mesh_trimesh(
                    name,
                    mesh,
                    wxyz=tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz,
                    position=T_parent_child[:3, 3] * scale,
                )
            else:
                target.add_mesh_simple(
                    name,
                    mesh.vertices,
                    mesh.faces,
                    color=mesh_color_override,
                    opacity=0.05,
                    wxyz=tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz,
                    position=T_parent_child[:3, 3] * scale,
                )
            for k, v in XELA_FLATTEN_ORDER.items():
                if k in name:
                    width, height, depth = mesh.bounding_box.extents
                    self.xela_extents.append((width, height, depth))
                    position = T_parent_child[:3, 3] + onp.array([width / 2, height / 2, depth + 0.001])
                    r = T_parent_child[:3, :3]
                    r = r @ onp.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    orientation = tf.SO3.from_matrix(r).wxyz
                    if "aftc" in k:  # curved fingertip sensor
                        position = T_parent_child[:3, 3] + onp.array([0, height / 2, depth + 0.001])
                    self.xela_link_names.append(k)
                    self.xela_image_handle_names.append(name + "_image")
                    self.xela_image_handles.append(
                        target.add_image(
                            name + "_image",
                            onp.random.randint(0, 256, (128, 128, 3), dtype=onp.uint8),
                            render_width=width * scale,
                            render_height=height * scale,
                            wxyz=orientation,
                            position=position,
                            visible=True,
                        )
                    )

    def update_cfg(
        self,
        configuration: onp.ndarray,
    ) -> None:
        """Update the joint angles of the visualized URDF."""
        self._urdf.update_cfg(configuration)
        with self._target.atomic():
            for joint, frame_handle in zip(self._urdf.joint_map.values(), self._joint_frames):
                assert isinstance(joint, yourdfpy.Joint)
                T_parent_child = self._urdf.get_transform(joint.child, joint.parent)
                frame_handle.wxyz = tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz
                frame_handle.position = T_parent_child[:3, 3] * self._scale

    def get_actuated_joint_limits(
        self,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Returns an ordered mapping from actuated joint names to position limits."""
        out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for joint_name, joint in zip(self._urdf.actuated_joint_names, self._urdf.actuated_joints):
            assert isinstance(joint_name, str)
            assert isinstance(joint, yourdfpy.Joint)
            assert joint.limit is not None
            out[joint_name] = (joint.limit.lower, joint.limit.upper)
        return out

    def get_actuated_joint_names(self) -> Tuple[str, ...]:
        """Returns a tuple of actuated joint names, in order."""
        return tuple(self._urdf.actuated_joint_names)


def _viser_name_from_frame(
    urdf: yourdfpy.URDF,
    frame_name: str,
    root_node_name: str = "/",
) -> str:
    """Given the (unique) name of a frame in our URDF's kinematic tree, return a
    scene node name for viser.

    For a robot manipulator with four frames, that looks like:


            ((shoulder)) == ((elbow))
               / /             |X|
              / /           ((wrist))
         ____/ /____           |X|
        [           ]       [=======]
        [ base_link ]        []   []
        [___________]


    this would map a name like "elbow" to "base_link/shoulder/elbow".
    """
    assert root_node_name.startswith("/")
    assert len(root_node_name) == 1 or not root_node_name.endswith("/")

    frames = []
    while frame_name != urdf.scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = urdf.scene.graph.transforms.parents[frame_name]
    return root_node_name + "/" + "/".join(frames[::-1])


# def create_magfield_image(mesh_id, magfield, sensor_positions):
#     coords = sensor_positions * 1000
#     height = 23.20
#     width = 22.20
#     resolution = 128
#     if "palm" in mesh_id:
#         height = 37.17
#         width = 96.46
#     image = plot_magnetic_heatmap(magfield, coords, resolution, height, width, scale=0.1, colormap="jet")
#     return image


def add_joint_angle_sliders(urdf: ViserUrdf, server: viser.ViserServer) -> None:
    initial_angles = onp.zeros([urdf._urdf.num_actuated_joints], dtype=onp.float32)
    initial_angles[12] = 0.4
    initial_angles[13] = 1.0
    with server.add_gui_folder("Joint Angles"):
        gui_joints: List[viser.GuiInputHandle[float]] = []
        for i, (joint_name, (lower, upper)) in enumerate(urdf.get_actuated_joint_limits().items()):
            lower = lower if lower is not None else -onp.pi
            upper = upper if upper is not None else onp.pi

            initial_angle = initial_angles[i]
            slider = server.add_gui_slider(
                label=joint_name,
                min=lower,
                max=upper,
                step=1e-3,
                initial_value=initial_angle,
            )
            slider.on_update(  # When sliders move, we update the URDF configuration.
                lambda _: urdf.update_cfg(onp.array([gui.value for gui in gui_joints]))
            )

            gui_joints.append(slider)

        reset_button = server.add_gui_button("Reset")

        @reset_button.on_click
        def _(_):
            for g, initial_angle in zip(gui_joints, initial_angles):
                g.value = initial_angle

    return gui_joints


def add_playback_controls(urdf: ViserUrdf, server: viser.ViserServer, num_frames: int) -> None:
    with server.add_gui_folder("Playback"):
        gui_timestep = server.add_gui_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.add_gui_button("Next Frame", disabled=True)
        gui_prev_frame = server.add_gui_button("Prev Frame", disabled=True)
        gui_playing = server.add_gui_checkbox("Playing", False)
        gui_framerate = server.add_gui_slider("FPS", min=1, max=20, step=1, initial_value=10)
        gui_framerate_options = server.add_gui_button_group("FPS options", ("10", "20"))

    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    return gui_timestep, gui_playing, gui_framerate


def add_thirdperson_cameraview(server: viser.ViserServer, image=None) -> None:
    if image is None:
        image = onp.random.randint(0, 256, (480, 640, 3), dtype=onp.uint8)
    orientation = tf.SO3.from_matrix(onp.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]))
    server.add_image(
        "realsense",
        image,
        render_width=(640 / 1000) * 0.25,
        render_height=(480 / 1000) * 0.25,
        wxyz=orientation.wxyz,
        position=(0.0, 0, 0.25),
        visible=True,
    )


def load_data(data_path: str, baseline_signal_path, urdf_path: str, num_frames: int) -> None:
    dataset = XelaSSLDataset(
        config=OmegaConf.create(
            {
                "window_time": 1.0 / 30,
                "interpolating_freq": 30,
                "subtract_baseline": True,
                "normalize": False,
                "smooth_data": False,
            }
        ),
        xela_urdf_path=urdf_path,
        baseline_signal_path=baseline_signal_path,
        data_path=data_path,
        load_images=False,
    )
    print(f"Length of dataset: {len(dataset)}")
    num_frames = min(num_frames, len(dataset))
    joint_states = []
    xela_images = []
    sensor_poses_ = []
    camera_images = []
    palm_sensor_grid_data = []
    xela_mean, xela_std = compute_xela_normalization([dataset])
    print(f"Xela mean: {xela_mean}, Xela std: {xela_std}")
    for i in tqdm(range(num_frames)):
        sample = dataset[i]
        joint_angles = None
        if "joint_angles" in sample.keys():
            joint_angles = sample["joint_angles"][0].numpy()

        if "sensor_poses" in sample.keys():
            sensor_poses = sample["sensor_poses"][0].numpy()

        sensor = sample["sensor"].numpy()
        sensor[..., :3] = (sensor[..., :3] - xela_mean[None, None]) / xela_std[None, None]
        sensor = pad_xela_sample(sample["sensor"].numpy())[0]
        # color_image = (sample["color_images"].numpy() * 255).astype(onp.uint8)
        # color_image = einops.rearrange(color_image, "k c h w -> k h w c")
        xela_sensor_data = {}
        prev_idx = 0
        for i, k in enumerate(XELA_FLATTEN_ORDER.keys()):
            sensor_ = sensor[i, :]
            sensor_ = einops.rearrange(sensor_, "(n c) -> n c", c=3)[..., :3]
            sensor_grid = None
            if "aftc" in k:  # curved fingertip sensor
                sensor_grid = onp.zeros((6, 6, 3))
                sensor_grid[2:, :] = einops.rearrange(sensor_[6:], "(h w) c -> h w c", h=4, w=6)
                sensor_grid[0, 2:-2] = sensor_[0:2]
                sensor_grid[1, 1:-1] = sensor_[2:6]
            elif "4x4" in k:  # 4x4 sensor flat sensor
                sensor_ = sensor_[:16]
                sensor_grid = einops.rearrange(sensor_, "(h w) c -> h w c", h=4, w=4)
            elif "4x6" in k:
                sensor_ = sensor_[:24]
                sensor_grid = einops.rearrange(sensor_, "(h w) c -> h w c", h=4, w=6)
                if "palm_2" in k:
                    palm_sensor_grid_data.append(sensor_grid)
            else:
                raise ValueError("Unknown sensor type")
            sensor_image = create_sensor_image(k, sensor_grid, resolution=256)
            xela_sensor_data[k] = sensor_image

        xela_images.append(xela_sensor_data)
        joint_states.append(joint_angles)
        sensor_poses_.append(sensor_poses)
    palm_sensor_grid_data = onp.stack(palm_sensor_grid_data, axis=0)
    import matplotlib.pyplot as plt

    # palm_sensor_grid_data = palm_sensor_grid_data.reshape(1000, -1, 3)
    # time = onp.arange(250, 300, 1)
    # for i in range(24):
    #     plt.plot(time, palm_sensor_grid_data[250:300, i, 0], label="x", color="r")
    #     plt.plot(time, palm_sensor_grid_data[250:300, i, 1], label="y", color="g")
    #     plt.plot(time, palm_sensor_grid_data[250:300, i, 2], label="z", color="b")
    # plt.show()

    # for i in range(color_image.shape[0]):
    #     camera_images.append(color_image[i])
    return joint_states, xela_images, num_frames, sensor_poses_


def main(urdf_path: str, data_path: str, baseline_signal_path: str) -> None:
    server = viser.ViserServer()
    urdf = ViserUrdf(server, Path(urdf_path), mesh_color_override=(0, 0, 255))

    num_frames = 3000
    joint_states, xela_images, num_frames, sensor_poses = load_data(
        data_path, baseline_signal_path, urdf_path, num_frames
    )
    add_thirdperson_cameraview(server)
    gui_timestep, gui_playing, gui_framerate = add_playback_controls(urdf, server, num_frames)
    gui_joints = add_joint_angle_sliders(urdf, server)

    # Apply initial joint angles to the URDF
    urdf.update_cfg(onp.array([gui.value for gui in gui_joints]))

    # Load the dataset

    @gui_timestep.on_update
    def _(_) -> None:
        with server.atomic():
            current_timestep = gui_timestep.value
            # Update URDF configuration via joint_states
            # urdf.update_cfg(joint_states[current_timestep])
            # sensor_pose = sensor_poses[current_timestep]
            # for k, sensor_frame_handle in enumerate(urdf.sensor_frames):
            #     sensor_frame_handle.position = sensor_pose[k, :3]
            #     # xyzw = sensor_pose[k, 3:]
            #     # print(f"xyzw: {xyzw}")
            #     sensor_frame_handle.wxyz = sensor_pose[k, 3:]  # onp.array([xyzw[3], xyzw[0], xyzw[1], xyzw[0]])

            # color_image = color_images[current_timestep]
            # color_image = onp.clip(color_image, 0, 255)
            # add_thirdperson_cameraview(server, image=color_image)

            # Update the xela images
            for link_name, image_handle_name, image_handle, extents in zip(
                urdf.xela_link_names,
                urdf.xela_image_handle_names,
                urdf.xela_image_handles,
                urdf.xela_extents,
            ):
                image = xela_images[current_timestep][link_name]
                width, height = extents[:2]

                server.add_image(
                    image_handle_name,
                    image,
                    render_width=width,
                    render_height=height,
                    wxyz=image_handle.wxyz,
                    position=image_handle.position,
                    visible=True,
                )
        server.flush()

    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
