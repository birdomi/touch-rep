from typing import List
import os
import glob
import numpy as np
import pickle
import h5py

from torch.utils import data

from tactile_ssl.data.xela_tdex.utils import TactileImage, get_tactile_augmentations


class TactileSSLDataset(data.Dataset):
    # Dataset for all possible tactile types (stacked, whole hand, one sensor)
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        if config.get("duration") is None:
            config.duration = 120
        if config.get("shuffle_type") is None:
            config.shuffle_type = None

        data_path = config.data_path
        tactile_information_type = config.tactile_information_type
        tactile_img_size = config.tactile_img_size
        duration = config.duration
        shuffle_type = config.shuffle_type

        self.roots = glob.glob(f"{data_path}/demonstration_*")
        self.data = self._load_data(demos_to_use=[], duration=duration)
        assert tactile_information_type in [
            "stacked",
            "whole_hand",
            "single_sensor",
        ], 'tactile_information_type can either be "stacked", "whole_hand" or "single_sensor"'
        self.tactile_information_type = tactile_information_type
        self.shuffle_type = shuffle_type

        # Set the transforms accordingly
        self.tactile_img = TactileImage(tactile_image_size=tactile_img_size, shuffle_type=shuffle_type)
        self.augmentations = get_tactile_augmentations(tactile_img_size)

    def _load_data(self, demos_to_use: List = [], duration: int = 120):
        roots = sorted(self.roots)

        tactile_indices = []
        allegro_indices = []
        allegro_action_indices = []
        kinova_indices = []
        image_indices = []

        tactile_values = {}
        allegro_tip_positions = {}
        allegro_joint_positions = {}
        allegro_joint_torques = {}
        allegro_actions = {}
        kinova_states = {}

        for demo_id, root in enumerate(roots):
            demo_num = int(root.split("/")[-1].split("_")[-1])
            if (len(demos_to_use) > 0 and demo_num in demos_to_use) or (
                len(demos_to_use) == 0
            ):  # If it's empty then it will be ignored
                with open(os.path.join(root, "tactile_indices.pkl"), "rb") as f:
                    tactile_indices += pickle.load(f)
                with open(os.path.join(root, "allegro_indices.pkl"), "rb") as f:
                    allegro_indices += pickle.load(f)
                with open(os.path.join(root, "allegro_action_indices.pkl"), "rb") as f:
                    allegro_action_indices += pickle.load(f)
                with open(os.path.join(root, "kinova_indices.pkl"), "rb") as f:
                    kinova_indices += pickle.load(f)
                with open(os.path.join(root, "image_indices.pkl"), "rb") as f:
                    image_indices += pickle.load(f)

                # Load the data
                with h5py.File(os.path.join(root, "allegro_fingertip_states.h5"), "r") as f:
                    allegro_tip_positions[demo_id] = f["positions"][()]
                with h5py.File(os.path.join(root, "allegro_joint_states.h5"), "r") as f:
                    allegro_joint_positions[demo_id] = f["positions"][()]
                    allegro_joint_torques[demo_id] = f["efforts"][()]
                with h5py.File(os.path.join(root, "allegro_commanded_joint_states.h5"), "r") as f:
                    allegro_actions[demo_id] = f["positions"][
                        ()
                    ]  # Positions are to be learned - since this is a position control
                with h5py.File(os.path.join(root, "touch_sensor_values.h5"), "r") as f:
                    tactile_values[demo_id] = f["sensor_values"][()]
                with h5py.File(os.path.join(root, "kinova_cartesian_states.h5"), "r") as f:
                    state = np.concatenate([f["positions"][()], f["orientations"][()]], axis=1)
                    kinova_states[demo_id] = state

        # Find the total lengths now
        whole_length = len(tactile_indices)
        desired_len = int((duration / 120) * whole_length)

        data = dict(
            tactile=dict(indices=tactile_indices[:desired_len], values=tactile_values),
            allegro_joint_states=dict(
                indices=allegro_indices[:desired_len],
                values=allegro_joint_positions,
                torques=allegro_joint_torques,
            ),
            allegro_tip_states=dict(indices=allegro_indices[:desired_len], values=allegro_tip_positions),
            allegro_actions=dict(indices=allegro_action_indices[:desired_len], values=allegro_actions),
            kinova=dict(indices=kinova_indices[:desired_len], values=kinova_states),
            image=dict(indices=image_indices[:desired_len]),
        )

        return data

    def _preprocess_tactile_indices(self):
        self.tactile_mapper = np.zeros(len(self.data["tactile"]["indices"]) * 15).astype(int)
        for data_id in range(len(self.data["tactile"]["indices"])):
            for sensor_id in range(15):
                self.tactile_mapper[data_id * 15 + sensor_id] = data_id  # Assign each finger to an index basically

    def _get_sensor_id(self, index):
        return index % 15

    def __len__(self):
        if self.tactile_information_type == "single_sensor":
            return len(self.tactile_mapper)
        else:
            return len(self.data["tactile"]["indices"])

    def _get_proper_tactile_value(self, index):
        if self.tactile_information_type == "single_sensor":
            data_id = self.tactile_mapper[index]
            demo_id, tactile_id = self.data["tactile"]["indices"][data_id]
            sensor_id = self._get_sensor_id(index)
            tactile_value = self.data["tactile"]["values"][demo_id][tactile_id][sensor_id]

            return tactile_value

        else:
            demo_id, tactile_id = self.data["tactile"]["indices"][index]
            tactile_values = self.data["tactile"]["values"][demo_id][tactile_id]

            return tactile_values

    def _get_tactile_image(self, tactile_values):
        return self.tactile_img.get(type=self.tactile_information_type, tactile_values=tactile_values)

    def __getitem__(self, index):
        tactile_value = self._get_proper_tactile_value(index)
        tactile_image = self._get_tactile_image(tactile_value)

        aug_tactile_image = self.augmentations(tactile_image)
        return {
            "image": tactile_image,
            "aug_image": aug_tactile_image,
            "tactile_values": tactile_value,
        }
