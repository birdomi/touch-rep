import os
import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig
from PIL import Image
import torchvision.transforms.functional as F


class ActionSenseSSLDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        sequences: List[str],
        data_path: str
    ):
        self.config = config
        self.sequences = sequences
        self.data_path = data_path

        self.sensor_data = []
        self.sensor_poses = []
        self.object_classes = []
        self.sequence_indices = []
        total_frames = 0

        for class_id, sequence in enumerate(self.sequences):
            # Construct the path to the gelsight_frame folder
            # Assuming the sequence corresponds to the folder name in data_path
            segments = [
                os.path.join(self.data_path, sequence, x, "segment_data.npz") for x in 
                os.listdir(os.path.join(self.data_path, str(sequence)))
            ]
            
            for segment in segments:
                if not os.path.exists(segment):
                    print(f"Warning: Path {segment} does not exist. Skipping.")
                    continue

                try:
                    with np.load(segment, allow_pickle=True) as npz:
                        # Assuming 'hand_joint_tactile' shape is (N, 16, 16) or similar
                        # Assuming 'hand_poses' shape is (N, 7) or (N, 4, 4)
                        tactile = npz["hand_joint_tactile"]
                        poses = npz["hand_poses"]
                        
                        n_frames = tactile.shape[0]
                        
                        self.sensor_data.append(tactile)
                        self.sensor_poses.append(poses)
                        self.object_classes.append(np.full(n_frames, class_id))
                        
                        self.sequence_indices.append((total_frames, total_frames + n_frames))
                        total_frames += n_frames

                except Exception as e:
                    print(f"Error loading segment {segment}: {e}")

        if not self.sensor_data:
            print("Warning: No data loaded!")
            return

        # Concatenate all data
        self.sensor_data = np.concatenate(self.sensor_data, axis=0)
        self.sensor_data = np.reshape(self.sensor_data, (self.sensor_data.shape[0], 1, 40, 25))
        self.sensor_poses = np.concatenate(self.sensor_poses, axis=0)
        self.sensor_poses = np.expand_dims(self.sensor_poses, axis=1)

        self.object_classes = np.concatenate(self.object_classes, axis=0)
        
        # Create valid indices (all frames are valid now)
        self.valid_indices = np.arange(len(self.sensor_data))

        print(f"Loaded {len(self.valid_indices)} frames from {len(self.sequences)} sequences.", self.sensor_data.shape)

        print(f"Loaded {len(self.valid_indices)} frames from {len(self.sequences)} sequences.", self.sensor_data.shape)
        print(f"Total frames: {total_frames}")

        self.sensor_mean = None
        self.sensor_std = None

    def update_normalization(self, sensor_mean, sensor_std):
        if not isinstance(sensor_mean, torch.Tensor):
            sensor_mean = torch.tensor(sensor_mean).float()
        if not isinstance(sensor_std, torch.Tensor):
            sensor_std = torch.tensor(sensor_std).float()
            
        # Ensure correct shape for broadcasting (1, C) or (C,) depending on data shape
        # sensor_data: (1, 40, 25). Normalizing across channels (dim 2)?
        # If mean is (C,), we need to reshape it.
        if sensor_mean.dim() == 1:
            # (C,) -> (1, 1, C)
            sensor_mean = sensor_mean.view(1, 1, -1)
            sensor_std = sensor_std.view(1, 1, -1)
            
        self.sensor_mean = sensor_mean
        self.sensor_std = sensor_std

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        start_frame = self.valid_indices[index]        
        class_id = self.object_classes[start_frame] 
        
        # Convert to torch tensor
        sensor_tensor = torch.from_numpy(self.sensor_data[start_frame]).float()
        # Convert to torch tensor
        sensor_tensor = torch.from_numpy(self.sensor_data[start_frame]).float()
        poses_tensor = torch.from_numpy(self.sensor_poses[start_frame]).float()
                
        sample_dict = {
            "sensor": sensor_tensor,
            "sensor_poses": poses_tensor,
            "object_classification": torch.tensor(class_id)
        }
        return sample_dict

if __name__ == '__main__':
    from omegaconf import OmegaConf
    sequences = [
        'Clean_a_pan_with_a_sponge'
    ]
    data_path = 'dataset/action_sense'
    
    # Create a dummy config
    config = OmegaConf.create({
        "window_time": 0.1,
        "interpolating_freq": 100,
        "window_overlap": 0.5
    })

    dataset = ActionSenseSSLDataset(config=config, sequences=sequences, data_path=data_path)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sensor shape:", sample['sensor'].shape)
        print("Poses shape:", sample['sensor_poses'].shape)
        print("Class ID:", sample['object_classification'])
    else:
        print("Dataset is empty.")
