from typing import Optional, List
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from omegaconf import DictConfig

from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

class DigitSSLDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        object_class: Optional[int] = None,
        # Unused arguments for compatibility with XelaSSLDataset signature
        xela_urdf_path: Optional[str] = None, 
        baseline_signal_path: Optional[str] = None,
        load_images: bool = False, 
    ):
        self.object_label = object_class
        
        self.data_path = Path(data_path)
        self.frames_path = self.data_path / "frames"
        
        if not self.frames_path.exists():
             raise FileNotFoundError(f"Frames directory not found at {self.frames_path}")

        # List all jpg files and sort them
        self.frame_filenames = sorted(list(self.frames_path.glob("*.jpg")))
        self.num_frames = len(self.frame_filenames)
        
        if self.num_frames == 0:
            log.warning(f"No frames found in {self.frames_path}")

        self.image_transform = transforms.ToTensor()

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        sample_dict = {}
        
        frame_path = str(self.frame_filenames[idx])
        # CV2 reads in BGR, convert to RGB
        img = cv2.imread(frame_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {frame_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.image_transform(img)
        # Shape: [C, H, W]

        sample_dict.update({"sensor": img_tensor})
        
        # Return dummy data for other fields expected by the pipeline
        # xela had [16] angles. providing same.
        sample_dict.update({"joint_angles": torch.zeros((16,))})
        # sensor_poses: [1, 7] (pos+quat)
        sample_dict.update({"sensor_poses": torch.zeros((1, 7))})

        if self.object_label is not None:
             sample_dict.update({"object_classification": torch.tensor(self.object_label)})

        return sample_dict


if __name__ == "__main__":
    import os
    import hydra
    from omegaconf import OmegaConf

    def get_digit_datasets(data_cfg: DictConfig):
        def get_digit_dataset(dataset_cfg: DictConfig, dataset_name: str, d_id: int):
            data_path = f"{dataset_cfg.data_path}"
            full_path = Path(data_path) / dataset_name / d_id
            if not full_path.exists():
                print(f"Dataset {full_path} not found")
                return None

            print(f"Loading dataset from: {full_path}")
            dataset = DigitSSLDataset(
                dataset_cfg,
                data_path=str(full_path),
                object_class=0, # Dummy class
            )
            return dataset

        target_path = Path("/home/user/yyg/sparsh-multisensory-touch/ycb-slide-dataset/real/004_sugar_box/dataset_0")
        if target_path.exists():
             ds = DigitSSLDataset(
                 data_cfg.dataset,
                 data_path=str(target_path),
                 object_class=0
             )
             return [ds], []
        
        return [], []

    print("Using manual config")
    config = OmegaConf.create({
        "data": {
            "dataset": {
                 "data_path": "/home/user/yyg/sparsh-multisensory-touch/ycb-slide-dataset/real"
            }
        }
    })

    print(OmegaConf.to_yaml(config, resolve=True))
    
    # Test loading
    train_dsets, val_dsets = get_digit_datasets(config.data)
    
    if len(train_dsets) > 0:
        dset = train_dsets[0]
        print(f"Dataset length: {len(dset)}")
        sample = dset[0]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {type(v)}")
    else:
        print("No datasets loaded.")
