"""
BrainCo Grasp Multimodal Dataset — tactile + RGB vision.

Extends BraincoGraspDetectionDataset by additionally loading image frames
from {episode}/colors/.  Each window sample returns:
  - sensor        : (W, num_sensors, 4)   — tactile
  - sensor_poses  : (W, 10, 6)            — fingertip 6D poses
  - frames        : (T, C, H, W)          — sampled RGB frames
  - label         : long scalar
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from omegaconf import DictConfig

from tactile_ssl.data.brainco_grasp_dataset import BraincoGraspDetectionDataset
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class BraincoGraspMultimodalDataset(BraincoGraspDetectionDataset):
    """Tactile + vision multimodal dataset.

    Inherits all tactile loading from BraincoGraspDetectionDataset and
    additionally loads RGB frames from colors/ for each window.
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        brainco_urdf_path: str = "dataset/brainco/urdf",
        num_frames_per_sample: int = 4,
        img_size: int = 224,
        augment: bool = False,
    ):
        # Load all tactile data via parent
        super().__init__(config=config, data_path=data_path, brainco_urdf_path=brainco_urdf_path)

        self.num_frames_per_sample = num_frames_per_sample

        if augment:
            self.img_transform = T.Compose([
                T.Resize((img_size + 32, img_size + 32)),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.img_transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        # Build episode_path → sorted image file list
        self._ep_img_files: dict = {}
        for ep in self.episode_data:
            colors_dir = Path(ep["path"]) / "colors"
            if colors_dir.exists():
                img_files = sorted(
                    f for f in colors_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
                )
                self._ep_img_files[ep["path"]] = img_files
            else:
                log.warning(f"No colors/ in {ep['path']}, vision will be zeros.")
                self._ep_img_files[ep["path"]] = []

        # Add episode_path + window_start_frame to each window (for frame lookup)
        ep_window_start = {}
        current_idx = 0
        for ep in self.episode_data:
            ep_window_start[ep["path"]] = current_idx
            current_idx += len(ep["window_starts"])

        for ep in self.episode_data:
            base = ep_window_start[ep["path"]]
            for i, start in enumerate(ep["window_starts"]):
                w = self.windows[base + i]
                w["episode_path"] = ep["path"]
                w["window_start_frame"] = int(start)

    def _sample_frames(self, ep_path: str, window_start: int) -> torch.Tensor:
        img_files = self._ep_img_files.get(ep_path, [])
        C, H, W = 3, 224, 224
        if not img_files:
            return torch.zeros(self.num_frames_per_sample, C, H, W)

        total = len(img_files)
        end   = window_start + self.num_frames_per_window - 1
        indices = np.linspace(window_start, end, self.num_frames_per_sample, dtype=int)
        indices = np.clip(indices, 0, total - 1)

        frames = []
        for idx in indices:
            img = Image.open(str(img_files[idx])).convert("RGB")
            frames.append(self.img_transform(img))
        return torch.stack(frames)  # (T, C, H, W)

    def __getitem__(self, idx):
        w = self.windows[idx]
        tactile_sample = super().__getitem__(idx)   # sensor, sensor_poses, label, episode_path, window_start_frame
        frames = self._sample_frames(w["episode_path"], w["window_start_frame"])
        tactile_sample["frames"] = frames
        return tactile_sample
