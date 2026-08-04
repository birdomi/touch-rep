"""
BrainCo Grasp Detection Dataset — Vision only (RGB frames from colors/).

Each episode is split into windows of frames using the same windowing
parameters as the tactile dataset.  For each window, N evenly-spaced
frames are sampled and returned as a (T, C, H, W) tensor.
"""

import json
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import DictConfig

from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEFAULT_LABEL_DIRS = {"grasp_success": 1, "grasp_fail": 0}


def _get_label_dirs(label_dirs):
    return [(str(name), int(label)) for name, label in label_dirs.items()]


class BraincoGraspVisionDataset(data.Dataset):
    """Episode-level grasp success/fail classification using RGB images only.

    Mirrors BraincoGraspDetectionDataset's episode_data / windows attributes
    so the same k-fold split logic in train_task_brainco_vision.py works
    without modification.
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        num_frames_per_sample: int = 4,
        img_size: int = 224,
        augment: bool = False,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.num_frames_per_sample = num_frames_per_sample
        self.img_size = int(img_size)
        self.crop_mode = str(config.get("crop_mode", "resize"))
        self.crop_size = int(config.get("crop_size", self.img_size))

        # Window parameters (kept identical to tactile dataset)
        self.window_time = config.window_time
        self.window_overlap = config.get("window_overlap", 0.0)
        self.interpolating_freq = config.interpolating_freq
        self.num_frames_per_window = int(round(self.window_time * self.interpolating_freq))
        self.shift_per_window = max(1, int(round(self.num_frames_per_window * (1.0 - self.window_overlap))))
        self.data_roots = self._get_data_roots(config, data_path)

        # "legacy" keeps the original window enumeration; "tactile" mirrors
        # BraincoSSLDataset.data_idxs so a vision run lines up window-for-window
        # with the tactile datasets.
        self.window_mode = str(config.get("window_mode", "legacy"))
        if self.window_mode not in {"legacy", "tactile"}:
            raise ValueError(
                f"window_mode must be 'legacy' or 'tactile', got {self.window_mode!r}"
            )
        # Explicit 1-indexed frame picks inside the window (e.g. [1, 5, 15]).
        # When unset, frames are spaced evenly across the window.
        frame_positions = config.get("frame_positions")
        self.frame_positions = (
            [int(position) for position in frame_positions]
            if frame_positions is not None else None
        )
        if self.frame_positions is not None:
            if not self.frame_positions:
                raise ValueError("frame_positions must not be empty")
            for position in self.frame_positions:
                if not 1 <= position <= self.num_frames_per_window:
                    raise ValueError(
                        "frame_positions entries must be within "
                        f"[1, {self.num_frames_per_window}], got {position}"
                    )
            if self.num_frames_per_sample != len(self.frame_positions):
                raise ValueError(
                    "num_frames_per_sample must match len(frame_positions): "
                    f"{self.num_frames_per_sample} vs {len(self.frame_positions)}"
                )

        # "box" keeps a fixed [left, top, width, height] region of the raw frame
        # — used to cut the room away and leave only the object and the fingers.
        crop_box = config.get("crop_box")
        self.crop_box = [int(v) for v in crop_box] if crop_box is not None else None
        if self.crop_mode == "box":
            if self.crop_box is None or len(self.crop_box) != 4:
                raise ValueError(
                    "crop_mode='box' needs crop_box=[left, top, width, height]"
                )
            if min(self.crop_box[2], self.crop_box[3]) <= 0:
                raise ValueError("crop_box width and height must be positive")

        # Transforms
        if self.crop_mode == "box":
            self.transform = self._build_crop_transform(augment, self._box_crop)
        elif self.crop_mode == "bottom_center":
            self.transform = self._build_bottom_center_transform(augment)
        elif augment:
            self.transform = T.Compose([
                T.Resize((self.img_size + 32, self.img_size + 32)),
                T.RandomCrop(self.img_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        # Discover + load episodes
        self.episode_data: List[dict] = []
        self.windows: List[dict] = []
        self._load_all_episodes()

        log.info(
            f"BraincoGraspVisionDataset: {len(self.episode_data)} episodes, "
            f"{len(self.windows)} total windows."
        )

    def _build_bottom_center_transform(self, augment: bool):
        return self._build_crop_transform(augment, self._bottom_center_crop)

    def _build_crop_transform(self, augment: bool, crop_fn):
        transforms = []
        if augment:
            transforms.extend([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomHorizontalFlip(),
            ])
        transforms.extend([
            T.Lambda(crop_fn),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return T.Compose(transforms)

    def _box_crop(self, img: Image.Image) -> Image.Image:
        left, top, width, height = self.crop_box
        cropped = TF.crop(img, top, left, height, width)
        if (width, height) != (self.img_size, self.img_size):
            cropped = TF.resize(cropped, [self.img_size, self.img_size])
        return cropped

    def _bottom_center_crop(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        crop_size = self.crop_size

        if width < crop_size or height < crop_size:
            scale = max(crop_size / width, crop_size / height)
            new_width = max(crop_size, int(round(width * scale)))
            new_height = max(crop_size, int(round(height * scale)))
            img = TF.resize(img, [new_height, new_width])
            width, height = img.size

        left = max(0, (width - crop_size) // 2)
        top = max(0, height - crop_size)
        cropped = TF.crop(img, top, left, crop_size, crop_size)
        if crop_size != self.img_size:
            cropped = TF.resize(cropped, [self.img_size, self.img_size])
        return cropped

    def _discover_episodes(self):
        episodes = []
        for data_root, label_dirs in self.data_roots:
            log.info(f"Discovering vision episodes from: {data_root}")
            for label_name, label_val in label_dirs:
                label_dir = data_root / label_name
                if not label_dir.exists():
                    log.warning(f"{label_dir} not found, skipping.")
                    continue
                for ep_dir in sorted(d for d in label_dir.iterdir()
                                      if d.is_dir() and d.name.startswith("episode_")):
                    if (ep_dir / "data.json").exists():
                        episodes.append((ep_dir, label_val))
        return episodes

    def _get_data_roots(self, config: DictConfig, data_path: str):
        data_roots = config.get("data_roots")
        if data_roots:
            roots = []
            for root_cfg in data_roots:
                label_dirs = root_cfg.get("label_dirs", config.get("label_dirs", DEFAULT_LABEL_DIRS))
                roots.append((Path(str(root_cfg.data_path)), _get_label_dirs(label_dirs)))
            return roots

        label_dirs = config.get("label_dirs", DEFAULT_LABEL_DIRS)
        return [(Path(data_path), _get_label_dirs(label_dirs))]

    def _load_all_episodes(self):
        for ep_path, label in self._discover_episodes():
            colors_dir = ep_path / "colors"
            if not colors_dir.exists():
                log.warning(f"No colors/ in {ep_path}, skipping.")
                continue

            img_files = sorted(
                f for f in colors_dir.iterdir()
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if not img_files:
                log.warning(f"No images in {colors_dir}, skipping.")
                continue

            with open(ep_path / "data.json") as f:
                num_frames = len(json.load(f)["data"])

            if self.window_mode == "tactile":
                max_start = num_frames - self.num_frames_per_window
                window_starts = np.arange(0, max(1, max_start + 1), self.shift_per_window)
            else:
                # Use the same window formula as BraincoGraspDetectionDataset
                max_length = num_frames - (num_frames % self.num_frames_per_window)
                max_length = max_length - self.num_frames_per_window
                window_starts = np.arange(0, max(1, max_length), self.shift_per_window)

            ep_data = {
                "path": str(ep_path),
                "label": label,
                "img_files": img_files,
                "num_frames": num_frames,
                "window_starts": window_starts,
            }
            self.episode_data.append(ep_data)

            for start in window_starts:
                self.windows.append({
                    "ep_idx": len(self.episode_data) - 1,
                    "window_start": int(start),
                    "label": torch.tensor(label, dtype=torch.long),
                    "episode_path": str(ep_path),
                    "window_start_frame": int(start),
                })

            print(f"Loaded {ep_path.name}: {num_frames} frames -> {len(window_starts)} windows")

    def _sample_frames(self, img_files, window_start: int) -> torch.Tensor:
        """Take the configured frames from the window (evenly spaced by default)."""
        total = len(img_files)
        end = window_start + self.num_frames_per_window - 1
        if self.frame_positions is not None:
            indices = np.array(
                [window_start + position - 1 for position in self.frame_positions],
                dtype=int,
            )
        else:
            indices = np.linspace(window_start, end, self.num_frames_per_sample, dtype=int)
        indices = np.clip(indices, 0, total - 1)

        frames = []
        for idx in indices:
            img = Image.open(str(img_files[idx])).convert("RGB")
            frames.append(self.transform(img))
        return torch.stack(frames)  # (T, C, H, W)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        ep = self.episode_data[w["ep_idx"]]
        frames = self._sample_frames(ep["img_files"], w["window_start"])
        return {
            "frames": frames,                          # (T, C, H, W)
            "label": w["label"],                       # scalar long
            "episode_path": w["episode_path"],
            "window_start_frame": w["window_start_frame"],
        }
