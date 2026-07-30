"""BrainCo slip-detection dataset over RGB frames (vision-only baseline).

Window construction mirrors
:class:`~tactile_ssl.data.brainco_xyz_slip_detection_dataset.BraincoXYZSlipDetectionDataset`
one-for-one: fixed-length windows of consecutive frames taken at a fixed
stride, labelled by the window's final frame, with windows touching the
configured slip-transition margins discarded. Frame indices in ``data.json``
address the tactile stream and ``colors/`` alike, so a ResNet baseline trained
here is scored on exactly the same samples as the tactile encoders.

Images inside one window share their augmentation parameters — an independent
random crop per frame would inject apparent motion, which is the very signal
slip detection reads.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from PIL import Image

from tactile_ssl.data.brainco_xyz_slip_detection_dataset import (
    _excluded_transition_frames,
    _labels_from_intervals,
    _read_slip_intervals,
)
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BraincoSlipVisionDataset(data.Dataset):
    """Aggregate slip episodes into non-overlapping windows of RGB frames.

    The public ``episode_data`` and ``windows`` attributes match the interface
    consumed by :mod:`train_task_brainco_vision`, which splits at the episode
    level and therefore avoids frame leakage across folds.
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        num_frames_per_sample: Optional[int] = None,
        img_size: int = 224,
        augment: bool = False,
    ):
        super().__init__()
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.crop_mode = str(config.get("crop_mode", "resize"))
        self.crop_size = int(config.get("crop_size", self.img_size))
        if self.crop_mode not in {"resize", "bottom_center"}:
            raise ValueError(
                f"crop_mode must be 'resize' or 'bottom_center', got {self.crop_mode!r}"
            )
        # Random-crop slack used only when augmenting, in pixels of the resized image.
        self.augment_pad = int(config.get("augment_pad", 16))
        self.augment_jitter = float(config.get("augment_jitter", 0.2))
        if self.augment_pad < 0:
            raise ValueError("augment_pad must be >= 0")

        camera_key = str(config.get("camera_key", "color_0"))
        input_window_frames = int(config.get("input_window_frames", 3))
        input_window_stride = int(config.get("input_window_stride", input_window_frames))
        exclude_before_slip_start_frames = int(
            config.get("exclude_before_slip_start_frames", 0)
        )
        exclude_after_slip_end_frames = int(
            config.get("exclude_after_slip_end_frames", 0)
        )
        # Keep only windows whose frames all carry the same label, so a window
        # never straddles a slip boundary while being scored by its last frame.
        require_uniform_label = bool(config.get("require_uniform_label", False))
        if input_window_frames <= 0:
            raise ValueError("input_window_frames must be positive")
        if input_window_stride < input_window_frames:
            raise ValueError(
                "input_window_stride must be at least input_window_frames "
                "to avoid overlap"
            )
        if exclude_before_slip_start_frames < 0 or exclude_after_slip_end_frames < 0:
            raise ValueError("Slip transition exclusion margins must be >= 0")

        self.input_window_frames = input_window_frames
        self.num_frames_per_sample = int(
            num_frames_per_sample
            if num_frames_per_sample is not None
            else input_window_frames
        )
        if self.num_frames_per_sample <= 0:
            raise ValueError("num_frames_per_sample must be positive")

        root = Path(data_path)
        class_names = list(config.get("classes", []))
        if not class_names:
            class_names = sorted(path.name for path in root.iterdir() if path.is_dir())

        self.episode_data: List[dict] = []
        self.windows: List[dict] = []
        num_excluded_windows = 0
        num_mixed_windows = 0

        for class_name in class_names:
            class_dir = root / class_name
            if not class_dir.exists():
                log.warning(f"Class directory not found, skipping: {class_dir}")
                continue

            episode_dirs = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_dir() and (path / "data.json").exists()
            )
            log.info(f"Loading {len(episode_dirs)} slip episodes from: {class_dir}")

            for episode_path in episode_dirs:
                try:
                    image_paths = self._episode_image_paths(episode_path, camera_key)
                    num_frames = len(image_paths)
                    intervals = _read_slip_intervals(episode_path, num_frames)
                    labels = _labels_from_intervals(intervals, num_frames)
                    excluded_frames = _excluded_transition_frames(
                        intervals,
                        num_frames,
                        before_start=exclude_before_slip_start_frames,
                        after_end=exclude_after_slip_end_frames,
                    )
                except Exception as exc:
                    log.warning(f"Skipping {episode_path}: {exc}")
                    continue

                window_specs = []
                for window_start in range(0, num_frames, input_window_stride):
                    frame_indices = np.clip(
                        np.arange(window_start, window_start + input_window_frames),
                        0,
                        num_frames - 1,
                    )
                    if excluded_frames[frame_indices].any():
                        num_excluded_windows += 1
                        continue
                    if require_uniform_label:
                        window_labels = labels[frame_indices]
                        if window_labels.min() != window_labels.max():
                            num_mixed_windows += 1
                            continue
                    window_specs.append((window_start, frame_indices))

                episode_index = len(self.episode_data)
                self.episode_data.append({
                    "path": str(episode_path),
                    "image_paths": image_paths,
                    "num_frames": num_frames,
                    "window_starts": [start for start, _ in window_specs],
                })

                for window_start, frame_indices in window_specs:
                    label_frame = min(
                        window_start + input_window_frames - 1, num_frames - 1
                    )
                    self.windows.append({
                        "ep_idx": episode_index,
                        "frame_indices": self._sample_frame_indices(frame_indices),
                        "label": torch.tensor(
                            int(labels[label_frame]), dtype=torch.long
                        ),
                        "episode_path": str(episode_path),
                        "window_start_frame": int(window_start),
                    })

        num_slip = sum(sample["label"].item() for sample in self.windows)
        log.info(
            f"BraincoSlipVisionDataset: {len(self.episode_data)} episodes, "
            f"{len(self.windows)} samples (slip={num_slip}, "
            f"non-slip={len(self.windows) - num_slip}, "
            f"excluded_transition_windows={num_excluded_windows}, "
            f"mixed_label_windows_dropped={num_mixed_windows} "
            f"(require_uniform_label={require_uniform_label}), "
            f"frames_per_sample={self.num_frames_per_sample}, "
            f"augment={self.augment})"
        )

    # ── episode parsing ───────────────────────────────────────────────────────

    def _episode_image_paths(self, episode_path: Path, camera_key: str) -> List[Path]:
        """Resolve one RGB path per frame, in ``data.json`` frame order."""
        with (episode_path / "data.json").open("r") as file:
            frames = json.load(file)["data"]

        paths = []
        for frame in frames:
            relative = frame.get("colors", {}).get(camera_key)
            if relative is None:
                raise ValueError(
                    f"Frame {frame.get('idx')} has no '{camera_key}' image in "
                    f"{episode_path / 'data.json'}"
                )
            paths.append(episode_path / str(relative))

        if not paths:
            raise ValueError(f"No frames listed in {episode_path / 'data.json'}")
        for probe in (paths[0], paths[-1]):
            if not probe.exists():
                raise FileNotFoundError(f"Missing RGB frame: {probe}")
        return paths

    def _sample_frame_indices(self, frame_indices: np.ndarray) -> np.ndarray:
        """Pick ``num_frames_per_sample`` frames evenly across the window."""
        if self.num_frames_per_sample == len(frame_indices):
            return frame_indices
        positions = np.linspace(
            0, len(frame_indices) - 1, self.num_frames_per_sample
        ).round().astype(int)
        return frame_indices[positions]

    # ── image loading ─────────────────────────────────────────────────────────

    def _bottom_center_crop(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        crop_size = self.crop_size
        if width < crop_size or height < crop_size:
            scale = max(crop_size / width, crop_size / height)
            img = TF.resize(
                img,
                [
                    max(crop_size, int(round(height * scale))),
                    max(crop_size, int(round(width * scale))),
                ],
            )
            width, height = img.size
        left = max(0, (width - crop_size) // 2)
        top = max(0, height - crop_size)
        return TF.crop(img, top, left, crop_size, crop_size)

    def _sample_augment_params(self) -> dict:
        """Draw one set of augmentation parameters, shared by the whole window."""
        pad = self.augment_pad
        jitter = self.augment_jitter
        return {
            "top": int(torch.randint(0, pad + 1, (1,)).item()),
            "left": int(torch.randint(0, pad + 1, (1,)).item()),
            "brightness": float(torch.empty(1).uniform_(1 - jitter, 1 + jitter)),
            "contrast": float(torch.empty(1).uniform_(1 - jitter, 1 + jitter)),
            "saturation": float(torch.empty(1).uniform_(1 - jitter, 1 + jitter)),
        }

    def _load_frame(self, path: Path, params: Optional[dict]) -> torch.Tensor:
        img = Image.open(str(path)).convert("RGB")
        if self.crop_mode == "bottom_center":
            img = self._bottom_center_crop(img)

        if params is None:
            img = TF.resize(img, [self.img_size, self.img_size])
        else:
            side = self.img_size + self.augment_pad
            img = TF.resize(img, [side, side])
            img = TF.crop(
                img, params["top"], params["left"], self.img_size, self.img_size
            )
            img = TF.adjust_brightness(img, params["brightness"])
            img = TF.adjust_contrast(img, params["contrast"])
            img = TF.adjust_saturation(img, params["saturation"])

        return TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        episode = self.episode_data[window["ep_idx"]]
        params = self._sample_augment_params() if self.augment else None
        frames = torch.stack([
            self._load_frame(episode["image_paths"][frame_index], params)
            for frame_index in window["frame_indices"]
        ])
        return {
            "frames": frames,                              # (T, 3, H, W)
            "label": window["label"],                      # scalar long
            "episode_path": window["episode_path"],
            "window_start_frame": window["window_start_frame"],
        }
