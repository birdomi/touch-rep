"""BraincoObjectClassificationDataset

Data structure:
    data_path/
        {class_name}/            e.g. basket/, box/, doll/, ...
            episode_XXXX/
                data.json
                tactiles/

Each class directory → integer label (0-indexed, alphabetical order).
Reuses BraincoSSLDataset per episode (FK is computed from URDF at load time).
"""

from pathlib import Path
from typing import List, Optional

import torch
from omegaconf import DictConfig

from tactile_ssl.data.brainco_tactile import BraincoSSLDataset
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class BraincoObjectClassificationDataset:
    """Multi-class object classification dataset for BrainCo pretraining data.

    Args:
        config:              Hydra DictConfig — forwarded to each BraincoSSLDataset.
        data_path:           Root directory containing per-class subdirectories.
        brainco_urdf_path:   URDF root directory (g1.urdf, hand URDFs).
        classes:             List of class folder names to include (in label order).
                             If None, auto-detects all episode_* subdirectories.
        use_skeleton_poses:  If True, use 42-joint skeleton_poses as sensor_poses.
        robot_to_human:      Forwarded to BraincoSSLDataset.
        retargeting_config_path_left:   Forwarded to BraincoSSLDataset.
        retargeting_config_path_right:  Forwarded to BraincoSSLDataset.
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        brainco_urdf_path: str = "dataset/brainco/urdf",
        classes: Optional[List[str]] = None,
        use_skeleton_poses: bool = False,
        robot_to_human: bool = False,
        retargeting_config_path_left: Optional[str] = None,
        retargeting_config_path_right: Optional[str] = None,
    ):
        self.episode_data: list = []
        self.windows:      list = []

        data_root = Path(data_path)

        # Auto-detect class directories if not specified
        if classes is None:
            classes = sorted(
                d.name for d in data_root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
        self.object_classes: List[str] = classes
        log.info(f"Object classes ({len(classes)}): {classes}")

        for label, class_name in enumerate(classes):
            class_path = data_root / class_name
            if not class_path.exists():
                log.warning(f"Class directory not found, skipping: {class_path}")
                continue

            episode_dirs = sorted(
                p for p in class_path.iterdir()
                if p.is_dir() and p.name.startswith("episode_")
            )
            log.info(f"[{class_name}] (label={label}) {len(episode_dirs)} episodes")

            for ep_path in episode_dirs:
                log.info(f"  Loading: {ep_path}")
                try:
                    ep_ds = BraincoSSLDataset(
                        config=config,
                        data_path=str(ep_path),
                        brainco_urdf_path=brainco_urdf_path,
                        object_class=label,
                        robot_to_human=robot_to_human,
                        retargeting_config_path_left=retargeting_config_path_left,
                        retargeting_config_path_right=retargeting_config_path_right,
                    )
                except Exception as exc:
                    log.warning(f"  Skipping {ep_path}: {exc}")
                    continue

                if ep_ds.num_frames < ep_ds.num_frames_per_window:
                    log.warning(
                        f"  Skipping {ep_path}: {ep_ds.num_frames} frames "
                        f"< window {ep_ds.num_frames_per_window}"
                    )
                    continue

                self.episode_data.append({
                    "path":          str(ep_path),
                    "window_starts": ep_ds.data_idxs.copy(),
                    "label":         label,
                })

                label_tensor = torch.tensor(label, dtype=torch.long)
                for w_idx in range(len(ep_ds)):
                    sample = ep_ds[w_idx]
                    new_sample = {
                        "label":       label_tensor,
                        "sensor":      sample["sensor"],
                        "wrist_poses": sample["wrist_poses"],
                    }
                    if use_skeleton_poses:
                        new_sample["sensor_poses"] = sample["skeleton_poses"]
                    else:
                        new_sample["sensor_poses"] = sample["sensor_poses"]
                    self.windows.append(new_sample)

        log.info(
            f"BraincoObjectClassificationDataset ready: "
            f"{len(self.episode_data)} episodes, {len(self.windows)} windows, "
            f"{len(classes)} classes"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        return self.windows[idx]
