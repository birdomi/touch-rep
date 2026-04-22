"""
BraincoGraspDetectionWorldDataset
---------------------------------
Multi-episode grasp detection dataset that aggregates BraincoSSLDataset
windows, but lifts fingertip poses into world coordinates.

__getitem__ output:
    sensor:             Tensor(W, 10, 7)   — tactile(4) + world xyz(3)
    sensor_poses:       Tensor(W, 10, 3)   — absolute fingertip coordinates in world frame
    label:              Tensor(,)  long    — class label; defaults to 0 for unlabeled pretraining layout
"""

from pathlib import Path

import numpy as np
import torch

from omegaconf import DictConfig

from tactile_ssl.data.brainco_tactile import BraincoSSLDataset
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


def _to_world_positions(
    local_positions: np.ndarray,
    wrist_pos_world: np.ndarray,
    wrist_rot_world: np.ndarray,
) -> np.ndarray:
    """Lift wrist-local positions into world coordinates.

    Args:
        local_positions: (N, K, 3)
        wrist_pos_world: (N, 2, 3)
        wrist_rot_world: (N, 2, 3, 3)

    Returns:
        world_positions: (N, K, 3)
    """
    world_positions = np.empty_like(local_positions, dtype=np.float32)
    num_left = local_positions.shape[1] // 2

    left_local = local_positions[:, :num_left]
    right_local = local_positions[:, num_left:]

    left_world = np.einsum("ntj,nij->nti", left_local, wrist_rot_world[:, 0])
    right_world = np.einsum("ntj,nij->nti", right_local, wrist_rot_world[:, 1])

    world_positions[:, :num_left] = left_world + wrist_pos_world[:, 0:1]
    world_positions[:, num_left:] = right_world + wrist_pos_world[:, 1:2]
    return world_positions.astype(np.float32)


class BraincoGraspDetectionWorldDataset(BraincoSSLDataset):
    """All grasp-success/fail episodes combined with world-frame poses."""

    _CLASS_DIRS = {
        "grasp_success": 1,
        "grasp_fail": 0,
    }

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        brainco_urdf_path: str = "dataset/brainco/urdf",
        robot_to_human: bool = False,
        retargeting_config_path_left: str = None,
        retargeting_config_path_right: str = None,
    ):
        self.episode_data: list = []
        self.windows: list = []

        data_root = Path(data_path)
        if any((data_root / class_dir).exists() for class_dir in self._CLASS_DIRS):
            episode_groups = []
            for class_dir, label in self._CLASS_DIRS.items():
                class_path = data_root / class_dir
                if not class_path.exists():
                    log.warning(f"Class directory not found, skipping: {class_path}")
                    continue
                episode_dirs = sorted(
                    p for p in class_path.iterdir()
                    if p.is_dir() and p.name.startswith("episode_")
                )
                log.info(f"[{class_dir}] Found {len(episode_dirs)} episodes")
                episode_groups.extend((ep_path, label) for ep_path in episode_dirs)
        else:
            object_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
            episode_groups = []
            for obj_dir in object_dirs:
                episode_dirs = sorted(
                    p for p in obj_dir.iterdir()
                    if p.is_dir() and p.name.startswith("episode_")
                )
                if episode_dirs:
                    log.info(f"[{obj_dir.name}] Found {len(episode_dirs)} episodes")
                    episode_groups.extend((ep_path, 0) for ep_path in episode_dirs)

        for ep_path, label in episode_groups:
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
                    f"  Skipping {ep_path}: {ep_ds.num_frames} frames < window size {ep_ds.num_frames_per_window}"
                )
                continue

            fingertip_world = _to_world_positions(
                ep_ds.fingertip_rel,
                ep_ds.wrist_pos_world,
                ep_ds.wrist_R_fk,
            )

            self.episode_data.append({
                "path": str(ep_path),
                "window_starts": ep_ds.data_idxs.copy(),
                "label": label,
            })

            label_tensor = torch.tensor(label, dtype=torch.long)
            for w_idx in range(len(ep_ds)):
                sample = ep_ds[w_idx]
                start = int(ep_ds.data_idxs[w_idx])
                end = start + ep_ds.num_frames_per_window

                fingertip_world_window = torch.from_numpy(
                    fingertip_world[start:end].copy()
                ).float()

                sensor = torch.cat(
                    [sample["sensor"], fingertip_world_window],
                    dim=-1,
                )

                new_sample = {
                    "label": label_tensor,
                    "sensor": sensor,
                    "sensor_poses": fingertip_world_window,
                    "fingertip_poses": fingertip_world_window,
                    "frame_indices": sample["frame_indices"],
                    "episode_path": str(ep_path),
                    "window_start_frame": start,
                }
                self.windows.append(new_sample)

        log.info(
            f"BraincoGraspDetectionWorldDataset ready: "
            f"{len(self.episode_data)} episodes, {len(self.windows)} windows"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        return self.windows[idx]
