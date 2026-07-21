"""BrainCo grasp prediction dataset with tactile contact and fingertip XYZ.

This downstream adapter reuses :class:`BraincoSSLDataset` for tactile loading
and forward kinematics, then exposes the same two streams used by the
``ours_3d`` pseudo-force pretraining setup:

``joint_contact``
    ``(W, 10, 4)`` normalized BrainCo fingertip tactile readings.
``finger_xyz``
    ``(W, 10, 3)`` fingertip positions in each hand's wrist-local frame,
    aligned to the XYZ convention used by the pseudo-force NPZ datasets.
"""

from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig

from tactile_ssl.data.brainco_tactile import BraincoSSLDataset
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

_CLASS_DIRS = {"grasp_success": 1, "grasp_fail": 0}


def _class_dirs(label_dirs):
    return [(str(name), int(label)) for name, label in label_dirs.items()]


def _data_roots(config: DictConfig, data_path: str):
    roots = config.get("data_roots")
    if roots:
        return [
            (
                Path(str(root.data_path)),
                _class_dirs(root.get("label_dirs", config.get("label_dirs", _CLASS_DIRS))),
            )
            for root in roots
        ]
    return [(Path(data_path), _class_dirs(config.get("label_dirs", _CLASS_DIRS)))]


def _align_xyz_to_npz(finger_xyz: torch.Tensor) -> torch.Tensor:
    """Convert BrainCo wrist-local XYZ to the pseudo-force NPZ convention.

    BrainCo's forward/lateral/thickness axes are ``(z, y, x)``, whereas the
    NPZ files use ``(y, x, z)`` semantically.  The NPZ lateral axis is mirrored
    between hands, so the left-hand output additionally negates BrainCo ``y``.

    Args:
        finger_xyz: ``(..., 10, 3)`` ordered left thumb..pinky, then right.

    Returns:
        Tensor with the same shape, in NPZ ``(x, y, z)`` convention.
    """
    aligned = finger_xyz[..., [1, 2, 0]].clone()
    aligned[..., :5, 0].neg_()
    return aligned


class BraincoXYZGraspDataset(data.Dataset):
    """Aggregate BrainCo grasp episodes as tactile + fingertip-XYZ windows.

    ``episode_data`` and ``windows`` intentionally match the interface used by
    ``train_task_brainco_angle.py`` for episode-level K-fold splitting.
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        brainco_urdf_path: str = "dataset/brainco/urdf",
    ):
        self.episode_data = []
        self.windows = []
        subtract_baseline = bool(config.get("subtract_baseline", False))
        align_xyz_to_npz = bool(config.get("align_xyz_to_npz", True))

        for root, class_dirs in _data_roots(config, data_path):
            log.info(f"Loading BrainCo XYZ grasp episodes from: {root}")
            for class_name, label in class_dirs:
                class_dir = root / class_name
                if not class_dir.exists():
                    log.warning(f"Class directory not found, skipping: {class_dir}")
                    continue

                episode_dirs = sorted(
                    path for path in class_dir.iterdir()
                    if path.is_dir() and (path / "data.json").exists()
                )
                log.info(f"  {class_name}: {len(episode_dirs)} episodes")

                for episode_path in episode_dirs:
                    try:
                        episode = BraincoSSLDataset(
                            config=config,
                            data_path=str(episode_path),
                            brainco_urdf_path=brainco_urdf_path,
                        )
                    except Exception as exc:
                        log.warning(f"  Skipping {episode_path}: {exc}")
                        continue

                    if episode.num_frames < episode.num_frames_per_window:
                        log.warning(
                            f"  Skipping {episode_path}: {episode.num_frames} frames < "
                            f"window size {episode.num_frames_per_window}"
                        )
                        continue

                    # Match the existing angle downstream preprocessing: remove
                    # each episode's initial tactile bias before max scaling.
                    if subtract_baseline:
                        tactile = episode.tactile_array
                        baseline = tactile[0].copy()
                        valid = (tactile >= 0) & (baseline[np.newaxis] >= 0)
                        baseline_full = np.broadcast_to(baseline, tactile.shape)
                        tactile[valid] -= baseline_full[valid]

                    starts = [int(start) for start in episode.data_idxs]
                    self.episode_data.append({
                        "path": str(episode_path),
                        "window_starts": starts,
                        "label": label,
                    })

                    label_tensor = torch.tensor(label, dtype=torch.long)
                    for window_index, start in enumerate(starts):
                        sample = episode[window_index]
                        finger_xyz = sample["sensor_poses"]
                        if align_xyz_to_npz:
                            finger_xyz = _align_xyz_to_npz(finger_xyz)
                        self.windows.append({
                            "joint_contact": sample["sensor"],
                            "finger_xyz": finger_xyz,
                            "label": label_tensor,
                            "episode_path": str(episode_path),
                            "window_start_frame": start,
                        })

        log.info(
            f"BraincoXYZGraspDataset: {len(self.episode_data)} episodes, "
            f"{len(self.windows)} windows"
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]
