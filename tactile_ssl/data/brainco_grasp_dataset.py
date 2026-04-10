"""
BraincoGraspDetectionDataset
-----------------------------
Multi-episode grasp detection dataset that inherits from BraincoSSLDataset.
Scans grasp_success/ and grasp_fail/ subdirectories, loads each episode via
BraincoSSLDataset (FK caching included), and aggregates all windows into a
flat list compatible with train_task_brainco.py.

__getitem__ output:
    sensor:             Tensor(W, 10, 4)   — tactile, ch2 null→-1
    sensor_poses:       Tensor(W, 10, 3)   — fingertip position in wrist-local frame
    wrist_poses:        Tensor(W, 2, 9)    — [translation(3), rotation_6d(6)] per wrist
    label:              Tensor(,)  long    — 0=fail, 1=success
    episode_path:       str
    window_start_frame: int
"""

from pathlib import Path

import numpy as np
import torch

from omegaconf import DictConfig

from tactile_ssl.data.brainco_tactile import BraincoSSLDataset
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class BraincoGraspDetectionDataset(BraincoSSLDataset):
    """All grasp-success/fail episodes combined into one indexable dataset.

    Inherits FK caching and tactile loading from BraincoSSLDataset (per episode).
    Does NOT call super().__init__(); instead instantiates BraincoSSLDataset
    per episode and aggregates the results.

    Required attributes for train_task_brainco.py:
        episode_data  — list of {path, window_starts, label}
        windows       — flat list of window dicts (indexed by __getitem__)

    Args:
        config:            Hydra DictConfig — forwarded to each BraincoSSLDataset.
                           Recognised keys: window_time, window_overlap,
                           interpolating_freq, bias_noise_std, bias_range,
                           (pose_type is accepted but currently unused).
        data_path:         Root directory containing grasp_success/ and grasp_fail/.
        brainco_urdf_path: URDF root directory (g1.urdf, hand URDFs).
    """

    # Mapping from subdirectory name to integer label
    _CLASS_DIRS = {
        "grasp_success": 1,
        "grasp_fail":    0,
    }

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        brainco_urdf_path: str = "dataset/brainco/urdf",
    ):
        # Do NOT call super().__init__() — we manage multiple episodes here.
        self.episode_data: list = []
        self.windows:      list = []

        data_root = Path(data_path)

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

            for ep_path in episode_dirs:
                log.info(f"  Loading: {ep_path}")
                try:
                    ep_ds = BraincoSSLDataset(
                        config=config,
                        data_path=str(ep_path),
                        brainco_urdf_path=brainco_urdf_path,
                        object_class=label,
                    )
                except Exception as exc:
                    log.warning(f"  Skipping {ep_path}: {exc}")
                    continue

                if ep_ds.num_frames < ep_ds.num_frames_per_window:
                    log.warning(
                        f"  Skipping {ep_path}: {ep_ds.num_frames} frames < window size {ep_ds.num_frames_per_window}"
                    )
                    continue

                # ── episode_data entry ──────────────────────────────────
                self.episode_data.append({
                    "path":          str(ep_path),
                    "window_starts": ep_ds.data_idxs.copy(),
                    "label":         label,
                })

                # ── Use wrist-local fingertip positions from SSL class ───────
                # fingertip_rel: (N, 10, 3) — fingertip in own-wrist local frame
                # wrist_poses:   (N, 2, 9)  — [translation(3), rotation_6d(6)]
                W = ep_ds.num_frames_per_window

                # ── Flatten windows ────────────────────────────────────
                label_tensor = torch.tensor(label, dtype=torch.long)
                for start in ep_ds.data_idxs:
                    end = start + W
                    self.windows.append({
                        "sensor":             torch.from_numpy(
                                                  ep_ds.tactile_array[start:end].copy()),
                        "sensor_poses":       torch.from_numpy(
                                                  ep_ds.fingertip_rel[start:end].copy()),
                        "wrist_poses":        torch.from_numpy(
                                                  ep_ds.wrist_poses[start:end].copy()),
                        "label":              label_tensor,
                        "episode_path":       str(ep_path),
                        "window_start_frame": int(start),
                    })

        log.info(
            f"BraincoGraspDetectionDataset ready: "
            f"{len(self.episode_data)} episodes, {len(self.windows)} windows"
        )

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        return self.windows[idx]
