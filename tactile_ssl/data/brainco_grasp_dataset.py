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
        use_skeleton_poses:          If True, use skeleton_poses instead of fingertip sensor_poses.
        robot_to_human:              Forwarded to BraincoSSLDataset — converts qpos to human skeleton.
        retargeting_config_path_left:  Forwarded to BraincoSSLDataset.
        retargeting_config_path_right: Forwarded to BraincoSSLDataset.
    """

    # Mapping from subdirectory name to integer label
    _CLASS_DIRS = {
        "grasp_success": 1,
        "grasp_fail":    0,
    }

    def _get_class_dirs(self, label_dirs):
        return [(str(name), int(label)) for name, label in label_dirs.items()]

    def _get_data_roots(self, config: DictConfig, data_path: str):
        data_roots = config.get("data_roots")
        if data_roots:
            roots = []
            for root_cfg in data_roots:
                label_dirs = root_cfg.get("label_dirs", config.get("label_dirs", self._CLASS_DIRS))
                roots.append((Path(str(root_cfg.data_path)), self._get_class_dirs(label_dirs)))
            return roots

        return [(Path(data_path), self._get_class_dirs(config.get("label_dirs", self._CLASS_DIRS)))]

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        brainco_urdf_path: str = "dataset/brainco/urdf",
        use_skeleton_poses: bool = False,
        robot_to_human: bool = False,
        retargeting_config_path_left: str = None,
        retargeting_config_path_right: str = None,
    ):
        # Do NOT call super().__init__() — we manage multiple episodes here.
        self.episode_data: list = []
        self.windows:      list = []

        for data_root, class_dirs in self._get_data_roots(config, data_path):
            log.info(f"Loading grasp episodes from: {data_root}")
            for class_dir, label in class_dirs:
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

                    # ── episode_data entry ──────────────────────────────────
                    print(ep_path, label)
                    self.episode_data.append({
                        "path":          str(ep_path),
                        "window_starts": ep_ds.data_idxs.copy(),
                        "label":         label,
                    })

                    # ── Flatten windows using ep_ds.__getitem__ ─────────────
                    label_tensor = torch.tensor(label, dtype=torch.long)
                    for w_idx in range(len(ep_ds)):
                        sample = ep_ds[w_idx]
                        sample["episode_path"] = str(ep_path)
                        sample["window_start_frame"] = int(ep_ds.data_idxs[w_idx])

                        new_sample = {}
                        new_sample["label"] = label_tensor
                        new_sample['sensor'] = sample['sensor']
                        if use_skeleton_poses:
                            new_sample['sensor_poses'] = sample['skeleton_poses']
                        else:
                            new_sample['sensor_poses'] = sample['sensor_poses']
                        new_sample['wrist_poses'] = sample['wrist_poses']

                        self.windows.append(new_sample)

        log.info(
            f"BraincoGraspDetectionDataset ready: "
            f"{len(self.episode_data)} episodes, {len(self.windows)} windows"
        )

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        return self.windows[idx]
