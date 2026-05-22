"""BraincoAngleGraspDataset: grasp prediction/detection with angle+contact inputs.

Scans grasp_success/ and grasp_fail/ subdirectories, loads each episode via
BraincoAngleTactileDataset logic (no FK), and aggregates windows.

Output per window
-----------------
joint_contact      : (W, 10, 4)  — tactile sensor values, normalized to [0, 1]
finger_angles      : (W, 10, 4)  — finger angle encoding [abd, MCP, PIP, DIP]
label              : ()  long    — 0=fail, 1=success
episode_path       : str
window_start_frame : int
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.data.brainco_angle_tactile import (
    _ee_qpos_to_finger_angles,
    _TACTILE_MAX,
)

log = get_pylogger(__name__)

_CLASS_DIRS = {"grasp_success": 1, "grasp_fail": 0}


class BraincoAngleGraspDataset(data.Dataset):
    """Grasp success/fail dataset using angle + contact inputs.

    Compatible with get_dataloader_brainco_grasp() in train_task_brainco_angle.py
    — exposes ``episode_data`` and ``windows`` for the fold splitter.

    Args:
        config    : DictConfig with window_time, window_overlap, interpolating_freq.
        data_path : Root containing grasp_success/ and grasp_fail/ subdirs.
    """

    def __init__(self, config: DictConfig, data_path: str):
        self.window_time        = float(config.window_time)
        self.interpolating_freq = int(config.interpolating_freq)
        self.num_frames_per_window = int(round(self.window_time * self.interpolating_freq))

        overlap = float(config.get("window_overlap", 0.0))
        assert 0.0 <= overlap < 1.0
        shift = int(round(self.num_frames_per_window * (1.0 - overlap)))
        self.shift_per_window = max(1, shift)
        self.subtract_baseline = bool(config.get("subtract_baseline", False))

        root = Path(data_path)
        self.episode_data: list = []
        self.windows:      list = []

        for class_name, label in _CLASS_DIRS.items():
            class_dir = root / class_name
            if not class_dir.exists():
                log.warning(f"  {class_dir} not found — skipping")
                continue

            ep_dirs = sorted(d for d in class_dir.iterdir() if d.is_dir())
            log.info(f"  {class_name}: {len(ep_dirs)} episodes")

            for ep_dir in ep_dirs:
                data_json = ep_dir / "data.json"
                if not data_json.exists():
                    continue

                with open(data_json) as f:
                    raw = json.load(f)
                frames = raw["data"]
                n      = len(frames)

                if n < self.num_frames_per_window:
                    log.warning(
                        f"  Skipping {ep_dir.name}: {n} frames < "
                        f"window size {self.num_frames_per_window}"
                    )
                    continue

                # ── Tactile ──────────────────────────────────────────────────
                tactile_list = []
                for frame in frames:
                    info = frame["tactiles"]
                    if isinstance(info["left_ee"], str):
                        lh = np.load(str(ep_dir / info["left_ee"])).reshape(-1, 4)
                    else:
                        lh = np.array(info["left_ee"], dtype=np.float32).reshape(-1, 4)
                    if isinstance(info["right_ee"], str):
                        rh = np.load(str(ep_dir / info["right_ee"])).reshape(-1, 4)
                    else:
                        rh = np.array(info["right_ee"], dtype=np.float32).reshape(-1, 4)
                    tactile_list.append(np.concatenate([lh, rh], axis=0))  # (10, 4)

                tactile = np.array(tactile_list, dtype=np.float32)          # (N, 10, 4)
                tactile[..., 2][tactile[..., 2] == 65535] = -1.0

                if self.subtract_baseline:
                    baseline = tactile[0].copy()                             # (10, 4)
                    valid_t  = tactile >= 0
                    valid_b  = baseline >= 0
                    sub_mask = valid_t & valid_b[np.newaxis]
                    tactile[sub_mask] -= np.broadcast_to(baseline, tactile.shape)[sub_mask]

                max_vals = np.array(_TACTILE_MAX, dtype=np.float32)
                tactile_norm = tactile.copy()
                valid = tactile_norm >= 0
                tactile_norm[valid] = (tactile_norm / max_vals)[valid]

                # ── Angles ───────────────────────────────────────────────────
                angle_list = []
                for frame in frames:
                    lh_q = np.array(frame["states"]["left_ee"]["qpos"],  dtype=np.float32)
                    rh_q = np.array(frame["states"]["right_ee"]["qpos"], dtype=np.float32)
                    angle_list.append(
                        np.concatenate([_ee_qpos_to_finger_angles(lh_q),
                                        _ee_qpos_to_finger_angles(rh_q)], axis=0)
                    )  # (10, 4)
                angles = np.array(angle_list, dtype=np.float32)   # (N, 10, 4)

                # ── Windows ──────────────────────────────────────────────────
                max_start  = n - self.num_frames_per_window
                win_starts = list(range(0, max(1, max_start + 1), self.shift_per_window))
                label_tensor = torch.tensor(label, dtype=torch.long)

                self.episode_data.append({
                    "path":          str(ep_dir),
                    "window_starts": win_starts,
                    "label":         label,
                })

                for ws in win_starts:
                    we = ws + self.num_frames_per_window
                    self.windows.append({
                        "joint_contact":      torch.from_numpy(tactile_norm[ws:we].copy()),
                        "finger_angles":      torch.from_numpy(angles[ws:we].copy()),
                        "label":              label_tensor,
                        "episode_path":       str(ep_dir),
                        "window_start_frame": ws,
                    })

        log.info(
            f"BraincoAngleGraspDataset: {len(self.episode_data)} episodes, "
            f"{len(self.windows)} windows"
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
