"""BraincoAngleTactileDataset: tactile sensor + finger angle pretraining dataset.

Output per sample
-----------------
joint_contact  : (W, 10, 4)  — tactile sensor values, normalized to [0, 1]
                               sensors 0-4 = left hand, 5-9 = right hand
finger_angles  : (W, 10, 4)  — per-finger joint angle encoding (human-hand layout)
                               fingers 0-4 = left hand, 5-9 = right hand
                               ch0: abduction  — always 0 (not available in brainco)
                               ch1: MCP flexion (metacarpal for thumb, 0 for others)
                               ch2: PIP flexion (proximal joint)
                               ch3: DIP flexion (distal ≈ proximal × ratio)

                               thumb  : [0, metacarpal, proximal, proximal]
                               others : [0, 0,          proximal, proximal × 1.155]

brainco hand qpos convention (6 DOF per hand)
---------------------------------------------
    q[0] thumb_metacarpal
    q[1] thumb_proximal   (thumb_distal ≈ q[1])
    q[2] index_proximal   (index_distal ≈ q[2] × 1.155)
    q[3] middle_proximal
    q[4] ring_proximal
    q[5] pinky_proximal
"""

from typing import Optional
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig

from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

# Normalization upper bounds per tactile channel (same as BraincoSSLDataset)
_TACTILE_MAX = [25000.0, 25000.0, 365.0, 500000.0]

# Distal-to-proximal coupling ratio for non-thumb fingers
_DISTAL_RATIO = 1.155


def _ee_qpos_to_finger_angles(qpos: np.ndarray) -> np.ndarray:
    """Convert 6-DOF hand qpos → (5, 4) per-finger angle encoding.

    Layout matches human hand anatomy:
        ch0  abduction (MCP lateral)  — 0, not available in brainco
        ch1  MCP flexion              — metacarpal (thumb only), 0 for others
        ch2  PIP flexion              — proximal joint
        ch3  DIP flexion              — distal joint (≈ proximal × ratio)

    Args:
        qpos : (6,) — [thumb_meta, thumb_prox, index_prox,
                        middle_prox, ring_prox, pinky_prox]

    Returns:
        (5, 4) — rows: [thumb, index, middle, ring, pinky]
    """
    tm, tp, ip, mp, rp, pp = qpos
    # print(f"qpos: {qpos}  →  angles: tm={tm:.3f}, tp={tp:.3f}, ip={ip:.3f}, mp={mp:.3f}, rp={rp:.3f}, pp={pp:.3f}")
    return np.array([
        [tm, 0.0,  tp,  tp              ],  # thumb : abduct=0, MCP=meta, PIP=prox, DIP≈prox
        [0.0, ip,  ip * _DISTAL_RATIO, 0.0],  # index
        [0.0, mp,  mp * _DISTAL_RATIO, 0.0],  # middle
        [0.0, rp,  rp * _DISTAL_RATIO, 0.0],  # ring
        [0.0, pp,  pp * _DISTAL_RATIO, 0.0],  # pinky
    ], dtype=np.float32)  # (5, 4)


class BraincoAngleTactileDataset(data.Dataset):
    """BrainCo SSL pretraining dataset returning tactile contact + finger angles.

    Loads a single episode directory (containing data.json). Produces sliding
    windows of joint_contact and finger_angles without any FK computation.

    Args:
        config            : Hydra DictConfig with window_time, window_overlap,
                            interpolating_freq.
        data_path         : Path to a single episode directory with data.json.
        object_class      : Optional integer label for object classification.
        normalization     : Optional DictConfig(mean, std) for contact stream.
                            If None, values are normalized to [0, 1] by channel max.
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        object_class: Optional[int] = None,
        normalization: Optional[DictConfig] = None,
        subtract_baseline: bool = False,
    ):
        self.window_time        = config.window_time
        self.interpolating_freq = config.interpolating_freq
        self.num_frames_per_window = int(round(self.window_time * self.interpolating_freq))

        overlap = float(config.get("window_overlap", 0.0))
        assert 0.0 <= overlap < 1.0, "window_overlap must be in [0, 1)"
        shift = int(round(self.num_frames_per_window * (1.0 - overlap)))
        self.shift_per_window = max(1, shift)

        self.subtract_baseline = subtract_baseline or bool(config.get("subtract_baseline", False))
        self.object_label = object_class
        self.data_path    = Path(data_path)

        # ── Load data.json ───────────────────────────────────────────────────
        data_json = self.data_path / "data.json"
        assert data_json.exists(), f"data.json not found: {data_json}"
        log.info(f"Loading BraincoAngle data from {data_json} (subtract_baseline={self.subtract_baseline})")

        with open(data_json, "r") as f:
            raw = json.load(f)
        frames = raw["data"]
        self.num_frames = len(frames)
        log.info(f"  {self.num_frames} frames")

        # ── Tactile data: (N, 10, 4) ─────────────────────────────────────────
        tactile_list = []
        for frame in frames:
            info = frame["tactiles"]
            if isinstance(info["left_ee"], str):
                lh = np.load(str(self.data_path / info["left_ee"])).reshape(-1, 4)
            else:
                lh = np.array(info["left_ee"], dtype=np.float32).reshape(-1, 4)

            if isinstance(info["right_ee"], str):
                rh = np.load(str(self.data_path / info["right_ee"])).reshape(-1, 4)
            else:
                rh = np.array(info["right_ee"], dtype=np.float32).reshape(-1, 4)

            tactile_list.append(np.concatenate([lh, rh], axis=0))  # (10, 4)

        tactile = np.array(tactile_list, dtype=np.float32)          # (N, 10, 4)
        # ch2: 65535 (invalid) → -1
        tactile[..., 2][tactile[..., 2] == 65535] = -1.0

        # Subtract frame-0 as baseline before normalization.
        # Only valid (>= 0) values are subtracted; invalid ch2 (-1) is kept as-is.
        if subtract_baseline:
            baseline = tactile[0].copy()                             # (10, 4)
            valid = tactile >= 0                                     # (N, 10, 4)
            baseline_valid = baseline >= 0                           # (10, 4)
            # subtract only where both frame and baseline are valid
            subtract_mask = valid & baseline_valid[np.newaxis]
            tactile[subtract_mask] -= np.broadcast_to(baseline, tactile.shape)[subtract_mask]

        # Normalize each channel to [0, 1] (invalid ch2 kept as -1 sentinel)
        max_vals = np.array(_TACTILE_MAX, dtype=np.float32)         # (4,)
        tactile_norm = tactile.copy()
        valid_mask = tactile_norm >= 0
        tactile_norm[valid_mask] = (tactile_norm / max_vals)[valid_mask]
        self.tactile_array = tactile_norm                            # (N, 10, 4)

        # ── Finger angles: (N, 10, 4) — lh 0-4, rh 5-9 ─────────────────────
        angle_list = []
        for frame in frames:
            lh_q = np.array(frame["states"]["left_ee"]["qpos"],  dtype=np.float32)  # (6,)
            rh_q = np.array(frame["states"]["right_ee"]["qpos"], dtype=np.float32)  # (6,)
            lh_angles = _ee_qpos_to_finger_angles(lh_q)   # (5, 4)
            rh_angles = _ee_qpos_to_finger_angles(rh_q)   # (5, 4)
            angle_list.append(np.concatenate([lh_angles, rh_angles], axis=0))  # (10, 4)

        self.angle_array = np.array(angle_list, dtype=np.float32)   # (N, 10, 4)

        log.info(f"  tactile_array : {self.tactile_array.shape}")
        log.info(f"  angle_array   : {self.angle_array.shape}")

        # ── Window indices ───────────────────────────────────────────────────
        if self.num_frames < self.num_frames_per_window:
            log.warning(
                f"  Episode has only {self.num_frames} frames, "
                f"less than window size {self.num_frames_per_window}. "
                "Creating a single padded window."
            )
        max_start = max(0, self.num_frames - self.num_frames_per_window)
        self.data_idxs = np.arange(0, max_start + 1, self.shift_per_window)
        log.info(f"  windows: {len(self.data_idxs)}  frames/window: {self.num_frames_per_window}")

    def __len__(self) -> int:
        return len(self.data_idxs)

    def __getitem__(self, idx: int) -> dict:
        start = self.data_idxs[idx]
        end   = start + self.num_frames_per_window

        joint_contact = torch.from_numpy(self.tactile_array[start:end].copy())  # (W, 10, 4)
        finger_angles = torch.from_numpy(self.angle_array[start:end].copy())    # (W, 10, 4)

        sample = {
            "joint_contact":  joint_contact,
            "finger_angles":  finger_angles,
        }

        if self.object_label is not None:
            sample["object_classification"] = torch.tensor(self.object_label)

        return sample
