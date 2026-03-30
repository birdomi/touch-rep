import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data

from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class _SequenceData:
    """Holds loaded data for one pkl file."""
    __slots__ = ("joint_data",)

    def __init__(self, joint_data: np.ndarray):
        # joint_data: (N_frames, 42, 4)  — x,y,z,c  (lh 0~20, rh 21~41)
        self.joint_data = joint_data


def _load_pkl_sequence(path: Path) -> _SequenceData:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    frame_indices = sorted(k for k in raw.keys() if isinstance(k, int))
    n = len(frame_indices)
    joint_data = np.zeros((n, 42, 4), dtype=np.float32)
    for i, t in enumerate(frame_indices):
        frame = raw[t]
        joint_data[i, :21] = frame["lh"]   # (21, 4)
        joint_data[i, 21:] = frame["rh"]   # (21, 4)
    return _SequenceData(joint_data)


class XYZCHandDataset(data.Dataset):
    """Base dataset for hand keypoint pretraining data in (x, y, z, c) format.

    pkl format: dict[frame_idx: int → {'lh': ndarray(21,4), 'rh': ndarray(21,4)}]
    - 21 keypoints per hand (wrist + 5 fingers × 4 joints, MediaPipe order)
    - channels: x, y, z (world coords, metres) + c (contact: 0.0 = no contact)

    Keypoint layout after concat:
        indices  0~20  = left hand  (lh[0] = lh wrist)
        indices 21~41  = right hand (rh[0] = rh wrist)

    Output per sample:
        sensor       : Tensor(T, 42, 1)  — contact channel c (normalised)
        sensor_poses : Tensor(T, 42, 6)  — relative xyz from own/opposite wrist
    """

    def __init__(
        self,
        data_root: str,
        window_size: int = 3,
        window_stride: int = 1,
        split: str = "train",
        train_val_split: float = 0.9,
        scenes: Optional[List[str]] = None,
        # accepted for API compatibility
        filter_valid: str = "both",
        normalize_xyz: bool = False,
        lazy: bool = False,
    ):
        self.window_size = window_size
        self.window_stride = window_stride

        # ── Scan pkl files ──────────────────────────────────────────────────
        root = Path(data_root)
        assert root.exists(), f"data_root not found: {root}"
        all_pkl = sorted(root.glob("*.pkl"))
        if scenes is not None:
            all_pkl = [p for p in all_pkl if any(p.name.startswith(s) for s in scenes)]
        assert len(all_pkl) > 0, f"No pkl files found in {root}"
        log.info(f"Found {len(all_pkl)} pkl files in {root}")

        # ── Train / val split (file-level) ─────────────────────────────────
        # train_val_split=1.0 → val reuses the last 10% of train files
        n_train = max(1, int(len(all_pkl) * min(train_val_split, 1.0)))
        if split == "train":
            pkl_files = all_pkl[:n_train]
        else:
            val_files = all_pkl[n_train:]
            if len(val_files) == 0:
                n_val = max(1, int(n_train * 0.1))
                val_files = all_pkl[n_train - n_val:n_train]
            pkl_files = val_files
        log.info(f"  {split}: {len(pkl_files)} files")

        # ── Load sequences ──────────────────────────────────────────────────
        self._sequences: List[_SequenceData] = []
        for p in pkl_files:
            self._sequences.append(_load_pkl_sequence(p))
        log.info(f"  Loaded {len(self._sequences)} sequences")

        # ── Build sliding window index ──────────────────────────────────────
        # Each entry: (seq_idx, frame_start) — never crosses file boundaries
        self.windows: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(self._sequences):
            n_frames = seq.joint_data.shape[0]
            max_start = n_frames - window_size
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, window_stride):
                self.windows.append((seq_idx, start))
        log.info(f"  Total windows: {len(self.windows)}")

        # Normalization stats for the c channel (injected externally)
        self.tactile_mean: Optional[float] = None
        self.tactile_std: Optional[float] = None

    def update_normalization(self, mean, std):
        """Inject c-channel normalization stats (called after computing on train set)."""
        self.tactile_mean = float(np.asarray(mean).ravel()[0])
        self.tactile_std  = float(np.asarray(std).ravel()[0])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        seq_idx, start = self.windows[idx]
        seq = self._sequences[seq_idx]

        window = seq.joint_data[start : start + self.window_size]  # (T, 42, 4)
        xyz = window[..., :3]   # (T, 42, 3)
        c   = window[..., 3:4]  # (T, 42, 1)

        # ── sensor_poses: 6D relative coords ──────────────────────────────
        # lh wrist = keypoint index 0,  rh wrist = keypoint index 21
        lh_wrist = xyz[:, 0:1, :]    # (T, 1, 3)
        rh_wrist = xyz[:, 21:22, :]  # (T, 1, 3)

        sensor_poses = np.empty((self.window_size, 42, 6), dtype=np.float32)
        # lh keypoints (0~20): own wrist=lh, opposite=rh
        sensor_poses[:, :21, :3] = xyz[:, :21] - lh_wrist
        sensor_poses[:, :21, 3:] = xyz[:, :21] - rh_wrist
        # rh keypoints (21~41): own wrist=rh, opposite=lh
        sensor_poses[:, 21:, :3] = xyz[:, 21:] - rh_wrist
        sensor_poses[:, 21:, 3:] = xyz[:, 21:] - lh_wrist

        # ── sensor: contact channel, normalised ────────────────────────────
        sensor = c.copy()
        if self.tactile_mean is not None:
            sensor = (sensor - self.tactile_mean) / max(self.tactile_std, 1e-6)

        return {
            "sensor":       torch.from_numpy(sensor).float(),        # (T, 42, 1)
            "sensor_poses": torch.from_numpy(sensor_poses).float(),  # (T, 42, 6)
            "sensor_id":    torch.tensor(0, dtype=torch.long),
        }


class GigaHandsTactileDataset(XYZCHandDataset):
    """GigaHands hand keypoint pretraining dataset.

    데이터 경로: pretraining_dataset/Gigahands/
    파일 포맷:  p{subject}-folder_{seq}.pkl
    """

    def __init__(self, data_root: str = "pretraining_dataset/Gigahands", **kwargs):
        super().__init__(data_root=data_root, **kwargs)
