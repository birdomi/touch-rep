import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data

from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

_FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # MediaPipe fingertip joint indices


class _Hot3DSequenceData:
    """Loaded HOT3D sequence with angle and contact data."""

    __slots__ = ("rh_angles", "lh_angles", "rh_contact", "lh_contact")

    def __init__(
        self,
        rh_angles: np.ndarray,   # (N, 5, 4)
        lh_angles: np.ndarray,   # (N, 5, 4)
        rh_contact: np.ndarray,  # (N, 21, 1)
        lh_contact: np.ndarray,  # (N, 21, 1)
    ):
        self.rh_angles = rh_angles
        self.lh_angles = lh_angles
        self.rh_contact = rh_contact
        self.lh_contact = lh_contact


def _load_hot3d_sequence(path: Path) -> _Hot3DSequenceData:
    with open(path, "rb") as f:
        raw = pickle.load(f)

    frame_indices = sorted(k for k in raw.keys() if isinstance(k, int))
    n = len(frame_indices)
    rh_angles = np.zeros((n, 5, 4), dtype=np.float32)
    lh_angles = np.zeros((n, 5, 4), dtype=np.float32)
    rh_contact = np.zeros((n, 21, 1), dtype=np.float32)
    lh_contact = np.zeros((n, 21, 1), dtype=np.float32)

    for i, t in enumerate(frame_indices):
        frame = raw[t]
        rh_angles[i] = frame["rh_angles"]
        lh_angles[i] = frame["lh_angles"]
        rh_contact[i] = frame["rh_contact"]
        lh_contact[i] = frame["lh_contact"]

    return _Hot3DSequenceData(
        rh_angles=rh_angles,
        lh_angles=lh_angles,
        rh_contact=rh_contact,
        lh_contact=lh_contact,
    )


class Hot3DAngleTactileDataset(data.Dataset):
    """HOT3D hand angle + contact pretraining dataset.

    데이터 경로: pretraining_dataset/vectors/hot3d/
    파일 포맷:  <participant>_<hash>_seg<NNN>.pkl

    각 프레임:
        rh_angles  (5, 4)  — 오른손 5개 손가락 각도
        lh_angles  (5, 4)  — 왼손 5개 손가락 각도
        rh_contact (21, 1) — 오른손 21개 관절 contact
        lh_contact (21, 1) — 왼손 21개 관절 contact
    """

    def __init__(
        self,
        data_root: str = "pretraining_dataset/vectors/hot3d",
        **kwargs,
    ):
        self.window_size = int(kwargs.pop("window_size", 3))
        self.window_stride = int(kwargs.pop("window_stride", 1))
        split = kwargs.pop("split", "train")
        train_val_split = float(kwargs.pop("train_val_split", 0.9))
        participants: Optional[List[str]] = kwargs.pop("participants", None)

        root = Path(data_root)
        assert root.exists(), f"data_root not found: {root}"
        all_pkl = sorted(root.glob("*.pkl"))
        if participants is not None:
            all_pkl = [p for p in all_pkl if any(p.name.startswith(s) for s in participants)]
        assert len(all_pkl) > 0, f"No pkl files found in {root}"
        log.info(f"Found {len(all_pkl)} pkl files in {root}")

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

        self._sequences: List[_Hot3DSequenceData] = []
        self._seq_paths: List[Path] = []
        for p in pkl_files:
            self._sequences.append(_load_hot3d_sequence(p))
            self._seq_paths.append(p)
        log.info(f"  Loaded {len(self._sequences)} sequences")

        self.windows: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(self._sequences):
            n_frames = seq.rh_angles.shape[0]
            max_start = n_frames - self.window_size
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.window_stride):
                self.windows.append((seq_idx, start))
        log.info(f"  Total windows: {len(self.windows)}")

        self.window_labels: List[int] = []
        for seq_idx, start in self.windows:
            seq = self._sequences[seq_idx]
            lh_c = seq.lh_contact[start: start + self.window_size]
            rh_c = seq.rh_contact[start: start + self.window_size]
            self.window_labels.append(self._compute_classification_id(lh_c, rh_c))

        _CLASS_NAMES = {
            0: "no touch",
            1: "LH only (no pinch)",
            2: "LH pinch (thumb+)",
            3: "RH only (no pinch)",
            4: "RH pinch (thumb+)",
            5: "both hands",
        }
        from collections import Counter
        total = len(self.window_labels)
        counts = Counter(self.window_labels)
        log.info("  classification_id distribution:")
        for cls in range(6):
            cnt = counts.get(cls, 0)
            log.info(f"    [{cls}] {_CLASS_NAMES[cls]:22s}: {cnt:6d}  ({100*cnt/total:.1f}%)")

    @staticmethod
    def _compute_classification_id(
        lh_contact: np.ndarray,  # (W, 21, 1)
        rh_contact: np.ndarray,  # (W, 21, 1)
        threshold: float = 0.0,
    ) -> int:
        lh_tips = lh_contact[:, _FINGERTIP_INDICES, 0]  # (W, 5)
        rh_tips = rh_contact[:, _FINGERTIP_INDICES, 0]  # (W, 5)

        lh_touch = (lh_tips > threshold).any(axis=0)  # (5,) bool
        rh_touch = (rh_tips > threshold).any(axis=0)  # (5,) bool

        lh_any = lh_touch.any()
        rh_any = rh_touch.any()

        if lh_any and rh_any:
            return 5
        if lh_any:
            if lh_touch[0] and lh_touch.sum() >= 2:
                return 2
            return 1
        if rh_any:
            if rh_touch[0] and rh_touch.sum() >= 2:
                return 4
            return 3
        return 0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        seq_idx, start = self.windows[idx]
        seq = self._sequences[seq_idx]
        end = start + self.window_size

        finger_angles = np.concatenate(
            [seq.lh_angles[start:end], seq.rh_angles[start:end]], axis=1
        )  # (W, 10, 4)
        joint_contact = np.concatenate(
            [seq.lh_contact[start:end], seq.rh_contact[start:end]], axis=1
        )  # (W, 42, 1)

        return {
            "finger_angles": torch.from_numpy(finger_angles).float(),
            "joint_contact": torch.from_numpy(joint_contact).float(),
            "classification_id": torch.tensor(self.window_labels[idx], dtype=torch.long),
        }
