"""Dataset for pseudo-force hand joints and fingertip XYZ positions.

Expected pickle format::

    {
        frame_index: {
            "lh": {
                "joints": [{"summary": {...}}, ...],  # 21 joints
                "tip_pos": np.ndarray,                 # (5, 3)
            },
            "rh": { ... },
        },
        ...,
    }

The four sensor channels are ``normal_force``, ``tangential_force``,
``tangential_direction``, and ``proximity`` in that order.
"""

import pickle
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data

from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

FORCE_CHANNELS = (
    "normal_force",
    "tangential_force",
    "tangential_direction",
    "proximity",
)
_FINGERTIP_INDICES = (4, 8, 12, 16, 20)


class _PseudoForceSequenceData:
    """Compact tensors retained after discarding the raw pickle dictionary."""

    __slots__ = ("joint_contact", "finger_xyz")

    def __init__(
        self,
        lh_contact: np.ndarray,
        rh_contact: np.ndarray,
        lh_tip_pos: np.ndarray,
        rh_tip_pos: np.ndarray,
    ):
        self.joint_contact = torch.from_numpy(
            np.concatenate([lh_contact, rh_contact], axis=1)
        )
        self.finger_xyz = torch.from_numpy(
            np.concatenate([lh_tip_pos, rh_tip_pos], axis=1)
        )

    @classmethod
    def from_compact(
        cls,
        joint_contact: np.ndarray,
        finger_xyz: np.ndarray,
    ) -> "_PseudoForceSequenceData":
        sequence = cls.__new__(cls)
        sequence.joint_contact = torch.from_numpy(
            np.ascontiguousarray(joint_contact, dtype=np.float32)
        )
        sequence.finger_xyz = torch.from_numpy(
            np.ascontiguousarray(finger_xyz, dtype=np.float32)
        )
        return sequence

    @property
    def num_frames(self) -> int:
        return self.joint_contact.shape[0]

    @property
    def lh_contact(self) -> np.ndarray:
        return self.joint_contact[:, :21].numpy()

    @property
    def rh_contact(self) -> np.ndarray:
        return self.joint_contact[:, 21:].numpy()


def _extract_hand(frame: dict, hand: str, path: Path, frame_index: int):
    hand_data = frame.get(hand)
    if hand_data is None:
        # Some source datasets contain single-hand frames. Keep the fixed
        # two-hand token layout and represent the absent hand as no contact.
        force = np.zeros((21, len(FORCE_CHANNELS)), dtype=np.float32)
        force[:, FORCE_CHANNELS.index("tangential_direction")] = -1.0
        tip_pos = np.zeros((5, 3), dtype=np.float32)
        return force, tip_pos

    try:
        joints = hand_data["joints"]
        tip_pos = np.asarray(hand_data["tip_pos"], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(f"{path}: frame {frame_index} missing {hand}.{exc.args[0]}") from exc

    if len(joints) != 21:
        raise ValueError(
            f"{path}: frame {frame_index} {hand}.joints must contain 21 joints, "
            f"got {len(joints)}"
        )
    if tip_pos.shape != (5, 3):
        raise ValueError(
            f"{path}: frame {frame_index} {hand}.tip_pos must have shape (5, 3), "
            f"got {tip_pos.shape}"
        )

    force = np.empty((21, len(FORCE_CHANNELS)), dtype=np.float32)
    for joint_offset, joint in enumerate(joints):
        summary = joint.get("summary")
        if summary is None:
            raise KeyError(
                f"{path}: frame {frame_index} {hand}.joints[{joint_offset}] "
                "missing summary"
            )
        try:
            force[joint_offset] = [float(summary[key]) for key in FORCE_CHANNELS]
        except KeyError as exc:
            raise KeyError(
                f"{path}: frame {frame_index} {hand}.joints[{joint_offset}].summary "
                f"missing {exc.args[0]}"
            ) from exc

    return force, tip_pos


def _load_pseudo_force_sequence(path: Path) -> _PseudoForceSequenceData:
    with open(path, "rb") as f:
        raw = pickle.load(f)

    frame_indices = sorted(k for k in raw if isinstance(k, int))
    if not frame_indices:
        raise ValueError(f"No integer frame keys found in {path}")

    n = len(frame_indices)
    lh_contact = np.empty((n, 21, len(FORCE_CHANNELS)), dtype=np.float32)
    rh_contact = np.empty_like(lh_contact)
    lh_tip_pos = np.empty((n, 5, 3), dtype=np.float32)
    rh_tip_pos = np.empty_like(lh_tip_pos)

    for offset, frame_index in enumerate(frame_indices):
        frame = raw[frame_index]
        lh_contact[offset], lh_tip_pos[offset] = _extract_hand(
            frame, "lh", path, frame_index
        )
        rh_contact[offset], rh_tip_pos[offset] = _extract_hand(
            frame, "rh", path, frame_index
        )

    return _PseudoForceSequenceData(
        lh_contact=lh_contact,
        rh_contact=rh_contact,
        lh_tip_pos=lh_tip_pos,
        rh_tip_pos=rh_tip_pos,
    )


def _load_compact_pseudo_force_sequence(path: Path) -> _PseudoForceSequenceData:
    with np.load(path, allow_pickle=False) as compact:
        if "joint_force" in compact:
            force_key = "joint_force"
        elif "joint_contact" in compact:
            force_key = "joint_contact"
        else:
            raise KeyError(f"{path}: expected 'joint_force' or 'joint_contact'")
        joint_contact = np.asarray(compact[force_key], dtype=np.float32)
        finger_xyz = np.asarray(compact["finger_xyz"], dtype=np.float32)

    if joint_contact.ndim != 3 or joint_contact.shape[1:] != (42, 4):
        raise ValueError(
            f"{path}: joint_contact must have shape (N, 42, 4), "
            f"got {joint_contact.shape}"
        )
    if finger_xyz.shape != (joint_contact.shape[0], 10, 3):
        raise ValueError(
            f"{path}: finger_xyz must have shape ({joint_contact.shape[0]}, 10, 3), "
            f"got {finger_xyz.shape}"
        )
    return _PseudoForceSequenceData.from_compact(joint_contact, finger_xyz)


class PseudoForceAngleDataset(data.Dataset):
    """Windowed pseudo-force dataset with fingertip XYZ as the position stream.

    Samples contain:

    - ``joint_contact``: ``(W, 42, 4)`` or ``(W, 42, 8)`` when
      ``include_force_delta=True``. The extra four channels are the current
      force minus the preceding frame's force.
    - ``finger_xyz``: ``(W, 10, 3)``
    - ``classification_id``: contact class derived from fingertip normal force
    """

    def __init__(
        self,
        data_root: str,
        window_size: int = 1,
        window_stride: int = 1,
        split: str = "train",
        train_val_split: float = 0.9,
        participants: Optional[List[str]] = None,
        sensor_output_key: str = "joint_contact",
        include_force_delta: bool = False,
        _file_pattern: str = "*.pkl",
        _sequence_loader=_load_pseudo_force_sequence,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if window_size <= 0 or window_stride <= 0:
            raise ValueError("window_size and window_stride must be positive")

        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.sensor_output_key = str(sensor_output_key)
        self.include_force_delta = bool(include_force_delta)

        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(f"data_root not found: {root}")
        all_pkl = sorted(root.glob(_file_pattern))
        if participants is not None:
            all_pkl = [
                p for p in all_pkl
                if any(p.name.startswith(participant) for participant in participants)
            ]
        if not all_pkl:
            raise FileNotFoundError(f"No files matching {_file_pattern!r} found in {root}")

        n_train = max(1, int(len(all_pkl) * min(max(train_val_split, 0.0), 1.0)))
        if split == "train":
            pkl_files = all_pkl[:n_train]
        else:
            pkl_files = all_pkl[n_train:]
            if not pkl_files:
                n_val = max(1, int(n_train * 0.1))
                pkl_files = all_pkl[n_train - n_val:n_train]

        log.info(f"Found {len(all_pkl)} pseudo-force files in {root}")
        log.info(f"  {split}: loading {len(pkl_files)} files")
        self._sequences = [_sequence_loader(path) for path in pkl_files]
        self._seq_paths = pkl_files

        self.windows: List[Tuple[int, int]] = []
        for seq_index, sequence in enumerate(self._sequences):
            max_start = sequence.num_frames - self.window_size
            for start in range(0, max_start + 1, self.window_stride):
                self.windows.append((seq_index, start))

        self.window_labels = torch.tensor(
            [self._classification_id(seq_index, start) for seq_index, start in self.windows],
            dtype=torch.long,
        )
        counts = Counter(self.window_labels.tolist())
        log.info(
            f"  Loaded {len(self._sequences)} sequences, {len(self.windows)} windows; "
            f"classification counts={dict(sorted(counts.items()))}"
        )

    def _classification_id(self, seq_index: int, start: int) -> int:
        sequence = self._sequences[seq_index]
        end = start + self.window_size
        lh_force = sequence.lh_contact[start:end, _FINGERTIP_INDICES, 0]
        rh_force = sequence.rh_contact[start:end, _FINGERTIP_INDICES, 0]
        lh_touch = (lh_force > 0).any(axis=0)
        rh_touch = (rh_force > 0).any(axis=0)

        if lh_touch.any() and rh_touch.any():
            return 5
        if lh_touch.any():
            return 2 if lh_touch[0] and lh_touch.sum() >= 2 else 1
        if rh_touch.any():
            return 4 if rh_touch[0] and rh_touch.sum() >= 2 else 3
        return 0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        seq_index, start = self.windows[index]
        sequence = self._sequences[seq_index]
        end = start + self.window_size
        joint_contact = sequence.joint_contact[start:end]
        if self.include_force_delta:
            previous = torch.empty_like(joint_contact)
            if start == 0:
                previous[0].zero_()
            else:
                previous[0] = sequence.joint_contact[start - 1]
            if self.window_size > 1:
                previous[1:] = joint_contact[:-1]

            delta = joint_contact - previous
            # There is no preceding observation at a sequence boundary.
            if start == 0:
                delta[0].zero_()
            joint_contact = torch.cat((joint_contact, delta), dim=-1)

        return {
            self.sensor_output_key: joint_contact,
            "finger_xyz": sequence.finger_xyz[start:end],
            "classification_id": self.window_labels[index],
        }


class CompactPseudoForceDataset(PseudoForceAngleDataset):
    """Pseudo-force dataset backed by compact ``joint_contact``/``finger_xyz`` NPZs."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            _file_pattern="*.npz",
            _sequence_loader=_load_compact_pseudo_force_sequence,
            **kwargs,
        )
