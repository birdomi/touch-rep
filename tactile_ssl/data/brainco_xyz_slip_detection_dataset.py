"""BrainCo slip-detection dataset with tactile contact and fingertip XYZ.

Each episode's ``data.json`` contains one or more inclusive slip intervals in
``slip_start_frame_index`` and ``slip_end_frame_index``. This adapter makes
non-overlapping three-frame samples ``[t, t+1, t+2]`` and uses the final
frame's label. At a final partial window, the final valid frame is repeated.
Windows touching the configured transition margins immediately before slip
onset or immediately after slip offset are discarded.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig

from tactile_ssl.data.brainco_tactile import BraincoSSLDataset
from tactile_ssl.data.brainco_xyz_grasp_dataset import _align_xyz_to_npz
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


def _as_index_list(value, field_name: str, episode_path: Path) -> list[int]:
    """Normalize scalar/list JSON metadata to a list of frame indices."""
    if value is None:
        raise ValueError(f"Missing {field_name} in {episode_path / 'data.json'}")
    if not isinstance(value, list):
        value = [value]
    try:
        return [int(index) for index in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {field_name} in {episode_path / 'data.json'}: {value!r}"
        ) from exc


def _read_slip_intervals(
    episode_path: Path, num_frames: int
) -> list[tuple[int, int]]:
    """Read and validate inclusive slip intervals from one episode."""
    with (episode_path / "data.json").open("r") as file:
        metadata = json.load(file)

    starts = _as_index_list(
        metadata.get("slip_start_frame_index"), "slip_start_frame_index", episode_path
    )
    ends = _as_index_list(
        metadata.get("slip_end_frame_index"), "slip_end_frame_index", episode_path
    )
    if len(starts) != len(ends):
        raise ValueError(
            f"Mismatched slip interval metadata in {episode_path / 'data.json'}: "
            f"{len(starts)} starts, {len(ends)} ends"
        )

    intervals = []
    for start, end in zip(starts, ends):
        if start < 0 or end < 0 or start >= num_frames or end >= num_frames:
            raise ValueError(
                f"Slip interval [{start}, {end}] is outside [0, {num_frames - 1}] "
                f"in {episode_path / 'data.json'}"
            )
        if end < start:
            raise ValueError(
                f"Slip interval end precedes start: [{start}, {end}] in "
                f"{episode_path / 'data.json'}"
            )
        intervals.append((start, end))
    return intervals


def _labels_from_intervals(
    intervals: list[tuple[int, int]], num_frames: int
) -> np.ndarray:
    labels = np.zeros(num_frames, dtype=np.int64)
    for start, end in intervals:
        labels[start : end + 1] = 1
    return labels


def _read_slip_labels(episode_path: Path, num_frames: int) -> np.ndarray:
    """Return a binary label for every frame from inclusive slip intervals."""
    return _labels_from_intervals(
        _read_slip_intervals(episode_path, num_frames), num_frames
    )


def _excluded_transition_frames(
    intervals: list[tuple[int, int]],
    num_frames: int,
    before_start: int,
    after_end: int,
) -> np.ndarray:
    """Mask ambiguous non-slip frames around slip transitions."""
    excluded = np.zeros(num_frames, dtype=bool)
    for start, end in intervals:
        excluded[max(0, start - before_start) : start] = True
        excluded[end + 1 : min(num_frames, end + 1 + after_end)] = True
    return excluded


class BraincoXYZSlipDetectionDataset(data.Dataset):
    """Aggregate slip episodes into non-overlapping tactile/XYZ windows.

    The public ``episode_data`` and ``windows`` attributes match the interface
    consumed by :mod:`train_task_brainco_angle`, which performs K-fold splits
    at the episode level and therefore avoids frame leakage across splits.
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        brainco_urdf_path: str = "dataset/brainco/urdf",
    ):
        self.episode_data = []
        self.windows = []

        root = Path(data_path)
        class_names = list(config.get("classes", []))
        if not class_names:
            class_names = sorted(path.name for path in root.iterdir() if path.is_dir())

        subtract_baseline = bool(config.get("subtract_baseline", False))
        align_xyz_to_npz = bool(config.get("align_xyz_to_npz", True))
        self.include_force_delta = bool(config.get("include_force_delta", False))
        input_window_frames = int(config.get("input_window_frames", 3))
        input_window_stride = int(
            config.get("input_window_stride", input_window_frames)
        )
        exclude_before_slip_start_frames = int(
            config.get("exclude_before_slip_start_frames", 0)
        )
        exclude_after_slip_end_frames = int(
            config.get("exclude_after_slip_end_frames", 0)
        )
        if input_window_frames != 3:
            raise ValueError(
                "BraincoXYZSlipDetectionDataset requires input_window_frames=3"
            )
        if input_window_stride < input_window_frames:
            raise ValueError(
                "input_window_stride must be at least input_window_frames "
                "to avoid overlap"
            )
        if (
            exclude_before_slip_start_frames < 0
            or exclude_after_slip_end_frames < 0
        ):
            raise ValueError("Slip transition exclusion margins must be >= 0")
        max_values = torch.tensor([25000, 25000, 365, 500000], dtype=torch.float32)
        num_excluded_windows = 0

        for class_name in class_names:
            class_dir = root / class_name
            if not class_dir.exists():
                log.warning(f"Class directory not found, skipping: {class_dir}")
                continue

            episode_dirs = sorted(
                path for path in class_dir.iterdir()
                if path.is_dir() and (path / "data.json").exists()
            )
            log.info(f"Loading {len(episode_dirs)} slip episodes from: {class_dir}")

            for episode_path in episode_dirs:
                try:
                    episode = BraincoSSLDataset(
                        config=config,
                        data_path=str(episode_path),
                        brainco_urdf_path=brainco_urdf_path,
                    )
                    intervals = _read_slip_intervals(
                        episode_path, episode.num_frames
                    )
                    labels = _labels_from_intervals(intervals, episode.num_frames)
                    excluded_frames = _excluded_transition_frames(
                        intervals,
                        episode.num_frames,
                        before_start=exclude_before_slip_start_frames,
                        after_end=exclude_after_slip_end_frames,
                    )
                except Exception as exc:
                    log.warning(f"Skipping {episode_path}: {exc}")
                    continue

                tactile = episode.tactile_array.copy()
                tactile_valid = tactile >= 0
                if subtract_baseline:
                    baseline = tactile[0].copy()
                    valid = tactile_valid & (baseline[np.newaxis] >= 0)
                    tactile -= np.where(valid, baseline[np.newaxis], 0)

                tactile = torch.from_numpy(tactile)
                tactile = tactile / max_values.view(1, 1, -1)
                if self.include_force_delta:
                    tactile_delta = torch.zeros_like(tactile)
                    valid_pairs = torch.from_numpy(
                        tactile_valid[1:] & tactile_valid[:-1]
                    )
                    frame_delta = tactile[1:] - tactile[:-1]
                    tactile_delta[1:] = torch.where(
                        valid_pairs, frame_delta, torch.zeros_like(frame_delta)
                    )

                candidate_starts = list(
                    range(0, episode.num_frames, input_window_stride)
                )
                window_specs = []
                for window_start in candidate_starts:
                    frame_indices = np.clip(
                        np.arange(window_start, window_start + input_window_frames),
                        0,
                        episode.num_frames - 1,
                    )
                    if excluded_frames[frame_indices].any():
                        num_excluded_windows += 1
                        continue
                    window_specs.append((window_start, frame_indices))

                self.episode_data.append({
                    "path": str(episode_path),
                    "window_starts": [start for start, _ in window_specs],
                })

                for window_start, frame_indices in window_specs:
                    label_frame = min(
                        window_start + input_window_frames - 1,
                        episode.num_frames - 1,
                    )
                    joint_contact = tactile[frame_indices]
                    if self.include_force_delta:
                        joint_contact = torch.cat(
                            (joint_contact, tactile_delta[frame_indices]), dim=-1
                        )
                    finger_xyz = torch.from_numpy(
                        episode.fingertip_rel[frame_indices].copy()
                    )
                    if align_xyz_to_npz:
                        finger_xyz = _align_xyz_to_npz(finger_xyz)

                    self.windows.append({
                        "joint_contact": joint_contact,
                        "finger_xyz": finger_xyz,
                        "label": torch.tensor(int(labels[label_frame]), dtype=torch.long),
                        "episode_path": str(episode_path),
                        "window_start_frame": window_start,
                    })

        num_slip = sum(sample["label"].item() for sample in self.windows)
        log.info(
            f"BraincoXYZSlipDetectionDataset: {len(self.episode_data)} episodes, "
            f"{len(self.windows)} samples (slip={num_slip}, "
            f"non-slip={len(self.windows) - num_slip}, "
            f"excluded_transition_windows={num_excluded_windows}, "
            f"force_delta={self.include_force_delta})"
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]
