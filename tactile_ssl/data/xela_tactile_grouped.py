"""XelaSSLDatasetGrouped: XELA 368-taxel data grouped into 10 tokens.

Groups the 18 XELA sensor patches (368 taxels total) into 10 anatomical
regions, matching BrainCo's 10-fingertip token layout.

10-token layout (mean-pooled within each group):
  Token 0: thumb tip    (3aftc,  30 taxels, taxels   0..29)
  Token 1: thumb body   (link_15+link_14, 32 taxels, 30..61)
  Token 2: finger0 tip  (0aftc,  30 taxels,          62..91)
  Token 3: finger0 body (link_2+link_1A+link_1B, 48, 92..139)
  Token 4: finger1 tip  (1aftc,  30 taxels,          140..169)
  Token 5: finger1 body (link_6+link_5A+link_5B, 48, 170..217)
  Token 6: finger2 tip  (2aftc,  30 taxels,          218..247)
  Token 7: finger2 body (link_10+link_9A+link_9B,48, 248..295)
  Token 8: palm front   (ahr_palm_2+ahr_palm_1, 48,  296..343)
  Token 9: palm back    (ahr_palm_3, 24 taxels,       344..367)

Output per sample:
  기본 (sparse_seq=False):
    sensor       : (1, 10, T*4=40) — T frames flattened into channel dim
    sensor_poses : (1, 10, 6)  — last-frame FK translation [xyz | xyz]

  sparse_seq=True (BrainCo seq 맞춤):
    sensor       : (len(sparse_seq_offsets), 10, 4) — offset 프레임만 선택
    sensor_poses : (len(sparse_seq_offsets), 10, 6) — 동일 프레임 FK pose
    e.g. sparse_seq_offsets=[-10, -5, 0] → (3, 10, 4)

  sensor_id    : scalar long tensor = 1  (XELA sensor type)
"""

from typing import Optional
import pickle
from pathlib import Path

import einops
import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R

from tactile_ssl.data.xela.utils import (
    read_xela_data,
    read_allegro_joint_data,
    compute_interp_timestamps,
    load_data_dict,
    XELA_FLATTEN_ORDER,
)
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

# ── 10-token grouping ─────────────────────────────────────────────────────────
# Each entry: (start_taxel_idx, end_taxel_idx) — exclusive end, in the flat
# 368-taxel array ordered by XELA_FLATTEN_ORDER.
XELA_10_GROUP_TAXEL_RANGES = [
    (0,   30),   # Token 0: thumb tip    (3aftc, 30 taxels)
    (30,  62),   # Token 1: thumb body   (link_15=16 + link_14=16)
    (62,  92),   # Token 2: finger0 tip  (0aftc, 30 taxels)
    (92,  140),  # Token 3: finger0 body (link_2=16 + link_1A=16 + link_1B=16)
    (140, 170),  # Token 4: finger1 tip  (1aftc, 30 taxels)
    (170, 218),  # Token 5: finger1 body (link_6=16 + link_5A=16 + link_5B=16)
    (218, 248),  # Token 6: finger2 tip  (2aftc, 30 taxels)
    (248, 296),  # Token 7: finger2 body (link_10=16 + link_9A=16 + link_9B=16)
    (296, 344),  # Token 8: palm front   (ahr_palm_2=24 + ahr_palm_1=24)
    (344, 368),  # Token 9: palm back    (ahr_palm_3=24)
]
assert sum(e - s for s, e in XELA_10_GROUP_TAXEL_RANGES) == 368

# Indices into the 18 FK joints (ordered as XELA_FLATTEN_ORDER keys) that
# contribute to each of the 10 output tokens.
XELA_10_GROUP_FK_JOINTS = [
    [0],           # Token 0: "3aftc_palm_link"
    [1, 2],        # Token 1: "link_15_4x4_palm_link", "link_14_4x4_palm_link"
    [3],           # Token 2: "0aftc_palm_link"
    [4, 5, 6],     # Token 3: "link_2", "link_1A", "link_1B"
    [7],           # Token 4: "1aftc_palm_link"
    [8, 9, 10],    # Token 5: "link_6", "link_5A", "link_5B"
    [11],          # Token 6: "2aftc_palm_link"
    [12, 13, 14],  # Token 7: "link_10", "link_9A", "link_9B"
    [15, 16],      # Token 8: "ahr_palm_2", "ahr_palm_1"
    [17],          # Token 9: "ahr_palm_3"
]
NUM_TOKENS = 10


class XelaSSLDatasetGrouped(data.Dataset):
    """XELA pretraining dataset with 368 taxels grouped into 10 tokens.

    Designed to produce samples in the same format as BraincoSSLDataset so
    both can be fed into the same BraincoDINOv2 model (in_dim=10, in_chans=4).
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        xela_urdf_path: str,
        baseline_signal_path: Optional[str] = None,
        object_class: Optional[int] = None,
        load_images: bool = False,
    ):
        # ── Config defaults ──────────────────────────────────────────────
        if config.get("window_overlap") is None:
            config.window_overlap = 0.0
        if config.get("subtract_baseline") is None:
            config.subtract_baseline = False
        if config.get("smooth_data") is None:
            config.smooth_data = False
        if config.get("bias_noise_std") is None:
            config.bias_noise_std = 0.0
        if config.get("bias_range") is None:
            config.bias_range = 0.0
        if config.get("sparse_seq") is None:
            config.sparse_seq = False
        if config.get("sparse_seq_offsets") is None:
            config.sparse_seq_offsets = [-10, -5, 0]
        if baseline_signal_path is None:
            config.subtract_baseline = False

        self.window_time = config.window_time
        assert 0 <= config.window_overlap < 1
        self.window_overlap = config.window_overlap
        self.interpolating_freq = config.interpolating_freq
        self.num_frames_per_window = int(round(self.window_time * self.interpolating_freq))
        self.shift_per_window = max(1, int(round(self.num_frames_per_window * (1.0 - self.window_overlap))))

        self.subtract_baseline = config.subtract_baseline
        self.smooth_data = config.smooth_data
        self.bias_noise_std = config.bias_noise_std
        self.bias_range = config.bias_range
        self.object_label = object_class

        # sparse_seq: window 마지막 프레임 기준 offset 프레임만 선택 → (S, 10, 4)
        self.sparse_seq = config.sparse_seq
        self.sparse_seq_offsets = list(config.sparse_seq_offsets)  # e.g. [-10, -5, 0]

        # ── Load raw XELA data ───────────────────────────────────────────
        self.data_path = data_path
        xela_dict, allegro_dict = load_data_dict(self.data_path)

        self.baseline_signal_path = baseline_signal_path
        if self.baseline_signal_path is not None:
            with open(self.baseline_signal_path, "rb") as f:
                baseline_signal = np.asarray(pickle.load(f))
            self.xela_baseline = np.mean(baseline_signal[:, :, 1:], axis=0)
        self.xela_mean, self.xela_std = None, None

        xela_array = np.array(xela_dict, copy=True)

        allegro_array = None
        if allegro_dict is not None:
            allegro_array = np.array(allegro_dict["joint_states"], copy=True)
            self.timestamps, self.num_frames = compute_interp_timestamps(
                [xela_array[:, 0, 0], allegro_array[:, 0]], self.interpolating_freq
            )
        else:
            self.timestamps, self.num_frames = compute_interp_timestamps(
                [xela_array[:, 0, 0]], self.interpolating_freq
            )

        self.xela_array = read_xela_data(
            xela_array, self.timestamps, self.interpolating_freq, self.smooth_data
        )  # (T, 368, 4)

        if allegro_array is not None:
            self.joint_angles, self.joint_effort = read_allegro_joint_data(
                allegro_array, self.timestamps, self.interpolating_freq, self.smooth_data
            )
        else:
            self.joint_angles = np.zeros([self.num_frames, 16], dtype=np.float32)
            self.joint_effort = np.zeros([self.num_frames, 16], dtype=np.float32)

        # ── Outlier removal ──────────────────────────────────────────────
        self.xela_array[..., 1:] = np.where(self.xela_array[..., 1:] < 20000, 0, self.xela_array[..., 1:])
        self.xela_array[..., 1:] = np.where(self.xela_array[..., 1:] > 60000, 0, self.xela_array[..., 1:])

        # ── Baseline subtraction ─────────────────────────────────────────
        mask = self.xela_array[:, ..., 1] != 0
        if self.subtract_baseline and hasattr(self, "xela_baseline"):
            baseline = einops.repeat(self.xela_baseline, "k c -> b k c", b=self.xela_array.shape[0])
            self.xela_array[mask, 1:] = self.xela_array[mask, 1:] - baseline[mask, :]

        # ── Window indices ───────────────────────────────────────────────
        max_length = self.num_frames - (self.num_frames % self.num_frames_per_window)
        max_length = max_length - self.num_frames_per_window
        self.data_idxs = np.arange(0, max(1, max_length), self.shift_per_window)

        # ── FK-based grouped positions ───────────────────────────────────
        assert Path(xela_urdf_path).exists(), f"{xela_urdf_path} does not exist"
        xela_chain = pk.build_chain_from_urdf(open(xela_urdf_path).read())
        self.grouped_poses = self._compute_grouped_poses(xela_chain)  # (T, 10, 6)

        log.info(
            f"XelaSSLDatasetGrouped: {len(self.data_idxs)} windows, "
            f"frames_per_window={self.num_frames_per_window}, "
            f"grouped_poses={self.grouped_poses.shape}"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_grouped_poses(self, xela_chain) -> np.ndarray:
        """Compute mean FK translation per 10-token group.

        Returns (T, 10, 6) where the last dim is [xyz | xyz] — the 3-D
        translation duplicated to match BrainCo's 6-D fingertip format.
        """
        T = self.num_frames
        joint_angles = torch.from_numpy(self.joint_angles.astype(np.float32))  # (T, 16)
        fk_result = xela_chain.forward_kinematics(joint_angles)

        translations_all = np.zeros((T, 18, 3), dtype=np.float32)
        for i, k in enumerate(XELA_FLATTEN_ORDER.keys()):
            t_mat = fk_result[k].get_matrix()              # (T, 4, 4)
            translations_all[:, i, :] = t_mat[:, :3, 3].numpy()

        # Average within each 10-token group
        grouped = np.zeros((T, NUM_TOKENS, 3), dtype=np.float32)
        for g, fk_idxs in enumerate(XELA_10_GROUP_FK_JOINTS):
            grouped[:, g, :] = np.mean(translations_all[:, fk_idxs, :], axis=1)

        # Duplicate xyz → 6-D [own-wrist-relative | opposite-wrist-relative proxy]
        sensor_poses_6d = np.concatenate([grouped, grouped], axis=-1)  # (T, 10, 6)
        return sensor_poses_6d.astype(np.float32)

    def _group_sensor_data(self, sensor_data: np.ndarray) -> np.ndarray:
        """Group (T, 368, 3) force data into (T, 10, 4) via mean pooling.

        Channel layout of the output:
          0-2 : mean force xyz
          3   : mean L2 norm (contact-intensity proxy, analogous to BrainCo approxi)
        """
        T = sensor_data.shape[0]
        grouped = np.zeros((T, NUM_TOKENS, 4), dtype=np.float32)
        for g, (s, e) in enumerate(XELA_10_GROUP_TAXEL_RANGES):
            group = sensor_data[:, s:e, :]          # (T, N_group, 3)
            grouped[:, g, :3] = np.mean(group, axis=1)
            grouped[:, g, 3] = np.mean(np.linalg.norm(group, axis=-1), axis=1)
        return grouped

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.data_idxs)

    def update_normalization(self, mean, std):
        self.xela_mean = mean
        self.xela_std = std

    def __getitem__(self, idx):
        index = self.data_idxs[idx]
        last = index + self.num_frames_per_window - 1

        if self.sparse_seq:
            # offset 프레임만 선택: window 마지막 기준 → (S, 368, 3)
            frame_indices = [last + o for o in self.sparse_seq_offsets]
            raw = self.xela_array[frame_indices, :, 1:]         # (S, 368, 3)
            sensor_data = self._group_sensor_data(raw)          # (S, 10, 4)
            sensor_data_out = torch.from_numpy(sensor_data).float()  # (S, 10, 4)

            sensor_poses = self.grouped_poses[frame_indices]    # (S, 10, 6)
            sensor_poses_out = torch.from_numpy(sensor_poses).float()
        else:
            # 기존 동작: T frames → channel dim flatten
            raw = self.xela_array[index : index + self.num_frames_per_window, :, 1:]
            sensor_data = self._group_sensor_data(raw)          # (T, 10, 4)
            sensor_data_t = torch.from_numpy(sensor_data).float()
            sensor_data_out = einops.rearrange(sensor_data_t, "t n c -> 1 n (t c)")  # (1, 10, T*4)

            sensor_poses = self.grouped_poses[last : last + 1]  # (1, 10, 6)
            sensor_poses_out = torch.from_numpy(sensor_poses).float()

        # print(sensor_data.shape, "xela")

        sample_dict = {
            "sensor": sensor_data_out,
            "sensor_poses": sensor_poses_out,
            "sensor_id": torch.tensor(1, dtype=torch.long),
        }
        if self.object_label is not None:
            sample_dict["object_classification"] = torch.tensor(self.object_label)
        return sample_dict
