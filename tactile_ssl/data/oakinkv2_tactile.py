import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data

from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

_MP_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
_MP_TIPS = [4, 8, 12, 16, 20]


class _OakInkV2SequenceData:
    """Loaded OakInkV2 sequence with precomputed wrist poses."""

    __slots__ = ("joint_data", "wrist_poses")

    def __init__(self, joint_data: np.ndarray, wrist_poses: np.ndarray):
        self.joint_data = joint_data      # (N, 42, 4)
        self.wrist_poses = wrist_poses    # (N, 2, 9)


def _load_oakinkv2_sequence(path: Path) -> _OakInkV2SequenceData:
    with open(path, "rb") as f:
        raw = pickle.load(f)

    frame_indices = sorted(k for k in raw.keys() if isinstance(k, int))
    n = len(frame_indices)
    joint_data = np.zeros((n, 42, 4), dtype=np.float32)
    wrist_poses = np.zeros((n, 2, 9), dtype=np.float32)

    for i, t in enumerate(frame_indices):
        frame = raw[t]
        joint_data[i, :21] = frame["lh"]
        joint_data[i, 21:] = frame["rh"]
        wrist_poses[i, 0, :3] = frame["lh_wrist_pos_vr"]
        wrist_poses[i, 0, 3:] = frame["lh_wrist_rot6d_vr"]
        wrist_poses[i, 1, :3] = frame["rh_wrist_pos_vr"]
        wrist_poses[i, 1, 3:] = frame["rh_wrist_rot6d_vr"]

    return _OakInkV2SequenceData(joint_data=joint_data, wrist_poses=wrist_poses)


class OakInkV2TactileDataset(data.Dataset):
    """OakInk-v2 hand keypoint pretraining dataset.

    데이터 경로: pretraining_dataset/OakInkv2/
    파일 포맷:  scene_XX__AXXX++seq__<hash>__<timestamp>.pkl

    scenes 파라미터로 특정 scene만 로드:
        scenes=['scene_01']  →  'scene_01__'로 시작하는 파일만 사용
    """

    def __init__(self, data_root: str = "pretraining_dataset/OakInkv2", **kwargs):
        self.window_size = int(kwargs.pop("window_size", 3))
        self.window_stride = int(kwargs.pop("window_stride", 1))
        split = kwargs.pop("split", "train")
        train_val_split = float(kwargs.pop("train_val_split", 0.9))
        scenes: Optional[List[str]] = kwargs.pop("scenes", None)

        root = Path(data_root)
        assert root.exists(), f"data_root not found: {root}"
        all_pkl = sorted(root.glob("*.pkl"))
        if scenes is not None:
            all_pkl = [p for p in all_pkl if any(p.name.startswith(s) for s in scenes)]
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

        self._sequences = []
        self._seq_paths = []
        for p in pkl_files:
            self._sequences.append(_load_oakinkv2_sequence(p))
            self._seq_paths.append(p)
        log.info(f"  Loaded {len(self._sequences)} sequences")

        self.windows: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(self._sequences):
            n_frames = seq.joint_data.shape[0]
            max_start = n_frames - self.window_size
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.window_stride):
                self.windows.append((seq_idx, start))
        log.info(f"  Total windows: {len(self.windows)}")

        self.tactile_mean = None
        self.tactile_std = None

    def update_normalization(self, mean, std):
        self.tactile_mean = float(np.asarray(mean).ravel()[0])
        self.tactile_std = float(np.asarray(std).ravel()[0])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        seq_idx, start = self.windows[idx]
        seq = self._sequences[seq_idx]

        window = seq.joint_data[start : start + self.window_size]
        wrist_poses = seq.wrist_poses[start : start + self.window_size]

        sensor = window[..., 3:4].copy()
        if self.tactile_mean is not None:
            sensor = (sensor - self.tactile_mean) / max(self.tactile_std, 1e-6)

        # OakInkV2 stores hand-joint xyz already in the hand-wrist frame.
        sensor_poses = window[..., :3].copy()

        return {
            "sensor": torch.from_numpy(sensor).float(),
            "sensor_poses": torch.from_numpy(sensor_poses).float(),
            "wrist_poses": torch.from_numpy(wrist_poses.copy()).float(),
            "sensor_id": torch.tensor(0, dtype=torch.long),
        }


if __name__ == "__main__":
    import argparse
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    parser = argparse.ArgumentParser(description="OakInkV2 양손 랜덤 시각화")
    parser.add_argument("--data_root", type=str, default="pretraining_dataset/OakInkv2")
    parser.add_argument("--n",     type=int, default=4,  help="시각화할 랜덤 샘플 수")
    parser.add_argument("--frame", type=int, default=0,  help="윈도우 내 프레임 인덱스")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--save",  type=str, default=None)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    args = parser.parse_args()

    dataset = OakInkV2TactileDataset(
        data_root=args.data_root,
        window_size=args.window_size,
        split=args.split,
    )
    if len(dataset) == 0:
        print("No data found. Check --data_root.")
        exit()
    print(f"Total windows: {len(dataset)}")

    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), min(args.n, len(dataset)))

    # ── Pre-load samples and compute global axis range ────────────────────────
    samples_data = []
    all_pos = []
    for idx in indices:
        sample    = dataset[idx]
        poses     = sample["sensor_poses"].numpy()   # (T, 42, 3)
        wrists    = sample["wrist_poses"].numpy()    # (T, 2, 9)
        frame_idx = min(args.frame, poses.shape[0] - 1)
        p         = poses[frame_idx]                 # (42, 3)
        left_pos  = p[:21]    # (21, 3) lh own-wrist-relative (world-aligned)
        right_pos = p[21:]    # (21, 3) rh own-wrist-relative (world-aligned)
        wrist_t   = wrists[frame_idx, :, :3]
        samples_data.append((sample, frame_idx, left_pos, right_pos, wrist_t))
        all_pos.append(left_pos)
        all_pos.append(right_pos)

    all_pos    = np.concatenate(all_pos, axis=0)
    global_min = all_pos.min()
    global_max = all_pos.max()
    pad        = (global_max - global_min) * 0.1
    ax_range   = (global_min - pad, global_max + pad)
    axis_len   = (global_max - global_min) * 0.15

    # ── Draw helper ───────────────────────────────────────────────────────────
    def draw_hand(ax, pos21, side, axis_len):
        color     = "tomato"    if side == "right" else "royalblue"
        tip_color = "red"       if side == "right" else "blue"
        for (i, j) in _MP_LINES:
            ax.plot([pos21[i,0], pos21[j,0]],
                    [pos21[i,1], pos21[j,1]],
                    [pos21[i,2], pos21[j,2]], color=color, linewidth=1.2)
        ax.scatter(*pos21.T, c=color,     s=10, zorder=3)
        ax.scatter(*pos21[_MP_TIPS].T, c=tip_color, s=40, zorder=4, marker="*")
        ax.scatter(0, 0, 0, c="black", s=60, zorder=5,
                   label=f"{'LH' if side=='left' else 'RH'} wrist (origin)")
        x_ax, y_ax, z_ax = _compute_hand_frame(pos21, side=side)
        for vec, col, lbl in [(x_ax,"red","X"), (y_ax,"green","Y"), (z_ax,"blue","Z")]:
            ax.quiver(0, 0, 0, vec[0]*axis_len, vec[1]*axis_len, vec[2]*axis_len,
                      color=col, linewidth=1.5)
            ax.text(*(vec * axis_len * 1.15), lbl, color=col, fontsize=7, fontweight="bold")
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7); ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=5)

    # ── Figure: N rows × 2 cols (LH | RH) ────────────────────────────────────
    n   = len(indices)
    fig = plt.figure(figsize=(14, 5 * n))

    for row, (sample, frame_idx, left_pos, right_pos, wrist_t) in enumerate(samples_data):
        seq_idx, start = dataset.windows[indices[row]]
        pkl_path  = dataset._seq_paths[seq_idx]
        abs_frame = start + frame_idx
        info = (f"[window #{indices[row]}]  seq={seq_idx}  "
                f"frame_start={start}  frame={abs_frame}\n{pkl_path.name}")

        ax_lh = fig.add_subplot(n, 2, row * 2 + 1, projection="3d")
        draw_hand(ax_lh, left_pos, "left", axis_len)
        ax_lh.set_xlim(*ax_range); ax_lh.set_ylim(*ax_range); ax_lh.set_zlim(*ax_range)
        ax_lh.set_title(
            f"LH  {info}\n"
            f"contact_lh={sample['sensor'].numpy()[frame_idx,:21,0].sum():.2f}  "
            f"wrist_t={np.array2string(wrist_t[0], precision=3)}",
                        fontsize=7)
        ax_lh.legend(loc="upper left", fontsize=6)

        ax_rh = fig.add_subplot(n, 2, row * 2 + 2, projection="3d")
        draw_hand(ax_rh, right_pos, "right", axis_len)
        ax_rh.set_xlim(*ax_range); ax_rh.set_ylim(*ax_range); ax_rh.set_zlim(*ax_range)
        ax_rh.set_title(
            f"RH  {info}\n"
            f"contact_rh={sample['sensor'].numpy()[frame_idx,21:,0].sum():.2f}  "
            f"wrist_t={np.array2string(wrist_t[1], precision=3)}",
                        fontsize=7)
        ax_rh.legend(loc="upper left", fontsize=6)

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=250)
        print(f"Saved to {args.save}")
    else:
        plt.show()
