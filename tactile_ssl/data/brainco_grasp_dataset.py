"""
BrainCo Grasp Detection Dataset.

Episode-level binary classification:
  - grasp_success/ → label=1
  - grasp_fail/    → label=0

Each episode contains data.json + tactiles/ following the same format as
BraincoSSLDataset. This dataset loads all episodes, computes per-episode
tactile/position features using windowed slicing, and returns them as
variable-length sequences for attentive pooling.
"""

from typing import Optional, List
import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.data.brainco_tactile import (
    build_combined_chain,
    compute_combined_fk,
    FINGERTIP_LINKS,
)

log = get_pylogger(__name__)


class BraincoGraspDetectionDataset(data.Dataset):
    """Episode-level grasp success/fail classification dataset.

    Each sample corresponds to one episode. The dataset loads tactile sensor
    data and fingertip positions for the entire episode, splits into windows,
    and returns (tactile_windows, position_windows, label).
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        brainco_urdf_path: str = "dataset/brainco/urdf",
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.brainco_urdf_path = Path(brainco_urdf_path)

        # Window parameters
        self.window_time = config.window_time
        self.window_overlap = config.get("window_overlap", 0.0)
        self.interpolating_freq = config.interpolating_freq
        self.num_frames_per_window = int(round(self.window_time * self.interpolating_freq))
        self.shift_per_window = int(round(self.num_frames_per_window * (1.0 - self.window_overlap)))
        self.shift_per_window = max(1, self.shift_per_window)

        # Build kinematic chain once (shared across all episodes)
        self.chain = build_combined_chain(self.brainco_urdf_path)

        # Discover episodes
        self.episodes = []  # list of (episode_path, label)
        self._discover_episodes()
        
        # Pre-load all episodes into memory
        self.episode_data = []  # list of dicts with tactile_windows, pos_windows, label
        self._load_all_episodes()
        
        # Flatten into individual windows for default dataloader collate
        self.windows = []
        for ep in self.episode_data:
            tactile_array = ep["tactile_array"]
            fingertip_6d = ep["fingertip_6d"]
            label = ep["label"]
            for start in ep["window_starts"]:
                end = start + self.num_frames_per_window
                sensor_win = tactile_array[start:end]
                pos_win = fingertip_6d[start:end]
                self.windows.append({
                    "sensor": torch.from_numpy(sensor_win).float(),
                    "sensor_poses": torch.from_numpy(pos_win).float(),
                    "label": torch.tensor(label, dtype=torch.long),
                    "episode_path": ep["path"],
                    "window_start_frame": int(start),
                })

        log.info(
            f"BraincoGraspDetectionDataset: {len(self.episodes)} episodes, "
            f"{len(self.windows)} total windows."
        )

    def _discover_episodes(self):
        """Find all episodes under grasp_success/ and grasp_fail/."""
        for label_name, label_val in [("grasp_success", 1), ("grasp_fail", 0)]:
            label_dir = self.data_path / label_name
            if not label_dir.exists():
                log.warning(f"Label directory {label_dir} not found, skipping.")
                continue
            episode_dirs = sorted(
                [d for d in label_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
            )
            for ep_dir in episode_dirs:
                data_json = ep_dir / "data.json"
                if data_json.exists():
                    self.episodes.append((ep_dir, label_val))
                else:
                    log.warning(f"No data.json in {ep_dir}, skipping.")

    def _load_episode(self, ep_path: Path):
        """Load and process a single episode into windowed tactile + position data."""
        data_json_path = ep_path / "data.json"
        with open(data_json_path, "r") as f:
            raw_data = json.load(f)

        frames = raw_data["data"]
        num_frames = len(frames)

        # ── Extract joint positions ──
        left_arm_list, right_arm_list = [], []
        left_ee_list, right_ee_list = [], []
        for frame in frames:
            states = frame["states"]
            left_arm_list.append(states["left_arm"]["qpos"])
            right_arm_list.append(states["right_arm"]["qpos"])
            left_ee_list.append(states["left_ee"]["qpos"])
            right_ee_list.append(states["right_ee"]["qpos"])

        left_arm_array = np.array(left_arm_list, dtype=np.float32)
        right_arm_array = np.array(right_arm_list, dtype=np.float32)
        left_ee_array = np.array(left_ee_list, dtype=np.float32)
        right_ee_array = np.array(right_ee_list, dtype=np.float32)

        # ── FK for fingertip positions ──
        fingertip_positions_list = []
        wrist_positions_list = []
        for i in range(num_frames):
            fk_res = compute_combined_fk(
                self.chain,
                left_arm_array[i], right_arm_array[i],
                left_ee_array[i], right_ee_array[i],
            )
            # Wrists
            left_wrist = fk_res["left_rubber_hand"].get_matrix()[:, :3, 3].squeeze(0).numpy()
            right_wrist = fk_res["right_rubber_hand"].get_matrix()[:, :3, 3].squeeze(0).numpy()
            wrist_positions_list.append(np.stack([left_wrist, right_wrist], axis=0))

            # Fingertips
            frame_positions = []
            for link_suffix in FINGERTIP_LINKS:
                frame_positions.append(fk_res[f"left_{link_suffix}"].get_matrix()[:, :3, 3].squeeze(0).numpy())
            for link_suffix in FINGERTIP_LINKS:
                frame_positions.append(fk_res[f"right_{link_suffix}"].get_matrix()[:, :3, 3].squeeze(0).numpy())
            fingertip_positions_list.append(np.stack(frame_positions, axis=0))

        fp_world = np.array(fingertip_positions_list, dtype=np.float32)      # (N, 10, 3)
        wrist_positions = np.array(wrist_positions_list, dtype=np.float32)    # (N, 2, 3)

        # ── 6D fingertip positions (relative to own + opposite wrist) ──
        left_idxs = [0, 1, 2, 3, 4]
        right_idxs = [5, 6, 7, 8, 9]
        fingertip_6d = np.zeros((num_frames, 10, 6), dtype=np.float32)
        for idx in left_idxs:
            fingertip_6d[:, idx, :3] = fp_world[:, idx, :] - wrist_positions[:, 0, :]
            fingertip_6d[:, idx, 3:] = fp_world[:, idx, :] - wrist_positions[:, 1, :]
        for idx in right_idxs:
            fingertip_6d[:, idx, :3] = fp_world[:, idx, :] - wrist_positions[:, 1, :]
            fingertip_6d[:, idx, 3:] = fp_world[:, idx, :] - wrist_positions[:, 0, :]

        # ── Load tactile data ──
        tactile_list = []
        for frame in frames:
            tactile_info = frame["tactiles"]
            if isinstance(tactile_info["left_ee"], str):
                left_path = ep_path / tactile_info["left_ee"]
                left_tactile = np.load(str(left_path)).reshape(-1, 4)
            else:
                left_tactile = np.array(tactile_info["left_ee"]).reshape(-1, 4)

            if isinstance(tactile_info["right_ee"], str):
                right_path = ep_path / tactile_info["right_ee"]
                right_tactile = np.load(str(right_path)).reshape(-1, 4)
            else:
                right_tactile = np.array(tactile_info["right_ee"]).reshape(-1, 4)

            tactile = np.concatenate([left_tactile, right_tactile], axis=0)
            tactile_list.append(tactile)

        tactile_array = np.array(tactile_list, dtype=np.float32)  # (N, num_sensors, 4)

        # Replace invalid values
        invalid_mask = tactile_array[..., 2] == 65535
        tactile_array[..., 2][invalid_mask] = -1

        # ── Window indices ──
        max_length = num_frames - (num_frames % self.num_frames_per_window)
        max_length = max_length - self.num_frames_per_window
        window_starts = np.arange(0, max(1, max_length), self.shift_per_window)

        print(f"Loaded {ep_path.name}: {num_frames} frames -> {len(window_starts)} windows")

        return {
            "tactile_array": tactile_array,
            "fingertip_6d": fingertip_6d,
            "window_starts": window_starts,
            "num_frames": num_frames,
        }

    def _load_all_episodes(self):
        """Pre-load all episodes into memory."""
        for ep_path, label in self.episodes:
            try:
                ep_data = self._load_episode(ep_path)
                ep_data["label"] = label
                ep_data["path"] = str(ep_path)
                self.episode_data.append(ep_data)
            except Exception as e:
                log.warning(f"Failed to load episode {ep_path}: {e}")
                

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
    # --- Main Block for Testing ---
if __name__ == "__main__":
    import argparse
    import torch
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser(description="Test BraincoGraspDetectionDataset")
    parser.add_argument("--data_path", type=str, default="dataset/brainco/downstream/grasp_detection",
                        help="Path to the grasp detection dataset root")
    parser.add_argument("--urdf_path", type=str, default="dataset/brainco/urdf",
                        help="Path to the BrainCo URDF directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for testing the dataloader")
    args = parser.parse_args()
    
    # Create a dummy config mapping what the dataset expects
    config = OmegaConf.create({
        "window_time": 0.1,  # 0.1s windows 
        "window_overlap": 0.5, # 50% overlap
        "interpolating_freq": 100, # 100 Hz
    })
    
    print("Initializing BraincoGraspDetectionDataset...")
    dataset = BraincoGraspDetectionDataset(
        config=config,
        data_path=args.data_path,
        brainco_urdf_path=args.urdf_path
    )
    
    print(f"\nTotal episodes loaded: {len(dataset)}")
    if len(dataset) == 0:
        print("No episodes found. Please check your data_path!")
        exit()
        
    # Inspect a single item
    sample = dataset[0]
    print("\n--- Single Item Inspection ---")
    print(f"Sensor shape (num_windows, W, num_sensors, 4): {sample['sensor'].shape}")
    print(f"Poses shape (num_windows, W, 10, 6): {sample['sensor_poses'].shape}")
    print(f"Label: {sample['label'].item()}")
    
    # Test DataLoader with default collate_fn
    print(f"\n--- DataLoader Inspection (Batch Size: {args.batch_size}) ---")
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Sensor       (B, W, num_sensors, 4): {batch['sensor'].shape}")
        print(f"  Poses        (B, W, 10, 6):          {batch['sensor_poses'].shape}")
        print(f"  Label        (B,):                   {batch['label'].shape}")
        break # Just test the first batch
