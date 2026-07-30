"""ACT (Action Chunking Transformer) dataset over BrainCo episodes.

Each episode directory holds a ``data.json`` whose frames carry both the
measured robot ``states`` and the commanded ``actions``:

``states``/``actions``
    ``left_arm``/``right_arm`` (7 joints each) and ``left_ee``/``right_ee``
    (6 joints each) → 26-D by default.
``colors``
    One JPEG per camera per frame under ``colors/``.
``tactiles``
    ``left_ee``/``right_ee``, 5 fingertips x 4 channels each → ``(10, 4)``.

One sample = one timestep ``t`` of one episode:

``qpos``    ``(state_dim,)``            normalized robot state at ``t``
``images``  ``(num_cameras, 3, H, W)``  normalized RGB at ``t``
``tactile`` ``(10, 4)``                 normalized tactile reading at ``t``
``actions`` ``(chunk_size, action_dim)`` normalized action chunk from ``t``
``is_pad``  ``(chunk_size,)``           True where the chunk ran past the episode

``episode_data`` mirrors the interface used by the other BrainCo downstream
datasets so ``train_act.py`` can do episode-level K-fold splitting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from omegaconf import DictConfig
import torchvision.transforms as T

from tactile_ssl.data.force_channels import force_direction_to_cartesian
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_QPOS_KEYS = ("left_arm", "right_arm", "left_ee", "right_ee")
NUM_TACTILE_SENSORS = 10
TACTILE_CHANNELS = 4


def _episode_dirs(root: Path) -> List[Path]:
    """Find every episode directory (a directory holding ``data.json``).

    Supports both ``root/episode_X/`` and ``root/<class>/episode_X/`` layouts.
    """
    if (root / "data.json").exists():
        return [root]
    found = {path.parent for path in root.glob("*/data.json")}
    found |= {path.parent for path in root.glob("*/*/data.json")}
    return sorted(found)


def _data_roots(config: DictConfig, data_path: str) -> List[Path]:
    roots = config.get("data_roots")
    if roots:
        return [Path(str(root)) for root in roots]
    return [Path(data_path)]


def _concat_qpos(entry: dict, keys: Sequence[str], episode_path: Path) -> np.ndarray:
    parts = []
    for key in keys:
        if key not in entry:
            raise ValueError(
                f"{episode_path}: '{key}' not found in data.json "
                f"(available: {sorted(entry.keys())})"
            )
        qpos = entry[key].get("qpos", [])
        if len(qpos) == 0:
            raise ValueError(
                f"{episode_path}: '{key}.qpos' is empty — drop it from "
                "state_keys/action_keys or pick a different episode root."
            )
        parts.append(np.asarray(qpos, dtype=np.float32))
    return np.concatenate(parts, axis=0)


def _frame_tactile(frame: dict, episode_path: Path) -> np.ndarray:
    """Read one frame's tactile reading as ``(10, 4)`` raw channels."""
    tactile_info = frame["tactiles"]
    hands = []
    for side in ("left_ee", "right_ee"):
        value = tactile_info[side]
        if isinstance(value, str):
            hand = np.load(str(episode_path / value)).reshape(-1, TACTILE_CHANNELS)
        else:
            hand = np.asarray(value, dtype=np.float32).reshape(-1, TACTILE_CHANNELS)
        hands.append(hand.astype(np.float32, copy=False))
    return np.concatenate(hands, axis=0)


class BraincoACTDataset(data.Dataset):
    """BrainCo episodes served as ACT (observation, action-chunk) pairs.

    Args:
        config: window/None-free config with the keys documented below.
        data_path: fallback episode root when ``config.data_roots`` is unset.

    Config keys:
        chunk_size (int):        number of future actions predicted per sample.
        camera_names (list):     camera keys inside ``data.json``'s ``colors``.
        state_keys (list):       ``states`` sub-keys concatenated into ``qpos``.
        action_keys (list):      ``actions`` sub-keys concatenated into targets.
        image_height/width(int): resize target for RGB frames.
        frame_stride (int):      keep every n-th timestep as a sample start.
        use_tactile (bool):      load and return the tactile stream.
        max_episodes (int):      cap the number of episodes (quick smoke tests).
        data_roots (list[str]):  episode roots to aggregate.
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        augment: bool = False,
    ):
        super().__init__()
        self.chunk_size = int(config.get("chunk_size", 60))
        self.camera_names = [str(name) for name in config.get("camera_names", ["color_0"])]
        self.state_keys = [str(key) for key in config.get("state_keys", DEFAULT_QPOS_KEYS)]
        self.action_keys = [str(key) for key in config.get("action_keys", DEFAULT_QPOS_KEYS)]
        self.image_height = int(config.get("image_height", 240))
        self.image_width = int(config.get("image_width", 320))
        self.frame_stride = max(1, int(config.get("frame_stride", 1)))
        self.use_tactile = bool(config.get("use_tactile", True))
        max_episodes = config.get("max_episodes", None)

        # Train and eval views of the same episodes. train_act.py registers the
        # validation sample indices via ``set_eval_indices`` so that held-out
        # samples are never augmented — train/val are Subsets of one dataset
        # object, so a single shared transform would augment validation too.
        self.augment = bool(augment)
        self.augment_color_jitter = float(config.get("augment_color_jitter", 0.2))
        self.augment_translate = float(config.get("augment_translate", 0.0))
        self.qpos_noise_std = float(config.get("qpos_noise_std", 0.0))
        self.transform_train = self._build_transform(self.augment)
        self.transform_eval = self._build_transform(False)
        self.eval_indices: set = set()

        self.episodes: List[dict] = []
        self.episode_data: List[dict] = []
        self.samples: List[tuple] = []

        for root in _data_roots(config, data_path):
            if not root.exists():
                log.warning(f"Episode root not found, skipping: {root}")
                continue
            episode_paths = _episode_dirs(root)
            log.info(f"Loading BrainCo ACT episodes from {root}: {len(episode_paths)} found")
            for episode_path in episode_paths:
                if max_episodes is not None and len(self.episodes) >= int(max_episodes):
                    break
                try:
                    episode = self._load_episode(episode_path)
                except Exception as exc:  # noqa: BLE001 — one bad episode must not kill a run
                    log.warning(f"  Skipping {episode_path}: {exc}")
                    continue
                self._register_episode(episode)

        if not self.episodes:
            raise RuntimeError(
                "BraincoACTDataset found no usable episodes. Check data.data_roots "
                "and that each episode directory contains data.json."
            )

        self.state_dim = int(self.episodes[0]["states"].shape[1])
        self.action_dim = int(self.episodes[0]["actions"].shape[1])

        # Identity normalization until train_act.py installs the train-split stats.
        self.qpos_mean = torch.zeros(self.state_dim)
        self.qpos_std = torch.ones(self.state_dim)
        self.action_mean = torch.zeros(self.action_dim)
        self.action_std = torch.ones(self.action_dim)
        self.tactile_mean = torch.zeros(TACTILE_CHANNELS)
        self.tactile_std = torch.ones(TACTILE_CHANNELS)

        log.info(
            f"BraincoACTDataset: {len(self.episodes)} episodes, {len(self.samples)} samples, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"cameras={self.camera_names}, chunk_size={self.chunk_size}"
        )

    # ── construction helpers ────────────────────────────────────────────────

    def _build_transform(self, augment: bool):
        """Build the RGB pipeline. Augmentation strength comes from config.

        Measured on BrainCo grasp episodes: mild colour jitter helps, while a
        random shift hurts — it breaks the spatial alignment between the image
        and the robot pose that manipulation depends on. ``augment_translate``
        therefore defaults to 0. No horizontal flip either: it would break the
        left/right hand correspondence between image and action.
        """
        stages = [T.Resize((self.image_height, self.image_width))]
        if augment:
            if self.augment_color_jitter > 0:
                strength = self.augment_color_jitter
                stages.append(
                    T.ColorJitter(
                        brightness=strength, contrast=strength, saturation=strength
                    )
                )
            if self.augment_translate > 0:
                stages.append(
                    T.RandomAffine(
                        degrees=0,
                        translate=(self.augment_translate, self.augment_translate),
                    )
                )
        stages += [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        return T.Compose(stages)

    def set_eval_indices(self, indices: Sequence[int]):
        """Mark samples that must bypass augmentation (the validation split)."""
        self.eval_indices = set(int(index) for index in indices)

    def _load_episode(self, episode_path: Path) -> dict:
        import json

        with open(episode_path / "data.json", "r") as handle:
            raw = json.load(handle)

        frames = raw["data"]
        num_frames = len(frames)
        if num_frames < 2:
            raise ValueError(f"only {num_frames} frames")

        states = np.stack(
            [_concat_qpos(frame["states"], self.state_keys, episode_path) for frame in frames]
        ).astype(np.float32)
        actions = np.stack(
            [_concat_qpos(frame["actions"], self.action_keys, episode_path) for frame in frames]
        ).astype(np.float32)

        image_paths: Dict[str, List[str]] = {}
        available = sorted(frames[0].get("colors", {}).keys())
        for camera in self.camera_names:
            if camera not in frames[0].get("colors", {}):
                raise ValueError(f"camera '{camera}' missing; available: {available}")
            paths = [str(episode_path / frame["colors"][camera]) for frame in frames]
            if not Path(paths[0]).exists():
                raise ValueError(
                    f"camera '{camera}' listed in data.json but {paths[0]} is missing on disk"
                )
            image_paths[camera] = paths

        if self.use_tactile:
            tactile = np.stack([_frame_tactile(frame, episode_path) for frame in frames])
            # ch2 (tangential direction) uses 65535 as its invalid marker; the
            # shared converter treats negatives as invalid and zeroes them out.
            tactile[..., 2][tactile[..., 2] == 65535] = -1
            tactile, _ = force_direction_to_cartesian(tactile)
        else:
            tactile = np.zeros((num_frames, NUM_TACTILE_SENSORS, TACTILE_CHANNELS), np.float32)

        return {
            "path": str(episode_path),
            "num_frames": num_frames,
            "states": states,
            "actions": actions,
            "tactile": tactile.astype(np.float32),
            "image_paths": image_paths,
            "goal": str(raw.get("text", {}).get("goal", "")),
        }

    def _register_episode(self, episode: dict):
        episode_index = len(self.episodes)
        self.episodes.append(episode)

        start = len(self.samples)
        # Every start timestep is a valid sample; the tail is zero-padded and
        # masked out by ``is_pad``, exactly as in the original ACT dataloader.
        for frame_index in range(0, episode["num_frames"], self.frame_stride):
            self.samples.append((episode_index, frame_index))

        self.episode_data.append({
            "path": episode["path"],
            "num_frames": episode["num_frames"],
            "sample_indices": list(range(start, len(self.samples))),
            "goal": episode["goal"],
        })

    # ── normalization ───────────────────────────────────────────────────────

    def compute_norm_stats(self, episode_indices: Optional[Sequence[int]] = None) -> dict:
        """Per-dimension mean/std over the given episodes (all episodes if None)."""
        if episode_indices is None:
            episode_indices = range(len(self.episodes))
        episode_indices = list(episode_indices)

        states = np.concatenate([self.episodes[i]["states"] for i in episode_indices], axis=0)
        actions = np.concatenate([self.episodes[i]["actions"] for i in episode_indices], axis=0)

        stats = {
            "qpos_mean": torch.from_numpy(states.mean(axis=0)),
            "qpos_std": torch.from_numpy(states.std(axis=0)).clamp(min=1e-3),
            "action_mean": torch.from_numpy(actions.mean(axis=0)),
            "action_std": torch.from_numpy(actions.std(axis=0)).clamp(min=1e-3),
        }

        if self.use_tactile:
            tactile = np.concatenate(
                [self.episodes[i]["tactile"] for i in episode_indices], axis=0
            ).reshape(-1, TACTILE_CHANNELS)
            stats["tactile_mean"] = torch.from_numpy(tactile.mean(axis=0))
            stats["tactile_std"] = torch.from_numpy(tactile.std(axis=0)).clamp(min=1e-3)
        else:
            stats["tactile_mean"] = torch.zeros(TACTILE_CHANNELS)
            stats["tactile_std"] = torch.ones(TACTILE_CHANNELS)
        return stats

    def set_norm_stats(self, stats: dict):
        for key, value in stats.items():
            setattr(self, key, torch.as_tensor(value, dtype=torch.float32))

    # ── torch Dataset ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        episode_index, frame_index = self.samples[index]
        episode = self.episodes[episode_index]
        num_frames = episode["num_frames"]

        is_eval = index in self.eval_indices
        transform = self.transform_eval if is_eval else self.transform_train
        images = torch.stack([
            transform(Image.open(episode["image_paths"][camera][frame_index]).convert("RGB"))
            for camera in self.camera_names
        ])  # (num_cameras, 3, H, W)

        qpos = torch.from_numpy(episode["states"][frame_index].copy())
        qpos = (qpos - self.qpos_mean) / self.qpos_std
        if not is_eval and self.qpos_noise_std > 0.0:
            # qpos nearly uniquely identifies a frame within an episode, so it
            # is a memorization channel that image augmentation cannot close.
            qpos = qpos + torch.randn_like(qpos) * self.qpos_noise_std

        end = min(frame_index + self.chunk_size, num_frames)
        chunk = torch.from_numpy(episode["actions"][frame_index:end].copy())
        chunk = (chunk - self.action_mean) / self.action_std

        actions = torch.zeros(self.chunk_size, self.action_dim, dtype=torch.float32)
        is_pad = torch.ones(self.chunk_size, dtype=torch.bool)
        actions[: chunk.shape[0]] = chunk
        is_pad[: chunk.shape[0]] = False

        tactile = torch.from_numpy(episode["tactile"][frame_index].copy())
        tactile = (tactile - self.tactile_mean) / self.tactile_std

        return {
            "qpos": qpos,
            "images": images,
            "tactile": tactile,
            "actions": actions,
            "is_pad": is_pad,
            "episode_path": episode["path"],
            "frame_index": frame_index,
        }


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Inspect the BrainCo ACT dataset")
    parser.add_argument("--data_path", default="dataset/brainco/downstream/grasp_detection")
    parser.add_argument("--chunk_size", type=int, default=60)
    parser.add_argument("--max_episodes", type=int, default=2)
    args = parser.parse_args()

    config = OmegaConf.create({
        "chunk_size": args.chunk_size,
        "camera_names": ["color_0"],
        "max_episodes": args.max_episodes,
        "use_tactile": True,
    })
    dataset = BraincoACTDataset(config=config, data_path=args.data_path)
    dataset.set_norm_stats(dataset.compute_norm_stats())

    sample = dataset[0]
    print("=== sample shapes ===")
    for key, value in sample.items():
        print(f"  {key:15s}: {tuple(value.shape) if torch.is_tensor(value) else value}")
    print(f"  episodes: {len(dataset.episodes)}, samples: {len(dataset)}")
    last = dataset[len(dataset) - 1]
    print(f"  last sample padded steps: {int(last['is_pad'].sum())}/{dataset.chunk_size}")
