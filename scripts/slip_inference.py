#!/usr/bin/env python
"""Real-time slip detection from a deployed BrainCo checkpoint.

Wraps the exact preprocessing BraincoXYZSlipDetectionDataset applies, so the
robot sees the same inputs the model was trained on. Every transform here is
imported from the training code rather than reimplemented -- the failure mode
this guards against is silent: a mismatched scale or axis convention produces
plausible-looking probabilities that are simply wrong.

Pipeline per frame (10 sensors = left thumb..pinky, then right thumb..pinky):

    raw tactile (10, 4) = [Fn, Ft, theta_deg, proximity]
      -> ch2 == 65535 becomes -1                (sensor's invalid marker)
      -> force_direction_to_cartesian           [Fn, Ft*cos, Ft*sin, prox] + valid
      -> zero the invalid entries
      -> divide by max_values [1000, 1000, 1000, 100000]
      -> zero the invalid entries again

    hand joints (6 per hand) -> hand-URDF FK -> *_touch_link positions,
    base_link-relative, left y negated -> _align_xyz_to_npz -> (10, 3)

The encoder then applies its own (x - signal_mean) / signal_std using buffers
that were fitted on the training split and travel inside the checkpoint.

Usage:
    from scripts.slip_inference import SlipDetector

    det = SlipDetector(
        checkpoint="experiments/brainco_xyz_slip_detection/<run>/checkpoints/epoch-0050.pth",
        config="experiments/brainco_xyz_slip_detection/<run>/config.yaml",
        device="cuda",
    )
    for frame in stream:                      # at the recording frame rate
        out = det.step(frame.tactile, frame.left_ee_qpos, frame.right_ee_qpos)
        if out is not None and out["slip"]:
            ...                               # out["slip_prob"] in [0, 1]

Self-test against a recorded episode (agreement with the training dataloader):
    python scripts/slip_inference.py --checkpoint ... --config ... \
        --episode dataset/brainco/downstream/slip_data_v2/slip_box/episode_0001
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

import hydra
from omegaconf import OmegaConf, open_dict

from tactile_ssl.data.brainco_tactile import (
    FINGERTIP_LINKS,
    build_hand_chain,
    compute_hand_fk,
)
from tactile_ssl.data.brainco_xyz_grasp_dataset import _align_xyz_to_npz
from tactile_ssl.data.force_channels import force_direction_to_cartesian

# Must stay identical to BraincoXYZSlipDetectionDataset.
MAX_VALUES = np.array([1000.0, 1000.0, 1000.0, 100000.0], dtype=np.float32)
INVALID_DIRECTION = 65535


def preprocess_tactile(raw: np.ndarray) -> np.ndarray:
    """Raw (..., 10, 4) sensor readings -> normalized model input.

    Mirrors BraincoSSLDataset's channel conversion followed by
    BraincoXYZSlipDetectionDataset's fixed-max scaling.
    """
    values = np.asarray(raw, dtype=np.float32).copy()
    if values.shape[-2:] != (10, 4):
        raise ValueError(f"Expected (..., 10, 4) tactile, got {values.shape}")
    values[..., 2][values[..., 2] == INVALID_DIRECTION] = -1
    converted, valid = force_direction_to_cartesian(values)
    converted = np.where(valid, converted, 0.0)
    converted = converted / MAX_VALUES
    return np.where(valid, converted, 0.0).astype(np.float32)


class FingertipFK:
    """Hand-only FK: 6 joint targets per hand -> 10 fingertip positions."""

    def __init__(self, urdf_path: str = "dataset/brainco/urdf"):
        root = Path(urdf_path)
        self.chains = {side: build_hand_chain(root, side) for side in ("left", "right")}

    def __call__(self, left_ee_qpos: Sequence[float],
                 right_ee_qpos: Sequence[float]) -> np.ndarray:
        """Returns (10, 3) in the NPZ convention the model was trained on."""
        tips = np.zeros((10, 3), dtype=np.float32)
        fk = {side: compute_hand_fk(self.chains[side], q, side)
              for side, q in (("left", left_ee_qpos), ("right", right_ee_qpos))}
        for index, link in enumerate(FINGERTIP_LINKS):
            left = fk["left"][f"left_{link}"].get_matrix()[:, :3, 3].squeeze(0).numpy().copy()
            left[1] *= -1.0   # sign convention from _compute_fk
            tips[index] = left
            tips[index + 5] = (
                fk["right"][f"right_{link}"].get_matrix()[:, :3, 3].squeeze(0).numpy()
            )
        return _align_xyz_to_npz(torch.from_numpy(tips)).numpy()


class SlipDetector:
    """Sliding-window slip classifier over a live BrainCo stream."""

    def __init__(
        self,
        checkpoint: str,
        config: str,
        device: str = "cuda",
        urdf_path: str = "dataset/brainco/urdf",
        window_frames: Optional[int] = None,
    ):
        cfg = OmegaConf.load(config)
        task_cfg = cfg.task
        with open_dict(task_cfg):
            # Both checkpoints off: the deployed weights supersede the SSL ones,
            # and task.checkpoint_task cannot be used here -- SLModule calls
            # load_task from its own __init__, before the subclass has assigned
            # self.classifier, so its strict=True load rejects the classifier.*
            # keys. Loading after construction is strict against the real model.
            task_cfg.checkpoint_encoder = None
            task_cfg.checkpoint_task = None
            task_cfg.train_encoder = False
        self.model = hydra.utils.instantiate(task_cfg)

        state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]   # fabric-saved last.ckpt
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(device)
        self.device = device

        if window_frames is None:
            window_frames = int(cfg.data.dataset.config.input_window_frames)
        self.window_frames = window_frames
        encoder_sequence = int(getattr(self.model.model_encoder, "sequence_length", 1))
        if encoder_sequence > 1 and window_frames % encoder_sequence:
            raise ValueError(
                f"window_frames={window_frames} is not divisible by the encoder "
                f"sequence_length={encoder_sequence}"
            )

        self.fk = FingertipFK(urdf_path)
        self.tactile: deque = deque(maxlen=window_frames)
        self.poses: deque = deque(maxlen=window_frames)

    def reset(self) -> None:
        self.tactile.clear()
        self.poses.clear()

    @property
    def ready(self) -> bool:
        return len(self.tactile) == self.window_frames

    def push(self, tactile_raw: np.ndarray, left_ee_qpos: Sequence[float],
             right_ee_qpos: Sequence[float]) -> None:
        """Append one frame. tactile_raw is (10, 4) left-then-right, raw units."""
        self.tactile.append(preprocess_tactile(tactile_raw))
        self.poses.append(self.fk(left_ee_qpos, right_ee_qpos))

    @torch.no_grad()
    def predict(self) -> Optional[dict]:
        """Classify the current window, or None until it is full."""
        if not self.ready:
            return None
        batch = {
            "joint_contact": torch.from_numpy(np.stack(self.tactile))[None].to(self.device),
            "finger_xyz": torch.from_numpy(np.stack(self.poses))[None].to(self.device),
        }
        logits = self.model(batch)
        probs = logits.softmax(dim=-1)[0]
        return {
            "slip": bool(probs[1] > probs[0]),
            "slip_prob": float(probs[1]),
            "logits": logits[0].detach().cpu().numpy(),
        }

    def step(self, tactile_raw: np.ndarray, left_ee_qpos: Sequence[float],
             right_ee_qpos: Sequence[float]) -> Optional[dict]:
        self.push(tactile_raw, left_ee_qpos, right_ee_qpos)
        return self.predict()


def _self_test(args) -> None:
    """Replay one recorded episode and compare against the training dataloader.

    Agreement here is the only cheap proof that the live preprocessing path
    reproduces what the model was trained on.
    """
    import json

    from tactile_ssl.data.brainco_xyz_slip_detection_dataset import (
        BraincoXYZSlipDetectionDataset,
    )

    cfg = OmegaConf.load(args.config)
    detector = SlipDetector(args.checkpoint, args.config, device=args.device,
                            urdf_path=args.urdf)

    # data_path is the dataset root and classes are object folders, so build the
    # one object folder and keep only this episode's windows.
    episode = Path(args.episode)
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset.config, resolve=True)
    reference = BraincoXYZSlipDetectionDataset(
        config=OmegaConf.create({**dataset_cfg, "classes": [episode.parent.name]}),
        data_path=str(episode.parent.parent),
        brainco_urdf_path=args.urdf,
    )
    windows = [w for w in reference.windows if Path(w["episode_path"]) == episode]
    if not windows:
        raise SystemExit(f"No windows found for {episode}")

    frames = json.loads((episode / "data.json").read_text())["data"]
    stride = int(dataset_cfg["input_window_stride"])
    max_tactile_error = 0.0
    max_pose_error = 0.0
    matched = 0

    for window_index, window in enumerate(windows):
        start = window["window_start_frame"]
        detector.reset()
        for offset in range(detector.window_frames):
            frame = frames[min(start + offset, len(frames) - 1)]
            # data.json stores tactile either inline or as an .npy path.
            tactile = np.concatenate(
                [(np.load(str(episode / entry)) if isinstance(entry, str)
                  else np.array(entry)).reshape(-1, 4)
                 for entry in (frame["tactiles"]["left_ee"],
                               frame["tactiles"]["right_ee"])],
                axis=0,
            )
            detector.push(tactile,
                          frame["states"]["left_ee"]["qpos"],
                          frame["states"]["right_ee"]["qpos"])
        live_tactile = np.stack(detector.tactile)
        live_poses = np.stack(detector.poses)
        max_tactile_error = max(
            max_tactile_error,
            float(np.abs(live_tactile - window["joint_contact"].numpy()).max()),
        )
        max_pose_error = max(
            max_pose_error,
            float(np.abs(live_poses - window["finger_xyz"].numpy()).max()),
        )
        out = detector.predict()
        matched += int(out["slip"]) == int(window["label"])
        if window_index < 5:
            print(f"  window @frame {start:4d}  p(slip)={out['slip_prob']:.3f}  "
                  f"label={int(window['label'])}")

    print(f"\n{len(windows)} windows (stride {stride})")
    print(f"  max |tactile diff| vs dataloader : {max_tactile_error:.3e}")
    print(f"  max |pose diff|    vs dataloader : {max_pose_error:.3e}")
    print(f"  accuracy on this episode         : {matched}/{len(windows)}")
    if max(max_tactile_error, max_pose_error) > 1e-5:
        raise SystemExit("Live preprocessing does not match the training dataloader")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="deployed .pth or .ckpt")
    parser.add_argument("--config", required=True, help="config.yaml from the run dir")
    parser.add_argument("--episode", help="episode dir to replay as a self-test")
    parser.add_argument("--urdf", default="dataset/brainco/urdf")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.episode:
        raise SystemExit("--episode is required for the self-test entry point")
    _self_test(args)
