# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This script is used to generate lerobot (version 1.5) format datasets for some of the imitation learning experiments
import sys
from typing import Optional, Dict, Any
import random
import json
import torch
import shutil
import tqdm
import numpy as np
import argparse
import pickle
from pathlib import Path

import pytorch_kinematics as pk
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import einops
import PIL.Image
import gc
import datasets
from datasets import Dataset, Value, Sequence, Features, Array3D, Array2D, Image
from tactile_ssl.data.policy.utils import flatten_dict
from tactile_ssl.data.policy.video_utils import (
    VideoFrame,
    get_default_encoding,
    save_images_concurrently,
    encode_video_frames,
)
from safetensors.torch import save_file
from tactile_ssl.data.xela.utils import (
    read_xela_data,
    read_allegro_joint_data,
    joint_angles_to_poses,
)
from tactile_ssl.data.policy.policy_dataset import PolicyDataset
from tactile_ssl.data.policy.compute_stats import compute_stats

np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

XELA_FREQ = 100
RGB_FREQ = 10
FRANKA_FREQ = 10
ALLEGRO_FREQ = 10


def read_franka_joint_data(
    joint_data: np.ndarray,
    resampling_timestamps: np.ndarray,
    nominal_freq: int,
    smooth_data: bool = False,
):
    joint_timestamps = joint_data[:, 0]
    joint_angles = joint_data[:, 1:8]
    joint_effort = joint_data[:, 8:]

    joint_angles_interpolated = CubicSpline(joint_timestamps, joint_angles, axis=0)(resampling_timestamps)
    joint_effort_interpolated = CubicSpline(joint_timestamps, joint_effort, axis=0)(resampling_timestamps)

    if smooth_data:
        joint_angles_interpolated = savgol_filter(joint_angles_interpolated, nominal_freq // 3, 3, axis=0)
        joint_effort_interpolated = savgol_filter(joint_effort_interpolated, nominal_freq // 3, 3, axis=0)

    return joint_angles_interpolated, joint_effort_interpolated


def concatenate_episodes(ep_dicts):
    concatenated_data = {}
    for key in ep_dicts[0].keys():
        concatenated_data[key] = np.concatenate([ep_dict[key] for ep_dict in ep_dicts], axis=0)
    total_frames = concatenated_data["frame_index"].shape[0]
    concatenated_data["index"] = np.arange(total_frames)
    return concatenated_data


def load_episode_dict_from_raw_data(
    episode_folders: list[Path],
    output_folder: Path,
    xela_baseline_path: Path,
    xela_kinematic_chain: pk.Chain,
    franka_fk_chain: pk.SerialChain,
    action_space: str,
    clip_length: bool,
    fps: int,
    video: bool,
    encoding: Optional[dict] = None,
):
    if clip_length:
        print("Requested to clip length of episode")

    with open(xela_baseline_path / "data.pkl", "rb") as f:
        xela_baseline = pickle.load(f)
    xela_baseline = np.asarray(xela_baseline)
    xela_baseline = np.mean(xela_baseline[..., 1:], axis=0, keepdims=True)

    ep_dicts = []
    for episode_idx in tqdm.tqdm(range(len(episode_folders))):
        episode_folder = episode_folders[episode_idx]

        with open(episode_folder / "xela/data.pkl", "rb") as f:
            xela_observations = pickle.load(f)
        xela_observations = np.asarray(xela_observations)

        with open(episode_folder / "allegro/data.pkl", "rb") as f:
            allegro_joint_positions = pickle.load(f)
        allegro_joint_positions = np.asarray(allegro_joint_positions["joint_states"])

        with open(episode_folder / "franka/data.pkl", "rb") as f:
            franka_joint_positions = pickle.load(f)
        franka_joint_positions = np.asarray(franka_joint_positions["joint_states"])

        timestamps = []
        timestamps.append(xela_observations[:, 0, 0])
        timestamps.append(allegro_joint_positions[:, 0])
        timestamps.append(franka_joint_positions[:, 0])

        rgb_observation_keys = ["left/color", "right/color", "top/color", "wrist/color"]
        for rgb_observation_key in rgb_observation_keys:
            rgb_timestamp = np.loadtxt(episode_folder / rgb_observation_key / "timestamps.txt")
            timestamps.append(rgb_timestamp)

        start_timestamp = max([timestamp[0] for timestamp in timestamps])
        end_timestamp = min([timestamp[-1] for timestamp in timestamps])

        xela_interp_timestamps = np.linspace(
            start_timestamp,
            end_timestamp,
            int((end_timestamp - start_timestamp) * XELA_FREQ),
        )
        allegro_interp_timestamps = np.linspace(
            start_timestamp,
            end_timestamp,
            int((end_timestamp - start_timestamp) * XELA_FREQ),
        )
        franka_interp_timestamps = np.linspace(
            start_timestamp,
            end_timestamp,
            int((end_timestamp - start_timestamp) * FRANKA_FREQ),
        )
        rgb_interp_timestamps = np.linspace(
            start_timestamp,
            end_timestamp,
            int((end_timestamp - start_timestamp) * RGB_FREQ),
        )

        rgb_idxs = {}
        max_episode_length = np.inf
        for rgb_observation_key in rgb_observation_keys:
            rgb_timestamp = np.loadtxt(episode_folder / rgb_observation_key / "timestamps.txt")
            idx = np.searchsorted(rgb_timestamp, rgb_interp_timestamps)
            rgb_idxs[rgb_observation_key] = idx
            max_episode_length = min(max_episode_length, idx.shape[0] * (XELA_FREQ / RGB_FREQ))

        xela_observations = read_xela_data(xela_observations, xela_interp_timestamps, XELA_FREQ)
        max_episode_length = min(max_episode_length, xela_observations.shape[0])

        allegro_joint_positions, allegro_joint_efforts = read_allegro_joint_data(
            allegro_joint_positions, allegro_interp_timestamps, XELA_FREQ
        )
        allegro_sensor_positions = joint_angles_to_poses(xela_kinematic_chain, allegro_joint_positions)
        allegro_joint_positions = allegro_joint_positions[
            :: int(XELA_FREQ // ALLEGRO_FREQ)
        ]  # Subsample to ALLEGRO_FREQ
        max_episode_length = min(
            max_episode_length,
            allegro_joint_positions.shape[0] * (XELA_FREQ / ALLEGRO_FREQ),
        )

        franka_joint_positions, franka_joint_efforts = read_franka_joint_data(
            franka_joint_positions, franka_interp_timestamps, FRANKA_FREQ
        )
        max_episode_length = min(
            max_episode_length,
            franka_joint_positions.shape[0] * (XELA_FREQ / FRANKA_FREQ),
        )
        max_episode_length = int(max_episode_length)

        xela_observations = xela_observations[:max_episode_length]
        allegro_sensor_positions = allegro_sensor_positions[:max_episode_length]

        mask = xela_observations[..., 1:] != 0
        xela_observations = np.where(mask, xela_observations[..., 1:] - xela_baseline, xela_observations[..., 1:])

        xela_observations = np.concatenate([xela_observations, allegro_sensor_positions], axis=-1)

        xela_observations = einops.rearrange(xela_observations, "(b t) n c -> b t n c", t=int(XELA_FREQ / RGB_FREQ))

        max_episode_length = xela_observations.shape[0]
        allegro_joint_positions = allegro_joint_positions[: int(max_episode_length)]
        allegro_joint_efforts = allegro_joint_efforts[: int(max_episode_length)]
        franka_joint_positions = franka_joint_positions[: int(max_episode_length)]
        franka_joint_efforts = franka_joint_efforts[: int(max_episode_length)]

        start_idx = 0
        end_idx = max_episode_length
        if clip_length and "2024" in str(episode_folder):
            # Find the index of xela_observations where significant signal changes are observed
            thumb_signal = xela_observations[:, :, :30, :3]
            thumb_signal = np.mean(thumb_signal, axis=1)
            thumb_signal_diff = np.diff(thumb_signal, axis=0)
            idxs = np.argwhere(thumb_signal_diff > 40.0)[:, 0]
            start_idx = np.min(idxs)
            end_idx = np.max(idxs)
            print(f"start_idx: {start_idx}, end_idx: {end_idx}, length = {end_idx - start_idx}")

        allegro_joint_positions = allegro_joint_positions[start_idx:end_idx]
        allegro_joint_efforts = allegro_joint_efforts[start_idx:end_idx]
        franka_joint_positions = franka_joint_positions[start_idx:end_idx]
        franka_joint_efforts = franka_joint_efforts[start_idx:end_idx]
        xela_observations = xela_observations[start_idx:end_idx]

        rgb_data = {}
        for rgb_observation_key in rgb_observation_keys:
            rgb_idxs[rgb_observation_key] = rgb_idxs[rgb_observation_key][start_idx:end_idx]

            rgb_frames = []
            for idx in rgb_idxs[rgb_observation_key]:
                image = PIL.Image.open(episode_folder / rgb_observation_key / f"{idx:06d}.jpg")
                if rgb_observation_key == "wrist/color":
                    image = image.rotate(180)
                if (
                    rgb_observation_key == "left/color"
                    or rgb_observation_key == "right/color"
                    or rgb_observation_key == "top/color"
                ):
                    image = image.crop((160, 0, 480, 320))
                image = image.resize((256, 256))
                rgb_frames.append(np.asarray(image))
                image.close()

            rgb_frames_ = np.stack(rgb_frames)
            image_dir = output_folder / "images" / f"episode_{episode_idx:06d}" / rgb_observation_key
            save_images_concurrently(rgb_frames_, image_dir)

            videos_dir = output_folder / "videos"
            num_frames = rgb_frames_.shape[0]
            fname = f"{rgb_observation_key}_episode_{episode_idx:06d}.mp4"
            video_path = videos_dir / fname
            if not video_path.exists():
                # save png images in temporary directory
                tmp_imgs_dir = videos_dir / "tmp_images"
                save_images_concurrently(rgb_frames_, tmp_imgs_dir)

                # encode images to a mp4 video
                encode_video_frames(tmp_imgs_dir, video_path, fps, **(encoding or {}))

                # clean temporary images directory
                shutil.rmtree(tmp_imgs_dir)

            if video:
                # store the reference to the video frame
                rgb_data[rgb_observation_key] = [
                    {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                ]
            else:
                rgb_data[rgb_observation_key] = rgb_frames_

        if action_space == "absolute_joint":
            actions = np.zeros_like(franka_joint_positions[1:])
            actions = franka_joint_positions[1:]
        elif action_space == "relative_joint":
            actions = np.zeros_like(franka_joint_positions[1:])
            actions = np.diff(franka_joint_positions, axis=0)
        elif "ee_pose" in action_space:
            actions = np.zeros([franka_joint_positions.shape[0] - 1, 6])
            franka_joint_tensor = torch.tensor(franka_joint_positions)
            franka_ee_pose = franka_fk_chain.forward_kinematics(franka_joint_tensor, end_only=True)
            franka_ee_matrix = franka_ee_pose.get_matrix().numpy()
            if action_space == "absolute_ee_pose":
                franka_position = franka_ee_matrix[:, :3, 3]
                franka_rotation = R.from_matrix(franka_ee_matrix[:, :3, :3]).as_rotvec()
                actions = np.concatenate([franka_position[1:], franka_rotation[1:]], axis=-1)
            if action_space == "relative_ee_pose":
                relative_franka_ee_matrix = np.linalg.inv(franka_ee_matrix[:-1]) @ franka_ee_matrix[1:]
                relative_franka_position = relative_franka_ee_matrix[:, :3, 3]
                relative_franka_rotation = R.from_matrix(relative_franka_ee_matrix[:, :3, :3]).as_rotvec()
                actions = np.concatenate([relative_franka_position, relative_franka_rotation], axis=-1)

        max_episode_length = actions.shape[0]

        xela_observations = xela_observations[:max_episode_length]
        allegro_joint_positions = allegro_joint_positions[:max_episode_length]
        allegro_joint_efforts = allegro_joint_efforts[:max_episode_length]
        franka_joint_positions = franka_joint_positions[:max_episode_length]
        franka_joint_efforts = franka_joint_efforts[:max_episode_length]
        rgb_data = {key: rgb_data[key][:max_episode_length] for key in rgb_data}

        episode_data = {
            "observation/tactile": xela_observations,
            "allegro_joint_position": allegro_joint_positions,
            "allegro_joint_effort": allegro_joint_efforts,
            "action": actions,
            "franka_joint_position": franka_joint_positions,
            "observation/state": franka_joint_positions,
            "franka_joint_effort": franka_joint_efforts,
            "observation/image/top": rgb_data["top/color"],
            "observation/image/left": rgb_data["left/color"],
            "observation/image/right": rgb_data["right/color"],
            "observation/image/wrist": rgb_data["wrist/color"],
            "frame_index": np.arange(0, max_episode_length),
            "timestamp": np.arange(0, max_episode_length) / RGB_FREQ,
            "episode_index": [episode_idx] * max_episode_length,
        }
        ep_dicts.append(episode_data)

        gc.collect()

    data_dict = concatenate_episodes(ep_dicts)

    return data_dict


def calculate_episode_data_index(
    hf_dataset: datasets.Dataset,
) -> Dict[str, torch.Tensor]:
    """
    This is adapted from huggingface/lerobot. URL: https://github.com/huggingface/lerobot
    Calculate episode data index for the provided HuggingFace Dataset. Relies on episode_index column of hf_dataset.

    Parameters:
    - hf_dataset (datasets.Dataset): A HuggingFace dataset containing the episode index.

    Returns:
    - episode_data_index: A dictionary containing the data index for each episode. The dictionary has two keys:
        - "from": A tensor containing the starting index of each episode.
        - "to": A tensor containing the ending index of each episode.
    """
    episode_data_index = {"from": [], "to": []}

    current_episode = None
    """
    The episode_index is a list of integers, each representing the episode index of the corresponding example.
    For instance, the following is a valid episode_index:
      [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    Below, we iterate through the episode_index and populate the episode_data_index dictionary with the starting and
    ending index of each episode. For the episode_index above, the episode_data_index dictionary will look like this:
        {
            "from": [0, 3, 7],
            "to": [3, 7, 12]
        }
    """
    if len(hf_dataset) == 0:
        episode_data_index = {
            "from": torch.tensor([]),
            "to": torch.tensor([]),
        }
        return episode_data_index
    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            # We encountered a new episode, so we append its starting location to the "from" list
            episode_data_index["from"].append(idx)
            # If this is not the first episode, we append the ending location of the previous episode to the "to" list
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            # Let's keep track of the current episode index
            current_episode = episode_idx
        else:
            # We are still in the same episode, so there is nothing for us to do here
            pass
    # We have reached the end of the dataset, so we append the ending location of the last episode to the "to" list
    episode_data_index["to"].append(idx + 1)

    for k in ["from", "to"]:
        episode_data_index[k] = torch.tensor(episode_data_index[k])

    return episode_data_index


def to_hf_dataset(data_dict, video):
    features = {}
    keys = [key for key in data_dict if "observation/image/" in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation/tactile"] = Array3D(shape=(10, 368, 6), dtype="float32", id=None)

    for key in [
        "allegro_joint_position",
        "allegro_joint_effort",
        "franka_joint_position",
        "franka_joint_effort",
        "action",
        "observation/state",
    ]:
        features[key] = Sequence(length=data_dict[key].shape[1], feature=Value(dtype="float32", id=None))

    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_format("torch")
    return hf_dataset


def save_metadata(
    info: Dict[str, Any],
    stats: Dict,
    episode_data_index: Dict[str, list],
    meta_data_dir: Path,
):
    meta_data_dir.mkdir(parents=True, exist_ok=True)

    # save info
    info_path = meta_data_dir / "info.json"
    with open(str(info_path), "w") as f:
        json.dump(info, f, indent=4)

    # save stats
    stats_path = meta_data_dir / "stats.safetensors"
    save_file(flatten_dict(stats, sep="|"), stats_path)

    # save episode_data_index
    episode_data_index = {key: episode_data_index[key].clone() for key in episode_data_index}
    ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
    save_file(episode_data_index, ep_data_idx_path)


def main(args):
    np.random.seed(42)
    random.seed(42)
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    video = args.video
    xela_baseline_path = Path(args.xela_baseline_path)
    xela_urdf_path = Path(args.xela_urdf_path)
    franka_urdf_path = Path(args.franka_urdf_path)
    xela_kinematic_chain = pk.build_chain_from_urdf(open(xela_urdf_path, "r").read())
    franka_fk_chain = pk.build_serial_chain_from_urdf(
        open(franka_urdf_path).read(),
        end_link_name="meta_hand_base_frame",
        root_link_name="base_link",
    )
    clip_length = args.clip_length
    episodes_subset = None if args.episodes is None else list(args.episodes)

    episode_folders = list(input_folder.glob("*"))
    # episode_folders2 = list(input_folder2.glob("*"))
    # episode_folders = episode_folders1 + episode_folders2
    episode_ids = np.arange(len(episode_folders))

    if episodes_subset is not None:
        assert all(episodes_subset) < len(episode_ids), f"Invalid episode indices: {episodes_subset}"
        episode_ids = episode_ids[episodes_subset]

    total_num_episodes = len(episode_ids)
    print(f"Processing total {total_num_episodes} episodes")

    permuted_episode_ids = np.random.permutation(episode_ids)
    training_episode_ids = permuted_episode_ids[: int(0.96 * total_num_episodes)]
    validation_episode_ids = permuted_episode_ids[int(0.96 * total_num_episodes) :]

    print(f"Training episodes: {training_episode_ids}")
    print(f"Validation episodes: {validation_episode_ids}")

    training_episode_folders = [episode_folders[i] for i in training_episode_ids]
    validation_episode_folders = [episode_folders[i] for i in validation_episode_ids]

    info = {
        "fps": 10,
        "video": video,
    }
    if video:
        info["video_encoding"] = get_default_encoding()

    for stage, folders in zip(["val", "train"], [validation_episode_folders, training_episode_folders]):
        data_dict = load_episode_dict_from_raw_data(
            folders,
            output_folder / stage,
            xela_baseline_path,
            xela_kinematic_chain,
            franka_fk_chain,
            action_space=args.action_space,
            clip_length=clip_length,
            fps=10,
            video=video,
            encoding=None,
        )
        hf_dataset = to_hf_dataset(data_dict, video=video)

        episode_data_index = calculate_episode_data_index(hf_dataset)

        videos_dir = None
        if video:
            videos_dir = output_folder / stage / "videos"
        policy_dataset = PolicyDataset.from_preloaded(
            hf_dataset=hf_dataset,
            episode_data_index=episode_data_index,
            info=info,
            videos_dir=videos_dir,
        )
        stats = compute_stats(policy_dataset)
        hf_dataset.set_transform(None)
        hf_dataset.set_format(None)
        hf_dataset.save_to_disk(str(output_folder / stage))

        save_metadata(info, stats, episode_data_index, output_folder / stage / "metadata")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, required=True)
    # parser.add_argument("--input-folder2", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--episodes", nargs="+", type=int, default=None)
    parser.add_argument("--xela-baseline-path", type=str, required=True)
    parser.add_argument("--clip-length", action="store_true", default=False)
    parser.add_argument(
        "--xela-urdf-path",
        type=str,
        default="/home/akashsharma/workspace/datasets/xela/pretraining/extracted/urdf/ahrcpcpn.urdf",
    )
    parser.add_argument(
        "--franka-urdf-path",
        type=str,
        default="/home/akashsharma/workspace/projects/gum_ws/src/GUM/gum/devices/metahand/ros/meta_hand_description/urdf/meta_hand_franka.urdf",
    )
    parser.add_argument(
        "--action-space",
        type=str,
        choices=[
            "absolute_joint",
            "relative_joint",
            "relative_ee_pose",
            "absolute_ee_pose",
        ],
        default="relative_ee_pose",
    )
    args = parser.parse_args()
    main(args)
