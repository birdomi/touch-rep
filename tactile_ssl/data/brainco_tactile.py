from typing import Optional, List
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import pytorch_kinematics as pk
from omegaconf import DictConfig

from tactile_ssl.utils.logging import get_pylogger

torch.set_printoptions(precision=4, sci_mode=False)

log = get_pylogger(__name__)

FINGERTIP_LINKS = [
    "thumb_touch_link",
    "index_touch_link",
    "middle_touch_link",
    "ring_touch_link",
    "pinky_touch_link",
]

SKELETON_LINKS = [
    "base_link", 
    "thumb_metacarpal_link", "thumb_proximal_link", "thumb_distal_link", "thumb_tip_link", "thumb_touch_link",
    "index_proximal_link", "index_distal_link", "index_tip_link", "index_touch_link",
    "middle_proximal_link", "middle_distal_link", "middle_tip_link", "middle_touch_link",
    "ring_proximal_link", "ring_distal_link", "ring_tip_link", "ring_touch_link",
    "pinky_proximal_link", "pinky_distal_link", "pinky_tip_link", "pinky_touch_link"
]
SKELETON_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),     # thumb
    (0, 6), (6, 7), (7, 8), (8, 9),             # index
    (0, 10), (10, 11), (11, 12), (12, 13),      # middle
    (0, 14), (14, 15), (15, 16), (16, 17),      # ring
    (0, 18), (18, 19), (19, 20), (20, 21),      # pinky
]


# All G1 revolute joints (29 DOF) — need to provide values for all of them
G1_ALL_JOINTS = [
    # Legs
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Per-side arm joint suffixes (7 DOF per arm)
G1_ARM_JOINTS = [
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
]

def build_combined_chain(urdf_path: Path):
    """Dynamically merge G1 URDF and Hand URDFs and build a single kinematic chain."""
    import xml.etree.ElementTree as ET
    g1_path = urdf_path / "g1.urdf"
    left_path = urdf_path / "revo2_left_hand.urdf"
    right_path = urdf_path / "revo2_right_hand.urdf"
    
    g1_tree = ET.parse(str(g1_path))
    g1_root = g1_tree.getroot()
    
    left_hand_tree = ET.parse(str(left_path))
    right_hand_tree = ET.parse(str(right_path))
    
    for elem in left_hand_tree.getroot():
        if elem.tag in ['link', 'joint', 'material']:
            g1_root.append(elem)
            
    joint_l = ET.SubElement(g1_root, 'joint', name='left_hand_fixed_joint', type='fixed')
    ET.SubElement(joint_l, 'parent', link='left_rubber_hand')
    ET.SubElement(joint_l, 'child', link='left_base_link')
    ET.SubElement(joint_l, 'origin', xyz='0 0 0', rpy='-1.5708 0 -1.5708')
    
    for elem in right_hand_tree.getroot():
        if elem.tag in ['link', 'joint', 'material']:
            g1_root.append(elem)
            
    joint_r = ET.SubElement(g1_root, 'joint', name='right_hand_fixed_joint', type='fixed')
    ET.SubElement(joint_r, 'parent', link='right_rubber_hand')
    ET.SubElement(joint_r, 'child', link='right_base_link')
    ET.SubElement(joint_r, 'origin', xyz='0 0 0', rpy='1.5708 0 1.5708')
    
    combined_xml = ET.tostring(g1_root, encoding='utf8')
    return pk.build_chain_from_urdf(combined_xml)

def compute_combined_fk(chain, left_arm, right_arm, left_ee, right_ee):
    """Compute forward kinematics for the combined G1 + Hands model."""
    joint_map = {name: 0.0 for name in G1_ALL_JOINTS}
    
    # Map G1 left arm joints
    l_arm = [float(v) for v in left_arm]
    for i, suffix in enumerate(G1_ARM_JOINTS):
        joint_map[f"left_{suffix}"] = l_arm[i]
        
    # Map G1 right arm joints    
    r_arm = [float(v) for v in right_arm]
    for i, suffix in enumerate(G1_ARM_JOINTS):
        joint_map[f"right_{suffix}"] = r_arm[i]
        
    # Map left hand joints
    l_ee = [float(v) for v in left_ee]
    joint_map["left_thumb_metacarpal_joint"] = l_ee[0]
    joint_map["left_thumb_proximal_joint"] = l_ee[1]
    joint_map["left_index_proximal_joint"] = l_ee[2]
    joint_map["left_middle_proximal_joint"] = l_ee[3]
    joint_map["left_ring_proximal_joint"] = l_ee[4]
    joint_map["left_pinky_proximal_joint"] = l_ee[5]
    joint_map["left_thumb_distal_joint"] = l_ee[1] * 1.0
    joint_map["left_index_distal_joint"] = l_ee[2] * 1.155
    joint_map["left_middle_distal_joint"] = l_ee[3] * 1.155
    joint_map["left_ring_distal_joint"] = l_ee[4] * 1.155
    joint_map["left_pinky_distal_joint"] = l_ee[5] * 1.155

    # Map right hand joints
    r_ee = [float(v) for v in right_ee]
    joint_map["right_thumb_metacarpal_joint"] = r_ee[0]
    joint_map["right_thumb_proximal_joint"] = r_ee[1]
    joint_map["right_index_proximal_joint"] = r_ee[2]
    joint_map["right_middle_proximal_joint"] = r_ee[3]
    joint_map["right_ring_proximal_joint"] = r_ee[4]
    joint_map["right_pinky_proximal_joint"] = r_ee[5]
    joint_map["right_thumb_distal_joint"] = r_ee[1] * 1.0
    joint_map["right_index_distal_joint"] = r_ee[2] * 1.155
    joint_map["right_middle_distal_joint"] = r_ee[3] * 1.155
    joint_map["right_ring_distal_joint"] = r_ee[4] * 1.155
    joint_map["right_pinky_distal_joint"] = r_ee[5] * 1.155

    return chain.forward_kinematics(joint_map)


class BraincoSSLDataset(data.Dataset):
    """Dataset for BrainCo pretraining tactile data.

    Reads tactile sensor data (npy files) and joint positions from data.json
    stored in dataset/brainco/pretraining/{object}/{episode}/.

    Per frame, data.json contains:
        - states.left_arm.qpos (7 joints)
        - states.right_arm.qpos (7 joints)
        - states.left_ee.qpos (6 joints)
        - states.right_ee.qpos (6 joints)
        - tactiles.left_ee / tactiles.right_ee (paths to npy files)
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        brainco_urdf_path: str = "dataset/brainco/urdf",
        object_class: Optional[int] = None,
        load_images: bool = False,
    ):
        # ── Config defaults ──────────────────────────────────────────────
        if config.get("window_overlap") is None:
            config.window_overlap = 0.0
        if config.get("bias_noise_std") is None:
            config.bias_noise_std = 0.0
        if config.get("bias_range") is None: 
            config.bias_range = 0.0

        self.window_time = config.window_time
        assert 0 <= config.window_overlap < 1, "Window overlap should be between 0 and 1"
        self.window_overlap = config.window_overlap
        self.interpolating_freq = config.interpolating_freq
        self.num_frames_per_window = int(round(self.window_time * self.interpolating_freq))
        self.shift_per_window = int(round(self.num_frames_per_window * (1.0 - self.window_overlap)))
        self.shift_per_window = max(1, self.shift_per_window)

        self.bias_noise_std = config.bias_noise_std
        self.bias_range = config.bias_range
        self.augment = not (self.bias_noise_std == 0.0 and self.bias_range == 0.0)
        self.object_label = object_class

        # ── Load data.json ───────────────────────────────────────────────
        self.data_path = Path(data_path)
        data_json_path = self.data_path / "data.json"
        assert data_json_path.exists(), f"data.json not found at {data_json_path}"

        log.info(f"Loading data from {data_json_path}")
        with open(data_json_path, "r") as f:
            raw_data = json.load(f)

        frames = raw_data["data"]
        self.num_frames = len(frames)
        log.info(f"  Found {self.num_frames} frames")

        # ── Extract joint positions (qpos) per frame ─────────────────────
        # Concatenate: left_arm(7) + right_arm(7) + left_ee(6) + right_ee(6) = 26
        joint_positions_list = []
        left_arm_list, right_arm_list = [], []
        left_ee_list, right_ee_list = [], []
        for frame in frames:
            states = frame["states"]
            qpos = (
                states["left_arm"]["qpos"]
                + states["right_arm"]["qpos"]
                + states["left_ee"]["qpos"]
                + states["right_ee"]["qpos"]
            )
            joint_positions_list.append(qpos)
            left_arm_list.append(states["left_arm"]["qpos"])
            right_arm_list.append(states["right_arm"]["qpos"])
            left_ee_list.append(states["left_ee"]["qpos"])
            right_ee_list.append(states["right_ee"]["qpos"])
        self.joint_positions = np.array(joint_positions_list, dtype=np.float32)  # (N, 26)
        left_arm_array = np.array(left_arm_list, dtype=np.float32)   # (N, 7)
        right_arm_array = np.array(right_arm_list, dtype=np.float32)  # (N, 7)
        left_ee_array = np.array(left_ee_list, dtype=np.float32)     # (N, 6)
        right_ee_array = np.array(right_ee_list, dtype=np.float32)    # (N, 6)
        log.info(f"  Joint positions shape: {self.joint_positions.shape}")

        # ── Compute positions via unified FK ──────────────────────────────────
        urdf_path = Path(brainco_urdf_path)
        combined_chain = build_combined_chain(urdf_path)

        fingertip_positions_list = []
        skeleton_positions_list = []
        wrist_positions_list = []
        for i in range(self.num_frames):
            fk_res = compute_combined_fk(
                combined_chain,
                left_arm_array[i],
                right_arm_array[i],
                left_ee_array[i],
                right_ee_array[i],
            )

            # Wrists
            left_wrist = fk_res["left_rubber_hand"].get_matrix()[:, :3, 3].squeeze(0).numpy()
            right_wrist = fk_res["right_rubber_hand"].get_matrix()[:, :3, 3].squeeze(0).numpy()
            wrist_positions_list.append(np.stack([left_wrist, right_wrist], axis=0))  # (2, 3)

            # Fingertips (Left 0-4, Right 5-9)
            frame_positions = []
            for link_suffix in FINGERTIP_LINKS:
                frame_positions.append(fk_res[f"left_{link_suffix}"].get_matrix()[:, :3, 3].squeeze(0).numpy())
            for link_suffix in FINGERTIP_LINKS:
                frame_positions.append(fk_res[f"right_{link_suffix}"].get_matrix()[:, :3, 3].squeeze(0).numpy())
            fingertip_positions_list.append(np.stack(frame_positions, axis=0))  # (10, 3)

            # Skeleton (Left 0-21, Right 22-43)
            skel_positions = []
            for link_suffix in SKELETON_LINKS:
                skel_positions.append(fk_res[f"left_{link_suffix}"].get_matrix()[:, :3, 3].squeeze(0).numpy())
            for link_suffix in SKELETON_LINKS:
                skel_positions.append(fk_res[f"right_{link_suffix}"].get_matrix()[:, :3, 3].squeeze(0).numpy())
            skeleton_positions_list.append(np.stack(skel_positions, axis=0))  # (44, 3)

        # Arrays in unified global frame natively!
        fp_world = np.array(fingertip_positions_list, dtype=np.float32)      # (N, 10, 3)
        self.skeleton_positions_world = np.array(skeleton_positions_list, dtype=np.float32)  # (N, 44, 3)
        self.wrist_positions = np.array(wrist_positions_list, dtype=np.float32)  # (N, 2, 3)

        log.info(f"  Wrist positions shape: {self.wrist_positions.shape}")

        # ── Compute 6D fingertip positions (relative to own + opposite wrist) ──
        # Layout: 0-4 = left hand (thumb~pinky), 5-9 = right hand (thumb~pinky)
        left_idxs = [0, 1, 2, 3, 4]
        right_idxs = [5, 6, 7, 8, 9]

        # 6D = [relative_to_own_wrist (3), relative_to_opposite_wrist (3)]
        fingertip_6d = np.zeros((self.num_frames, 10, 6), dtype=np.float32)
        for idx in left_idxs:
            fingertip_6d[:, idx, :3] = fp_world[:, idx, :] - self.wrist_positions[:, 0, :]  # own (left)
            fingertip_6d[:, idx, 3:] = fp_world[:, idx, :] - self.wrist_positions[:, 1, :]  # opposite (right)
        for idx in right_idxs:
            fingertip_6d[:, idx, :3] = fp_world[:, idx, :] - self.wrist_positions[:, 1, :]  # own (right)
            fingertip_6d[:, idx, 3:] = fp_world[:, idx, :] - self.wrist_positions[:, 0, :]  # opposite (left)

        self.fingertip_positions = fingertip_6d  # (N, 10, 6)
        self.fingertip_positions_world = fp_world  # (N, 10, 3) — for visualization

        log.info(f"  Fingertip positions (6D) shape: {self.fingertip_positions.shape}")

        # ── Load tactile data per frame ──────────────────────────────────
        # Each npy file is small (~288 bytes), so load all eagerly
        tactile_list = []
        for frame in frames:
            tactile_info = frame["tactiles"]
            
            if isinstance(tactile_info["left_ee"], str):
                left_path = self.data_path / tactile_info["left_ee"]
                left_tactile = np.load(str(left_path)).reshape(-1, 4)   # (num_sensors_left, 4)
            else:
                left_tactile = np.array(tactile_info["left_ee"]).reshape(-1, 4)
                
            if isinstance(tactile_info["right_ee"], str):
                right_path = self.data_path / tactile_info["right_ee"]
                right_tactile = np.load(str(right_path)).reshape(-1, 4)  # (num_sensors_right, 4)
            else:
                right_tactile = np.array(tactile_info["right_ee"]).reshape(-1, 4)
                
            # Concatenate left and right tactile data along sensor axis
            tactile = np.concatenate([left_tactile, right_tactile], axis=0)  # (num_sensors, 4)
            tactile_list.append(tactile)
        
        self.tactile_array = np.array(tactile_list, dtype=np.float32)  # (N, num_sensors, 4)
        
        # Replace 65535 with -1 in the 3rd sensor value channel (index 2) # 65535 refers to the invalid value
        invalid_mask = self.tactile_array[..., 2] == 65535
        self.tactile_array[..., 2][invalid_mask] = -1
        
        log.info(f"  Tactile array shape: {self.tactile_array.shape}")

        # ── Compute window indices ───────────────────────────────────────
        max_length = self.num_frames - (self.num_frames % self.num_frames_per_window)
        max_length = max_length - self.num_frames_per_window
        self.data_idxs = np.arange(0, max(1, max_length), self.shift_per_window)
        log.info(f"  Number of windows: {len(self.data_idxs)}, frames_per_window: {self.num_frames_per_window}")

    def __len__(self):
        return len(self.data_idxs)

    def read_joint_sample(self, index):
        """Read a window of joint positions."""
        return self.joint_positions[index : index + self.num_frames_per_window]

    def read_fingertip_sample(self, index):
        """Read a window of fingertip positions."""
        return self.fingertip_positions[index : index + self.num_frames_per_window]

    def read_wrist_sample(self, index):
        """Read a window of wrist positions."""
        return self.wrist_positions[index : index + self.num_frames_per_window]

    def update_normalization(self, mean, std):
        self.tactile_mean = mean
        self.tactile_std = std

    def __getitem__(self, idx):
        sample_dict = {}
        index = self.data_idxs[idx]

        # Tactile sensor data for this window
        sensor_data = self.tactile_array[index : index + self.num_frames_per_window]
        # print(sensor_data.shape)

        # Joint positions for this window
        joint_positions = self.read_joint_sample(index)

        # Fingertip positions from FK
        fingertip_positions = self.read_fingertip_sample(index)

        # Wrist positions from FK
        wrist_positions = self.read_wrist_sample(index)

        sensor_data = torch.from_numpy(sensor_data).float()
        joint_positions = torch.from_numpy(joint_positions).float()
        fingertip_positions = torch.from_numpy(fingertip_positions).float()
        wrist_positions = torch.from_numpy(wrist_positions).float()

        # sample_dict["sensor"] = sensor_data
        # sample_dict["joint_angles"] = joint_positions
        # sample_dict["sensor_poses"] = fingertip_positions
        # sample_dict["wrist_positions"] = wrist_positions
        sample_dict.update({"sensor": sensor_data})
        sample_dict.update({"joint_angles": joint_positions})
        sample_dict.update({"sensor_poses": fingertip_positions})
        sample_dict.update({"wrist_positions": wrist_positions})
        if self.object_label is not None:
            sample_dict["object_classification"] = torch.tensor(self.object_label)

        return sample_dict


if __name__ == "__main__":
    import os
    import argparse
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from omegaconf import OmegaConf

    data_root = "dataset/brainco/pretraining"

    parser = argparse.ArgumentParser(description="Visualize BrainCo tactile data")
    parser.add_argument("--object", "-o", type=str, default="basket",
                        help="Object name (e.g. basket, cable, cleaner, doll, hammer, pot, towel)")
    parser.add_argument("--episode", "-e", type=str, default=None,
                        help="Episode name (e.g. episode_0000). If not given, uses first available.")
    args = parser.parse_args()

    # List available objects
    available_objects = sorted([d for d in os.listdir(data_root)
                                if os.path.isdir(os.path.join(data_root, d))])
    print(f"Available objects: {available_objects}")

    obj_path = os.path.join(data_root, args.object)
    assert os.path.exists(obj_path), f"Object '{args.object}' not found in {data_root}"

    # List available episodes
    available_episodes = sorted([d for d in os.listdir(obj_path) if d.startswith("episode_")])
    print(f"Available episodes for '{args.object}': {available_episodes}")

    episode = args.episode if args.episode else available_episodes[0]
    ep_path = os.path.join(obj_path, episode)
    assert os.path.exists(ep_path), f"Episode '{episode}' not found in {obj_path}"
    print(f"\nSelected: {args.object}/{episode}")

    config = OmegaConf.create({
        "window_time": 0.01,
        "window_overlap": 0.0,
        "interpolating_freq": 100,
        "bias_noise_std": 0,
        "bias_range": 0,
    })

    print("Loading dataset...")
    dataset = BraincoSSLDataset(config=config, data_path=ep_path, brainco_urdf_path="dataset/brainco/urdf")
    print(f"Dataset length: {len(dataset)}")
    print(f"Fingertip positions: {dataset.fingertip_positions.shape}")
    print(f"Wrist positions: {dataset.wrist_positions.shape}")

    # ── Open video ────────────────────────────────────────────────
    video_path = os.path.join(ep_path, "colors.mp4")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}, fps={video_fps}, total_frames={video_total_frames}")

    # ── Fingertip layout: 0-4 = left, 5-9 = right ──
    left_finger_idxs = [0, 1, 2, 3, 4]
    right_finger_idxs = [5, 6, 7, 8, 9]
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]

    fingertips_world = dataset.fingertip_positions_world  # (N, 10, 3) — already in world frame
    skeletons_world = dataset.skeleton_positions_world    # (N, 44, 3) — already in world frame
    wrists = dataset.wrist_positions                      # (N, 2, 3)
    tactile_data = dataset.tactile_array                  # (N, 10, 4)
    num_data_frames = fingertips_world.shape[0]
    print(f"Fingertip 6D shape: {dataset.fingertip_positions.shape}")
    print(f"Tactile array shape: {tactile_data.shape}")

    # Map data frame index to video frame index
    data_to_video_ratio = video_total_frames / num_data_frames

    # ── Setup figure: video (left) + 3D (right) ──────────────────
    fig = plt.figure(figsize=(16, 7))
    ax_video = fig.add_subplot(121)
    ax_video.set_title("Camera View")
    ax_video.axis("off")

    # Read first frame for initial display
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    else:
        first_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    img_display = ax_video.imshow(first_frame)

    ax_3d = fig.add_subplot(122, projection="3d")

    # Compute axis limits from all data (world-space)
    all_positions = np.concatenate([
        skeletons_world.reshape(-1, 3),
        wrists.reshape(-1, 3),
    ], axis=0)
    margin = 0.02
    x_min, x_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    y_min, y_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2

    xlim = (x_mid + max_range/2 + margin, x_mid - max_range/2 - margin)
    ylim = (y_mid - max_range/2 - margin, y_mid + max_range/2 + margin)
    zlim = (z_mid - max_range/2 - margin, z_mid + max_range/2 + margin)
    
    ax_3d.set_box_aspect([1, 1, 1])

    def update(frame_idx):
        # ── Update video frame ──
        video_frame_idx = int(frame_idx * data_to_video_ratio)
        video_frame_idx = min(video_frame_idx, video_total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_display.set_data(frame)
        ax_video.set_title(f"Camera (video frame {video_frame_idx})")

        # ── Update 3D plot ──
        ax_3d.cla()
        ax_3d.set_title(f"Positions (data frame {frame_idx}/{num_data_frames})")
        ax_3d.set_xlabel("Y")
        ax_3d.set_ylabel("X")
        ax_3d.set_zlabel("Z")
        ax_3d.set_xlim(*xlim)
        ax_3d.set_ylim(*ylim)
        ax_3d.set_zlim(*zlim)

        fp = fingertips_world[frame_idx]  # (10, 3) — world-space
        sk = skeletons_world[frame_idx]   # (44, 3)
        wp = wrists[frame_idx]            # (2, 3)
        tac = tactile_data[frame_idx]     # (10, 4)

        # Draw hand skeletons
        left_sk = sk[:22]
        right_sk = sk[22:]
        for (i, j) in SKELETON_LINES:
            ax_3d.plot([left_sk[i, 1], left_sk[j, 1]], 
                       [left_sk[i, 0], left_sk[j, 0]], 
                       [left_sk[i, 2], left_sk[j, 2]], c="blue", alpha=0.5, linewidth=2)
            ax_3d.plot([right_sk[i, 1], right_sk[j, 1]], 
                       [right_sk[i, 0], right_sk[j, 0]], 
                       [right_sk[i, 2], right_sk[j, 2]], c="red", alpha=0.5, linewidth=2)

        # Left fingertips (blue)
        left_fp = fp[left_finger_idxs]
        left_tac = tac[left_finger_idxs]
        
        # Scale sizes based on the tactile value (4th element), with a base size
        left_sizes = 20 + (np.clip(left_tac[:, 3], 0, 5) * 50)  # Adjust scaling factor as needed
        
        ax_3d.scatter(left_fp[:, 1], left_fp[:, 0], left_fp[:, 2],
                      c="blue", s=left_sizes, marker="o", label="Left fingers")
        for i, name in enumerate(finger_names):
            ax_3d.text(left_fp[i, 1], left_fp[i, 0], left_fp[i, 2], f" L_{name}", fontsize=7, color="blue")

        # Right fingertips (red)
        right_fp = fp[right_finger_idxs]
        right_tac = tac[right_finger_idxs]
        
        # Scale sizes based on the tactile value (4th element), with a base size
        right_sizes = 20 + (np.clip(right_tac[:, 3], 0, 5) * 50) # Adjust scaling factor as needed
        
        ax_3d.scatter(right_fp[:, 1], right_fp[:, 0], right_fp[:, 2],
                      c="red", s=right_sizes, marker="o", label="Right fingers")
        for i, name in enumerate(finger_names):
            ax_3d.text(right_fp[i, 1], right_fp[i, 0], right_fp[i, 2], f" R_{name}", fontsize=7, color="red")

        # Wrists
        ax_3d.scatter(wp[0, 1], wp[0, 0], wp[0, 2],
                      c="cyan", s=120, marker="s", label="Left wrist", edgecolors="black")
        ax_3d.scatter(wp[1, 1], wp[1, 0], wp[1, 2],
                      c="magenta", s=120, marker="s", label="Right wrist", edgecolors="black")

        # Lines from wrist to hand base_link instead of fingertips
        ax_3d.plot([wp[0, 1], left_sk[0, 1]], [wp[0, 0], left_sk[0, 0]], [wp[0, 2], left_sk[0, 2]],
                   c="cyan", alpha=0.6, linewidth=2, linestyle='--')
        ax_3d.plot([wp[1, 1], right_sk[0, 1]], [wp[1, 0], right_sk[0, 0]], [wp[1, 2], right_sk[0, 2]],
                   c="magenta", alpha=0.6, linewidth=2, linestyle='--')

        ax_3d.legend(loc="upper left", fontsize=8)

        # HUD Text for positions (using Y, X, Z to match the plot axes)
        left_thumb = left_fp[0]
        right_thumb = right_fp[0]
        info_text = (
            f"L Wrist: {wp[0, 1]:.3f}, {wp[0, 0]:.3f}, {wp[0, 2]:.3f}\n"
            f"L Thumb: {left_thumb[1]:.3f}, {left_thumb[0]:.3f}, {left_thumb[2]:.3f}\n"
            f"R Wrist: {wp[1, 1]:.3f}, {wp[1, 0]:.3f}, {wp[1, 2]:.3f}\n"
            f"R Thumb: {right_thumb[1]:.3f}, {right_thumb[0]:.3f}, {right_thumb[2]:.3f}"
        )
        ax_3d.text2D(0.02, 0.02, info_text, transform=ax_3d.transAxes, 
                     fontsize=9, verticalalignment='bottom', 
                     bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7})

    # Animate all frames, slower playback
    frame_indices = list(range(0, num_data_frames))
    anim = FuncAnimation(fig, update, frames=frame_indices, interval=10, repeat=True)
    plt.tight_layout()
    plt.show()
    cap.release()
