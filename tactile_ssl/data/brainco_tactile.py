from typing import Optional
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import pytorch_kinematics as pk
from omegaconf import DictConfig

from tactile_ssl.utils.logging import get_pylogger

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
    "pinky_proximal_link", "pinky_distal_link", "pinky_tip_link", "pinky_touch_link",
]

SKELETON_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (0, 6), (6, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16), (16, 17),
    (0, 18), (18, 19), (19, 20), (20, 21),
]

G1_ALL_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

G1_ARM_JOINTS = [
    "shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
    "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint",
]


def build_g1_chain(urdf_path: Path):
    """Build kinematic chain for G1 body only (arm → wrist positions)."""
    g1_path = urdf_path / "g1.urdf"
    with open(str(g1_path), "rb") as f:
        return pk.build_chain_from_urdf(f.read())


def build_hand_chain(urdf_path: Path, side: str):
    """Build kinematic chain for a single hand URDF (finger positions relative to base_link).

    Args:
        side: "left" or "right"
    """
    hand_path = urdf_path / f"revo2_{side}_hand.urdf"
    with open(str(hand_path), "rb") as f:
        return pk.build_chain_from_urdf(f.read())


# ── Joint-name helpers ────────────────────────────────────────────────────────
_HAND_JOINT_SUFFIXES = [
    "thumb_metacarpal_joint",
    "thumb_proximal_joint",
    "thumb_distal_joint",
    "index_proximal_joint",
    "index_distal_joint",
    "middle_proximal_joint",
    "middle_distal_joint",
    "ring_proximal_joint",
    "ring_distal_joint",
    "pinky_proximal_joint",
    "pinky_distal_joint",
]


def compute_g1_fk(chain, left_arm, right_arm):
    """Compute G1 arm FK → wrist poses (world frame).

    Returns fk result dict with at least 'left_rubber_hand' and 'right_rubber_hand'.
    """
    joint_map = {name: 0.0 for name in G1_ALL_JOINTS}
    for i, suffix in enumerate(G1_ARM_JOINTS):
        joint_map[f"left_{suffix}"]  = float(left_arm[i])
        joint_map[f"right_{suffix}"] = float(right_arm[i])
    return chain.forward_kinematics(joint_map)


def compute_hand_fk(chain, ee_qpos, side: str):
    """Compute hand FK → finger joint positions relative to base_link.

    Args:
        chain:    hand kinematic chain (left or right)
        ee_qpos:  (6,) — [thumb_meta, thumb_prox, index_prox,
                           middle_prox, ring_prox, pinky_prox]
        side:     "left" or "right"

    Returns fk result dict; all positions are relative to {side}_base_link.
    """
    q = [float(v) for v in ee_qpos]
    p = f"{side}_"
    joint_map = {
        p + _HAND_JOINT_SUFFIXES[0]: q[0],           # thumb_metacarpal
        p + _HAND_JOINT_SUFFIXES[1]: q[1],            # thumb_proximal
        p + _HAND_JOINT_SUFFIXES[2]: q[1] * 1.0,     # thumb_distal
        p + _HAND_JOINT_SUFFIXES[3]: q[2],            # index_proximal
        p + _HAND_JOINT_SUFFIXES[4]: q[2] * 1.155,   # index_distal
        p + _HAND_JOINT_SUFFIXES[5]: q[3],            # middle_proximal
        p + _HAND_JOINT_SUFFIXES[6]: q[3] * 1.155,   # middle_distal
        p + _HAND_JOINT_SUFFIXES[7]: q[4],            # ring_proximal
        p + _HAND_JOINT_SUFFIXES[8]: q[4] * 1.155,   # ring_distal
        p + _HAND_JOINT_SUFFIXES[9]: q[5],            # pinky_proximal
        p + _HAND_JOINT_SUFFIXES[10]: q[5] * 1.155,  # pinky_distal
    }
    return chain.forward_kinematics(joint_map)


def _rot_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D representation (first two columns, flattened).

    Args:
        R: (3, 3) rotation matrix
    Returns:
        (6,) 6D rotation: [R[:,0], R[:,1]]
    """
    return np.concatenate([R[:, 0], R[:, 1]], axis=0)


def _rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation back to 3x3 rotation matrix via Gram-Schmidt.

    Args:
        rot6d: (..., 6) — [R[:,0], R[:,1]] (first two columns concatenated)
    Returns:
        (..., 3, 3) orthonormal rotation matrix
    """
    c0 = rot6d[..., :3]
    c1 = rot6d[..., 3:]
    c0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-8)
    c1 = c1 - (c1 * c0).sum(axis=-1, keepdims=True) * c0
    c1 = c1 / (np.linalg.norm(c1, axis=-1, keepdims=True) + 1e-8)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=-1)   # (..., 3, 3)


def _compute_fk(
    urdf_path: Path,
    frames: list,
) -> dict:
    """Run FK for all frames and return cache arrays.

    Wrist poses are computed via G1 arm FK (world frame).
    Fingertip and skeleton positions are base_link-relative (wrist frame) —
    no world transform applied. For the left hand, the local y-axis is flipped
    during extraction so that +y points from pinky toward thumb.

    Returns dict with keys:
        fingertip_base: (N, 10, 3)  base_link-relative; left 0-4, right 5-9
        skeleton_base:  (N, 44, 3)  base_link-relative; left 0-21, right 22-43
        wrist_pos:      (N, 2, 3)   world positions [left, right]
        wrist_rot6d:    (N, 2, 6)   world 6D rotations [left, right]
    """
    num_frames = len(frames)
    fingertip_base = np.zeros((num_frames, 10, 3), dtype=np.float32)
    skeleton_base  = np.zeros((num_frames, 44, 3), dtype=np.float32)
    wrist_pos      = np.zeros((num_frames, 2, 3),  dtype=np.float32)
    wrist_rot6d    = np.zeros((num_frames, 2, 6),  dtype=np.float32)

    left_arm_array  = np.array([f["states"]["left_arm"]["qpos"]  for f in frames], dtype=np.float32)
    right_arm_array = np.array([f["states"]["right_arm"]["qpos"] for f in frames], dtype=np.float32)
    left_ee_array   = np.array([f["states"]["left_ee"]["qpos"]   for f in frames], dtype=np.float32)
    right_ee_array  = np.array([f["states"]["right_ee"]["qpos"]  for f in frames], dtype=np.float32)

    log.info(f"  Building kinematic chains from {urdf_path} ...")
    g1_chain = build_g1_chain(urdf_path)
    lh_chain = build_hand_chain(urdf_path, "left")
    rh_chain = build_hand_chain(urdf_path, "right")
    log.info(f"  Running FK for {num_frames} frames ...")

    for i in range(num_frames):
        # ── G1 arm FK → wrist world pose ─────────────────────────────────
        g1_fk = compute_g1_fk(g1_chain, left_arm_array[i], right_arm_array[i])
        lh_mat = g1_fk["left_rubber_hand"].get_matrix().squeeze(0).numpy()
        rh_mat = g1_fk["right_rubber_hand"].get_matrix().squeeze(0).numpy()
        wrist_pos[i, 0]   = lh_mat[:3, 3]
        wrist_pos[i, 1]   = rh_mat[:3, 3]
        wrist_rot6d[i, 0] = _rot_to_6d(lh_mat[:3, :3])
        wrist_rot6d[i, 1] = _rot_to_6d(rh_mat[:3, :3])

        # ── Hand FK → positions relative to base_link (no world transform) ──
        lh_fk = compute_hand_fk(lh_chain, left_ee_array[i],  "left")
        rh_fk = compute_hand_fk(rh_chain, right_ee_array[i], "right")

        for fi, link in enumerate(FINGERTIP_LINKS):
            left_tip = lh_fk[f"left_{link}"].get_matrix()[:, :3, 3].squeeze(0).numpy()
            left_tip[1] *= -1.0
            fingertip_base[i, fi]     = left_tip
            fingertip_base[i, fi + 5] = rh_fk[f"right_{link}"].get_matrix()[:, :3, 3].squeeze(0).numpy()

        for li, link in enumerate(SKELETON_LINKS):
            left_joint = lh_fk[f"left_{link}"].get_matrix()[:, :3, 3].squeeze(0).numpy()
            left_joint[1] *= -1.0
            skeleton_base[i, li]      = left_joint
            skeleton_base[i, li + 22] = rh_fk[f"right_{link}"].get_matrix()[:, :3, 3].squeeze(0).numpy()

    return {
        "fingertip_base": fingertip_base,
        "skeleton_base":  skeleton_base,
        "wrist_pos":      wrist_pos,
        "wrist_rot6d":    wrist_rot6d,
    }


def compute_fk(data_path: Path, urdf_path: Path, frames: list) -> dict:
    result = _compute_fk(urdf_path, frames)
    return result


def compute_wrist_poses_vroot(wrist_pos: np.ndarray, wrist_rot6d: np.ndarray) -> np.ndarray:
    """Convert world-frame wrist pos/rot into virtual-root-relative wrist poses.

    Applies the URDF→model axis remap, constructs a per-frame virtual root frame
    (origin = midpoint of two wrists, x-axis = left→right, y-axis = finger direction),
    and expresses each wrist's pose relative to that frame.

    Args:
        wrist_pos:   (N, 2, 3)  world positions  [left, right]
        wrist_rot6d: (N, 2, 6)  world 6D rotations [left, right]

    Returns:
        wrist_poses: (N, 2, 9)  vroot-relative [pos(3) + rot6d(6)] per wrist
    """
    # ── Axis remap: URDF world → model convention ────────────────────────
    wrist_pos   = wrist_pos.copy()
    wrist_rot6d = wrist_rot6d.copy()
    wrist_pos   = wrist_pos[..., [1, 0, 2]]
    wrist_pos[..., 0] = -wrist_pos[..., 0]
    wrist_rot6d = wrist_rot6d[..., [1, 0, 2, 4, 3, 5]]
    wrist_rot6d[..., 0] = -wrist_rot6d[..., 0]
    wrist_rot6d[..., 3] = -wrist_rot6d[..., 3]

    # ── Virtual root frame ────────────────────────────────────────────────
    R_wrists = _rot6d_to_mat(wrist_rot6d)            # (N, 2, 3, 3)

    lr_vec = wrist_pos[:, 1] - wrist_pos[:, 0]
    x_ax   = lr_vec / (np.linalg.norm(lr_vec, axis=-1, keepdims=True) + 1e-8)

    finger_avg  = R_wrists[:, 0, :, 1] + R_wrists[:, 1, :, 1]
    finger_norm = np.linalg.norm(finger_avg, axis=-1, keepdims=True)
    palm_avg    = R_wrists[:, 0, :, 2] + R_wrists[:, 1, :, 2]
    palm_norm   = np.linalg.norm(palm_avg,   axis=-1, keepdims=True)
    use_finger  = (finger_norm > 1e-6).squeeze(-1)
    y_candidate = np.where(use_finger[:, None],
                           finger_avg / (finger_norm + 1e-8),
                           palm_avg   / (palm_norm  + 1e-8))
    y_candidate = y_candidate - (y_candidate * x_ax).sum(-1, keepdims=True) * x_ax
    y_ax = y_candidate / (np.linalg.norm(y_candidate, axis=-1, keepdims=True) + 1e-8)
    z_ax = np.cross(x_ax, y_ax)
    z_ax = z_ax / (np.linalg.norm(z_ax, axis=-1, keepdims=True) + 1e-8)

    R_vroot   = np.stack([x_ax, y_ax, z_ax], axis=-1)   # (N, 3, 3)
    R_vroot_T = R_vroot.transpose(0, 2, 1)
    vroot_pos = (wrist_pos[:, 0] + wrist_pos[:, 1]) / 2.0

    # ── Express each wrist relative to vroot ─────────────────────────────
    N = wrist_pos.shape[0]
    wrist_poses = np.zeros((N, 2, 9), dtype=np.float32)
    for side in range(2):
        t_vroot = np.einsum('nij,nj->ni', R_vroot_T,
                            wrist_pos[:, side] - vroot_pos)
        R_rel   = np.einsum('nij,njk->nik', R_vroot_T, R_wrists[:, side])
        rot6d   = np.concatenate([R_rel[:, :, 0], R_rel[:, :, 1]], axis=-1)
        wrist_poses[:, side] = np.concatenate([t_vroot, rot6d], axis=-1)

    return wrist_poses


class BraincoSSLDataset(data.Dataset):
    """BrainCo SSL pretraining dataset (single episode).

    Args:
        config:            Hydra DictConfig with window_time, window_overlap,
                           interpolating_freq, bias_noise_std, bias_range.
        data_path:         Path to a single episode directory containing data.json.
        brainco_urdf_path: Root directory containing g1.urdf and hand URDFs.
        object_class:      Integer label for object classification (optional).
        load_images:       Reserved, not yet used.
        joint_poses:       If False (default), sensor_poses is (W, 10, 3) fingertip
                           positions relative to own wrist.
                           If True, sensor_poses is (W, 42, 3) for all skeleton
                           joints (left 0-20, right 21-41) relative to own wrist.

    Output per sample:
        sensor:       Tensor(W, 10, 4)  — tactile data, ch2 null(65535)→-1
        sensor_poses: Tensor(W, 10, 3) or (W, 42, 3)
        wrist_poses:  Tensor(W, 2, 9)  — [3D translation rel. virtual root, 6D rotation]
        object_classification: Tensor  — only if object_class is not None
    """

    def __init__(
        self,
        config: DictConfig,
        data_path: str,
        brainco_urdf_path: str = "dataset/brainco/urdf",
        object_class: Optional[int] = None,
        load_images: bool = False,
        joint_poses: bool = False,
    ):
        self._load_images = load_images  # reserved for future RGB loading
        # ── Window config ────────────────────────────────────────────────
        self.window_time = config.window_time
        self.interpolating_freq = config.interpolating_freq
        self.num_frames_per_window = int(round(self.window_time * self.interpolating_freq))

        overlap = config.get("window_overlap", 0.0)
        assert 0 <= overlap < 1, "window_overlap must be in [0, 1)"
        self.window_overlap = overlap
        shift = int(round(self.num_frames_per_window * (1.0 - overlap)))
        self.shift_per_window = max(1, shift)

        self.object_label = object_class
        self.joint_poses = joint_poses

        # ── Load data.json ───────────────────────────────────────────────
        self.data_path = Path(data_path)
        data_json_path = self.data_path / "data.json"
        assert data_json_path.exists(), f"data.json not found at {data_json_path}"

        log.info(f"Loading BrainCo SSL data from {data_json_path}")
        with open(data_json_path, "r") as f:
            raw_data = json.load(f)

        frames = raw_data["data"]
        self.num_frames = len(frames)
        log.info(f"  Found {self.num_frames} frames")

        # ── FK ──────────────────────────────────────────────────
        fk = compute_fk(self.data_path, Path(brainco_urdf_path), frames)
        fingertip_base  = fk["fingertip_base"]   # (N, 10, 3) base_link-relative
        skeleton_base   = fk["skeleton_base"]    # (N, 44, 3) base_link-relative
        wrist_pos       = fk["wrist_pos"]        # (N, 2, 3)  world
        wrist_rot6d     = fk["wrist_rot6d"]      # (N, 2, 6)  world

        # ── fingertip_rel: base_link coords are already wrist-local ──────
        # Hand FK output is relative to base_link = wrist frame directly.
        self.fingertip_rel  = fingertip_base               # (N, 10, 3)
        self.fingertip_base = fingertip_base               # (N, 10, 3) — kept for visualization

        # ── skeleton_rel: skip base_link (idx 0 / 22) ────────────────────
        # skeleton_base: left 0-21 (0=base_link), right 22-43 (22=base_link)
        skeleton_rel = np.concatenate([
            skeleton_base[:, 1:22],   # left  joints 1-21 (skip base_link)
            skeleton_base[:, 23:44],  # right joints 23-43 (skip base_link)
        ], axis=1)  # (N, 42, 3)
        self.skeleton_rel  = skeleton_rel
        self.skeleton_base = skeleton_base                 # (N, 44, 3) — kept for visualization

        # ── wrist world poses (URDF frame, before axis remap) ────────────
        self.wrist_pos_world   = wrist_pos.copy()    # (N, 2, 3)
        self.wrist_rot6d_world = wrist_rot6d.copy()  # (N, 2, 6)

        # ── wrist_poses: position + orientation in vroot frame ───────────
        self.wrist_poses = compute_wrist_poses_vroot(wrist_pos, wrist_rot6d)  # (N, 2, 9)

        log.info(f"  fingertip_rel shape:  {self.fingertip_rel.shape}")
        log.info(f"  skeleton_rel shape:   {self.skeleton_rel.shape}")
        log.info(f"  wrist_poses shape:    {self.wrist_poses.shape}")

        # ── Tactile data ─────────────────────────────────────────────────
        tactile_list = []
        for frame in frames:
            tactile_info = frame["tactiles"]

            if isinstance(tactile_info["left_ee"], str):
                left_tactile = np.load(str(self.data_path / tactile_info["left_ee"])).reshape(-1, 4)
            else:
                left_tactile = np.array(tactile_info["left_ee"]).reshape(-1, 4)

            if isinstance(tactile_info["right_ee"], str):
                right_tactile = np.load(str(self.data_path / tactile_info["right_ee"])).reshape(-1, 4)
            else:
                right_tactile = np.array(tactile_info["right_ee"]).reshape(-1, 4)

            tactile_list.append(np.concatenate([left_tactile, right_tactile], axis=0))

        self.tactile_array = np.array(tactile_list, dtype=np.float32)  # (N, 10, 4)

        # ch2: 65535 (invalid) → -1
        invalid_mask = self.tactile_array[..., 2] == 65535
        self.tactile_array[..., 2][invalid_mask] = -1

        log.info(f"  tactile_array shape: {self.tactile_array.shape}")

        # ── Window indices ───────────────────────────────────────────────
        max_start = self.num_frames - self.num_frames_per_window
        self.data_idxs = np.arange(0, max(1, max_start + 1), self.shift_per_window)
        log.info(f"  Windows: {len(self.data_idxs)}, frames_per_window: {self.num_frames_per_window}")

    def __len__(self) -> int:
        return len(self.data_idxs)

    def update_normalization(self, mean, std):
        self.tactile_mean = mean
        self.tactile_std = std

    def __getitem__(self, idx: int) -> dict:
        start = self.data_idxs[idx]
        end = start + self.num_frames_per_window

        sensor = torch.from_numpy(self.tactile_array[start:end].copy())  # (W, 10, 4)

        if self.joint_poses:
            sensor_poses = torch.from_numpy(self.skeleton_rel[start:end].copy())  # (W, 42, 3)
        else:
            sensor_poses = torch.from_numpy(self.fingertip_rel[start:end].copy())  # (W, 10, 3)

        wrist_poses = torch.from_numpy(self.wrist_poses[start:end].copy())  # (W, 2, 9)

        sample = {
            "sensor":       sensor,
            "sensor_poses": sensor_poses,
            "wrist_poses":  wrist_poses,
        }

        if self.object_label is not None:
            sample["object_classification"] = torch.tensor(self.object_label)

        return sample


if __name__ == "__main__":
    import argparse
    import os
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for saving
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from omegaconf import OmegaConf

    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    parser = argparse.ArgumentParser(description="Visualize BrainCo SSL dataset")
    parser.add_argument("--data_root", default="dataset/brainco/pretraining")
    parser.add_argument("--object", "-o", default=None)
    parser.add_argument("--episode", "-e", default=None)
    parser.add_argument("--urdf", default="dataset/brainco/urdf")
    parser.add_argument("--window_time", type=float, default=0.1)
    parser.add_argument("--window_overlap", type=float, default=0.5)
    parser.add_argument("--joint_poses", action="store_true")
    parser.add_argument("--out", default=".",
                        help="Output directory for saved files")
    parser.add_argument("--fps", type=int, default=15,
                        help="FPS for animation output")
    args = parser.parse_args()

    data_root = args.data_root
    available_objects = sorted(d for d in os.listdir(data_root)
                               if os.path.isdir(os.path.join(data_root, d)))
    print(f"Available objects: {available_objects}")

    obj = args.object or available_objects[0]
    obj_path = os.path.join(data_root, obj)
    available_episodes = sorted(d for d in os.listdir(obj_path) if d.startswith("episode_"))
    print(f"Episodes for '{obj}': {available_episodes}")

    ep = args.episode or available_episodes[0]
    ep_path = os.path.join(obj_path, ep)
    print(f"\nLoading: {obj}/{ep}\n")

    os.makedirs(args.out, exist_ok=True)
    prefix = os.path.join(args.out, f"{obj}_{ep}")

    config = OmegaConf.create({
        "window_time": args.window_time,
        "window_overlap": args.window_overlap,
        "interpolating_freq": 100,
        "bias_noise_std": 0,
        "bias_range": 0,
    })

    dataset = BraincoSSLDataset(
        config=config,
        data_path=ep_path,
        brainco_urdf_path=args.urdf,
        joint_poses=args.joint_poses,
    )

    # ── Shape verification ────────────────────────────────────────────────
    sample = dataset[0]
    print("=== Sample shapes ===")
    for k, v in sample.items():
        print(f"  {k:30s}: {tuple(v.shape)}")
    print(f"  dataset length   : {len(dataset)}")
    print(f"  num_frames       : {dataset.num_frames}")
    print(f"  frames_per_window: {dataset.num_frames_per_window}")
    print(f"  shift_per_window : {dataset.shift_per_window}")
    null_ratio = (dataset.tactile_array[..., 2] == -1).mean()
    print(f"  ch2 null ratio   : {null_ratio:.1%}")
    wrist_t = dataset.wrist_poses[:, :, :3]
    d = np.linalg.norm(wrist_t[:, 0] - wrist_t[:, 1], axis=-1)
    print(f"  wrist-wrist dist : min={d.min():.3f}  mean={d.mean():.3f}  max={d.max():.3f} m\n")

    # ═══════════════════════════════════════════════════════════════════════
    # Visualize directly in base_link (wrist-local) frame — no reconstruction.
    # skeleton_base: (N, 44, 3)  left 0-21, right 22-43
    # fingertip_base: (N, 10, 3) left 0-4,  right 5-9
    # ═══════════════════════════════════════════════════════════════════════
    sk_base = dataset.skeleton_base   # (N, 44, 3)
    fp_base = dataset.fingertip_base  # (N, 10, 3)
    tactile = dataset.tactile_array   # (N, 10, 4)
    N       = dataset.num_frames

    def _hand_vis_coords(points: np.ndarray, side: str) -> np.ndarray:
        return points.copy()

    # Axis limits: computed from all skeleton joints after applying hand-view convention
    _all_pts  = np.concatenate([
        _hand_vis_coords(sk_base[:, :22].reshape(-1, 3), "left"),
        _hand_vis_coords(sk_base[:, 22:].reshape(-1, 3), "right"),
    ], axis=0)
    margin    = 0.01
    max_range = max(np.ptp(_all_pts[:, i]) for i in range(3)) / 2 + margin
    mids      = np.array([
        (_all_pts[:, i].max() + _all_pts[:, i].min()) / 2 for i in range(3)
    ])

    def _set_hand_lim(ax, label):
        ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
        ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
        ax.set_zlim(mids[2] - max_range, mids[2] + max_range)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7); ax.set_zlabel("Z", fontsize=7)
        ax.set_title(label, fontsize=8)

    def _draw_hand(ax, fi, side: str, show_labels=True):
        """Draw one hand in its own base_link frame.

        side: 'left'  → skeleton_base[:22], fingertip_base[:5]
              'right' → skeleton_base[22:], fingertip_base[5:]
        """
        if side == "left":
            sk  = _hand_vis_coords(sk_base[fi, :22], "left")   # (22, 3)
            fps = _hand_vis_coords(fp_base[fi, :5], "left")    # (5, 3)
            tac = tactile[fi, :5]        # (5, 4)
            col = "royalblue"
            fp_col = "blue"
            lbl_pfx = "L"
        else:
            sk  = _hand_vis_coords(sk_base[fi, 22:], "right")  # (22, 3)
            fps = _hand_vis_coords(fp_base[fi, 5:], "right")   # (5, 3)
            tac = tactile[fi, 5:]        # (5, 4)
            col = "tomato"
            fp_col = "red"
            lbl_pfx = "R"

        for (i, j) in SKELETON_LINES:
            ax.plot([sk[i, 0], sk[j, 0]],
                    [sk[i, 1], sk[j, 1]],
                    [sk[i, 2], sk[j, 2]], c=col, alpha=0.7, lw=1.5)

        # Base_link origin marker
        ax.scatter(0, 0, 0, c="gold", s=60, marker="*", zorder=6)

        for li in range(5):
            force = max(float(tac[li, 0]), 0.0)
            size  = 30 + force * 0.002
            p = fps[li]
            ax.scatter(p[0], p[1], p[2], c=fp_col, s=size, zorder=5, depthshade=False)
            if show_labels:
                ax.text(p[0], p[1], p[2],
                        f" {lbl_pfx}_{FINGER_NAMES[li][0]}", fontsize=6, color=fp_col)

    def _draw_hand_projection(ax, fi, side: str, plane: str, show_labels=True):
        plane_axes = {
            "xy": (0, 1),
            "yz": (1, 2),
            "xz": (0, 2),
        }
        axis_labels = {
            "xy": ("X", "Y"),
            "yz": ("Y", "Z"),
            "xz": ("X", "Z"),
        }
        i0, i1 = plane_axes[plane]
        label0, label1 = axis_labels[plane]

        if side == "left":
            sk = _hand_vis_coords(sk_base[fi, :22], "left")
            fps = _hand_vis_coords(fp_base[fi, :5], "left")
            tac = tactile[fi, :5]
            col = "royalblue"
            fp_col = "blue"
            lbl_pfx = "L"
        else:
            sk = _hand_vis_coords(sk_base[fi, 22:], "right")
            fps = _hand_vis_coords(fp_base[fi, 5:], "right")
            tac = tactile[fi, 5:]
            col = "tomato"
            fp_col = "red"
            lbl_pfx = "R"

        for (i, j) in SKELETON_LINES:
            ax.plot([sk[i, i0], sk[j, i0]],
                    [sk[i, i1], sk[j, i1]], c=col, alpha=0.7, lw=1.5)

        ax.scatter(0, 0, c="gold", s=60, marker="*", zorder=6)

        for li in range(5):
            force = max(float(tac[li, 0]), 0.0)
            size = 30 + force * 0.002
            p = fps[li]
            ax.scatter(p[i0], p[i1], c=fp_col, s=size, zorder=5)
            if show_labels:
                ax.text(p[i0], p[i1], f" {lbl_pfx}_{FINGER_NAMES[li][0]}", fontsize=6, color=fp_col)

        pts = np.vstack([sk[:, [i0, i1]], fps[:, [i0, i1]], np.zeros((1, 2), dtype=np.float32)])
        margin = 0.01
        center = pts.mean(axis=0)
        radius = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-3) / 2 + margin
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(label0, fontsize=7)
        ax.set_ylabel(label1, fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{'Left' if side=='left' else 'Right'} {plane.upper()}  f{fi}", fontsize=8)

    # ═══════════════════════════════════════════════════════════════════════
    # RGB loader: colors/ directory → colors.mp4 fallback
    # ═══════════════════════════════════════════════════════════════════════
    import cv2

    _img_files = None
    _video_cap = None
    _video_total = 0

    colors_dir = Path(ep_path) / "colors"
    if colors_dir.exists():
        _img_files = sorted(
            f for f in colors_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        _video_total = len(_img_files)
    else:
        video_path = Path(ep_path) / "colors.mp4"
        if video_path.exists():
            _video_cap = cv2.VideoCapture(str(video_path))
            _video_total = int(_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _load_rgb(data_frame_idx: int):
        """Return HxWx3 uint8 RGB for the given data frame, or None."""
        if _video_total == 0:
            return None
        vid_idx = int(round(data_frame_idx * _video_total / N))
        vid_idx = min(vid_idx, _video_total - 1)
        if _img_files is not None:
            bgr = cv2.imread(str(_img_files[vid_idx]))
        else:
            _video_cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
            ret, bgr = _video_cap.read()
            if not ret:
                return None
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    has_rgb = (_video_total > 0)

    # ═══════════════════════════════════════════════════════════════════════
    # Output 1: static 3D snapshot grid  (N_SNAP evenly-spaced frames)
    #           Row 0: 3D skeleton   Row 1: RGB (if available)
    # ═══════════════════════════════════════════════════════════════════════
    # ── rows: left-hand | right-hand | RGB(optional) ──────────────────
    N_SNAP = 6
    snap_frames = np.linspace(0, N - 1, N_SNAP, dtype=int)
    n_3d_rows = 2                          # left hand row + right hand row
    n_rows    = n_3d_rows + (1 if has_rgb else 0)

    fig_snap = plt.figure(figsize=(N_SNAP * 3.5, n_rows * 3.5))
    fig_snap.suptitle(f"{obj}/{ep}  —  wrist-local skeleton (base_link frame)", fontsize=11)

    for col_idx, fi in enumerate(snap_frames):
        # Left hand
        ax_l = fig_snap.add_subplot(n_rows, N_SNAP, col_idx + 1, projection="3d")
        _set_hand_lim(ax_l, f"Left  f{fi}")
        _draw_hand(ax_l, fi, "left",  show_labels=False)

        # Right hand
        ax_r = fig_snap.add_subplot(n_rows, N_SNAP, N_SNAP + col_idx + 1, projection="3d")
        _set_hand_lim(ax_r, f"Right f{fi}")
        _draw_hand(ax_r, fi, "right", show_labels=False)

        # RGB row
        if has_rgb:
            ax_rgb = fig_snap.add_subplot(n_rows, N_SNAP, 2 * N_SNAP + col_idx + 1)
            rgb = _load_rgb(fi)
            if rgb is not None:
                ax_rgb.imshow(rgb)
            else:
                ax_rgb.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax_rgb.transAxes)
            ax_rgb.axis("off")
            ax_rgb.set_title(f"RGB f{fi}", fontsize=7)

    snap_path = f"{prefix}_3d_snapshots.png"
    fig_snap.savefig(snap_path, dpi=100, bbox_inches="tight")
    plt.close(fig_snap)
    print(f"Saved: {snap_path}")


    def _add_ax(fig, nrows, ncols, idx, is_3d):
        if is_3d:
            return fig.add_subplot(nrows, ncols, idx, projection="3d")
        return fig.add_subplot(nrows, ncols, idx)

    def _draw_vroot_projection(ax, plane: str):
        plane_axes = {
            "xy": (0, 1),
            "yz": (1, 2),
            "xz": (0, 2),
        }
        axis_labels = {
            "xy": ("X", "Y"),
            "yz": ("Y", "Z"),
            "xz": ("X", "Z"),
        }
        i0, i1 = plane_axes[plane]
        label0, label1 = axis_labels[plane]

        _VR_AXIS_SCALE  = 0.04
        _WR_AXIS_SCALE  = 0.025
        _AXIS_COLORS    = ["red", "green", "blue"]
        _WRIST_COL      = ["cyan", "magenta"]
        _WRIST_LBL      = ["Left wrist", "Right wrist"]

        for side, (col, lbl) in enumerate(zip(_WRIST_COL, _WRIST_LBL)):
            p = wp_s[side]
            ax.scatter(p[i0], p[i1], c=col, s=70, zorder=6, label=lbl)
            R = R_wrists_s[side]
            for j, ac in enumerate(_AXIS_COLORS):
                axis_vec = R[:, j] * _WR_AXIS_SCALE
                ax.arrow(
                    p[i0], p[i1], axis_vec[i0], axis_vec[i1],
                    color=ac, width=0.0008, length_includes_head=True, alpha=0.7
                )

        vo = vroot_orig_s
        ax.scatter(vo[i0], vo[i1], c="gold", s=100, marker="*", zorder=7, label="VRoot")
        for j, (ac, albl) in enumerate(zip(_AXIS_COLORS, ["VR-X", "VR-Y", "VR-Z"])):
            axis_vec = R_vroot_s[:, j] * _VR_AXIS_SCALE
            ax.arrow(
                vo[i0], vo[i1], axis_vec[i0], axis_vec[i1],
                color=ac, width=0.0012, length_includes_head=True, alpha=0.9, label=albl
            )

        ax.plot(
            [wp_s[0, i0], wp_s[1, i0]],
            [wp_s[0, i1], wp_s[1, i1]],
            c="gray", ls="--", lw=1.0, alpha=0.6
        )

        pts = np.vstack([wp_s[:, [i0, i1]], vroot_orig_s[[i0, i1]][None]])
        margin = 0.05
        center = pts.mean(axis=0)
        radius = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-3) / 2 + margin
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(label0, fontsize=7)
        ax.set_ylabel(label1, fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=5, loc="upper left")

    # ═══════════════════════════════════════════════════════════════════════
    # Output 3: Multi-frame snapshots  ─  RGB | Left hand | Right hand | VRoot
    # ═══════════════════════════════════════════════════════════════════════
    single_frames = np.unique(np.linspace(0, N - 1, 5, dtype=int))

    # ── Wrist poses for visualization: keep world pose fixed to frame 0 ───
    wp_s = dataset.wrist_pos_world[0].copy()       # (2, 3)
    wr_s = dataset.wrist_rot6d_world[0].copy()     # (2, 6)
    R_wrists_s = _rot6d_to_mat(wr_s)               # (2, 3, 3)

    # ── Virtual root in world frame: midpoint origin, world-aligned axes ──
    R_vroot_s    = np.eye(3, dtype=np.float32)
    vroot_orig_s = (wp_s[0] + wp_s[1]) / 2.0

    panels = []
    if has_rgb:
        panels.append("rgb")
    panels += [
        "left_3d", "left_xy", "left_yz", "left_xz",
        "right_3d", "right_xy", "right_yz", "right_xz",
        "vroot", "xy", "yz", "xz",
    ]
    ncols = len(panels)

    for fi_s in single_frames:
        fig_sf = plt.figure(figsize=(ncols * 3.8, 4.2))
        fig_sf.suptitle(f"{obj}/{ep}  frame {fi_s}  —  single snapshot", fontsize=10)

        for pi, ptype in enumerate(panels):
            ax_idx = pi + 1
            if ptype == "rgb":
                ax = _add_ax(fig_sf, 1, ncols, ax_idx, False)
                rgb = _load_rgb(fi_s)
                if rgb is not None:
                    ax.imshow(rgb)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=12)
                ax.axis("off")
                ax.set_title(f"RGB  f{fi_s}", fontsize=8)

            elif ptype in ("left_3d", "right_3d"):
                side = "left" if ptype == "left_3d" else "right"
                ax = _add_ax(fig_sf, 1, ncols, ax_idx, True)
                _set_hand_lim(ax, f"{'Left' if side=='left' else 'Right'} hand  f{fi_s}")
                _draw_hand(ax, fi_s, side, show_labels=True)

            elif ptype.startswith("left_") or ptype.startswith("right_"):
                side, plane = ptype.split("_", 1)
                ax = _add_ax(fig_sf, 1, ncols, ax_idx, False)
                _draw_hand_projection(ax, fi_s, side, plane, show_labels=True)

            elif ptype == "vroot":  # world-frame wrist positions + virtual root axes
                ax = _add_ax(fig_sf, 1, ncols, ax_idx, True)
                ax.set_title(f"Wrist + VRoot  world@f0 (snapshot f{fi_s})", fontsize=8)

                _VR_AXIS_SCALE  = 0.04
                _WR_AXIS_SCALE  = 0.025
                _AXIS_COLORS    = ["red", "green", "blue"]   # x, y, z
                _WRIST_COL      = ["cyan", "magenta"]
                _WRIST_LBL      = ["Left wrist", "Right wrist"]

                for side, (col, lbl) in enumerate(zip(_WRIST_COL, _WRIST_LBL)):
                    p = wp_s[side]
                    ax.scatter(*p, c=col, s=100, zorder=6, depthshade=False, label=lbl)
                    R = R_wrists_s[side]
                    for j, ac in enumerate(_AXIS_COLORS):
                        ax.quiver(*p, *(R[:, j] * _WR_AXIS_SCALE),
                                  color=ac, arrow_length_ratio=0.4, lw=0.8, alpha=0.7)

                vo = vroot_orig_s
                ax.scatter(*vo, c="gold", s=150, marker="*", zorder=7,
                           depthshade=False, label="VRoot")
                for j, (ac, albl) in enumerate(zip(_AXIS_COLORS, ["VR-X", "VR-Y", "VR-Z"])):
                    ax.quiver(*vo, *(R_vroot_s[:, j] * _VR_AXIS_SCALE),
                              color=ac, arrow_length_ratio=0.3, lw=2.0,
                              label=albl)

                ax.plot([wp_s[0, 0], wp_s[1, 0]],
                        [wp_s[0, 1], wp_s[1, 1]],
                        [wp_s[0, 2], wp_s[1, 2]],
                        c="gray", ls="--", lw=1.0, alpha=0.6)

                ax.set_xlabel("X", fontsize=7)
                ax.set_ylabel("Y", fontsize=7)
                ax.set_zlabel("Z", fontsize=7)
                ax.set_box_aspect([1, 1, 1])
                ax.legend(fontsize=6, loc="upper left")

            else:
                ax = _add_ax(fig_sf, 1, ncols, ax_idx, False)
                ax.set_title(f"{ptype.upper()}  world@f0", fontsize=8)
                _draw_vroot_projection(ax, ptype)

        sf_path = f"{prefix}_single_frame_f{fi_s:04d}.png"
        fig_sf.savefig(sf_path, dpi=120, bbox_inches="tight")
        plt.close(fig_sf)
        print(f"Saved: {sf_path}")
