from typing import Optional
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import pytorch_kinematics as pk
from omegaconf import DictConfig

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.data.force_channels import force_direction_to_cartesian

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



def _build_g1_R_to_world_R(R, side="left"):
    # Visualization code expects row-major axes, so keep transpose.
    return R.T

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
    wrist_R_raw    = np.zeros((num_frames, 2, 3, 3), dtype=np.float32)
    wrist_R_fk     = np.zeros((num_frames, 2, 3, 3), dtype=np.float32)

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

        raw_L = lh_mat[:3, :3].copy()
        raw_R = rh_mat[:3, :3].copy()
        lh_mat = _build_g1_R_to_world_R(raw_L, 'left')
        rh_mat = _build_g1_R_to_world_R(raw_R, 'right')
        wrist_R_raw[i, 0] = lh_mat
        wrist_R_raw[i, 1] = rh_mat
        # URDF fixed joint: rubber_hand → hand base_link (per side)
        R_fxL = np.array([[ 0., 0.,  1.], [-1., 0.,  0.], [ 0., -1., 0.]], dtype=np.float32)
        R_fxR = np.array([[ 0., 0.,  1.], [ 1., 0.,  0.], [ 0.,  1., 0.]], dtype=np.float32)
        
        Rz_cw  = np.array([[ 0.,  1., 0.], [-1.,  0., 0.], [ 0.,  0., 1.]], dtype=np.float32)  # CW  -90°
        Rz_ccw = np.array([[ 0., -1., 0.], [ 1.,  0., 0.], [ 0.,  0., 1.]], dtype=np.float32)  # CCW +90°
        _S     = np.array([[ 0., -1., 0.], [ 1.,  0., 0.], [ 0.,  0., 1.]], dtype=np.float32)
        wrist_R_fk[i, 0] = _S @ (raw_L @ R_fxL @ Rz_cw)  @ _S.T  # left:  CW  -90°
        wrist_R_fk[i, 1] = _S @ (raw_R @ R_fxR @ Rz_ccw) @ _S.T  # right: CCW +90°

        # ── Hand FK → positions relative to base_link (no world transform) ──
        lh_fk = compute_hand_fk(lh_chain, left_ee_array[i],  "left")
        rh_fk = compute_hand_fk(rh_chain, right_ee_array[i], "right")

        for fi, link in enumerate(FINGERTIP_LINKS):
            left_tip = lh_fk[f"left_{link}"].get_matrix()[:, :3, 3].squeeze(0).numpy().copy()
            left_tip[1] *= -1.0
            fingertip_base[i, fi]     = left_tip
            fingertip_base[i, fi + 5] = rh_fk[f"right_{link}"].get_matrix()[:, :3, 3].squeeze(0).numpy()

        for li, link in enumerate(SKELETON_LINKS):
            left_joint = lh_fk[f"left_{link}"].get_matrix()[:, :3, 3].squeeze(0).numpy()
            left_joint[1] *= -1.0
            skeleton_base[i, li]      = left_joint
            skeleton_base[i, li + 22] = rh_fk[f"right_{link}"].get_matrix()[:, :3, 3].squeeze(0).numpy()

    P_SWAP_XY = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=np.float32)
    wrist_pos = np.einsum("ij,nsj->nsi", P_SWAP_XY, wrist_pos)

    return {
        "fingertip_base": fingertip_base,
        "skeleton_base":  skeleton_base,
        "wrist_pos":      wrist_pos,
        "wrist_R":        wrist_R_raw,
        "wrist_R_fk":     wrist_R_fk,
    }


def compute_fk(data_path: Path, urdf_path: Path, frames: list) -> dict:
    result = _compute_fk(urdf_path, frames)
    return result


def compute_wrist_poses_vroot(wrist_pos: np.ndarray, wrist_rot6d: np.ndarray) -> np.ndarray:
    """Convert world-frame wrist pos/rot into virtual-root-relative wrist poses.

    Virtual root origin = midpoint of the two wrists.
    Virtual root axes are aligned with world axes, so each wrist pose is stored as:
      [xyz relative to virtual root, rot6d(local-to-world wrist rotation)]

    Args:
        wrist_pos:   (N, 2, 3)  world positions  [left, right]
        wrist_rot6d: (N, 2, 6)  world 6D rotations [left, right]

    Returns:
        wrist_poses: (N, 2, 9)  vroot-relative [pos(3) + rot6d(6)] per wrist
    """
    root_pos = (wrist_pos[:, 0] + wrist_pos[:, 1]) / 2.0   # (N, 3)
    N = wrist_pos.shape[0]
    wrist_poses = np.zeros((N, 2, 9), dtype=np.float32)
    for side in range(2):
        pos_vr = (wrist_pos[:, side] - root_pos).astype(np.float32)   # (N, 3)
        rot6d = wrist_rot6d[:, side].astype(np.float32)                # (N, 6)
        wrist_poses[:, side] = np.concatenate([pos_vr, rot6d], axis=-1)
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

        robot_to_human:    If True, converts both hand qpos to human 21-joint skeleton
                           (MediaPipe format) via RobotToHumanRetargeter and uses the
                           result as skeleton_poses (W, 42, 3) [left 0-20, right 21-41].
                           Requires dex-retargeting config YAMLs at brainco_urdf_path.
        retargeting_config_path_left:  Path to dex-retargeting YAML for left hand.
                           Defaults to {brainco_urdf_path}/revo2_left_hand.yml.
        retargeting_config_path_right: Path to dex-retargeting YAML for right hand.
                           Defaults to {brainco_urdf_path}/revo2_right_hand.yml.

    Output per sample:
        sensor:       Tensor(W, 10, 4)  — [Fn, Ft*cosθ, Ft*sinθ, proximity]
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
        robot_to_human: bool = False,
        retargeting_config_path_left: Optional[str] = None,
        retargeting_config_path_right: Optional[str] = None,
    ):
        self._load_images = load_images  # reserved for future RGB loading
        # ── Window config ────────────────────────────────────────────────
        self.window_time = config.window_time
        self.interpolating_freq = config.interpolating_freq
        self.num_frames_per_window = int(round(self.window_time * self.interpolating_freq))
        self.max_values = [25000, 25000, 25000, 500000]

        overlap = config.get("window_overlap", 0.0)
        assert 0 <= overlap < 1, "window_overlap must be in [0, 1)"
        self.window_overlap = overlap
        shift = int(round(self.num_frames_per_window * (1.0 - overlap)))
        self.shift_per_window = max(1, shift)

        self.object_label = object_class
        self.joint_poses = joint_poses
        self.robot_to_human = robot_to_human

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

        # ── fingertip_rel: base_link coords are already wrist-local ──────
        # Hand FK output is relative to base_link = wrist frame directly.
        self.fingertip_rel  = fingertip_base               # (N, 10, 3)
        self.fingertip_base = fingertip_base               # (N, 10, 3) — kept for visualization

        # ── skeleton_rel: keep base_link (idx 0/22), skip thumb_metacarpal (idx 1/23) ──
        # skeleton_base: left 0-21 (0=base_link, 1=thumb_metacarpal_link), right 22-43
        skeleton_rel = np.concatenate([
            skeleton_base[:, 0:1],    # left  base_link
            skeleton_base[:, 2:22],   # left  joints 2-21 (skip thumb_metacarpal)
            skeleton_base[:, 22:23],  # right base_link
            skeleton_base[:, 24:44],  # right joints 24-43 (skip thumb_metacarpal)
        ], axis=1)  # (N, 42, 3)
        self.skeleton_rel  = skeleton_rel
        self.skeleton_base = skeleton_base                 # (N, 44, 3) — kept for visualization

        # ── wrist world poses: 6D rotation from wrist_R_fk ──────────────────
        wrist_R_fk_arr = fk["wrist_R_fk"]                        # (N, 2, 3, 3)
        wrist_rot6d = np.concatenate([wrist_R_fk_arr[:, :, 0, :], wrist_R_fk_arr[:, :, 1, :]], axis=-1)  # (N, 2, 6)
        self.wrist_pos_world   = wrist_pos            # (N, 2, 3)
        self.wrist_rot6d_world = wrist_rot6d          # (N, 2, 6)
        self.wrist_R_fk        = wrist_R_fk_arr       # (N, 2, 3, 3) raw rubber_hand rotation in vis world

        # ── wrist_poses: position + orientation in vroot frame ───────────
        self.wrist_poses = compute_wrist_poses_vroot(wrist_pos, wrist_rot6d)  # (N, 2, 9)

        log.info(f"  fingertip_rel shape:  {self.fingertip_rel.shape}")
        log.info(f"  skeleton_rel shape:   {self.skeleton_rel.shape}")
        log.info(f"  wrist_poses shape:    {self.wrist_poses.shape}")

        # ── Robot → Human skeleton retargeting ───────────────────────────
        if robot_to_human:
            from tactile_ssl.data.robot_to_hand import RobotToHumanRetargeter
            urdf_root = Path(brainco_urdf_path)
            cfg_left  = retargeting_config_path_left  or str(urdf_root / "revo2_left_hand.yml")
            cfg_right = retargeting_config_path_right or str(urdf_root / "revo2_right_hand.yml")

            log.info(f"  Building RobotToHumanRetargeter (left:  {cfg_left})")
            left_retargeter  = RobotToHumanRetargeter(cfg_left,  side="left")
            log.info(f"  Building RobotToHumanRetargeter (right: {cfg_right})")
            right_retargeter = RobotToHumanRetargeter(cfg_right, side="right")

            left_ee_array  = np.array([f["states"]["left_ee"]["qpos"]  for f in frames], dtype=np.float32)
            right_ee_array = np.array([f["states"]["right_ee"]["qpos"] for f in frames], dtype=np.float32)

            log.info(f"  Running human retargeting for {self.num_frames} frames ...")
            human_left  = left_retargeter.batch_forward(list(left_ee_array))   # (N, 21, 3)
            human_right = right_retargeter.batch_forward(list(right_ee_array)) # (N, 21, 3)

            # Mirror left hand: negate y-axis so +y points pinky→thumb,
            # matching the sign convention applied in _compute_fk (left_joint[1] *= -1).
            human_left = human_left.copy()
            human_left[:, :, 1] *= -1.0

            # Concatenate: left joints 0-20, right joints 21-41 → (N, 42, 3)
            self.human_skeleton = np.concatenate(
                [human_left, human_right], axis=1
            ).astype(np.float32)
            log.info(f"  human_skeleton shape: {self.human_skeleton.shape}")
        else:
            self.human_skeleton = None

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
        self.tactile_array, self.tactile_valid_array = (
            force_direction_to_cartesian(self.tactile_array)
        )

        log.info(f"tactile_array shape: {self.tactile_array.shape}")

        # ── Window indices ───────────────────────────────────────────────
        max_start = self.num_frames - self.num_frames_per_window
        self.data_idxs = np.arange(0, max(1, max_start + 1), self.shift_per_window)
        log.info(f"  Windows: {len(self.data_idxs)}, frames_per_window: {self.num_frames_per_window}")

    def __len__(self) -> int:
        return len(self.data_idxs)


    def __getitem__(self, idx: int) -> dict:
        start = self.data_idxs[idx]
        end = start + self.num_frames_per_window

        sensor = torch.from_numpy(self.tactile_array[start:end].copy())  # (W, 10, 4)
        sensor = sensor / torch.tensor(self.max_values).view(1, 1, -1)  # fixed channel-max scaling
        fingertip_poses = torch.from_numpy(self.fingertip_rel[start:end].copy())  # (W, 10, 3)
        if self.human_skeleton is not None:
            skeleton_poses = torch.from_numpy(self.human_skeleton[start:end].copy())  # (W, 42, 3)
        else:
            skeleton_poses = torch.from_numpy(self.skeleton_rel[start:end].copy())  # (W, 42, 3)
        frame_indices = torch.arange(start, end, dtype=torch.long)

        if self.joint_poses:
            sensor_poses = skeleton_poses
        else:
            sensor_poses = fingertip_poses

        wrist_poses = torch.from_numpy(self.wrist_poses[start:end].copy())  # (W, 2, 9)

        sample = {
            "sensor":       sensor,
            "sensor_poses": sensor_poses,
            "sensor_id": torch.tensor([0]),
            "fingertip_poses": fingertip_poses,
            "skeleton_poses": skeleton_poses,
            "wrist_poses":  wrist_poses,
            "frame_indices": frame_indices,
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
    parser.add_argument("--object", "-o", default='doll')
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
    null_ratio = (~dataset.tactile_valid_array[..., 1]).mean()
    print(f"  tangential null ratio: {null_ratio:.1%}")
    wrist_t = dataset.wrist_poses[:, :, :3]
    d = np.linalg.norm(wrist_t[:, 0] - wrist_t[:, 1], axis=-1)
    print(f"  wrist-wrist dist : min={d.min():.3f}  mean={d.mean():.3f}  max={d.max():.3f} m\n")

    # ═══════════════════════════════════════════════════════════════════════
    # VR-root frame visualization  (style adapted from brainco_tactile_prev.py)
    # VR root origin = midpoint of two wrists; axes = world axes (xy-swapped).
    # fingertip / skeleton positions converted to world in main using wrist_R_raw.
    # ═══════════════════════════════════════════════════════════════════════
    def _rot6d_rows_to_mat(rot6d: np.ndarray) -> np.ndarray:
        """Convert row-major wrist rot6d ([R[0,:], R[1,:]]) to rotation matrices."""
        r0 = rot6d[..., :3]
        r1 = rot6d[..., 3:]
        r0 = r0 / (np.linalg.norm(r0, axis=-1, keepdims=True) + 1e-8)
        r1 = r1 - (r1 * r0).sum(axis=-1, keepdims=True) * r0
        r1 = r1 / (np.linalg.norm(r1, axis=-1, keepdims=True) + 1e-8)
        r2 = np.cross(r0, r1)
        return np.stack([r0, r1, r2], axis=-2)


    N = dataset.num_frames
    _margin = 0.02
    def _set_lim_3d(ax):
        _max_rng = getattr(ax, "_vr_max_rng", 0.2)
        ax.set_xlim(-_max_rng, _max_rng)
        ax.set_ylim(-_max_rng, _max_rng)
        ax.set_zlim(-_max_rng, _max_rng)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    def _set_lim_2d(ax, i0, i1):
        _max_rng = getattr(ax, "_vr_max_rng", 0.2)
        ax.set_xlim(-_max_rng, _max_rng)
        ax.set_ylim(-_max_rng, _max_rng)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    def _draw_frame(axes_3d, axes_2d, sample: dict, frame_idx: int):
        """Draw one frame by inverting wrist_poses 9D back into world coordinates."""
        fi = int(sample["frame_indices"][frame_idx].item())
        wrist_poses = sample["wrist_poses"][frame_idx].cpu().numpy()
        fingertip_local = sample["fingertip_poses"][frame_idx].cpu().numpy()
        skeleton_local = sample["skeleton_poses"][frame_idx].cpu().numpy()
        tactile_frame = sample["sensor"][frame_idx].cpu().numpy()

        wrists_vr = wrist_poses[:, :3].astype(np.float32)
        root_world = ((dataset.wrist_pos_world[fi, 0] + dataset.wrist_pos_world[fi, 1]) * 0.5).astype(np.float32)
        wrists_world = wrists_vr + root_world[None]
        wrist_R = _rot6d_to_mat(wrist_poses[:, 3:]).astype(np.float32)

        left_sk_local = np.concatenate([np.zeros((1, 3), dtype=np.float32), skeleton_local[:21]], axis=0)
        right_sk_local = np.concatenate([np.zeros((1, 3), dtype=np.float32), skeleton_local[21:]], axis=0)
        left_sk_local[:, 0] *= -1.0
        fingertip_local[:5, 0] *= -1.0

        left_fp = wrists_world[0] + (wrist_R[0] @ fingertip_local[:5].T).T
        right_fp = wrists_world[1] + (wrist_R[1] @ fingertip_local[5:].T).T
        left_sk = wrists_world[0] + (wrist_R[0] @ left_sk_local.T).T
        right_sk = wrists_world[1] + (wrist_R[1] @ right_sk_local.T).T

        frame = {
            "wrists_vr": wrists_vr,
            "wrists_world": wrists_world,
            "virtual_root_world": root_world,
            "wrist_R": wrist_R,
            "wrist_rot6d": wrist_poses[:, 3:].astype(np.float32),
            "fingertips": np.concatenate([left_fp, right_fp], axis=0),
            "skeleton": np.concatenate([left_sk, right_sk], axis=0),
            "tactile": tactile_frame,
        }

        fp_w = frame["fingertips"]
        sk_w = frame["skeleton"]
        wp = frame["wrists_world"]
        tactile = frame["tactile"]
        root = frame["virtual_root_world"]

        fp = fp_w
        sk = sk_w
        wrists = wp
        tac = tactile        # (10, 4)
        fixed_min = -0.3
        fixed_max = 0.7
        fixed_elev = 20.0
        fixed_azim = -60.0

        planes = [("xy",(0,1),("X","Y")), ("xz",(0,2),("X","Z")), ("yz",(1,2),("Y","Z"))]

        for ax, (_, (i0, i1), (xl, yl)) in zip(axes_2d, planes):
            ax.cla()
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(fixed_min, fixed_max)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel(xl); ax.set_ylabel(yl)

        axes_3d.cla()
        axes_3d.set_xlim(fixed_min, fixed_max)
        axes_3d.set_ylim(fixed_min, fixed_max)
        axes_3d.set_zlim(fixed_min, fixed_max)
        axes_3d.set_box_aspect([1, 1, 1])
        axes_3d.view_init(elev=fixed_elev, azim=fixed_azim)
        axes_3d.set_xlabel("X"); axes_3d.set_ylabel("Y"); axes_3d.set_zlabel("Z")
        axes_3d.set_title(f"World From wrist_poses  f{fi}", fontsize=8)

        # Skeleton lines
        left_sk, right_sk = sk[:22], sk[22:]
        for i, j in SKELETON_LINES:
            axes_3d.plot([left_sk[i,0],  left_sk[j,0]],
                         [left_sk[i,1],  left_sk[j,1]],
                         [left_sk[i,2],  left_sk[j,2]],  c="blue",  alpha=0.5, lw=2)
            axes_3d.plot([right_sk[i,0], right_sk[j,0]],
                         [right_sk[i,1], right_sk[j,1]],
                         [right_sk[i,2], right_sk[j,2]], c="red",   alpha=0.5, lw=2)
            for ax, (_, (i0,i1), _) in zip(axes_2d, planes):
                ax.plot([left_sk[i,i0],  left_sk[j,i0]],
                        [left_sk[i,i1],  left_sk[j,i1]],  c="blue",  alpha=0.5, lw=1.5)
                ax.plot([right_sk[i,i0], right_sk[j,i0]],
                        [right_sk[i,i1], right_sk[j,i1]], c="red",   alpha=0.5, lw=1.5)

        # Fingertips (size scaled by normal force)
        left_fp,  right_fp  = fp[:5],  fp[5:]
        left_tac, right_tac = tac[:5], tac[5:]
        left_sz  = 20 + np.clip(left_tac[:,  0], 0, None) * 0.002
        right_sz = 20 + np.clip(right_tac[:, 0], 0, None) * 0.002
        axes_3d.scatter(left_fp[:,0],  left_fp[:,1],  left_fp[:,2],
                        c="blue",    s=left_sz,  marker="o", label="Left fingers",  zorder=5)
        axes_3d.scatter(right_fp[:,0], right_fp[:,1], right_fp[:,2],
                        c="red",     s=right_sz, marker="o", label="Right fingers", zorder=5)
        for ax, (_, (i0,i1), _) in zip(axes_2d, planes):
            ax.scatter(left_fp[:,i0],  left_fp[:,i1],  c="blue",  s=left_sz,  marker="o", zorder=5)
            ax.scatter(right_fp[:,i0], right_fp[:,i1], c="red",   s=right_sz, marker="o", zorder=5)

        # Labels for fingertips
        for li, name in enumerate(FINGER_NAMES):
            axes_3d.text(*left_fp[li],  f" L_{name}", fontsize=7, color="blue")
            axes_3d.text(*right_fp[li], f" R_{name}", fontsize=7, color="red")

        # Wrist coordinate axes reconstructed from wrist_poses 9D.
        R_wax = frame["wrist_R"]
        _ql = 0.1   # quiver length (metres)
        _ax_colors = ["r", "g", "b"]   # x=red, y=green, z=blue
        for side, (w_pos, side_R) in enumerate(zip(wrists, R_wax)):
            for ax_idx in range(3):
                col = _ax_colors[ax_idx]
                d   = side_R[:, ax_idx] * _ql   # direction vector (world)
                axes_3d.quiver(*w_pos, *d, color=col, linewidth=1.5, arrow_length_ratio=0.3)
                for ax2d, (_, (i0, i1), _) in zip(axes_2d, planes):
                    ax2d.annotate("", xy=(w_pos[i0]+d[i0], w_pos[i1]+d[i1]),
                                  xytext=(w_pos[i0], w_pos[i1]),
                                  arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

        # Wrists
        axes_3d.scatter(*wrists[0], c="cyan",    s=120, marker="s", label="Left wrist",  edgecolors="black", zorder=6)
        axes_3d.scatter(*wrists[1], c="magenta", s=120, marker="s", label="Right wrist", edgecolors="black", zorder=6)
        for ax, (_, (i0,i1), _) in zip(axes_2d, planes):
            ax.scatter(wrists[0,i0], wrists[0,i1], c="cyan",    s=80, marker="s", edgecolors="black", zorder=6)
            ax.scatter(wrists[1,i0], wrists[1,i1], c="magenta", s=80, marker="s", edgecolors="black", zorder=6)

        # VR root origin
        axes_3d.scatter(*root, c="gold", s=150, marker="*", zorder=7, label="VR root")
        for ax, (_, (i0,i1), _) in zip(axes_2d, planes):
            ax.scatter(root[i0], root[i1], c="gold", s=100, marker="*", zorder=7)
        for axis_idx, col in enumerate(_ax_colors):
            root_axis = np.zeros(3, dtype=np.float32)
            root_axis[axis_idx] = _ql
            axes_3d.quiver(*root, *root_axis, color=col, linewidth=2.0, arrow_length_ratio=0.3)
            for ax2d, (_, (i0, i1), _) in zip(axes_2d, planes):
                ax2d.annotate(
                    "",
                    xy=(root[i0] + root_axis[i0], root[i1] + root_axis[i1]),
                    xytext=(root[i0], root[i1]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4),
                )

        # Wrist-to-base_link lines
        axes_3d.plot([wrists[0,0], left_sk[0,0]],  [wrists[0,1], left_sk[0,1]],
                     [wrists[0,2], left_sk[0,2]],  c="cyan",    alpha=0.6, lw=2, ls="--")
        axes_3d.plot([wrists[1,0], right_sk[0,0]], [wrists[1,1], right_sk[0,1]],
                     [wrists[1,2], right_sk[0,2]], c="magenta", alpha=0.6, lw=2, ls="--")
        axes_3d.legend(loc="upper left", fontsize=7)
        return frame

    # ── RGB loader ──────────────────────────────────────────────────────
    import cv2
    _img_files  = None
    _video_cap  = None
    _vid_total  = 0
    colors_dir  = Path(ep_path) / "colors"
    if colors_dir.exists():
        _img_files = sorted(f for f in colors_dir.iterdir()
                            if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        _vid_total = len(_img_files)
    else:
        _vp = Path(ep_path) / "colors.mp4"
        if _vp.exists():
            _video_cap = cv2.VideoCapture(str(_vp))
            _vid_total = int(_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _load_rgb(fi: int):
        if _vid_total == 0:
            return None
        vid_idx = min(int(round(fi * _vid_total / N)), _vid_total - 1)
        if _img_files is not None:
            bgr = cv2.imread(str(_img_files[vid_idx]))
        else:
            _video_cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
            ret, bgr = _video_cap.read()
            if not ret:
                return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

    def _draw_local_hand_axes(axes_3d, axes_2d, sample: dict, frame_idx: int):
        """Draw left/right hands in wrist-local coordinates with local axes."""
        fi = int(sample["frame_indices"][frame_idx].item())
        skeleton_local = sample["skeleton_poses"][frame_idx].cpu().numpy()
        fingertip_local = sample["fingertip_poses"][frame_idx].cpu().numpy()
        tac = sample["sensor"][frame_idx].cpu().numpy()
        sk = np.concatenate([
            np.zeros((1, 3), dtype=np.float32),
            skeleton_local[:21],
            np.zeros((1, 3), dtype=np.float32),
            skeleton_local[21:],
        ], axis=0)
        fp = fingertip_local
        planes = [("xy", (0, 1), ("X", "Y")), ("xz", (0, 2), ("X", "Z")), ("yz", (1, 2), ("Y", "Z"))]
        offsets = {
            "left": np.array([-0.12, 0.0, 0.0], dtype=np.float32),
            "right": np.array([0.12, 0.0, 0.0], dtype=np.float32),
        }
        axis_colors = ["r", "g", "b"]
        axis_len = 0.04

        for ax, (_, (i0, i1), (xl, yl)) in zip(axes_2d, planes):
            ax.cla()
            ax.set_xlim(-0.22, 0.22)
            ax.set_ylim(-0.12, 0.12)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)

        axes_3d.cla()
        axes_3d.set_xlim(-0.22, 0.22)
        axes_3d.set_ylim(-0.12, 0.12)
        axes_3d.set_zlim(-0.12, 0.12)
        axes_3d.set_box_aspect([1, 1, 1])
        axes_3d.set_xlabel("X")
        axes_3d.set_ylabel("Y")
        axes_3d.set_zlabel("Z")
        axes_3d.set_title(f"Row 1: Hand Local Axes  f{fi}", fontsize=9)

        for side, color, wrist_color in [("left", "blue", "cyan"), ("right", "red", "magenta")]:
            if side == "left":
                sk_side = sk[:22].copy()
                fp_side = fp[:5].copy()
                tac_side = tac[:5]
            else:
                sk_side = sk[22:].copy()
                fp_side = fp[5:].copy()
                tac_side = tac[5:]

            offset = offsets[side]
            sk_side = sk_side + offset
            fp_side = fp_side + offset
            wrist = offset

            for i, j in SKELETON_LINES:
                axes_3d.plot([sk_side[i, 0], sk_side[j, 0]],
                             [sk_side[i, 1], sk_side[j, 1]],
                             [sk_side[i, 2], sk_side[j, 2]], c=color, alpha=0.6, lw=2)
                for ax, (_, (i0, i1), _) in zip(axes_2d, planes):
                    ax.plot([sk_side[i, i0], sk_side[j, i0]],
                            [sk_side[i, i1], sk_side[j, i1]], c=color, alpha=0.6, lw=1.5)

            fp_sizes = 20 + np.clip(tac_side[:, 0], 0, None) * 0.002
            axes_3d.scatter(fp_side[:, 0], fp_side[:, 1], fp_side[:, 2],
                            c=color, s=fp_sizes, marker="o", zorder=5, label=f"{side} fingers")
            for ax, (_, (i0, i1), _) in zip(axes_2d, planes):
                ax.scatter(fp_side[:, i0], fp_side[:, i1], c=color, s=fp_sizes, marker="o", zorder=5)

            axes_3d.scatter(*wrist, c=wrist_color, s=120, marker="s",
                            edgecolors="black", zorder=6, label=f"{side} wrist")
            for ax, (_, (i0, i1), _) in zip(axes_2d, planes):
                ax.scatter(wrist[i0], wrist[i1], c=wrist_color, s=80, marker="s",
                           edgecolors="black", zorder=6)

            for axis_idx, axis_color in enumerate(axis_colors):
                axis_vec = np.zeros(3, dtype=np.float32)
                axis_vec[axis_idx] = axis_len
                axes_3d.quiver(*wrist, *axis_vec, color=axis_color, linewidth=1.5, arrow_length_ratio=0.3)
                for ax, (_, (i0, i1), _) in zip(axes_2d, planes):
                    ax.annotate(
                        "",
                        xy=(wrist[i0] + axis_vec[i0], wrist[i1] + axis_vec[i1]),
                        xytext=(wrist[i0], wrist[i1]),
                        arrowprops=dict(arrowstyle="->", color=axis_color, lw=1.2),
                    )

        axes_3d.legend(loc="upper left", fontsize=7)

    # ── Generate sample frames ──────────────────────────────────────────
    sample_indices = np.unique(np.linspace(0, len(dataset) - 1, min(5, len(dataset)), dtype=int))

    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        mid    = sample["sensor"].shape[0] // 2
        fi     = int(sample["frame_indices"][mid].item())
        rgb    = _load_rgb(fi)
        local_sensor_pose = sample["sensor_poses"][mid].cpu().numpy()
        wrist_pose = sample["wrist_poses"][mid].cpu().numpy()

        ncols = 5 if rgb is not None else 4
        fig = plt.figure(figsize=(4.8 * ncols, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(3, ncols, height_ratios=[1.0, 1.0, 0.65])
        fig.suptitle(f"{obj}/{ep}  sample {sample_idx}  frame {fi}", fontsize=10)

        if rgb is not None:
            ax_rgb_top = fig.add_subplot(gs[0, 0])
            ax_rgb_top.imshow(rgb)
            ax_rgb_top.axis("off")
            ax_rgb_top.set_title(f"RGB  f{fi}", fontsize=9)
            ax_rgb_bot = fig.add_subplot(gs[1, 0])
            ax_rgb_bot.imshow(rgb)
            ax_rgb_bot.axis("off")
            ax_rgb_bot.set_title(f"RGB  f{fi}", fontsize=9)
            off = 1
        else:
            off = 0

        ax3d_local = fig.add_subplot(gs[0, off + 0], projection="3d")
        ax_xy_local = fig.add_subplot(gs[0, off + 1]); ax_xy_local.set_title("XY")
        ax_xz_local = fig.add_subplot(gs[0, off + 2]); ax_xz_local.set_title("XZ")
        ax_yz_local = fig.add_subplot(gs[0, off + 3]); ax_yz_local.set_title("YZ")
        _draw_local_hand_axes(ax3d_local, [ax_xy_local, ax_xz_local, ax_yz_local], sample, mid)

        ax3d = fig.add_subplot(gs[1, off + 0], projection="3d")
        ax_xy = fig.add_subplot(gs[1, off + 1]); ax_xy.set_title("XY")
        ax_xz = fig.add_subplot(gs[1, off + 2]); ax_xz.set_title("XZ")
        ax_yz = fig.add_subplot(gs[1, off + 3]); ax_yz.set_title("YZ")
        ax3d.set_title(f"Row 2: VR-Root Frame  f{fi}", fontsize=9)
        frame = _draw_frame(ax3d, [ax_xy, ax_xz, ax_yz], sample, mid)

        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis("off")

        def _fmt_vec(name: str, value: np.ndarray) -> str:
            return f"{name}: [{', '.join(f'{float(v): .4f}' for v in value)}]"

        info_lines = [
            f"Row 3: Coordinates  sample={sample_idx}  frame={fi}",
            _fmt_vec("left_wrist_xyz_vr", wrist_pose[0, :3]),
            _fmt_vec("left_wrist_rot6d", wrist_pose[0, 3:]),
            _fmt_vec("right_wrist_xyz_vr", wrist_pose[1, :3]),
            _fmt_vec("right_wrist_rot6d", wrist_pose[1, 3:]),
            _fmt_vec("left_wrist_world", frame["wrists_world"][0]),
            _fmt_vec("right_wrist_world", frame["wrists_world"][1]),
            _fmt_vec("virtual_root_world", frame["virtual_root_world"]),
        ]
        if local_sensor_pose.shape[0] == 10:
            for tip_idx, tip in enumerate(local_sensor_pose):
                info_lines.append(_fmt_vec(f"local_tip_{tip_idx:02d}", tip))
            for tip_idx, tip in enumerate(frame["fingertips"]):
                info_lines.append(_fmt_vec(f"world_tip_{tip_idx:02d}", tip))
        else:
            info_lines.append(
                f"sensor_poses has shape {tuple(local_sensor_pose.shape)} "
                f"(joint_poses={dataset.joint_poses}), so showing sensor_poses rows directly."
            )
            for pose_idx, pose in enumerate(local_sensor_pose):
                info_lines.append(_fmt_vec(f"local_pose_{pose_idx:02d}", pose))
        ax_text.text(
            0.01,
            0.98,
            "\n".join(info_lines),
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            transform=ax_text.transAxes,
        )

        out_path = f"{prefix}_sample_{sample_idx:04d}_world.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
