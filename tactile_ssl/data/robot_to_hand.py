"""
Reverse retargeting: Robot hand qpos -> Human hand 3D skeleton (21 joints, MediaPipe format)

Usage:
    python robot_to_human.py --pickle-path /tmp/revo2_output.pkl --output-path /tmp/human_joints.npy
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.robot_wrapper import RobotWrapper

# MediaPipe 21-joint index reference:
#  0: wrist
#  1-4:  thumb  (MCP, PIP, DIP, tip)
#  5-8:  index  (MCP, PIP, DIP, tip)
#  9-12: middle (MCP, PIP, DIP, tip)
# 13-16: ring   (MCP, PIP, DIP, tip)
# 17-20: pinky  (MCP, PIP, DIP, tip)

# revo2 link -> MediaPipe index mapping (suffix-only, without side prefix)
# Fingers without MCP link use linear interpolation between wrist and proximal.
_LINK_SUFFIX_TO_HUMAN = {
    "base_link": 0,
    # Thumb has metacarpal, so full 4-joint chain is available
    "thumb_metacarpal_link": 1,
    "thumb_proximal_link": 2,
    "thumb_distal_link": 3,
    "thumb_tip_link": 4,
    # Index: proximal=PIP(6), distal=DIP(7), tip(8) — MCP(5) interpolated
    "index_proximal_link": 6,
    "index_distal_link": 7,
    "index_tip_link": 8,
    # Middle
    "middle_proximal_link": 10,
    "middle_distal_link": 11,
    "middle_tip_link": 12,
    # Ring
    "ring_proximal_link": 14,
    "ring_distal_link": 15,
    "ring_tip_link": 16,
    # Pinky
    "pinky_proximal_link": 18,
    "pinky_distal_link": 19,
    "pinky_tip_link": 20,
}

# MCP joints (5,9,13,17) are interpolated: wrist + t*(proximal - wrist)
# t=0.6 places MCP roughly 60% of the way from wrist to proximal link
MCP_INTERP_T = 0.6
_MCP_PROXIMAL_SUFFIX = {
    5: "index_proximal_link",
    9: "middle_proximal_link",
    13: "ring_proximal_link",
    17: "pinky_proximal_link",
}

# Keep original names for backward compatibility
REVO2_LINK_TO_HUMAN = {f"right_{k}": v for k, v in _LINK_SUFFIX_TO_HUMAN.items()}
MCP_PROXIMAL_LINK = {k: f"right_{v}" for k, v in _MCP_PROXIMAL_SUFFIX.items()}

# 6 independent joints stored in the dataset (order matches data["states"][side_ee]["qpos"])
_INDEPENDENT_JOINT_SUFFIXES = [
    "thumb_metacarpal_joint",  # q[0]
    "thumb_proximal_joint",    # q[1]
    "index_proximal_joint",    # q[2]
    "middle_proximal_joint",   # q[3]
    "ring_proximal_joint",     # q[4]
    "pinky_proximal_joint",    # q[5]
]

# Mimic joints: suffix -> (source_suffix, multiplier)
# Matches URDF mimic constraints in revo2_{side}_hand.urdf
_MIMIC_JOINT_SUFFIXES = {
    "thumb_distal_joint":  ("thumb_proximal_joint",  1.0),
    "index_distal_joint":  ("index_proximal_joint",  1.155),
    "middle_distal_joint": ("middle_proximal_joint", 1.155),
    "ring_distal_joint":   ("ring_proximal_joint",   1.155),
    "pinky_distal_joint":  ("pinky_proximal_joint",  1.155),
}


class RobotToHumanRetargeter:
    def __init__(self, config_path: str, side: str = "right"):
        """
        Args:
            config_path: Path to the dex-retargeting YAML config for the robot hand.
            side:        "right" or "left" — determines which link name prefix to use.
        """
        assert side in ("left", "right"), f"side must be 'left' or 'right', got {side!r}"
        self.side = side
        p = f"{side}_"

        self.link_to_human = {f"{p}{k}": v for k, v in _LINK_SUFFIX_TO_HUMAN.items()}
        self.mcp_proximal = {k: f"{p}{v}" for k, v in _MCP_PROXIMAL_SUFFIX.items()}

        retargeting = RetargetingConfig.load_from_file(config_path).build()
        self.robot: RobotWrapper = retargeting.optimizer.robot

        # Pre-cache link IDs
        self.link_ids = {
            name: self.robot.get_link_index(name)
            for name in self.link_to_human
        }

        # Build qpos expansion table: dof_joint_names index -> (source_idx, multiplier)
        # source_idx: index into the 6 independent joints; -1 if it IS an independent joint.
        p = f"{side}_"
        indep_names = [f"{p}{s}" for s in _INDEPENDENT_JOINT_SUFFIXES]
        self._indep_idx = {name: i for i, name in enumerate(indep_names)}

        # For each dof joint, record how to fill it from the 6-element qpos
        # (indep_index, mimic_multiplier); mimic_multiplier=1.0 for independent joints.
        self._qpos_expand = []
        for jname in self.robot.dof_joint_names:
            suffix = jname[len(p):]
            if jname in self._indep_idx:
                self._qpos_expand.append((self._indep_idx[jname], 1.0))
            elif suffix in _MIMIC_JOINT_SUFFIXES:
                src_suffix, mult = _MIMIC_JOINT_SUFFIXES[suffix]
                src_name = f"{p}{src_suffix}"
                self._qpos_expand.append((self._indep_idx[src_name], mult))
            else:
                raise ValueError(f"Unknown joint {jname!r} — not independent or mimic")

    def _expand_qpos(self, qpos_6: np.ndarray) -> np.ndarray:
        """Expand 6 independent joint values to full DOF qpos (11 joints).

        Args:
            qpos_6: (6,) independent joint angles in dataset order.

        Returns:
            qpos_full: (n_dof,) full qpos vector matching robot.dof_joint_names.
        """
        full = np.empty(len(self._qpos_expand), dtype=np.float64)
        for i, (src_idx, mult) in enumerate(self._qpos_expand):
            full[i] = qpos_6[src_idx] * mult
        return full

    def _get_pos(self, link_name: str) -> np.ndarray:
        """Returns (3,) world-frame position of the link."""
        lid = self.link_ids[link_name]
        return self.robot.get_link_pose(lid)[:3, 3]

    def forward(self, qpos: np.ndarray) -> np.ndarray:
        """
        Args:
            qpos: Robot joint positions — either (6,) independent joints (dataset format)
                  or (n_dof,) full qpos matching robot.dof_joint_names.

        Returns:
            joints: (21, 3) human hand joints in robot base frame.
        """
        if len(qpos) == 6:
            qpos = self._expand_qpos(qpos)
        self.robot.compute_forward_kinematics(qpos)

        joints = np.zeros((21, 3))

        # Direct link mappings
        for link_name, human_idx in self.link_to_human.items():
            joints[human_idx] = self._get_pos(link_name)

        # Interpolate MCP joints (5, 9, 13, 17)
        wrist = joints[0]
        for mcp_idx, proximal_link in self.mcp_proximal.items():
            proximal_pos = self._get_pos(proximal_link)
            joints[mcp_idx] = wrist + MCP_INTERP_T * (proximal_pos - wrist)

        return joints

    def batch_forward(self, qpos_seq: list) -> np.ndarray:
        """
        Args:
            qpos_seq: List of qpos arrays, length T.

        Returns:
            joints_seq: (T, 21, 3) human hand joints.
        """
        results = [self.forward(qpos) for qpos in qpos_seq]
        return np.stack(results, axis=0)


def visualize_skeleton(joints_seq: np.ndarray, fps: float = 30.0):
    """Simple matplotlib 3D visualization of the human hand skeleton."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # MediaPipe skeleton connectivity
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # index
        (0, 9), (9, 10), (10, 11), (11, 12),   # middle
        (0, 13), (13, 14), (14, 15), (15, 16), # ring
        (0, 17), (17, 18), (18, 19), (19, 20), # pinky
        (5, 9), (9, 13), (13, 17),             # palm
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Compute axis limits from all frames
    all_pos = joints_seq.reshape(-1, 3)
    margin = 0.02
    xlim = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    zlim = (all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    lines = [ax.plot([], [], [], "b-", lw=2)[0] for _ in connections]
    dots = ax.plot([], [], [], "ro", ms=5)[0]
    title = ax.set_title("")

    def init():
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return lines + [dots]

    def update(frame_idx):
        joints = joints_seq[frame_idx]
        for line, (i, j) in zip(lines, connections):
            xs = [joints[i, 0], joints[j, 0]]
            ys = [joints[i, 1], joints[j, 1]]
            zs = [joints[i, 2], joints[j, 2]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        dots.set_data(joints[:, 0], joints[:, 1])
        dots.set_3d_properties(joints[:, 2])
        title.set_text(f"Frame {frame_idx + 1} / {len(joints_seq)}")
        return lines + [dots, title]

    interval_ms = int(1000 / fps)
    anim = FuncAnimation(
        fig, update, frames=len(joints_seq),
        init_func=init, interval=interval_ms, blit=False
    )
    plt.tight_layout()
    plt.show()
    return anim


def main(
    pickle_path: str,
    config_path: str = "urdf/revo2_right_hand.yml",
    output_path: Optional[str] = None,
    visualize: bool = True,
):
    """
    Converts robot hand qpos sequence to human 3D skeleton (21 joints, MediaPipe format).

    Args:
        pickle_path: Path to .pkl file produced by detect_from_video.py.
        config_path: Path to retargeting config YAML for the robot.
        output_path: If set, saves (T, 21, 3) numpy array as .npy file.
        visualize:  Show matplotlib 3D animation.
    """
    # Load robot qpos sequence
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    qpos_seq = data["data"]
    print(f"Loaded {len(qpos_seq)} frames from {pickle_path}")

    # Build retargeter
    retargeter = RobotToHumanRetargeter(config_path)

    # Run FK for all frames
    joints_seq = retargeter.batch_forward(qpos_seq)
    print(f"Output shape: {joints_seq.shape}")  # (T, 21, 3)

    # Save
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, joints_seq)
        print(f"Saved to {output_path}")

    # Visualize
    if visualize:
        visualize_skeleton(joints_seq)


if __name__ == "__main__":
    tyro.cli(main)
