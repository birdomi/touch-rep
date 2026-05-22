"""test_visualization_angle.py

Visualize BraincoAngleTactileDataset samples:
  - Panel 0: RGB frame  (colors/ or colors.mp4)
  - Panel 1: Tactile contact heatmap  (10 sensors × 4 channels)
  - Panel 2: Abduction  θ_abd  (top-down stylized)
  - Panel 3: Flexion    θ_MCP / θ_PIP / θ_DIP  (side-view per finger)

Usage:
    # single episode
    python test_visualization_angle.py \\
        --data_path dataset/brainco/pretraining/doll/episode_0000

    # custom window config
    python test_visualization_angle.py \\
        --data_path dataset/brainco/pretraining/doll/episode_0000 \\
        --window_time 0.1 --window_overlap 0.5 --n_samples 6

    # save to custom dir
    python test_visualization_angle.py \\
        --data_path dataset/brainco/pretraining/doll/episode_0000 \\
        --out visualization_angle
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from tactile_ssl.data.brainco_angle_tactile import BraincoAngleTactileDataset

# ── 색상 상수 ──────────────────────────────────────────────────────────────────

FINGER_NAMES  = ("thumb", "index", "middle", "ring", "pinky")
FINGER_COLORS = {
    "thumb":  "#ff6b6b",
    "index":  "#ffd166",
    "middle": "#06d6a0",
    "ring":   "#4ecdc4",
    "pinky":  "#a78bfa",
}
LH_COLOR = "#3dd68c"
RH_COLOR = "#38bdf8"

CH_NAMES  = ["Normal", "Tangential", "Depth", "Proximity"]
CH_COLORS = ["#ff6b6b", "#ffd166", "#06d6a0", "#a78bfa"]


# ── RGB 로더 ──────────────────────────────────────────────────────────────────

class _RGBLoader:
    """Load RGB frames from `colors/` image folder or `colors.mp4`."""

    def __init__(self, data_path: str):
        colors_dir = Path(data_path) / "colors"
        video_path = Path(data_path) / "colors.mp4"

        self._img_files = None
        self._cap       = None
        self._n         = 0

        if colors_dir.exists():
            self._img_files = sorted(
                p for p in colors_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            self._n = len(self._img_files)
        elif video_path.exists():
            self._cap = cv2.VideoCapture(str(video_path))
            self._n   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __bool__(self):
        return self._n > 0

    def load(self, tactile_frame_idx: int, total_tactile_frames: int) -> Optional[np.ndarray]:
        """Return (H, W, 3) uint8 RGB array, or None if unavailable."""
        if self._n == 0:
            return None
        vid_idx = min(
            int(round(tactile_frame_idx * self._n / max(total_tactile_frames, 1))),
            self._n - 1,
        )
        if self._img_files is not None:
            bgr = cv2.imread(str(self._img_files[vid_idx]))
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
            ret, bgr = self._cap.read()
            if not ret:
                return None
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._cap is not None:
            self._cap.release()


# ── Panel 0: RGB ──────────────────────────────────────────────────────────────

def draw_rgb_panel(ax, rgb: Optional[np.ndarray], frame_idx: int, data_path: str):
    ax.set_facecolor("#0d0d0d")
    ax.axis("off")
    if rgb is not None:
        ax.imshow(rgb, aspect="auto")
    else:
        ax.text(0.5, 0.5, "RGB N/A", color="#555555",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title(f"RGB  frame {frame_idx}", color="#cccccc", fontsize=8, pad=3)
    ax.text(0.01, 0.99,
            Path(data_path).name,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=6, color="#ffffff", family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000",
                      alpha=0.5, edgecolor="none"))


# ── 손가락 기하 파라미터 ───────────────────────────────────────────────────────

_META_LEN = {"thumb": 0.35, "index": 0.50, "middle": 0.52, "ring": 0.48, "pinky": 0.40}
_PROX_LEN = {"thumb": 0.32, "index": 0.40, "middle": 0.42, "ring": 0.38, "pinky": 0.30}
_MID_LEN  = {"thumb": 0.22, "index": 0.28, "middle": 0.30, "ring": 0.27, "pinky": 0.22}
_DIST_LEN = {"thumb": 0.18, "index": 0.20, "middle": 0.20, "ring": 0.18, "pinky": 0.15}

_MCP_YZ = {
    "thumb":  np.array([ 0.28,  0.22]),
    "index":  np.array([ 0.16,  0.50]),
    "middle": np.array([ 0.00,  0.52]),
    "ring":   np.array([-0.13,  0.50]),
    "pinky":  np.array([-0.24,  0.42]),
}
_PALM_POLY_RH = np.array([
    [ 0.00, -0.08], [-0.25, -0.03], [-0.28,  0.45],
    [-0.13,  0.52], [ 0.00,  0.54], [ 0.16,  0.52],
    [ 0.22,  0.10], [ 0.10, -0.06],
])


# ── 기하 함수 ──────────────────────────────────────────────────────────────────

def _abd_points(fname: str, theta_abd: float, flip_y: bool = False):
    mcp = _MCP_YZ[fname].copy()
    if flip_y:
        mcp[0] *= -1
        theta_abd = -theta_abd
    meta_dir   = mcp / (np.linalg.norm(mcp) + 1e-8)
    meta_angle = np.arctan2(meta_dir[0], meta_dir[1])
    prox_angle = meta_angle + theta_abd
    prox_dir   = np.array([np.sin(prox_angle), np.cos(prox_angle)])
    pip = mcp + _PROX_LEN[fname] * prox_dir
    tip = pip + 0.10 * prox_dir
    return np.zeros(2), mcp, pip, tip


def _flex_points(fname: str, mcp_flex: float, pip_flex: float, dip_flex: float):
    wrist    = np.zeros(2)
    MCP      = wrist + np.array([0.0, _META_LEN[fname]])
    d_prox   = np.array([np.sin(mcp_flex),            np.cos(mcp_flex)])
    PIP      = MCP + _PROX_LEN[fname] * d_prox
    ang_mid  = mcp_flex + pip_flex
    d_mid    = np.array([np.sin(ang_mid),              np.cos(ang_mid)])
    DIP      = PIP + _MID_LEN[fname] * d_mid
    ang_dist = ang_mid + dip_flex
    d_dist   = np.array([np.sin(ang_dist),             np.cos(ang_dist)])
    tip      = DIP + _DIST_LEN[fname] * d_dist
    return wrist, MCP, PIP, DIP, tip


# ── ax 스타일 헬퍼 ─────────────────────────────────────────────────────────────

def _style_ax(ax, title: str = "", equal_aspect: bool = False):
    ax.set_facecolor("#0d0d0d")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a2a")
    ax.tick_params(colors="#444444", labelsize=6)
    ax.grid(color="#181818", linewidth=0.5)
    if title:
        ax.set_title(title, color="#cccccc", fontsize=8, pad=4)


# ── Panel 1: Tactile heatmap ──────────────────────────────────────────────────

def draw_tactile_panel(ax, contact: np.ndarray):
    """contact : (10, 4)  — 10 sensors × 4 channels, values in [0,1] (-1=invalid)."""
    _style_ax(ax, "Tactile contact  (10 sensors × 4 ch)")

    data = contact.copy()   # (10, 4)
    invalid = data < 0

    # heatmap
    im = ax.imshow(
        np.ma.masked_where(invalid, data).T,
        aspect="auto", vmin=0.0, vmax=1.0,
        cmap="hot", origin="lower",
        extent=[-0.5, 9.5, -0.5, 3.5],
    )
    # grey for invalid
    grey = np.ma.masked_where(~invalid, np.zeros_like(data))
    ax.imshow(
        grey.T, aspect="auto", vmin=0, vmax=1,
        cmap="Greys", alpha=0.6, origin="lower",
        extent=[-0.5, 9.5, -0.5, 3.5],
    )

    ax.set_xticks(range(10))
    ax.set_xticklabels(
        [f"L{n}" for n in FINGER_NAMES] + [f"R{n}" for n in FINGER_NAMES],
        rotation=35, ha="right", fontsize=6, color="#aaaaaa",
    )
    ax.set_yticks(range(4))
    ax.set_yticklabels(CH_NAMES, fontsize=6, color="#aaaaaa")

    # vertical separator between lh / rh
    ax.axvline(4.5, color="#555555", lw=1.2, ls="--", alpha=0.7)
    ax.text(2.0, 3.7, "Left hand",  color=LH_COLOR, fontsize=7, ha="center", va="bottom")
    ax.text(7.0, 3.7, "Right hand", color=RH_COLOR, fontsize=7, ha="center", va="bottom")

    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                 label="normalized [0, 1]")

    # value annotations
    for si in range(10):
        for ci in range(4):
            v = data[si, ci]
            if v < 0:
                txt, col = "inv", "#555555"
            else:
                txt, col = f"{v:.2f}", "#ffffff" if v < 0.5 else "#000000"
            ax.text(si, ci, txt, ha="center", va="center",
                    fontsize=4.5, color=col)


# ── Panel 2: Abduction ────────────────────────────────────────────────────────

def draw_abduction_panel(ax, lh_angles: np.ndarray, rh_angles: np.ndarray):
    """lh_angles / rh_angles : (5, 4)  — ch0 = abduction."""
    _style_ax(ax, "Abduction  θ_abd  (top-down)", equal_aspect=True)
    ax.set_xlabel("← Left hand  |  Right hand →", color="#555555", fontsize=6)
    ax.set_ylabel("Proximal direction", color="#555555", fontsize=6)

    SEP = 1.5
    for angles, is_left in [(lh_angles, True), (rh_angles, False)]:
        flip_y  = not is_left
        offset  = np.array([-SEP / 2, 0.0]) if is_left else np.array([SEP / 2, 0.0])
        h_color = LH_COLOR if is_left else RH_COLOR
        h_label = "Left" if is_left else "Right"

        # palm polygon
        poly = _PALM_POLY_RH.copy()
        if flip_y:
            poly[:, 0] *= -1
        ax.add_patch(mpatches.Polygon(
            poly + offset, closed=True,
            facecolor=h_color, alpha=0.07, edgecolor=h_color, linewidth=0.8,
        ))
        ax.scatter(*offset, color=h_color, s=60, zorder=7, marker="s", alpha=0.9)
        ax.text(offset[0], offset[1] - 0.13, h_label,
                color=h_color, ha="center", va="top", fontsize=7, fontweight="bold")

        for fi, fname in enumerate(FINGER_NAMES):
            theta_abd = float(angles[fi, 0])
            color     = FINGER_COLORS[fname]
            wrist, mcp, pip, tip = _abd_points(fname, theta_abd, flip_y=flip_y)
            pts = np.array([wrist, mcp, pip, tip]) + offset

            ax.plot(pts[[0, 1], 0], pts[[0, 1], 1], color=color, lw=1.4, alpha=0.35)
            ax.plot(pts[[1, 2], 0], pts[[1, 2], 1], color=color, lw=2.8, alpha=0.95,
                    solid_capstyle="round")
            ax.plot(pts[[2, 3], 0], pts[[2, 3], 1], color=color, lw=1.2, alpha=0.45)
            ax.scatter(*pts[1], color=color, s=22, zorder=6)
            ax.scatter(*pts[2], color=color, s=12, zorder=6)

            prox_mid = (pts[1] + pts[2]) / 2
            ax.text(prox_mid[0], prox_mid[1], f"{np.degrees(theta_abd):.1f}°",
                    color=color, fontsize=5, ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#0d0d0d",
                              alpha=0.65, edgecolor="none"))

    legend_elems = (
        [mpatches.Patch(color=FINGER_COLORS[f], label=f) for f in FINGER_NAMES]
        + [mpatches.Patch(color=LH_COLOR, alpha=0.5, label="Left"),
           mpatches.Patch(color=RH_COLOR, alpha=0.5, label="Right")]
    )
    ax.legend(handles=legend_elems, loc="lower center", ncol=4,
              facecolor="#141414", labelcolor="white", edgecolor="#333333",
              fontsize=5.5, framealpha=0.85)


# ── Panel 3: Flexion ─────────────────────────────────────────────────────────

def draw_flexion_panel(fig, outer_spec, lh_angles: np.ndarray, rh_angles: np.ndarray):
    """lh_angles / rh_angles : (5, 4)  — ch1=MCP, ch2=PIP, ch3=DIP."""
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 5, subplot_spec=outer_spec, wspace=0.05, hspace=0.25,
    )
    for row, (angles, is_left) in enumerate([(lh_angles, True), (rh_angles, False)]):
        h_label = "Left" if is_left else "Right"
        h_color = LH_COLOR if is_left else RH_COLOR
        for fi, fname in enumerate(FINGER_NAMES):
            ax = fig.add_subplot(inner[row, fi])
            _style_ax(ax)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            mcp_flex = float(angles[fi, 1])
            pip_flex = float(angles[fi, 2])
            dip_flex = float(angles[fi, 3])
            color    = FINGER_COLORS[fname]

            wrist, MCP, PIP, DIP, tip = _flex_points(fname, mcp_flex, pip_flex, dip_flex)
            pts = np.array([wrist, MCP, PIP, DIP, tip])
            for (i, j_), lw, alpha in zip(
                [(0, 1), (1, 2), (2, 3), (3, 4)],
                [1.4, 2.8, 2.5, 2.2],
                [0.35, 0.95, 0.90, 0.85],
            ):
                ax.plot([pts[i, 0], pts[j_, 0]], [pts[i, 1], pts[j_, 1]],
                        color=color, lw=lw, alpha=alpha, solid_capstyle="round")
            for pt, sz, mk in zip([wrist, MCP, PIP, DIP, tip],
                                  [24, 20, 14, 12, 14],
                                  ["s", "o", "o", "o", "^"]):
                c = h_color if mk == "s" else color
                ax.scatter(*pt, color=c, s=sz, marker=mk, zorder=6)

            deg_str = (f"MCP {np.degrees(mcp_flex):.1f}°\n"
                       f"PIP {np.degrees(pip_flex):.1f}°\n"
                       f"DIP {np.degrees(dip_flex):.1f}°")
            ax.text(0.03, 0.03, deg_str,
                    transform=ax.transAxes, color=color, fontsize=5.5, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d0d0d",
                              alpha=0.75, edgecolor="none"))

            if row == 0:
                ax.set_title(fname, color=color, fontsize=7, pad=2)
            if fi == 0:
                ax.set_ylabel(h_label, color=h_color, fontsize=7, rotation=90, labelpad=4)

            xs, ys = pts[:, 0], pts[:, 1]
            x_margin = max(abs(xs).max() + 0.15, 0.35)
            ax.set_xlim(-x_margin, x_margin)
            ax.set_ylim(ys.min() - 0.08, ys.max() + 0.10)


# ── Figure 조합 ────────────────────────────────────────────────────────────────

def make_figure(
    contact: np.ndarray,              # (10, 4) tactile, middle frame of window
    angles: np.ndarray,               # (10, 4) finger angles, middle frame of window
    sample_idx: int,
    frame_idx: int,
    data_path: str,
    rgb: Optional[np.ndarray] = None, # (H, W, 3) uint8 or None
):
    """Build a 4-panel figure for one sample window."""
    lh_angles = angles[:5]   # (5, 4)
    rh_angles = angles[5:]   # (5, 4)

    fig = plt.figure(figsize=(34, 8), facecolor="#0d0d0d")
    outer = gridspec.GridSpec(
        1, 4, figure=fig,
        left=0.02, right=0.98, top=0.90, bottom=0.08,
        wspace=0.10, width_ratios=[0.85, 1.1, 1.0, 2.0],
    )

    # Panel 0: RGB
    ax_rgb = fig.add_subplot(outer[0])
    draw_rgb_panel(ax_rgb, rgb, frame_idx, data_path)

    # Panel 1: tactile heatmap
    ax_tac = fig.add_subplot(outer[1])
    draw_tactile_panel(ax_tac, contact)

    # Panel 2: abduction
    ax_abd = fig.add_subplot(outer[2])
    draw_abduction_panel(ax_abd, lh_angles, rh_angles)

    # Panel 3: flexion (2 rows × 5 fingers)
    draw_flexion_panel(fig, outer[3], lh_angles, rh_angles)

    x_flex = (outer[3].get_position(fig).x0 + outer[3].get_position(fig).x1) / 2
    fig.text(x_flex, 0.94,
             "Flexion per finger  (side view)  θ_MCP / θ_PIP / θ_DIP",
             color="#cccccc", fontsize=9, ha="center", va="top")

    fig.text(
        0.01, 0.96,
        f"{data_path}  |  sample {sample_idx}  frame {frame_idx}",
        color="#aaaaaa", fontsize=8, va="top",
    )
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.create({
        "window_time":        args.window_time,
        "window_overlap":     args.window_overlap,
        "interpolating_freq": args.interpolating_freq,
    })

    print(f"Loading dataset from: {args.data_path}")
    dataset = BraincoAngleTactileDataset(
        config=config,
        data_path=args.data_path,
    )
    print(f"  windows : {len(dataset)}")
    print(f"  frames/window : {dataset.num_frames_per_window}")

    if len(dataset) == 0:
        print("No windows found. Check --window_time / data_path.")
        return

    rgb_loader = _RGBLoader(args.data_path)
    print(f"  RGB source  : {'OK' if rgb_loader else 'N/A (colors/ or colors.mp4 not found)'}")

    # Sample indices to visualize
    n = min(args.n_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)

    try:
        for sample_idx in indices:
            sample = dataset[sample_idx]

            # Use the middle frame of the window
            W = sample["joint_contact"].shape[0]
            mid = W // 2

            contact = sample["joint_contact"][mid].numpy()   # (10, 4)
            angles  = sample["finger_angles"][mid].numpy()   # (10, 4)

            # Absolute frame index
            frame_idx = int(dataset.data_idxs[sample_idx]) + mid

            rgb = rgb_loader.load(frame_idx, dataset.num_frames)

            fig = make_figure(
                contact, angles,
                sample_idx=sample_idx,
                frame_idx=frame_idx,
                data_path=args.data_path,
                rgb=rgb,
            )

            out_path = out_dir / f"sample_{sample_idx:04d}_frame_{frame_idx:05d}.png"
            fig.savefig(str(out_path), dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  saved {out_path}")
    finally:
        rgb_loader.close()

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brainco angle tactile visualizer")
    parser.add_argument(
        "--data_path", type=str,
        default="dataset/brainco/pretraining/doll/episode_0000",
        help="경로: dataset/brainco/pretraining/<object>/<episode>",
    )
    parser.add_argument(
        "--window_time", type=float, default=0.1,
        help="윈도우 길이 (초, default=0.1)",
    )
    parser.add_argument(
        "--window_overlap", type=float, default=0.5,
        help="윈도우 겹침 비율 (default=0.5)",
    )
    parser.add_argument(
        "--interpolating_freq", type=int, default=100,
        help="데이터 주파수 Hz (default=100)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=6,
        help="시각화할 샘플 수 (default=6)",
    )
    parser.add_argument(
        "--out", type=str, default="visualization_angle",
        help="출력 이미지 폴더 (default=visualization_angle)",
    )
    main(parser.parse_args())
