"""test_visualization.py — BrainCo vs OakInkV2 visualization with identical pipeline.

Both datasets go through the exact same post-processing and visualization code.
Row 0: BrainCo  (local hand frame)
Row 1: OakInkV2 (local hand frame)
Row 2: BrainCo  (global vroot frame — wrist pos + rotation axes)
Row 3: OakInkV2 (global vroot frame — wrist pos + rotation axes)

Usage:
    python test_visualization.py \
        --brainco_ep   dataset/brainco/pretraining/doll/episode_0000 \
        --brainco_urdf dataset/brainco/urdf \
        --oakink_root  pretraining_dataset/OakInkv2 \
        --out          vis/compare.png
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from tactile_ssl.data.brainco_tactile import BraincoSSLDataset, SKELETON_LINES
from tactile_ssl.data.oakinkv2_tactile import OakInkV2TactileDataset

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

_MP_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
_MP_TIPS = [4, 8, 12, 16, 20]
_BC_TIPS = [5, 9, 13, 17, 21]   # touch_link indices in 22-joint BrainCo skeleton


# ── Shared math ───────────────────────────────────────────────────────────────

def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) → (..., 3, 3) rotation matrix via Gram-Schmidt."""
    c0 = rot6d[..., :3]
    c1 = rot6d[..., 3:]
    c0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-8)
    c1 = c1 - (c1 * c0).sum(axis=-1, keepdims=True) * c0
    c1 = c1 / (np.linalg.norm(c1, axis=-1, keepdims=True) + 1e-8)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=-1)  # columns = x, y, z axes


def _collect_image_files(rgb_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not rgb_dir.exists() or not rgb_dir.is_dir():
        return []
    return sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in exts and p.is_file())


def _find_brainco_rgb_files(ep_path: str) -> tuple[list[Path], Optional[Path]]:
    ep = Path(ep_path)
    for name in ["colors", "rgb", "RGB", "images", "image"]:
        files = _collect_image_files(ep / name)
        if files:
            return files, ep / name

    # fallback: choose the folder under episode that has the most image files
    best_dir = None
    best_files: list[Path] = []
    for d in ep.rglob("*"):
        if not d.is_dir():
            continue
        files = _collect_image_files(d)
        if len(files) > len(best_files):
            best_files = files
            best_dir = d
    return best_files, best_dir


def _load_brainco_rgb(ep_path: str, abs_frame: int, total_frames: int) -> tuple[Optional[np.ndarray], str]:
    files, rgb_dir = _find_brainco_rgb_files(ep_path)
    if not files:
        return None, "RGB not found"

    denom = max(total_frames, 1)
    vid_idx = min(int(round(abs_frame * len(files) / denom)), len(files) - 1)
    img = plt.imread(str(files[vid_idx]))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    src = f"{rgb_dir.name if rgb_dir is not None else '?'}[{vid_idx}]/{len(files)}"
    return img, src


# ── Sample extraction — returns identical dict format for both datasets ────────

def extract_brainco(ep_path: str, urdf_path: str,
                    sample_idx: int = -1, frame_idx: int = -1) -> dict:
    config = OmegaConf.create({
        "window_time": 0.1,
        "window_overlap": 0.5,
        "interpolating_freq": 100,
        "bias_noise_std": 0,
        "bias_range": 0,
    })
    dataset = BraincoSSLDataset(
        config=config,
        data_path=ep_path,
        brainco_urdf_path=urdf_path,
        joint_poses=True,
        robot_to_human=True, 
        retargeting_config_path_left="assets/brainco/revo2_left_hand.yml", 
        retargeting_config_path_right="assets/brainco/revo2_right_hand.yml"
    )
    if sample_idx < 0:
        sample_idx = len(dataset) // 2
    sample_idx = min(sample_idx, len(dataset) - 1)
    sample    = dataset[sample_idx]
    if frame_idx < 0:
        frame_idx = sample["sensor"].shape[0] // 2
    frame_idx = min(frame_idx, sample["sensor"].shape[0] - 1)
    fi        = int(sample["frame_indices"][frame_idx].item())
    print(f"  BrainCo: dataset size={len(dataset)}, sample_idx={sample_idx}, frame_idx={frame_idx}, abs_frame={fi}")

    skeleton  = sample["skeleton_poses"][frame_idx].cpu().numpy()   # (42, 3) wrist-local
    wrist_p   = sample["wrist_poses"][frame_idx].cpu().numpy()      # (2, 9)
    tac       = sample["sensor"][frame_idx].cpu().numpy()           # (10, 4)

    # prepend wrist origin → (22, 3) per hand
    sk_left  = np.concatenate([np.zeros((1, 3), np.float32), skeleton[:21]],  axis=0)
    sk_right = np.concatenate([np.zeros((1, 3), np.float32), skeleton[21:]], axis=0)

    fp_left  = sk_left[_BC_TIPS]   # (5, 3)
    fp_right = sk_right[_BC_TIPS]

    fp_sizes_l = 20 + np.clip(tac[:5,  0], 0, None) * 0.002
    fp_sizes_r = 20 + np.clip(tac[5:, 0],  0, None) * 0.002

    wrist_R_fk = dataset.wrist_R_fk[fi]   # (2, 3, 3) raw rubber_hand rotation in vis world

    return dict(
        sk_left=sk_left, sk_right=sk_right,
        fp_left=fp_left, fp_right=fp_right,
        fp_sizes_l=fp_sizes_l, fp_sizes_r=fp_sizes_r,
        wrist_poses=wrist_p,
        wrist_R_fk=wrist_R_fk,
        sk_lines=SKELETON_LINES,
        abs_frame=fi,
        num_frames=dataset.num_frames,
        ep_path=ep_path,
        info=f"BrainCo  ep={os.path.basename(ep_path)}  frame={fi}",
    )


def extract_oakinkv2(data_root: str,
                     seq_idx: int = -1, frame_idx: int = 0) -> dict:
    all_pkls = sorted(Path(data_root).glob("*.pkl"))
    assert all_pkls, f"No pkl files in {data_root}"
    pkl_idx = (len(all_pkls) // 2) if seq_idx < 0 else min(seq_idx, len(all_pkls) - 1)
    _tmp = tempfile.mkdtemp()
    shutil.copy(all_pkls[pkl_idx], _tmp)
    dataset = OakInkV2TactileDataset(data_root=_tmp, window_size=3, split="train",
                                     train_val_split=1.0)
    assert len(dataset) > 0, f"No OakInkV2 data in {data_root}"

    idx       = frame_idx #len(dataset) // 2
    sample    = dataset[idx]
    frame_idx = min(frame_idx, sample["sensor"].shape[0] - 1)
    print(sample["sensor"].shape, frame_idx)
    print(f"  OakInkV2: total_pkls={len(all_pkls)}, pkl_idx={pkl_idx}({all_pkls[pkl_idx].name}), window={idx}, frame_idx={frame_idx}")

    skeleton = sample["sensor_poses"][frame_idx].cpu().numpy()  # (42, 3) wrist-local
    wrist_p  = sample["wrist_poses"][frame_idx].cpu().numpy()   # (2, 9)
    tac      = sample["sensor"][frame_idx].cpu().numpy()         # (42, 1)

    sk_left  = skeleton[:21]   # idx 0 = wrist origin
    sk_right = skeleton[21:]

    fp_left  = sk_left[_MP_TIPS]   # (5, 3)
    fp_right = sk_right[_MP_TIPS]

    fp_sizes_l = 20 + np.clip(tac[_MP_TIPS, 0], 0, None) * 200
    fp_sizes_r = 20 + np.clip(tac[[21 + t for t in _MP_TIPS], 0], 0, None) * 200

    seq_idx, start = dataset.windows[idx]
    return dict(
        sk_left=sk_left, sk_right=sk_right,
        fp_left=fp_left, fp_right=fp_right,
        fp_sizes_l=fp_sizes_l, fp_sizes_r=fp_sizes_r,
        wrist_poses=wrist_p,
        sk_lines=_MP_LINES,
        info=f"OakInkV2  seq={seq_idx}  frame_start={start}",
    )


# ── Shared visualization functions ────────────────────────────────────────────

PLANES = [("XY", (0, 1), ("X", "Y")),
          ("XZ", (0, 2), ("X", "Z")),
          ("YZ", (1, 2), ("Y", "Z"))]
AX_COLORS = ["r", "g", "b"]


def _setup_axes(axes_3d, axes_2d, title,
                lim3d=(-0.25, 0.25), lim2d=(-0.25, 0.25),
                elev=20.0, azim=-60.0):
    for ax, (_, _, (xl, yl)) in zip(axes_2d, PLANES):
        ax.cla()
        ax.set_xlim(*lim2d); ax.set_ylim(*lim2d)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    axes_3d.cla()
    axes_3d.set_xlim(*lim3d); axes_3d.set_ylim(*lim3d); axes_3d.set_zlim(*lim3d)
    axes_3d.set_box_aspect([1, 1, 1])
    axes_3d.view_init(elev=elev, azim=azim)
    axes_3d.set_xlabel("X"); axes_3d.set_ylabel("Y"); axes_3d.set_zlabel("Z")
    axes_3d.set_title(title, fontsize=9)


_JOINT_PT_SIZE_3D = 6    # non-tip joints in 3D view
_JOINT_PT_SIZE_2D = 10   # non-tip joints in 2D views


def _draw_hand_skeleton(axes_3d, axes_2d, sk, fp, fp_sz, color, sk_lines):
    # ── Bones ─────────────────────────────────────────────────────────────
    for i, j in sk_lines:
        axes_3d.plot([sk[i,0], sk[j,0]], [sk[i,1], sk[j,1]], [sk[i,2], sk[j,2]],
                     c=color, alpha=0.6, lw=2)
        for ax, (_, (i0, i1), _) in zip(axes_2d, PLANES):
            ax.plot([sk[i,i0], sk[j,i0]], [sk[i,i1], sk[j,i1]], c=color, alpha=0.6, lw=1.5)

    # ── All joints — small dots ────────────────────────────────────────────
    axes_3d.scatter(sk[:,0], sk[:,1], sk[:,2],
                    c=color, s=_JOINT_PT_SIZE_3D, marker="o", alpha=0.8, zorder=4)
    for ax, (_, (i0, i1), _) in zip(axes_2d, PLANES):
        ax.scatter(sk[:,i0], sk[:,i1],
                   c=color, s=_JOINT_PT_SIZE_2D, marker="o", alpha=0.8, zorder=4)

    # ── Fingertips — larger dots (drawn on top) ───────────────────────────
    axes_3d.scatter(fp[:,0], fp[:,1], fp[:,2], c=color, s=fp_sz * 0.4, marker="o", zorder=5)
    for ax, (_, (i0, i1), _) in zip(axes_2d, PLANES):
        ax.scatter(fp[:,i0], fp[:,i1], c=color, s=fp_sz, marker="o", zorder=5)

    # ── Finger labels ─────────────────────────────────────────────────────
    for li, name in enumerate(FINGER_NAMES):
        axes_3d.text(*fp[li], f" {name[0]}", fontsize=7, color=color)


def _draw_wrist_axes(axes_3d, axes_2d, wrist_pos, wrist_R, wrist_color, label, quiver_len):
    axes_3d.scatter(*wrist_pos, c=wrist_color, s=50, marker="s",
                    edgecolors="black", zorder=6, label=label)
    for ax, (_, (i0, i1), _) in zip(axes_2d, PLANES):
        ax.scatter(wrist_pos[i0], wrist_pos[i1], c=wrist_color, s=80,
                   marker="s", edgecolors="black", zorder=6)
    for ai, ac in enumerate(AX_COLORS):
        d = wrist_R[:, ai] * quiver_len
        axes_3d.quiver(*wrist_pos, *d, color=ac, linewidth=1.5, arrow_length_ratio=0.3)
        for ax, (_, (i0, i1), _) in zip(axes_2d, PLANES):
            ax.annotate("", xy=(wrist_pos[i0]+d[i0], wrist_pos[i1]+d[i1]),
                        xytext=(wrist_pos[i0], wrist_pos[i1]),
                        arrowprops=dict(arrowstyle="->", color=ac, lw=1.2))


def visualize_local(axes_3d, axes_2d, data: dict):
    """Local wrist frame: left hand offset left, right hand offset right."""
    offsets = [np.array([-0.12, 0., 0.], np.float32),
               np.array([ 0.12, 0., 0.], np.float32)]
    quiver_len = 0.04

    print(data)
    _setup_axes(axes_3d, axes_2d, f"[Local] {data['info']}",
                lim3d=(-0.25, 0.25), lim2d=(-0.25, 0.25))

    for (sk_raw, fp_raw, fp_sz, color, wrist_color, label), off in zip(
        [
            (data["sk_left"],  data["fp_left"],  data["fp_sizes_l"], "blue",   "cyan",    "Left wrist"),
            (data["sk_right"], data["fp_right"], data["fp_sizes_r"], "red",    "magenta", "Right wrist"),
        ],
        offsets,
    ):
        sk = sk_raw + off
        fp = fp_raw + off
        _draw_hand_skeleton(axes_3d, axes_2d, sk, fp, fp_sz, color, data["sk_lines"])
        # identity axes at wrist origin (local frame = identity R)
        _draw_wrist_axes(axes_3d, axes_2d, off, np.eye(3, dtype=np.float32),
                         wrist_color, label, quiver_len)

    axes_3d.legend(loc="upper left", fontsize=7)


def visualize_global(axes_3d, axes_2d, data: dict):
    """Global vroot frame: reconstruct world positions from wrist_poses + local joints."""
    quiver_len = 0.08

    wrist_pos = data["wrist_poses"][:, :3].astype(np.float32)             # (2, 3)
    print(f"\n=== visualize_global ===")
    print(f"  {data['info']}")
    for side, idx in [("Left ", 0), ("Right", 1)]:
        pos = wrist_pos[idx]
        r6d = data["wrist_poses"][idx, 3:]
        print(f"  {side}  pos(xyz) = [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]"
              f"  rot6d = [{', '.join(f'{v:+.4f}' for v in r6d)}]")
    wrist_R   = rot6d_to_mat(data["wrist_poses"][:, 3:].astype(np.float32))  # (2, 3, 3)
    wrist_R = wrist_R.transpose(0, 2, 1)
    root      = np.zeros(3, dtype=np.float32)

    sk_left  = data["sk_left"].copy()
    fp_left  = data["fp_left"].copy()
    sk_left[:, 0]  = -sk_left[:, 0]
    fp_left[:, 0]  = -fp_left[:, 0]
    sk_left_g  = wrist_pos[0] + (wrist_R[0] @ sk_left.T).T
    sk_right_g = wrist_pos[1] + (wrist_R[1] @ data["sk_right"].T).T
    fp_left_g  = wrist_pos[0] + (wrist_R[0] @ fp_left.T).T
    fp_right_g = wrist_pos[1] + (wrist_R[1] @ data["fp_right"].T).T

    _setup_axes(axes_3d, axes_2d, f"[Global vroot] {data['info']}",
                lim3d=(-0.75, 0.75), lim2d=(-0.75, 0.75))

    _draw_hand_skeleton(axes_3d, axes_2d, sk_left_g,  fp_left_g,  data["fp_sizes_l"], "blue", data["sk_lines"])
    _draw_hand_skeleton(axes_3d, axes_2d, sk_right_g, fp_right_g, data["fp_sizes_r"], "red",  data["sk_lines"])

    _draw_wrist_axes(axes_3d, axes_2d, wrist_pos[0], wrist_R[0], "cyan",    "Left wrist",  quiver_len)
    _draw_wrist_axes(axes_3d, axes_2d, wrist_pos[1], wrist_R[1], "magenta", "Right wrist", quiver_len)

    # ── rubber_hand raw FK axes (before any additional rotation fix) ──────
    # Dark-tone XYZ axes drawn as lines to distinguish from corrected axes (quiver).
    # Only available for BrainCo data which carries wrist_R_fk.
    _RAW_COLORS = ["#FF4444", "#44FF44", "#4444FF"]   # bright red / green / blue (dashed)
    _raw_len = quiver_len * 2.5
    if "wrist_R_fk" in data:
        for side, pos, side_lbl in [(0, wrist_pos[0], "L"), (1, wrist_pos[1], "R")]:
            R_raw = data["wrist_R_fk"][side]   # (3, 3) columns = X, Y, Z axes
            for ai, ac in enumerate(_RAW_COLORS):
                end = pos + R_raw[:, ai] * _raw_len
                axes_3d.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                             c=ac, lw=3.0, alpha=1.0, linestyle="--",
                             label=f"{side_lbl} rubber_hand {'XYZ'[ai]}" if side == 0 else None)
                for ax, (_, (i0, i1), _) in zip(axes_2d, PLANES):
                    ax.annotate("", xy=(end[i0], end[i1]), xytext=(pos[i0], pos[i1]),
                                arrowprops=dict(arrowstyle="->", color=ac, lw=2.0,
                                                linestyle="dashed"))

    # vroot origin + world axes
    axes_3d.scatter(*root, c="gold", s=60, marker="*", zorder=7, label="VR root")
    for ax, (_, (i0, i1), _) in zip(axes_2d, PLANES):
        ax.scatter(root[i0], root[i1], c="gold", s=110, marker="*", zorder=7)
    for ai, ac in enumerate(AX_COLORS):
        v = np.zeros(3, dtype=np.float32); v[ai] = quiver_len
        axes_3d.quiver(*root, *v, color=ac, linewidth=2.0, arrow_length_ratio=0.3)
        for ax, (_, (i0, i1), _) in zip(axes_2d, PLANES):
            ax.annotate("", xy=(root[i0]+v[i0], root[i1]+v[i1]),
                        xytext=(root[i0], root[i1]),
                        arrowprops=dict(arrowstyle="->", color=ac, lw=1.4))

    # wrist-to-skeleton-root dashed lines
    axes_3d.plot([wrist_pos[0,0], sk_left_g[0,0]], [wrist_pos[0,1], sk_left_g[0,1]],
                 [wrist_pos[0,2], sk_left_g[0,2]], c="cyan", alpha=0.6, lw=2, ls="--")
    axes_3d.plot([wrist_pos[1,0], sk_right_g[0,0]], [wrist_pos[1,1], sk_right_g[0,1]],
                 [wrist_pos[1,2], sk_right_g[0,2]], c="magenta", alpha=0.6, lw=2, ls="--")

    axes_3d.legend(loc="upper left", fontsize=7)


# ── Main ─────────────────────────────────────────────────────────────────────

def _render_and_save(bc, ok, out_path: str):
    bc_rgb, bc_rgb_src = _load_brainco_rgb(
        bc["ep_path"],
        abs_frame=int(bc["abs_frame"]),
        total_frames=int(bc["num_frames"]),
    )
    if bc_rgb is None:
        print("  BrainCo RGB: not found")
    else:
        print(f"  BrainCo RGB: loaded from {bc_rgb_src}")

    ncols = 5  # RGB + 3D + XY + XZ + YZ
    nrows = 4  # local BC, local OK, global BC, global OK

    fig = plt.figure(figsize=(4.8 * ncols, 5.5 * nrows), constrained_layout=True)
    fig.suptitle("BrainCo vs Arctic  —  Local (rows 0-1) / Global vroot (rows 2-3)", fontsize=11)

    def make_row(row, with_rgb: bool = False, rgb: Optional[np.ndarray] = None, rgb_title: str = "RGB"):
        if with_rgb:
            ax_rgb = fig.add_subplot(nrows, ncols, row * ncols + 1)
            if rgb is not None:
                ax_rgb.imshow(rgb)
                ax_rgb.set_title(rgb_title, fontsize=9)
            else:
                ax_rgb.text(0.5, 0.5, "RGB not found", ha="center", va="center", fontsize=9)
                ax_rgb.set_title("RGB", fontsize=9)
            ax_rgb.axis("off")
        else:
            ax_blank = fig.add_subplot(nrows, ncols, row * ncols + 1)
            ax_blank.axis("off")

        ax3d = fig.add_subplot(nrows, ncols, row * ncols + 2, projection="3d")
        axs  = [fig.add_subplot(nrows, ncols, row * ncols + c) for c in [3, 4, 5]]
        for ax, lbl in zip(axs, ["XY", "XZ", "YZ"]):
            ax.set_title(lbl)
        return ax3d, axs

    visualize_local(*make_row(0, with_rgb=True, rgb=bc_rgb, rgb_title=f"BrainCo RGB  f{bc['abs_frame']}"), bc)
    visualize_local(*make_row(1), ok)
    visualize_global(*make_row(2, with_rgb=True, rgb=bc_rgb, rgb_title=f"BrainCo RGB  f{bc['abs_frame']}"), bc)
    visualize_global(*make_row(3), ok)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainco_ep",     default="dataset/brainco/pretraining/pot/episode_0000")
    # parser.add_argument("--brainco_ep",     default="dataset/brainco/downstream/grasp_prediction/grasp_fail/episode_0000")

    parser.add_argument("--brainco_urdf",   default="dataset/brainco/urdf")
    parser.add_argument("--brainco_frame",  type=int, default=0, help="frame within window (-1=middle)")
    parser.add_argument("--frame_step",     type=int, default=100, help="sample index step for multi-frame output")
    parser.add_argument("--oakink_root",    default="pretraining_dataset/Arctic")
    parser.add_argument("--oakink_seq",     type=int, default=1, help="pkl file index (-1=middle)")
    parser.add_argument("--oakink_frame",   type=int, default=0,  help="frame within window")
    parser.add_argument("--out",            default="vis/compare.png")
    args = parser.parse_args()

    out_path = Path(args.out)
    os.makedirs(out_path.parent or ".", exist_ok=True)

    # Build dataset once to know total sample count
    config = OmegaConf.create({
        "window_time": 0.15,
        "window_overlap": 0.0,
        "interpolating_freq": 100,
        "bias_noise_std": 0,
        "bias_range": 0,
    })
    from tactile_ssl.data.brainco_tactile import BraincoSSLDataset
    _ds = BraincoSSLDataset(config=config, data_path=args.brainco_ep,
                            brainco_urdf_path=args.brainco_urdf, joint_poses=True)
    total_samples = len(_ds)
    del _ds
    print(f"Total BrainCo samples: {total_samples}, step: {args.frame_step}")

    print("Loading OakInkV2 sample...")
    ok = extract_oakinkv2(args.oakink_root,
                          seq_idx=args.oakink_seq, frame_idx=args.oakink_frame)

    sample_indices = range(0, total_samples, args.frame_step)
    for sample_idx in sample_indices:
        print(f"\n--- sample_idx={sample_idx} ---")
        bc = extract_brainco(args.brainco_ep, args.brainco_urdf,
                             sample_idx=sample_idx, frame_idx=args.brainco_frame)
        stem = out_path.stem
        suffix = out_path.suffix
        out_file = out_path.parent / f"{stem}_sample{sample_idx:04d}{suffix}"
        _render_and_save(bc, ok, str(out_file))


if __name__ == "__main__":
    main()
