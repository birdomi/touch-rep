"""test_visualization_all.py

Unified visualizer that stitches 5 datasets into a single image per sample:
  brainco | arctic | taco | hot3d | oakinkv2

Each output PNG contains one row per dataset, with the same 4-panel layout:
  - Panel 0: RGB frame              (brainco only — others show "RGB N/A")
  - Panel 1: Contact heatmap         (BrainCo: 10 sensors × 4 ch;  others: 10 fingertips × 1 ch)
  - Panel 2: Abduction  θ_abd        (top-down stylized)
  - Panel 3: Flexion    θ_MCP/PIP/DIP (side-view per finger)

Usage:
    # default — produce 1 combined image with all 5 datasets stacked
    python test_visualization_all.py

    # multiple combined images (each with all datasets, sampled at different positions)
    python test_visualization_all.py --n_samples 4

    # pick a subset of datasets
    python test_visualization_all.py --datasets brainco arctic

    # custom output directory
    python test_visualization_all.py --out my_vis

    # override path for a single dataset (and run only that one)
    python test_visualization_all.py --datasets brainco \\
        --data_path dataset/brainco/pretraining/towel/episode_0001
"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from test_visualization_angle import (
    FINGER_NAMES, LH_COLOR, RH_COLOR, CH_NAMES,
    _RGBLoader, _style_ax,
    draw_rgb_panel, draw_abduction_panel, draw_flexion_panel,
)

# HOT3D / ARCTIC / TACO / OAKINKV2 are 21-joint hands; we extract the
# 5 fingertip joints per hand to fit the BrainCo-style 10-sensor panel.
_FINGERTIP_INDICES = [4, 8, 12, 16, 20]

# Built-in default paths per dataset.
_DEFAULT_PATHS = {
    "brainco":  "dataset/brainco/pretraining/towel/episode_0000",
    "arctic":   "pretraining_dataset/vector_dataset/ARCTIC",
    "taco":     "pretraining_dataset/vector_dataset/TACO",
    "hot3d":    "pretraining_dataset/vector_dataset/HOT3D/hot3d",
    "oakinkv2": "pretraining_dataset/vector_dataset/OAKINKV2",
}
_ALL_DATASETS = list(_DEFAULT_PATHS.keys())

# Root of the external OakInkv2 archive that ships meshes for all 4 datasets.
_OAKINKV2_ROOT = Path("/media/etri/02363C66363C5CBB/workspace/OakInkv2")


# ── Mesh resolution + rendering (RGB fallback for non-brainco) ───────────────

def _resolve_mesh_path(dataset_name: str, seq_path: Path, seq_idx: int) -> Optional[Path]:
    """Best-effort: map a sequence file to a representative mesh on disk."""
    root = _OAKINKV2_ROOT
    name = seq_path.stem

    if dataset_name == "oakinkv2":
        info_file = root / "program" / "program_info" / f"{name}.json"
        if info_file.exists():
            try:
                with open(info_file) as f:
                    info = json.load(f)
            except Exception:
                info = {}
            for entry in info.values():
                obj_list = entry.get("obj_list") or []
                for obj_id in obj_list:
                    for parent in ("object_raw", "object_repair"):
                        obj_dir = root / parent / "align_ds" / obj_id
                        if obj_dir.exists():
                            plys = sorted(obj_dir.glob("*.ply"))
                            if plys:
                                return plys[0]

    elif dataset_name == "arctic":
        # filename: s<NN>_<obj>_<action>_<NN>.pkl
        parts = name.split("_")
        if len(parts) >= 2:
            obj = parts[1]
            mesh = (root / "arctic" / "data" / "arctic_data" / "data"
                    / "meta" / "object_vtemplates" / obj / "mesh.obj")
            if mesh.exists():
                return mesh

    elif dataset_name == "taco":
        # TACO meshes (NNN_cm.obj) don't map 1:1 to filenames; use seq_idx % count.
        mesh_dir = root / "TACO" / "TACO_dataset" / "object_models_released"
        if mesh_dir.exists():
            meshes = sorted(mesh_dir.glob("*.obj"))
            if meshes:
                return meshes[seq_idx % len(meshes)]

    elif dataset_name == "hot3d":
        glb_dir = root / "hot3d" / "hot3d" / "dataset" / "assets"
        if glb_dir.exists():
            glbs = sorted(glb_dir.glob("*.glb"))
            if glbs:
                return glbs[seq_idx % len(glbs)]

    return None


def _mesh_to_rgb_array(mesh_path: Path, size_px: int = 800) -> Optional[np.ndarray]:
    """Render a mesh to an (H, W, 3) uint8 RGB array via matplotlib 3D."""
    try:
        import trimesh
        loaded = trimesh.load(str(mesh_path), force="mesh")
        verts = np.asarray(loaded.vertices, dtype=np.float32)
        faces = np.asarray(loaded.faces, dtype=np.int64)
    except Exception as e:
        print(f"  mesh load failed ({mesh_path.name}): {type(e).__name__}: {e}")
        return None
    if verts.size == 0 or faces.size == 0:
        return None

    # Keep enough faces for solid shading; downsample very dense meshes.
    MAX_FACES = 15000
    if len(faces) > MAX_FACES:
        step = max(1, len(faces) // MAX_FACES)
        faces = faces[::step]

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(size_px / 100, size_px / 100), facecolor="#0d0d0d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d0d0d")
    ax.set_axis_off()

    tri = verts[faces]                          # (F, 3, 3)
    poly = Poly3DCollection(
        tri, alpha=0.92, facecolor="#4ecdc4",
        edgecolor="#1a4040", linewidth=0.05,
    )
    ax.add_collection3d(poly)

    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    extent = mx - mn
    extent[extent < 1e-6] = 1.0
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_zlim(mn[2], mx[2])
    # Match axis box aspect to actual mesh extent so it fills the canvas.
    try:
        ax.set_box_aspect(extent / extent.max())
    except Exception:
        pass
    ax.view_init(elev=20, azim=40)

    # Tighten the axes to remove default padding.
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


# ── Adaptive tactile panel ────────────────────────────────────────────────────

def draw_tactile_panel_adaptive(ax, contact: np.ndarray, ch_names):
    """contact : (10, C)  — C=4 for BrainCo, C=1 for joint-only datasets.

    Invalid entries (<0) are shown as grey cells with 'inv' label.
    """
    C = contact.shape[1]
    _style_ax(ax, f"Contact  (10 sensors × {C} ch)")

    data = contact.copy()
    invalid = data < 0

    extent = [-0.5, 9.5, -0.5, C - 0.5]
    im = ax.imshow(
        np.ma.masked_where(invalid, data).T,
        aspect="auto", vmin=0.0, vmax=1.0,
        cmap="hot", origin="lower",
        extent=extent,
    )
    grey = np.ma.masked_where(~invalid, np.zeros_like(data))
    ax.imshow(
        grey.T, aspect="auto", vmin=0, vmax=1,
        cmap="Greys", alpha=0.6, origin="lower",
        extent=extent,
    )

    ax.set_xticks(range(10))
    ax.set_xticklabels(
        [f"L{n}" for n in FINGER_NAMES] + [f"R{n}" for n in FINGER_NAMES],
        rotation=35, ha="right", fontsize=6, color="#aaaaaa",
    )
    ax.set_yticks(range(C))
    ax.set_yticklabels(ch_names[:C], fontsize=6, color="#aaaaaa")

    ax.axvline(4.5, color="#555555", lw=1.2, ls="--", alpha=0.7)
    ax.text(2.0, C - 0.3, "Left hand",  color=LH_COLOR, fontsize=7, ha="center", va="bottom")
    ax.text(7.0, C - 0.3, "Right hand", color=RH_COLOR, fontsize=7, ha="center", va="bottom")

    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="normalized [0, 1]")

    for si in range(10):
        for ci in range(C):
            v = data[si, ci]
            if v < 0:
                txt, col = "inv", "#555555"
            else:
                txt, col = f"{v:.2f}", "#ffffff" if v < 0.5 else "#000000"
            ax.text(si, ci, txt, ha="center", va="center", fontsize=5.5, color=col)


# ── Figure assembly ──────────────────────────────────────────────────────────

def make_figure(
    contact: np.ndarray,              # (10, C)
    angles: np.ndarray,               # (10, 4)
    sample_idx: int,
    frame_idx: int,
    title_path: str,
    ch_names,
    rgb: Optional[np.ndarray] = None,
):
    lh_angles = angles[:5]
    rh_angles = angles[5:]

    fig = plt.figure(figsize=(34, 8), facecolor="#0d0d0d")
    outer = gridspec.GridSpec(
        1, 4, figure=fig,
        left=0.02, right=0.98, top=0.90, bottom=0.08,
        wspace=0.10, width_ratios=[0.85, 1.1, 1.0, 2.0],
    )

    ax_rgb = fig.add_subplot(outer[0])
    draw_rgb_panel(ax_rgb, rgb, frame_idx, title_path)

    ax_tac = fig.add_subplot(outer[1])
    draw_tactile_panel_adaptive(ax_tac, contact, ch_names)

    ax_abd = fig.add_subplot(outer[2])
    draw_abduction_panel(ax_abd, lh_angles, rh_angles)

    draw_flexion_panel(fig, outer[3], lh_angles, rh_angles)

    x_flex = (outer[3].get_position(fig).x0 + outer[3].get_position(fig).x1) / 2
    fig.text(x_flex, 0.94,
             "Flexion per finger  (side view)  θ_MCP / θ_PIP / θ_DIP",
             color="#cccccc", fontsize=9, ha="center", va="top")

    fig.text(
        0.01, 0.96,
        f"{title_path}  |  sample {sample_idx}  frame {frame_idx}",
        color="#aaaaaa", fontsize=8, va="top",
    )
    return fig


# ── Dataset adapters ─────────────────────────────────────────────────────────

class _BraincoAdapter:
    """Adapter for BraincoAngleTactileDataset."""

    def __init__(self, data_path: str, args, dataset_name: str = "brainco"):
        from tactile_ssl.data.brainco_angle_tactile import BraincoAngleTactileDataset
        config = OmegaConf.create({
            "window_time":        args.window_time,
            "window_overlap":     args.window_overlap,
            "interpolating_freq": args.interpolating_freq,
        })
        self.dataset = BraincoAngleTactileDataset(
            config=config, data_path=data_path,
        )
        self.rgb_loader = _RGBLoader(data_path)
        self.data_path  = data_path
        self.ch_names   = CH_NAMES   # ["Normal", "Tangential", "Depth", "Proximity"]
        self._dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def get_sample(self, idx: int):
        sample = self.dataset[idx]
        W = sample["joint_contact"].shape[0]
        mid = W // 2
        contact = sample["joint_contact"][mid].numpy()        # (10, 4)
        angles  = sample["finger_angles"][mid].numpy()        # (10, 4)
        frame_idx = int(self.dataset.data_idxs[idx]) + mid
        rgb = self.rgb_loader.load(frame_idx, self.dataset.num_frames)
        return contact, angles, frame_idx, self.data_path, rgb

    def close(self):
        self.rgb_loader.close()


class _AngleAdapter:
    """Adapter for AngleTactileDataset (HOT3D / ARCTIC / TACO / OAKINKV2).

    No RGB stream is available — falls back to rendering a representative
    object mesh from ../OakInkv2/.  Rendered meshes are cached per sequence.
    """

    def __init__(self, data_path: str, args, dataset_name: str):
        from tactile_ssl.data.angle_tactile import AngleTactileDataset
        self.dataset = AngleTactileDataset(
            data_root=data_path,
            window_size=1, window_stride=1, train_val_split=1.0, split="train",
        )
        self.ch_names = ["Contact"]
        self._dataset_name = dataset_name
        self._mesh_cache: Dict[int, Optional[np.ndarray]] = {}

    def __len__(self):
        return len(self.dataset)

    def _get_mesh_rgb(self, seq_idx: int) -> Optional[np.ndarray]:
        if seq_idx in self._mesh_cache:
            return self._mesh_cache[seq_idx]
        seq_path = self.dataset._seq_paths[seq_idx]
        mesh_path = _resolve_mesh_path(self._dataset_name, seq_path, seq_idx)
        if mesh_path is None or not mesh_path.exists():
            self._mesh_cache[seq_idx] = None
            return None
        rgb = _mesh_to_rgb_array(mesh_path)
        self._mesh_cache[seq_idx] = rgb
        return rgb

    def get_sample(self, idx: int):
        sample = self.dataset[idx]
        W = sample["joint_contact"].shape[0]
        mid = W // 2
        contact_raw = sample["joint_contact"][mid].numpy()    # (42, 1)
        angles      = sample["finger_angles"][mid].numpy()    # (10, 4)

        lh_tips = contact_raw[:21, 0][_FINGERTIP_INDICES]     # (5,)
        rh_tips = contact_raw[21:, 0][_FINGERTIP_INDICES]     # (5,)
        contact = np.concatenate([lh_tips, rh_tips])[:, None] # (10, 1)

        seq_idx, start = self.dataset.windows[idx]
        seq_path = self.dataset._seq_paths[seq_idx]
        title    = f"{seq_path.parent.name}/{seq_path.stem}"
        rgb = self._get_mesh_rgb(seq_idx)
        return contact, angles, int(start) + mid, title, rgb

    def close(self):
        pass


_ADAPTERS = {
    "brainco":  _BraincoAdapter,
    "arctic":   _AngleAdapter,
    "taco":     _AngleAdapter,
    "hot3d":    _AngleAdapter,
    "oakinkv2": _AngleAdapter,
}


# ── Stitching helper ─────────────────────────────────────────────────────────

def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _stitch_vertical(images, bg="#0d0d0d") -> Image.Image:
    """Pad to common width and stack vertically."""
    target_w = max(im.width for im in images)
    resized = []
    for im in images:
        if im.width != target_w:
            new_h = int(im.height * target_w / im.width)
            im = im.resize((target_w, new_h), Image.LANCZOS)
        resized.append(im)
    total_h = sum(im.height for im in resized)
    canvas = Image.new("RGB", (target_w, total_h), color=bg)
    y = 0
    for im in resized:
        canvas.paste(im, (0, y))
        y += im.height
    return canvas


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    targets = args.datasets or _ALL_DATASETS

    if args.data_path is not None:
        if len(targets) != 1:
            raise SystemExit(
                "--data_path can only be used with a single --datasets entry "
                f"(got {targets})."
            )
        path_overrides = {targets[0]: args.data_path}
    else:
        path_overrides = {}

    # ── Open every requested adapter; skip any that fail to load ────────────
    adapters: dict = {}
    for name in targets:
        path = path_overrides.get(name, _DEFAULT_PATHS[name])
        print(f"[{name}] Loading from: {path}")
        if not Path(path).exists():
            print(f"  SKIP: path does not exist")
            continue
        try:
            adapter = _ADAPTERS[name](path, args, dataset_name=name)
        except Exception as e:
            print(f"  SKIP: {type(e).__name__}: {e}")
            continue
        if len(adapter) == 0:
            print("  SKIP: 0 windows")
            adapter.close()
            continue
        print(f"  windows : {len(adapter)}")
        adapters[name] = (adapter, path)

    if not adapters:
        print("\nNo dataset successfully loaded.")
        return

    # ── For each combined image, take the k-th evenly-spaced index per dataset ─
    try:
        for k in range(args.n_samples):
            row_imgs = []
            for name, (adapter, path) in adapters.items():
                n_dataset = len(adapter)
                if args.n_samples == 1:
                    idx = n_dataset // 2
                else:
                    idx = int(np.linspace(0, n_dataset - 1, args.n_samples)[k])
                contact, angles, frame_idx, title_path, rgb = adapter.get_sample(idx)
                fig = make_figure(
                    contact, angles,
                    sample_idx=idx,
                    frame_idx=frame_idx,
                    title_path=f"[{name.upper()}] {title_path}",
                    ch_names=adapter.ch_names,
                    rgb=rgb,
                )
                row_imgs.append(_fig_to_pil(fig))
                plt.close(fig)

            combined = _stitch_vertical(row_imgs)
            out_path = out_root / f"combined_{k:03d}.png"
            combined.save(str(out_path))
            print(f"saved {out_path}  ({combined.width}×{combined.height})")
    finally:
        for adapter, _ in adapters.values():
            adapter.close()

    print(f"\nDone — {args.n_samples} combined image(s) under {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified visualizer for brainco / arctic / taco / hot3d / oakinkv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=_ALL_DATASETS, default=None,
        help=f"하나 이상 선택 (default: 전체 {_ALL_DATASETS})",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="단일 dataset 사용 시 기본 경로 override (--datasets 한 개와 함께 사용)",
    )
    parser.add_argument("--window_time",        type=float, default=0.1,
                        help="(brainco) 윈도우 길이 초")
    parser.add_argument("--window_overlap",     type=float, default=0.5,
                        help="(brainco) 윈도우 겹침 비율")
    parser.add_argument("--interpolating_freq", type=int,   default=100,
                        help="(brainco) 데이터 주파수 Hz")
    parser.add_argument("--n_samples",          type=int,   default=6,
                        help="데이터셋당 시각화할 샘플 수")
    parser.add_argument("--out",                type=str,   default="visualization_all",
                        help="출력 폴더 (각 데이터셋은 <out>/<dataset>/ 하위에 저장)")
    main(parser.parse_args())
