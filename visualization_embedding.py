#!/usr/bin/env python3
"""
visualization_embedding.py

Standalone script to visualize per-window register-token embeddings
from a trained MultiSensorTransformer using t-SNE.

Supports two dataset types:
  brainco   — BraincoGraspDetectionDataset (grasp_success / grasp_fail episodes)
  oakinkv2  — OakInkV2TactileDataset       (pretraining sequences, no labels)

For each episode/sequence:
  - Each window → one embedding (register token, averaged over temporal chunks)
  - Points are colored by window order within the episode (frame progression)

Usage:
    # BrainCo grasp dataset
    python visualization_embedding.py \
        --checkpoint checkpoints/epoch-0210.ckpt \
        --data_path dataset/brainco/downstream/grasp_prediction \
        --dataset_type brainco

    # OakInkV2 pretraining dataset
    python visualization_embedding.py \
        --checkpoint checkpoints/epoch-0210.ckpt \
        --data_path pretraining_dataset/OakInkv2 \
        --dataset_type oakinkv2 \
        --window_size 3 --window_stride 30 \
        --num_sequences 20
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap.umap_ as umap_lib
from omegaconf import OmegaConf

# Register custom Hydra resolvers used elsewhere in the codebase
OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b), replace=True)
OmegaConf.register_new_resolver("int_divide",   lambda a, b: a // b,     replace=True)
OmegaConf.register_new_resolver("capitalize",   lambda s: s.title(),     replace=True)

from tactile_ssl.model.multi_sensor_transformer import multi_sensor_tiny
from tactile_ssl.data.brainco_grasp_dataset import BraincoGraspDetectionDataset
from tactile_ssl.data.oakinkv2_tactile import OakInkV2TactileDataset


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="t-SNE visualization of MultiSensorTransformer register-token embeddings"
    )

    # Required
    p.add_argument("--checkpoint", required=True,
                   help="Path to encoder checkpoint (.ckpt / .pt)")
    p.add_argument("--data_path", required=True,
                   help="Dataset root (grasp_success/grasp_fail for brainco; "
                        "OakInkv2 pkl dir for oakinkv2)")

    # Dataset type
    p.add_argument("--dataset_type", default="brainco",
                   choices=["brainco", "oakinkv2"],
                   help="Dataset to visualize (default: brainco)")

    # Output
    p.add_argument("--output", default="embedding_tsne.png",
                   help="Output image path (default: embedding_tsne.png)")

    # Model architecture — must match the checkpoint
    p.add_argument("--sequence_length", type=int, default=3,
                   help="Encoder sequence_length (default: 3)")
    p.add_argument("--time_chunk_size", type=int, default=3,
                   help="Encoder time_chunk_size (default: 3)")
    p.add_argument("--in_dim",          type=int, default=42,
                   help="Number of joints/sensors for the encoder (default: 42)")
    p.add_argument("--in_chans",        type=int, default=4,
                   help="Sensor channels expected by the encoder (default: 4)")

    # ── BrainCo-specific ──────────────────────────────────────────────────────
    p.add_argument("--urdf_path", default="dataset/brainco/urdf",
                   help="[brainco] URDF directory")
    p.add_argument("--use_skeleton_poses", action="store_true",
                   help="[brainco] Use 42-joint skeleton poses instead of 10 fingertip poses")
    p.add_argument("--window_time",    type=float, default=0.1,
                   help="[brainco] Window duration in seconds (default: 0.1)")
    p.add_argument("--window_overlap", type=float, default=0.5,
                   help="[brainco] Window overlap ratio (default: 0.5)")
    p.add_argument("--num_episodes", type=int, default=None,
                   help="[brainco] Number of episodes to visualize (default: all)")
    p.add_argument("--max_episodes_per_class", type=int, default=None,
                   help="[brainco] Cap episodes per class independently")

    # ── OakInkV2-specific ─────────────────────────────────────────────────────
    p.add_argument("--window_size",   type=int, default=3,
                   help="[oakinkv2] Frames per window (default: 3)")
    p.add_argument("--window_stride", type=int, default=30,
                   help="[oakinkv2] Stride between windows (default: 30 — sparse sampling)")
    p.add_argument("--split",         default="train", choices=["train", "val"],
                   help="[oakinkv2] Dataset split (default: train)")
    p.add_argument("--scenes",        nargs="*", default=None,
                   help="[oakinkv2] Scene filter, e.g. --scenes scene_01 scene_02")
    p.add_argument("--num_sequences", type=int, default=None,
                   help="[oakinkv2] Number of sequences to visualize (default: all)")

    # UMAP hyper-parameters
    p.add_argument("--n_neighbors", type=int,   default=15,
                   help="UMAP n_neighbors (default: 15)")
    p.add_argument("--min_dist",    type=float, default=0.1,
                   help="UMAP min_dist (default: 0.1)")

    # Misc
    p.add_argument("--batch_size", type=int, default=64,
                   help="Inference batch size (default: 64)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device (default: cuda if available)")

    return p.parse_args()


# ── checkpoint loading ────────────────────────────────────────────────────────

def load_encoder(checkpoint_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """Load encoder weights from a checkpoint, stripping common key prefixes."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Unwrap state dict from various checkpoint formats
    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("state_dict")
            or ckpt.get("model")
            or ckpt.get("encoder")
            or ckpt
        )
    else:
        state_dict = ckpt

    # Strip common key prefixes produced by SLModule / DDP / Lightning
    prefixes = [
        "teacher_encoder.backbone.",
    ]
    cleaned = {}
    # print(state_dict.keys())

    for k, v in state_dict.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
                cleaned[k] = v
                break

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[load_encoder] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[load_encoder] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    print(f"[load_encoder] Loaded from {checkpoint_path}")
    return model


# ── embedding extraction ──────────────────────────────────────────────────────

def _run_encoder(
    model: torch.nn.Module,
    sensor_in: torch.Tensor,
    poses_in: torch.Tensor,
    wrist_in: Optional[torch.Tensor],
    sensor_ids: torch.Tensor,
) -> torch.Tensor:
    """Forward one batch through the encoder and return register tokens (B, D)."""
    out = model.forward_features(
        sensor_in, poses_in,
        wrist_poses=wrist_in,
        sensor_ids=sensor_ids,
    )
    return out["x_norm_regtokens"][:, 0, :]   # (B, D)


@torch.no_grad()
def extract_embeddings_brainco(
    model: torch.nn.Module,
    dataset: BraincoGraspDetectionDataset,
    num_episodes: Optional[int],
    max_episodes_per_class: Optional[int],
    batch_size: int,
    device: str,
):
    """Extract one register-token embedding per window for each selected episode (BrainCo).

    Returns:
        embeddings : np.ndarray (total_windows, embed_dim)
        metadata   : list of dicts { episode_idx, window_idx, num_windows, label }
    """
    model.eval()
    model.to(device)

    episodes = dataset.episode_data  # list of {path, window_starts, label}

    # ── episode selection ─────────────────────────────────────────────────────
    if max_episodes_per_class is not None:
        success_eps = [e for e in episodes if e["label"] == 1][:max_episodes_per_class]
        fail_eps    = [e for e in episodes if e["label"] == 0][:max_episodes_per_class]
        episodes = success_eps + fail_eps

    if num_episodes is not None:
        episodes = episodes[:num_episodes]

    # ── build episode → window index range map ────────────────────────────────
    # dataset.windows is flat; episodes are appended in order, so we can
    # reconstruct ranges by counting window_starts per episode.
    flat_ranges: List[tuple] = []   # (start, end) index into dataset.windows
    cursor = 0
    for ep in dataset.episode_data:   # iterate full dataset to get correct offsets
        n = len(ep["window_starts"])
        flat_ranges.append((cursor, cursor + n))
        cursor += n

    # Map selected episodes to their window ranges
    selected_ep_path_to_range = {}
    for ep_full, (start, end) in zip(dataset.episode_data, flat_ranges):
        selected_ep_path_to_range[ep_full["path"]] = (start, end)

    T   = model.sequence_length   # frames per encoder call
    all_embeddings: List[torch.Tensor] = []
    all_metadata:   List[dict]         = []

    for ep_idx, ep_info in enumerate(episodes):
        ep_path = ep_info["path"]
        label   = ep_info["label"]

        start, end = selected_ep_path_to_range[ep_path]
        windows = dataset.windows[start:end]   # list of window dicts (temporal order)

        if not windows:
            continue

        ep_embeddings: List[torch.Tensor] = []

        # ── batch-wise inference ──────────────────────────────────────────────
        for b_start in range(0, len(windows), batch_size):
            batch = windows[b_start : b_start + batch_size]
            B = len(batch)

            # Stack tensors: each window has shape (W, *, *)
            sensor      = torch.stack([w["sensor"]       for w in batch])        # (B, W, 10, 4)
            sensor_poses = torch.stack([w["sensor_poses"] for w in batch])       # (B, W, N_pos, 3or6)
            wrist_raw   = [w.get("wrist_poses") for w in batch]
            wrist_poses = (
                torch.stack(wrist_raw).to(device)                                # (B, W, 2, 9)
                if wrist_raw[0] is not None else None
            )

            sensor       = sensor.to(device)
            sensor_poses = sensor_poses.to(device)

            W      = sensor.shape[1]
            N_pos  = sensor_poses.shape[2]

            # ── split W frames into G = W // T temporal chunks ────────────────
            # If W is not divisible, drop the remainder (tail frames).
            G = W // T
            if G == 0:
                # Window shorter than sequence_length: pad by repeating last frame
                pad = T - W
                sensor       = torch.cat([sensor,       sensor[:, -1:].expand(-1, pad, -1, -1)], dim=1)
                sensor_poses = torch.cat([sensor_poses, sensor_poses[:, -1:].expand(-1, pad, -1, -1)], dim=1)
                if wrist_poses is not None:
                    wrist_poses = torch.cat([wrist_poses, wrist_poses[:, -1:].expand(-1, pad, -1, -1)], dim=1)
                G = 1

            usable = G * T
            sensor_in = sensor[:, :usable].reshape(B * G, T, sensor.shape[2], 4)        # (B*G, T, 10, 4)
            poses_in  = sensor_poses[:, :usable, :, :3].reshape(B * G, T, N_pos, 3)    # (B*G, T, N_pos, 3)
            wrist_in  = (
                wrist_poses[:, :usable].reshape(B * G, T, 2, 9)
                if wrist_poses is not None else None
            )

            # sensor_ids = 1: consistent with multi_sensor_grasp_sl.py
            sensor_ids = torch.ones(B * G, dtype=torch.long, device=device)

            reg = _run_encoder(model, sensor_in, poses_in, wrist_in, sensor_ids)  # (B*G, D)
            reg = reg.reshape(B, G, -1).mean(dim=1)                               # (B,   D)

            ep_embeddings.append(reg.cpu())

        ep_emb = torch.cat(ep_embeddings, dim=0)   # (num_windows_in_ep, D)
        all_embeddings.append(ep_emb)

        win_size = windows[0]["sensor"].shape[0]   # W frames per window (BrainCo)
        n_wins = ep_emb.shape[0]
        for w_idx in range(n_wins):
            all_metadata.append({
                "episode_idx": ep_idx,
                "window_idx":  w_idx,
                "num_windows": n_wins,
                "window_size": win_size,
                "label":       label,   # 0=fail / 1=success
            })

        print(f"  episode {ep_idx:3d} | label={label} | windows={n_wins} | path={Path(ep_path).name}")

    embeddings_np = torch.cat(all_embeddings, dim=0).float().numpy()
    return embeddings_np, all_metadata


@torch.no_grad()
def extract_embeddings_oakinkv2(
    model: torch.nn.Module,
    dataset: OakInkV2TactileDataset,
    num_sequences: Optional[int],
    batch_size: int,
    device: str,
):
    """Extract one register-token embedding per window for each sequence (OakInkV2).

    OakInkV2 windows are (seq_idx, start_frame) tuples; this function groups
    them by sequence and processes them in temporal order.

    sensor shape from dataset : (T, 42, 1)  — only contact channel
    encoder expects           : (B, T, N, in_chans)
    → channels are zero-padded to match model.in_chans before inference.

    Returns:
        embeddings : np.ndarray (total_windows, embed_dim)
        metadata   : list of dicts { episode_idx, window_idx, num_windows, label=-1 }
    """
    model.eval()
    model.to(device)

    in_chans = model.in_chans   # e.g. 4

    # ── group windows by sequence index, sorted by start frame ───────────────
    seq_to_wins: dict = {}
    for flat_idx, (seq_idx, start) in enumerate(dataset.windows):
        seq_to_wins.setdefault(seq_idx, []).append((start, flat_idx))
    for seq_idx in seq_to_wins:
        seq_to_wins[seq_idx].sort(key=lambda x: x[0])   # sort by start frame

    seq_ids = sorted(seq_to_wins.keys())
    if num_sequences is not None:
        seq_ids = seq_ids[:num_sequences]

    T = model.sequence_length
    all_embeddings: List[torch.Tensor] = []
    all_metadata:   List[dict]         = []

    for ep_i, seq_idx in enumerate(seq_ids):
        wins = seq_to_wins[seq_idx]   # [(start_frame, flat_idx), ...]
        flat_idxs = [fi for _, fi in wins]

        ep_embeddings: List[torch.Tensor] = []

        for b_start in range(0, len(flat_idxs), batch_size):
            batch_idxs = flat_idxs[b_start : b_start + batch_size]
            samples = [dataset[i] for i in batch_idxs]
            B = len(samples)

            # sensor: (B, T, 42, 1) → pad channels → (B, T, 42, in_chans)
            sensor = torch.stack([s["sensor"] for s in samples]).to(device)          # (B, T, 42, 1)
            if sensor.shape[-1] < in_chans:
                pad = torch.zeros(*sensor.shape[:-1], in_chans - sensor.shape[-1],
                                  device=device, dtype=sensor.dtype)
                sensor = torch.cat([sensor, pad], dim=-1)                            # (B, T, 42, in_chans)

            # sensor_poses: (B, T, 42, 3)  — already xyz
            sensor_poses = torch.stack([s["sensor_poses"] for s in samples]).to(device)

            # wrist_poses: (B, T, 2, 9)
            wrist_raw = [s.get("wrist_poses") for s in samples]
            wrist_poses = (
                torch.stack(wrist_raw).to(device)
                if wrist_raw[0] is not None else None
            )

            W = sensor.shape[1]
            G = W // T
            if G == 0:
                pad_len = T - W
                sensor       = torch.cat([sensor,       sensor[:, -1:].expand(-1, pad_len, -1, -1)], dim=1)
                sensor_poses = torch.cat([sensor_poses, sensor_poses[:, -1:].expand(-1, pad_len, -1, -1)], dim=1)
                if wrist_poses is not None:
                    wrist_poses = torch.cat([wrist_poses, wrist_poses[:, -1:].expand(-1, pad_len, -1, -1)], dim=1)
                G = 1

            usable    = G * T
            N         = sensor.shape[2]
            sensor_in = sensor[:, :usable].reshape(B * G, T, N, in_chans)
            poses_in  = sensor_poses[:, :usable, :, :3].reshape(B * G, T, N, 3)
            wrist_in  = (
                wrist_poses[:, :usable].reshape(B * G, T, 2, 9)
                if wrist_poses is not None else None
            )

            sensor_ids = torch.ones(B * G, dtype=torch.long, device=device)

            reg = _run_encoder(model, sensor_in, poses_in, wrist_in, sensor_ids)  # (B*G, D)
            reg = reg.reshape(B, G, -1).mean(dim=1)                               # (B,   D)

            ep_embeddings.append(reg.cpu())

        ep_emb = torch.cat(ep_embeddings, dim=0)
        all_embeddings.append(ep_emb)

        n_wins = ep_emb.shape[0]
        for w_idx in range(n_wins):
            all_metadata.append({
                "episode_idx": ep_i,
                "window_idx":  w_idx,
                "num_windows": n_wins,
                "window_size": dataset.window_size,
                "label":       -1,       # OakInkV2 has no grasp label
            })

        seq_name = Path(dataset._seq_paths[seq_idx]).name
        print(f"  seq {ep_i:3d} (seq_idx={seq_idx}) | windows={n_wins} | {seq_name}")

    embeddings_np = torch.cat(all_embeddings, dim=0).float().numpy()
    return embeddings_np, all_metadata


# ── UMAP + plot ───────────────────────────────────────────────────────────────

# Colormaps cycling for up to ~20 episodes
_EPISODE_CMAPS = [
    "Blues", "Oranges", "Greens", "Reds", "Purples",
    "YlOrBr", "PuRd", "BuGn", "GnBu", "RdPu",
    "YlGn",  "OrRd",  "PuBu",  "BuPu", "YlGnBu",
    "RdYlBu", "Spectral", "coolwarm", "bwr", "seismic",
]


def plot_umap(
    embeddings:  np.ndarray,
    metadata:    List[dict],
    output_path: str,
    n_neighbors: int,
    min_dist:    float,
):
    """Run UMAP and produce two side-by-side plots.

    Left : each episode/sequence gets its own colormap; color encodes frame order
           (light = early frame, dark = late frame).  ▲ = first window, ■ = last.
    Right: BrainCo → grasp label (success/fail).
           OakInkV2 (label=-1) → sequence index coloring.
    """
    n_total = embeddings.shape[0]
    print(f"\nRunning UMAP on {n_total} embeddings (dim={embeddings.shape[1]}) ...")

    reducer = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=min_dist, random_state=42)
    coords  = reducer.fit_transform(embeddings)   # (N_total, 2)

    _, axes = plt.subplots(1, 2, figsize=(16, 7))

    episode_indices = sorted(set(m["episode_idx"] for m in metadata))
    has_labels      = any(m["label"] >= 0 for m in metadata)

    # ── left: frame-order coloring ────────────────────────────────────────────
    ax_order = axes[0]

    for ep_i, ep_idx in enumerate(episode_indices):
        mask = [i for i, m in enumerate(metadata) if m["episode_idx"] == ep_idx]
        if not mask:
            continue

        num_wins = metadata[mask[0]]["num_windows"]
        cmap     = cm.get_cmap(_EPISODE_CMAPS[ep_i % len(_EPISODE_CMAPS)])

        for rank, data_idx in enumerate(mask):
            frac  = rank / max(num_wins - 1, 1)
            color = cmap(0.25 + 0.75 * frac)
            ax_order.scatter(
                coords[data_idx, 0], coords[data_idx, 1],
                c=[color], s=18, alpha=0.85, linewidths=0,
            )

        ax_order.scatter(
            coords[mask[0], 0],  coords[mask[0], 1],
            c="black", s=55, marker="^", zorder=6,
            label=f"ep{ep_idx} start" if ep_i < 5 else "",
        )
        ax_order.scatter(
            coords[mask[-1], 0], coords[mask[-1], 1],
            c="black", s=55, marker="s", zorder=6,
        )

    # ── 100 window마다 실제 프레임 번호(window_idx * window_size)를 레이블 표시 ──
    for i, m in enumerate(metadata):
        if m["window_idx"] % 100 == 0:
            frame_no = m["window_idx"] * m["window_size"]
            for ax in axes:
                ax.annotate(
                    str(frame_no),
                    xy=(coords[i, 0], coords[i, 1]),
                    fontsize=5,
                    color="dimgray",
                    ha="center",
                    va="bottom",
                    zorder=10,
                )

    ax_order.set_title(
        "UMAP — colored by frame order within episode\n"
        "(light=early  dark=late  ▲=first  ■=last)",
        fontsize=11,
    )
    ax_order.set_xlabel("UMAP dim 1")
    ax_order.set_ylabel("UMAP dim 2")
    if len(episode_indices) <= 5:
        ax_order.legend(fontsize=8, markerscale=0.8)

    # ── right: label / sequence-index coloring ────────────────────────────────
    ax_right = axes[1]

    if has_labels:
        # BrainCo: success vs fail
        label_style = {
            0: dict(c="tomato",    label="fail    (0)", marker="x", s=20),
            1: dict(c="steelblue", label="success (1)", marker="o", s=18),
        }
        for lv, style in label_style.items():
            idxs = [i for i, m in enumerate(metadata) if m["label"] == lv]
            if not idxs:
                continue
            ax_right.scatter(coords[idxs, 0], coords[idxs, 1],
                             alpha=0.65, linewidths=0.5, **style)
        ax_right.set_title("UMAP — colored by grasp label", fontsize=11)
        ax_right.legend(fontsize=9)
    else:
        # OakInkV2: color by sequence index
        seq_cmap = cm.get_cmap("tab20")
        n_eps    = len(episode_indices)
        for ep_i, ep_idx in enumerate(episode_indices):
            idxs = [i for i, m in enumerate(metadata) if m["episode_idx"] == ep_idx]
            if not idxs:
                continue
            color = seq_cmap(ep_i / max(n_eps - 1, 1))
            ax_right.scatter(coords[idxs, 0], coords[idxs, 1],
                             c=[color], s=18, alpha=0.7, linewidths=0,
                             label=f"seq{ep_idx}" if ep_i < 10 else "")
        ax_right.set_title("UMAP — colored by sequence index", fontsize=11)
        if n_eps <= 10:
            ax_right.legend(fontsize=8, markerscale=0.8)

    ax_right.set_xlabel("t-SNE dim 1")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out.resolve()}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── build encoder ─────────────────────────────────────────────────────────
    print(f"\nBuilding MultiSensorTransformer (tiny)  "
          f"seq={args.sequence_length}  chunk={args.time_chunk_size}  "
          f"in_dim={args.in_dim}  in_chans={args.in_chans}")
    model = multi_sensor_tiny(
        in_dim=args.in_dim,
        in_chans=args.in_chans,
        sequence_length=args.sequence_length,
        time_chunk_size=args.time_chunk_size,
    )
    model = load_encoder(args.checkpoint, model)
    model.eval()

    # ── build dataset + extract embeddings ───────────────────────────────────
    if args.dataset_type == "brainco":
        data_cfg = OmegaConf.create({
            "window_time":        args.window_time,
            "window_overlap":     args.window_overlap,
            "interpolating_freq": 100,
        })
        print(f"\nLoading BrainCo dataset from {args.data_path} ...")
        dataset = BraincoGraspDetectionDataset(
            config=data_cfg,
            data_path=args.data_path,
            brainco_urdf_path=args.urdf_path,
            use_skeleton_poses=args.use_skeleton_poses,
        )
        print(f"Dataset: {len(dataset.episode_data)} episodes, "
              f"{len(dataset.windows)} windows total")

        print(f"\nExtracting embeddings (device={args.device}, batch={args.batch_size}) ...")
        embeddings, metadata = extract_embeddings_brainco(
            model=model,
            dataset=dataset,
            num_episodes=args.num_episodes,
            max_episodes_per_class=args.max_episodes_per_class,
            batch_size=args.batch_size,
            device=args.device,
        )

    else:  # oakinkv2
        print(f"\nLoading OakInkV2 dataset from {args.data_path} ...")
        dataset = OakInkV2TactileDataset(
            data_root=args.data_path,
            window_size=args.window_size,
            window_stride=args.window_stride,
            split=args.split,
            scenes=args.scenes,
        )
        print(f"Dataset: {len(dataset._sequences)} sequences, "
              f"{len(dataset.windows)} windows total")

        print(f"\nExtracting embeddings (device={args.device}, batch={args.batch_size}) ...")
        embeddings, metadata = extract_embeddings_oakinkv2(
            model=model,
            dataset=dataset,
            num_sequences=args.num_sequences,
            batch_size=args.batch_size,
            device=args.device,
        )

    print(f"\nTotal embeddings: {embeddings.shape[0]}  dim: {embeddings.shape[1]}")

    # ── UMAP + plot ───────────────────────────────────────────────────────────
    plot_umap(
        embeddings=embeddings,
        metadata=metadata,
        output_path=args.output,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )


if __name__ == "__main__":
    main()
