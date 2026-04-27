#!/usr/bin/env python3
"""
retrieve_embedding.py

Given one input `.pkl` file and one or more target `.pkl` files, compute frame-level
embeddings with MultiSensorTransformer and retrieve the nearest target frame for
each input frame.

The script assumes the `.pkl` format stores integer frame ids whose values contain:
  - "lh", "rh"                 : (21, 4) joint xyz + contact
  - "lh_wrist_pos_vr", "rh_wrist_pos_vr"
  - "lh_wrist_rot6d_vr", "rh_wrist_rot6d_vr"

Usage:
    python retrieve_embedding.py \
      --checkpoint checkpoints/dinov2_multi_sensor_pretrained/epoch-0200-all.ckpt \
      --input_pkl pretraining_dataset/TACO/foo.pkl \
      --target_pkls pretraining_dataset/TACO/bar.pkl pretraining_dataset/TACO/baz.pkl \
      --output_json retrieve_results.json

    python retrieve_embedding.py \
      --checkpoint checkpoints/dinov2_multi_sensor_pretrained/epoch-0200-all.ckpt \
      --input_pkl pretraining_dataset/TACO/foo.pkl \
      --target_list target_pkls.txt \
      --output_json retrieve_results.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from tactile_ssl.model.multi_sensor_transformer import multi_sensor_tiny


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieve nearest target frame indices using frame-level embeddings."
    )
    p.add_argument("--checkpoint", required=True, help="Path to encoder checkpoint")
    p.add_argument("--input_pkl", required=True, help="Query/input .pkl file")
    p.add_argument("--target_pkls", nargs="*", default=None, help="One or more target .pkl files")
    p.add_argument(
        "--target_list",
        default=None,
        help="Text file containing target .pkl paths, one per line",
    )
    p.add_argument("--output_json", default="retrieve_results.json", help="Output JSON path")
    p.add_argument(
        "--video_root",
        default="../OakInkv2/TACO/TACO_dataset/Egocentric_RGB_Videos",
        help="Root directory for TACO egocentric RGB videos",
    )
    p.add_argument(
        "--visualize_png",
        default=None,
        help="If set, save retrieval visualization PNG using RGB video frames",
    )
    p.add_argument(
        "--max_visualizations",
        type=int,
        default=10,
        help="Maximum number of query/match rows to draw in the visualization",
    )

    p.add_argument("--sequence_length", type=int, default=3, help="Encoder sequence length")
    p.add_argument("--time_chunk_size", type=int, default=3, help="Encoder time chunk size")
    p.add_argument("--in_dim", type=int, default=42, help="Number of joints")
    p.add_argument("--in_chans", type=int, default=1, help="Input channels before model-side padding")

    p.add_argument("--metric", choices=["cosine", "l2"], default="cosine", help="Nearest-neighbor metric")
    p.add_argument("--top_k", type=int, default=5, help="How many nearest target frames to return")
    p.add_argument("--query_frame", type=int, default=None, help="If set, retrieve only this input frame index")
    p.add_argument(
        "--context_mode",
        choices=["center", "repeat"],
        default="center",
        help="How to build a sequence_length context around each frame",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    return p.parse_args()


def load_encoder(checkpoint_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """Load encoder weights from a checkpoint, stripping common prefixes."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("encoder") or ckpt
    else:
        state_dict = ckpt

    cleaned = {}
    prefixes = ["teacher_encoder.backbone."]
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


def resolve_target_pkls(
    input_pkl: str,
    target_pkls: Optional[List[str]],
    target_list: Optional[str],
) -> List[str]:
    """Merge CLI target list and target-list file into one deduplicated path list."""
    resolved: List[str] = []
    input_resolved = str(Path(input_pkl).resolve())

    if target_pkls:
        resolved.extend(target_pkls)

    if target_list is not None:
        list_path = Path(target_list)
        if not list_path.exists():
            raise FileNotFoundError(f"--target_list file not found: {list_path}")
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                resolved.append(line)

    deduped: List[str] = []
    seen = set()
    for p in resolved:
        rp = str(Path(p).resolve())
        if rp == input_resolved:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(rp)

    if not deduped:
        raise ValueError(
            "Provide at least one target via --target_pkls or --target_list "
            "(excluding the input_pkl itself)."
        )

    return deduped


def load_sequence_from_pkl(path: str) -> Dict[str, np.ndarray]:
    """Load a single OakInkV2/TACO/Arctic-style pkl into dense arrays."""
    with open(path, "rb") as f:
        raw = pickle.load(f)

    frame_indices = sorted(k for k in raw.keys() if isinstance(k, int))
    if not frame_indices:
        raise ValueError(f"No integer frame keys found in {path}")

    n = len(frame_indices)
    joint_data = np.zeros((n, 42, 4), dtype=np.float32)
    wrist_poses = np.zeros((n, 2, 9), dtype=np.float32)

    for i, t in enumerate(frame_indices):
        frame = raw[t]
        joint_data[i, :21] = frame["lh"]
        joint_data[i, 21:] = frame["rh"]
        wrist_poses[i, 0, :3] = frame["lh_wrist_pos_vr"]
        wrist_poses[i, 0, 3:] = frame["lh_wrist_rot6d_vr"]
        wrist_poses[i, 1, :3] = frame["rh_wrist_pos_vr"]
        wrist_poses[i, 1, 3:] = frame["rh_wrist_rot6d_vr"]

    return {
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "sensor": joint_data[..., 3:4].copy(),
        "sensor_poses": joint_data[..., :3].copy(),
        "wrist_poses": wrist_poses,
    }


def build_frame_context(seq: Dict[str, np.ndarray], center_idx: int, seq_len: int, mode: str) -> Dict[str, np.ndarray]:
    """Create a sequence_length window for a given frame index."""
    if mode == "repeat":
        idxs = np.full(seq_len, center_idx, dtype=np.int64)
    else:
        left = seq_len // 2
        raw = np.arange(center_idx - left, center_idx - left + seq_len)
        idxs = np.clip(raw, 0, seq["sensor"].shape[0] - 1)

    return {
        "sensor": seq["sensor"][idxs],
        "sensor_poses": seq["sensor_poses"][idxs],
        "wrist_poses": seq["wrist_poses"][idxs],
    }


def run_encoder(
    model: torch.nn.Module,
    sensor_in: torch.Tensor,
    poses_in: torch.Tensor,
    wrist_in: Optional[torch.Tensor],
    sensor_ids: torch.Tensor,
) -> torch.Tensor:
    out = model.forward_features(
        sensor_in,
        poses_in,
        wrist_poses=wrist_in,
        sensor_ids=sensor_ids,
    )
    return out["x_norm_regtokens"][:, 0, :]


@torch.no_grad()
def embed_sequence_frames(
    model: torch.nn.Module,
    seq: Dict[str, np.ndarray],
    device: str,
    context_mode: str,
) -> torch.Tensor:
    """Compute one embedding per frame using temporal context around that frame."""
    model.eval()
    model.to(device)

    seq_len = model.sequence_length
    in_chans = model.in_chans
    all_embeds: List[torch.Tensor] = []

    for frame_idx in range(seq["sensor"].shape[0]):
        sample = build_frame_context(seq, frame_idx, seq_len, context_mode)

        sensor = torch.from_numpy(sample["sensor"]).float().unsqueeze(0).to(device)          # (1, T, 42, 1)
        if sensor.shape[-1] < in_chans:
            pad = torch.zeros(*sensor.shape[:-1], in_chans - sensor.shape[-1], device=device, dtype=sensor.dtype)
            sensor = torch.cat([sensor, pad], dim=-1)

        poses = torch.from_numpy(sample["sensor_poses"]).float().unsqueeze(0).to(device)     # (1, T, 42, 3)
        wrist = torch.from_numpy(sample["wrist_poses"]).float().unsqueeze(0).to(device)      # (1, T, 2, 9)

        sensor_ids = torch.zeros(1, dtype=torch.long, device=device)
        emb = run_encoder(model, sensor, poses, wrist, sensor_ids)                            # (1, D)
        all_embeds.append(emb.cpu())

    return torch.cat(all_embeds, dim=0)


def retrieve_nearest(
    input_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    target_infos: List[Dict],
    top_k: int,
    metric: str,
) -> tuple[List[List[int]], List[List[float]]]:
    """Return top-k nearest target indices and scores with unique target sequences."""
    if metric == "cosine":
        input_norm = torch.nn.functional.normalize(input_embeddings, dim=1)
        target_norm = torch.nn.functional.normalize(target_embeddings, dim=1)
        scores = input_norm @ target_norm.T
        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        score_matrix = scores
        largest_is_better = True
    else:
        dists = torch.cdist(input_embeddings, target_embeddings, p=2)
        sorted_indices = torch.argsort(dists, dim=1, descending=False)
        score_matrix = dists
        largest_is_better = False

    selected_indices_all: List[List[int]] = []
    selected_scores_all: List[List[float]] = []

    for row_idx in range(sorted_indices.shape[0]):
        used_pkls = set()
        row_selected_indices: List[int] = []
        row_selected_scores: List[float] = []

        for flat_idx_t in sorted_indices[row_idx]:
            flat_idx = int(flat_idx_t.item())
            target_pkl = target_infos[flat_idx]["target_pkl"]
            if target_pkl in used_pkls:
                continue
            used_pkls.add(target_pkl)
            row_selected_indices.append(flat_idx)
            row_selected_scores.append(float(score_matrix[row_idx, flat_idx].item()))
            if len(row_selected_indices) >= top_k:
                break

        selected_indices_all.append(row_selected_indices)
        selected_scores_all.append(row_selected_scores)

    return selected_indices_all, selected_scores_all


def build_results(
    input_path: str,
    input_seq: Dict[str, np.ndarray],
    target_infos: List[Dict],
    top_indices: List[List[int]],
    top_scores: List[List[float]],
    metric: str,
    query_frame: Optional[int],
) -> Dict:
    """Convert retrieval outputs to a JSON-serializable structure."""
    frame_ids = range(input_seq["frame_indices"].shape[0]) if query_frame is None else [query_frame]
    results = []

    for row_idx, input_frame_idx in enumerate(frame_ids):
        matches = []
        for rank, flat_target_idx in enumerate(top_indices[row_idx]):
            score = float(top_scores[row_idx][rank])
            target_info = target_infos[flat_target_idx]
            matches.append({
                "rank": rank + 1,
                "target_pkl": target_info["target_pkl"],
                "target_pkl_index": int(target_info["target_pkl_index"]),
                "target_frame_index": int(target_info["target_frame_index"]),
                "target_frame_id": int(target_info["target_frame_id"]),
                "num_target_frames": int(target_info["num_target_frames"]),
                metric: score,
            })

        results.append({
            "input_frame_index": int(input_frame_idx),
            "input_frame_id": int(input_seq["frame_indices"][input_frame_idx]),
            "matches": matches,
        })

    return {
        "input_pkl": str(Path(input_path).resolve()),
        "target_pkls": sorted({str(Path(info["target_pkl"]).resolve()) for info in target_infos}),
        "metric": metric,
        "num_input_frames": int(input_seq["frame_indices"].shape[0]),
        "num_target_frames_total": int(len(target_infos)),
        "results": results,
    }


def taco_video_path_from_pkl(pkl_path: str, video_root: str) -> Path:
    """Map `foo__bar__baz__YYYYMMDD_NNN.pkl` to TACO RGB video path."""
    p = Path(pkl_path)
    parts = p.stem.split("__")
    if len(parts) < 4:
        raise ValueError(f"Unsupported TACO pkl naming format: {p.name}")

    action, tool, obj = parts[0], parts[1], parts[2]
    timestamp = "__".join(parts[3:])
    root = Path(video_root)

    # TACO folder names sometimes use spaces where pkl names use underscores,
    # e.g. `put_in` -> `put in`, `pour_in_some` -> `pour in some`.
    candidates = [
        (action, tool, obj),
        (action.replace("_", " "), tool.replace("_", " "), obj.replace("_", " ")),
    ]

    for action_name, tool_name, obj_name in candidates:
        video_dir = root / f"({action_name}, {tool_name}, {obj_name})" / timestamp
        video_path = video_dir / "color.mp4"
        if video_path.exists():
            return video_path

    raise FileNotFoundError(
        f"RGB video not found for {p.name}. Tried: "
        + ", ".join(
            str(root / f"({a}, {t}, {o})" / timestamp / "color.mp4")
            for a, t, o in candidates
        )
    )


def load_video_frame(video_path: Path, frame_idx: int):
    """Load one RGB frame from an mp4 using OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Visualization requires opencv-python in the active environment."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, max(total - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), total


def get_video_num_frames(video_path: Path) -> int:
    """Return the total frame count for an mp4."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Visualization requires opencv-python in the active environment."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def map_pkl_frame_to_video_frame(pkl_frame_idx: int, num_pkl_frames: int, num_video_frames: int) -> int:
    """Approximate frame alignment by relative position in the sequence."""
    if num_video_frames <= 0:
        return 0
    return min(int(round(pkl_frame_idx * num_video_frames / max(num_pkl_frames, 1))), num_video_frames - 1)


def visualize_retrieval(
    result_dict: Dict,
    input_seq: Dict[str, np.ndarray],
    video_root: str,
    output_png: str,
    max_rows: int,
) -> None:
    """Render query and top-k retrieved target RGB frames as a comparison grid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib in the active environment."
        ) from exc

    def _draw_missing(ax, title: str, message: str) -> None:
        ax.axis("off")
        ax.text(
            0.5, 0.5, message,
            ha="center", va="center",
            fontsize=9, color="crimson",
            wrap=True,
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=9)

    input_video = taco_video_path_from_pkl(result_dict["input_pkl"], video_root)
    input_video_total = get_video_num_frames(input_video)
    target_video_totals: Dict[str, int] = {}

    all_rows = result_dict["results"]
    if len(all_rows) <= max_rows:
        rows = all_rows
    else:
        sampled_idxs = np.linspace(0, len(all_rows) - 1, num=max_rows, dtype=int)
        rows = [all_rows[i] for i in sampled_idxs]
    if not rows:
        raise ValueError("No retrieval rows available for visualization.")

    num_match_cols = max(len(row["matches"]) for row in rows)
    num_cols = 1 + num_match_cols
    fig, axes = plt.subplots(len(rows), num_cols, figsize=(4.8 * num_cols, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    if num_cols == 1:
        axes = axes.reshape(len(rows), 1)

    fig.suptitle(
        f"Nearest-frame retrieval\ninput={Path(result_dict['input_pkl']).name}  "
        f"targets={len(result_dict['target_pkls'])} files  top-k={num_match_cols}",
        fontsize=11,
    )

    for row_idx, row in enumerate(rows):
        input_frame_idx = row["input_frame_index"]

        input_video_frame_idx = map_pkl_frame_to_video_frame(
            input_frame_idx,
            result_dict["num_input_frames"],
            input_video_total,
        )
        input_rgb, _ = load_video_frame(input_video, input_video_frame_idx)

        ax_q = axes[row_idx, 0]
        ax_q.imshow(input_rgb)
        ax_q.axis("off")

        score_key = result_dict["metric"]
        ax_q.set_title(
            f"Query\npkl frame={input_frame_idx}  video frame={input_video_frame_idx}",
            fontsize=9,
        )

        for match_idx in range(num_match_cols):
            ax_t = axes[row_idx, match_idx + 1]
            ax_t.axis("off")

            if match_idx >= len(row["matches"]):
                _draw_missing(ax_t, f"Top-{match_idx + 1}", "No match")
                continue

            match = row["matches"][match_idx]
            target_frame_idx = match["target_frame_index"]
            target_pkl = match["target_pkl"]

            try:
                target_video = taco_video_path_from_pkl(target_pkl, video_root)
                if target_pkl not in target_video_totals:
                    target_video_totals[target_pkl] = get_video_num_frames(target_video)

                target_video_frame_idx = map_pkl_frame_to_video_frame(
                    target_frame_idx,
                    match["num_target_frames"],
                    target_video_totals[target_pkl],
                )
                target_rgb, _ = load_video_frame(target_video, target_video_frame_idx)
                ax_t.imshow(target_rgb)
                ax_t.set_title(
                    f"Top-{match_idx + 1}\npkl frame={target_frame_idx}  video frame={target_video_frame_idx}\n"
                    f"{Path(target_pkl).name}\n{score_key}={match[score_key]:.4f}",
                    fontsize=9,
                )
            except FileNotFoundError as exc:
                _draw_missing(
                    ax_t,
                    f"Top-{match_idx + 1}\npkl frame={target_frame_idx}\n{Path(target_pkl).name}",
                    f"RGB video missing\n{exc}",
                )

    plt.tight_layout()
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved retrieval visualization to {out_path.resolve()}")


def main() -> None:
    args = parse_args()
    target_pkls = resolve_target_pkls(args.input_pkl, args.target_pkls, args.target_list)

    print(
        f"Building MultiSensorTransformer (tiny)  seq={args.sequence_length}  "
        f"chunk={args.time_chunk_size}  in_dim={args.in_dim}  in_chans={args.in_chans}"
    )
    model = multi_sensor_tiny(
        in_dim=args.in_dim,
        in_chans=args.in_chans,
        sequence_length=args.sequence_length,
        time_chunk_size=args.time_chunk_size,
    )
    model = load_encoder(args.checkpoint, model)

    print(f"\nLoading input sequence: {args.input_pkl}")
    input_seq = load_sequence_from_pkl(args.input_pkl)
    print(f"Input frames: {input_seq['frame_indices'].shape[0]}")

    print(f"\nEmbedding input frames on {args.device} ...")
    input_embeddings = embed_sequence_frames(model, input_seq, args.device, args.context_mode)

    print(f"\nLoading {len(target_pkls)} target sequence(s) ...")
    target_embedding_list: List[torch.Tensor] = []
    target_infos: List[Dict] = []
    for target_pkl_idx, target_pkl in enumerate(target_pkls):
        print(f"Loading target sequence: {target_pkl}")
        target_seq = load_sequence_from_pkl(target_pkl)
        print(f"Target frames: {target_seq['frame_indices'].shape[0]}")

        print(f"Embedding target frames on {args.device} ...")
        target_embeddings_one = embed_sequence_frames(model, target_seq, args.device, args.context_mode)
        target_embedding_list.append(target_embeddings_one)
        for frame_idx, frame_id in enumerate(target_seq["frame_indices"]):
            target_infos.append({
                "target_pkl": str(Path(target_pkl).resolve()),
                "target_pkl_index": target_pkl_idx,
                "target_frame_index": int(frame_idx),
                "target_frame_id": int(frame_id),
                "num_target_frames": int(target_seq["frame_indices"].shape[0]),
            })

    target_embeddings = torch.cat(target_embedding_list, dim=0)

    if args.query_frame is not None:
        if args.query_frame < 0 or args.query_frame >= input_embeddings.shape[0]:
            raise ValueError(
                f"--query_frame {args.query_frame} is out of range for {input_embeddings.shape[0]} frames"
            )
        query_embeddings = input_embeddings[args.query_frame : args.query_frame + 1]
    else:
        query_embeddings = input_embeddings

    print(f"\nRetrieving nearest target frames with metric={args.metric} ...")
    top_indices, top_scores = retrieve_nearest(
        query_embeddings,
        target_embeddings,
        target_infos,
        args.top_k,
        args.metric,
    )

    result_dict = build_results(
        input_path=args.input_pkl,
        input_seq=input_seq,
        target_infos=target_infos,
        top_indices=top_indices,
        top_scores=top_scores,
        metric=args.metric,
        query_frame=args.query_frame,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    print(f"Saved retrieval results to {out_path.resolve()}")
    if result_dict["results"]:
        first = result_dict["results"][0]
        best = first["matches"][0]
        print(
            f"Example: input frame {first['input_frame_index']} "
            f"-> target frame {best['target_frame_index']} "
            f"({args.metric}={best[args.metric]:.6f})"
        )

    if args.visualize_png is not None:
        visualize_retrieval(
            result_dict=result_dict,
            input_seq=input_seq,
            video_root=args.video_root,
            output_png=args.visualize_png,
            max_rows=args.max_visualizations,
        )


if __name__ == "__main__":
    main()
