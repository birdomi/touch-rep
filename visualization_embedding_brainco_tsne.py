#!/usr/bin/env python3
"""
Visualize AngleTransformer embeddings with t-SNE.

The class label is derived from the episode/file parent folder by default:
  dataset/brainco/downstream/grasp_prediction_0611/box_succ/episode_0002
  -> class "box_succ"

Example:
    XFORMERS_DISABLED=TRUE python visualization_embedding_brainco_tsne.py \
        --data_path dataset/brainco/downstream/grasp_prediction_0611
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from visualization_embedding_brainco import (
    DEFAULT_LABEL_DIRS,
    FrameRecord,
    build_dataset,
    build_model,
    compute_contact_stats,
    embed_records,
    load_encoder,
    validate_window_shapes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="t-SNE visualization of BrainCo/TACO AngleTransformer embeddings."
    )

    p.add_argument(
        "--checkpoint",
        default="checkpoints/dinov2_angle/epoch-5000-brainco.ckpt",
        help="Path to a pretrained AngleTransformer encoder checkpoint.",
    )
    p.add_argument(
        "--data_path",
        nargs="+",
        default=["dataset/brainco/downstream/grasp_prediction_0611"],
        help="One or more BrainCo roots/episodes or angle-vector .pkl directories.",
    )
    p.add_argument(
        "--select",
        nargs="*",
        default=None,
        help="Optional paths/globs/substrings to select from data_path.",
    )
    p.add_argument("--label_dirs", default=DEFAULT_LABEL_DIRS)

    p.add_argument("--window_time", type=float, default=0.01)
    p.add_argument("--window_overlap", type=float, default=0.0)
    p.add_argument("--interpolating_freq", type=int, default=100)
    p.add_argument(
        "--subtract_baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--brainco_contact_mode",
        choices=["auto", "all", "first", "first_to_42"],
        default="auto",
    )

    p.add_argument("--model_size", choices=["tiny", "small"], default="tiny")
    p.add_argument("--in_dim", type=int, default=None)
    p.add_argument("--in_chans", type=int, default=None)
    p.add_argument("--pos_in_dim", type=int, default=10)
    p.add_argument("--pos_in_chans", type=int, default=4)
    p.add_argument("--sequence_length", type=int, default=1)
    p.add_argument("--time_chunk_size", type=int, default=1)
    p.add_argument(
        "--with_masktoken",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--use_null_token", action="store_true")
    p.add_argument(
        "--normalize_from_data",
        action="store_true",
        help="Recompute contact normalization stats from the loaded data.",
    )

    p.add_argument("--pkl_window_size", type=int, default=1)
    p.add_argument("--pkl_window_stride", type=int, default=1)

    p.add_argument(
        "--embedding_unit",
        choices=["window", "frame"],
        default="window",
        help="Use one embedding per window or per frame.",
    )
    p.add_argument(
        "--pool_frame_stride",
        type=int,
        default=1,
        help="[window] Use every Nth frame in a window when averaging embeddings.",
    )
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument(
        "--max_points",
        type=int,
        default=3000,
        help="Evenly subsample to at most this many points. Set <=0 to disable.",
    )
    p.add_argument(
        "--max_points_per_class",
        type=int,
        default=None,
        help="Optional per-class cap before global --max_points.",
    )
    p.add_argument(
        "--class_parent_depth",
        type=int,
        default=1,
        help="0 uses episode/file name, 1 uses parent folder, 2 uses grandparent folder.",
    )

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--learning_rate", type=parse_learning_rate, default="auto")
    p.add_argument("--tsne_iter", type=int, default=1000)
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument(
        "--normalize_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize embeddings before t-SNE.",
    )

    p.add_argument("--point_size", type=float, default=20.0)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument(
        "--output_png",
        default="outputs/brainco_embedding_tsne.png",
    )
    p.add_argument(
        "--output_json",
        default="outputs/brainco_embedding_tsne.json",
    )

    # build_dataset() checks these optional query fields.
    p.set_defaults(query_path=None, query_episode=None)
    return p.parse_args()


def parse_learning_rate(value):
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    return float(value)


def make_window_center_records(dataset) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    for window_index, sample in enumerate(dataset.windows):
        window_len = int(sample["joint_contact"].shape[0])
        local = window_len // 2
        frame_ids = sample.get("frame_ids")
        frame_id = int(frame_ids[local]) if frame_ids and local < len(frame_ids) else None
        label = int(sample["label"].item() if hasattr(sample["label"], "item") else sample["label"])
        records.append(
            FrameRecord(
                record_index=len(records),
                window_index=window_index,
                local_frame=local,
                abs_frame=int(sample["window_start_frame"]) + local,
                episode_path=str(sample["episode_path"]),
                label=label,
                frame_id=frame_id,
                source_type=str(sample.get("source_type", "brainco")),
            )
        )
    return records


def make_frame_records(dataset, frame_stride: int) -> List[FrameRecord]:
    if frame_stride <= 0:
        raise ValueError("--frame_stride must be >= 1")

    records: List[FrameRecord] = []
    for window_index, sample in enumerate(dataset.windows):
        window_len = int(sample["joint_contact"].shape[0])
        frame_ids = sample.get("frame_ids")
        label = int(sample["label"].item() if hasattr(sample["label"], "item") else sample["label"])
        for local in range(0, window_len, frame_stride):
            frame_id = int(frame_ids[local]) if frame_ids and local < len(frame_ids) else None
            records.append(
                FrameRecord(
                    record_index=len(records),
                    window_index=window_index,
                    local_frame=local,
                    abs_frame=int(sample["window_start_frame"]) + local,
                    episode_path=str(sample["episode_path"]),
                    label=label,
                    frame_id=frame_id,
                    source_type=str(sample.get("source_type", "brainco")),
                )
            )
    return records


def class_name_for_record(record: FrameRecord, parent_depth: int) -> str:
    path = Path(record.episode_path)
    if parent_depth <= 0:
        return path.stem if path.is_file() else path.name

    parents = path.parents
    if len(parents) >= parent_depth:
        return parents[parent_depth - 1].name
    return path.name


def _even_indices(indices: Sequence[int], limit: int) -> List[int]:
    if limit <= 0 or len(indices) <= limit:
        return list(indices)
    selected = np.linspace(0, len(indices) - 1, num=limit, dtype=int)
    return [indices[int(i)] for i in selected]


def subsample_by_class(
    records: Sequence[FrameRecord],
    class_names: Sequence[str],
    max_points: int,
    max_points_per_class: Optional[int],
) -> Tuple[List[FrameRecord], List[str]]:
    selected_indices = list(range(len(records)))

    if max_points_per_class is not None and max_points_per_class > 0:
        grouped: Dict[str, List[int]] = defaultdict(list)
        for idx, class_name in enumerate(class_names):
            grouped[class_name].append(idx)
        selected_indices = []
        for class_name in sorted(grouped):
            selected_indices.extend(_even_indices(grouped[class_name], max_points_per_class))
        selected_indices.sort()

    if max_points is not None and max_points > 0:
        selected_indices = _even_indices(selected_indices, max_points)

    return [records[i] for i in selected_indices], [class_names[i] for i in selected_indices]


def run_tsne(embeddings: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if len(embeddings) < 2:
        raise ValueError("t-SNE needs at least 2 points.")

    perplexity = min(float(args.perplexity), max(1.0, (len(embeddings) - 1) / 3.0))
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        learning_rate=args.learning_rate,
        init="pca",
        random_state=args.random_state,
        metric="euclidean",
    )
    for use_max_iter in (True, False):
        try:
            if use_max_iter:
                tsne = TSNE(max_iter=args.tsne_iter, **kwargs)
            else:
                tsne = TSNE(n_iter=args.tsne_iter, **kwargs)
            return tsne.fit_transform(embeddings)
        except TypeError:
            continue
        except ValueError:
            if kwargs["learning_rate"] == "auto":
                kwargs["learning_rate"] = 200.0
                continue
            raise
    tsne = TSNE(n_iter=args.tsne_iter, **kwargs)
    return tsne.fit_transform(embeddings)



def plot_tsne(
    coords: np.ndarray,
    class_names: Sequence[str],
    args: argparse.Namespace,
    class_counts: Counter,
) -> None:
    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    unique_classes = sorted(class_counts)
    if len(unique_classes) <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(len(unique_classes))]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / max(1, len(unique_classes))) for i in range(len(unique_classes))]
    color_by_class = dict(zip(unique_classes, colors))

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    class_array = np.asarray(class_names)
    for class_name in unique_classes:
        mask = class_array == class_name
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=args.point_size,
            alpha=args.alpha,
            color=color_by_class[class_name],
            label=f"{class_name} ({int(mask.sum())})",
            linewidths=0,
        )

    ax.set_title(
        f"Angle embedding t-SNE ({args.embedding_unit}, n={len(coords)})",
        fontsize=12,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        markerscale=1.8,
        frameon=False,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    coords: np.ndarray,
    records: Sequence[FrameRecord],
    class_names: Sequence[str],
    args: argparse.Namespace,
) -> None:
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    points = []
    for i, (record, class_name) in enumerate(zip(records, class_names)):
        points.append(
            {
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "class": class_name,
                "episode_path": record.episode_path,
                "abs_frame": record.abs_frame,
                "frame_id": record.frame_id,
                "source_type": record.source_type,
            }
        )

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_path": [str(Path(p).resolve()) for p in args.data_path],
        "embedding_unit": args.embedding_unit,
        "num_points": len(points),
        "class_counts": dict(Counter(class_names)),
        "brainco_contact_mode": getattr(args, "resolved_brainco_contact_mode", args.brainco_contact_mode),
        "output_png": str(Path(args.output_png).resolve()),
        "points": points,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.data_path} ...")
    dataset = build_dataset(args)
    print(f"Dataset: {len(dataset.episode_data)} episodes, {len(dataset.windows)} windows")

    inferred_in_dim, inferred_in_chans = validate_window_shapes(dataset)
    if args.in_dim is None:
        args.in_dim = inferred_in_dim
    if args.in_chans is None:
        args.in_chans = inferred_in_chans
    if (args.in_dim, args.in_chans) != (inferred_in_dim, inferred_in_chans):
        raise ValueError(
            "Model input shape does not match selected data: "
            f"model=({args.in_dim}, {args.in_chans}), "
            f"data=({inferred_in_dim}, {inferred_in_chans})"
        )
    if args.with_masktoken is None:
        args.with_masktoken = args.in_dim == 42

    if args.embedding_unit == "window":
        records = make_window_center_records(dataset)
    else:
        records = make_frame_records(dataset, args.frame_stride)

    class_names = [class_name_for_record(r, args.class_parent_depth) for r in records]
    records, class_names = subsample_by_class(
        records,
        class_names,
        args.max_points,
        args.max_points_per_class,
    )
    if len(records) < 2:
        raise ValueError("Need at least 2 records after sampling.")

    print("Class counts:")
    for name, count in sorted(Counter(class_names).items()):
        print(f"  {name}: {count}")
    print(f"Embedding {len(records)} {args.embedding_unit}s on {args.device} ...")

    print(
        f"Building AngleTransformer ({args.model_size}) "
        f"in_dim={args.in_dim}, in_chans={args.in_chans}, "
        f"with_masktoken={args.with_masktoken}"
    )
    model = build_model(args)
    model = load_encoder(args.checkpoint, model)

    if args.normalize_from_data:
        mean, std = compute_contact_stats(dataset)
        model.update_stats(mean, std)
        print(f"[normalization] signal_mean={mean.tolist()}")
        print(f"[normalization] signal_std ={std.tolist()}")

    embeddings = embed_records(
        model=model,
        dataset=dataset,
        records=records,
        batch_size=args.batch_size,
        device=args.device,
        retrieval_unit=args.embedding_unit,
        pool_frame_stride=args.pool_frame_stride,
    ).numpy()
    if args.normalize_embeddings:
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norm, 1e-12, None)

    print("Running t-SNE ...")
    coords = run_tsne(embeddings, args)
    plot_tsne(coords, class_names, args, Counter(class_names))
    write_metadata(coords, records, class_names, args)

    print(f"Saved t-SNE:   {Path(args.output_png).resolve()}")
    print(f"Saved metadata:{Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
