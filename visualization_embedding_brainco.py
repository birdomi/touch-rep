#!/usr/bin/env python3
"""
visualization_embedding_brainco.py

Retrieve and visualize the Top-K angle/contact frames whose AngleTransformer
embeddings are most similar to a query frame.

The script can load BrainCo data.json episodes or angle-vector .pkl files, so
each frame/window is represented by:
  - joint_contact: BrainCo (10, 4), angle-vector pkl/TACO-like (42, 1)
  - finger_angles: (10, 4)

By default it uses the pretrained BrainCo angle checkpoint and the combined
0611 grasp-prediction dataset layout. Missing class directories are skipped, so
the same defaults also work for the older grasp_success / grasp_fail layout.

Example:
    # BrainCo root, selecting one episode inside data_path
    python visualization_embedding_brainco.py \
        --checkpoint checkpoints/dinov2_angle/epoch-5000-brainco.ckpt \
        --data_path dataset/brainco/downstream/grasp_prediction_0611 \
        --select box_succ/episode_0000_ep0001 \
        --query_frame 40 \
        --output_png outputs/brainco_top5.png

    # TACO-like angle-vector pkl directory
    python visualization_embedding_brainco.py \
        --checkpoint checkpoints/dinov2_angle/epoch-0200-all.ckpt \
        --data_path pretraining_dataset/vector_dataset/TACO \
        --select '*brush*bowl*.pkl' \
        --query_path brush__brush__bowl__20230919_036.pkl \
        --query_frame 40
"""

import argparse
import fnmatch
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from tactile_ssl.data.brainco_angle_tactile import BraincoAngleTactileDataset
from tactile_ssl.model.angle_transformer import (
    FULL_SKELETON_SIZE,
    TACTILE_SENSOR_IDXS,
    angle_small,
    angle_tiny,
)


OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b), replace=True)
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b, replace=True)
OmegaConf.register_new_resolver("capitalize", lambda s: s.title(), replace=True)


DEFAULT_LABEL_DIRS = (
    "grasp_success:1,grasp_fail:0,"
    "box_succ:1,box_fail:0,"
    "tumbler_succ:1,tumbler_fail:0,"
    "eraser_succ:1,eraser_fail:0,"
    "driver_succ:1,driver_fail:0"
)


@dataclass(frozen=True)
class FrameRecord:
    record_index: int
    window_index: int
    local_frame: int
    abs_frame: int
    episode_path: str
    label: int
    frame_id: Optional[int] = None
    source_type: str = "brainco"


class SimpleWindowDataset:
    def __init__(self):
        self.episode_data: List[dict] = []
        self.windows: List[dict] = []

    def __len__(self) -> int:
        return len(self.windows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find and visualize Top-K BrainCo frames by pretrained embedding similarity."
    )

    p.add_argument(
        "--checkpoint",
        default="checkpoints/dinov2_angle/epoch-5000-brainco.ckpt",
        help="Path to a pretrained AngleTransformer or downstream task checkpoint.",
    )
    p.add_argument(
        "--data_path",
        nargs="+",
        default=["dataset/brainco/downstream/grasp_prediction_0611"],
        help="One or more BrainCo root/episode dirs, or angle-vector .pkl/.pkl dirs such as TACO.",
    )
    p.add_argument(
        "--select",
        nargs="*",
        default=None,
        help=(
            "Optional paths, glob patterns, or substrings to select from data_path. "
            "Examples: --select box_succ/episode_0000_ep0001 or --select '*brush*bowl*.pkl'."
        ),
    )
    p.add_argument(
        "--label_dirs",
        default=DEFAULT_LABEL_DIRS,
        help="Comma-separated class mapping, e.g. 'grasp_success:1,grasp_fail:0'.",
    )

    p.add_argument("--window_time", type=float, default=0.01)
    p.add_argument("--window_overlap", type=float, default=0.0)
    p.add_argument("--interpolating_freq", type=int, default=100)
    p.add_argument(
        "--subtract_baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match angle-grasp training default by subtracting the first tactile frame.",
    )
    p.add_argument(
        "--brainco_contact_mode",
        choices=["auto", "all", "first", "first_to_42"],
        default="auto",
        help=(
            "BrainCo tactile channel handling. auto keeps (10,4) for BrainCo-only runs, "
            "but uses first_to_42 when comparing with TACO-like pkl data."
        ),
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
        help="Defaults to True for 42-joint pkl data and False for BrainCo 10-sensor data.",
    )
    p.add_argument("--use_null_token", action="store_true")
    p.add_argument(
        "--normalize_from_data",
        action="store_true",
        help="Recompute contact normalization stats from the loaded BrainCo data.",
    )

    p.add_argument(
        "--query_window_index",
        type=int,
        default=None,
        help="Flat dataset window index to use as the query.",
    )
    p.add_argument(
        "--query_frame_in_window",
        type=int,
        default=None,
        help="Frame offset inside the query window. Default: middle frame.",
    )
    p.add_argument(
        "--query_episode",
        default=None,
        help="Episode path or substring used with --query_frame.",
    )
    p.add_argument(
        "--query_frame",
        type=int,
        default=None,
        help="Frame row index within the matched episode data.json.",
    )
    p.add_argument(
        "--query_path",
        default=None,
        help="Query episode/pkl path or substring. Alias-friendly alternative to --query_episode.",
    )

    p.add_argument("--pkl_window_size", type=int, default=1)
    p.add_argument("--pkl_window_stride", type=int, default=1)

    p.add_argument(
        "--retrieval_unit",
        choices=["window", "frame"],
        default="window",
        help="window compares mean embeddings over each window; frame compares single-frame embeddings.",
    )
    p.add_argument(
        "--pool_frame_stride",
        type=int,
        default=1,
        help="[window] Use every Nth frame inside a window when averaging embeddings.",
    )
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument(
        "--rank_mode",
        choices=["top", "bottom", "both"],
        default="top",
        help="Visualize most similar, least similar, or both rankings.",
    )
    p.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    p.add_argument("--include_query", action="store_true")
    p.add_argument(
        "--exclude_same_episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude candidates from the query episode.",
    )
    p.add_argument(
        "--one_per_episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep at most one retrieved result per episode.",
    )
    p.add_argument(
        "--min_frame_gap",
        type=int,
        default=0,
        help="Exclude candidates whose data.json frame row is within this gap of the query row. Set 0 to disable.",
    )
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on candidate frames. Frames are sampled evenly.",
    )
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )

    p.add_argument(
        "--image_mode",
        choices=["auto", "rgb", "tactile"],
        default="auto",
        help="auto/rgb uses colors from data.json when available; tactile draws a contact heatmap.",
    )
    p.add_argument(
        "--camera_key",
        default="color_0",
        help="Preferred key under frame['colors']; falls back to the first available color.",
    )
    p.add_argument(
        "--taco_video_root",
        default="../OakInkv2/TACO/TACO_dataset/Egocentric_RGB_Videos",
        help="Root containing TACO egocentric RGB videos arranged as '(action, tool, object)/timestamp/color.mp4'.",
    )
    p.add_argument(
        "--output_png",
        default="outputs/brainco_embedding_top5.png",
        help="Combined query + Top-K visualization image.",
    )
    p.add_argument(
        "--output_json",
        default="outputs/brainco_embedding_top5.json",
        help="JSON metadata for the retrieval results.",
    )

    return p.parse_args()


def parse_label_dirs(spec: str) -> Dict[str, int]:
    label_dirs: Dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --label_dirs item '{item}'. Expected name:label.")
        name, label = item.split(":", 1)
        label_dirs[name.strip()] = int(label)
    if not label_dirs:
        raise ValueError("--label_dirs did not contain any class mappings.")
    return label_dirs


def _source_label(path: Path, label_dirs: Dict[str, int]) -> int:
    return int(label_dirs.get(path.parent.name, -1))


def discover_sources(data_path: str, label_dirs: Dict[str, int]) -> List[Tuple[str, Path, int]]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"--data_path not found: {root}")

    if root.is_file():
        if root.suffix.lower() == ".pkl":
            return [("angle_pkl", root, -1)]
        if root.name == "data.json":
            ep = root.parent
            return [("brainco", ep, _source_label(ep, label_dirs))]
        raise ValueError(f"Unsupported file for --data_path: {root}")

    if (root / "data.json").exists():
        return [("brainco", root, _source_label(root, label_dirs))]

    pkl_files = sorted(root.glob("*.pkl"))
    if pkl_files:
        return [("angle_pkl", p, -1) for p in pkl_files]

    json_files = sorted(root.rglob("data.json"))
    sources = [("brainco", p.parent, _source_label(p.parent, label_dirs)) for p in json_files]
    if not sources:
        raise ValueError(
            f"No BrainCo data.json episodes or angle-vector .pkl files found in {root}"
        )
    return sources


def filter_sources(
    sources: Sequence[Tuple[str, Path, int]],
    selectors: Optional[Sequence[str]],
    data_path: str,
) -> List[Tuple[str, Path, int]]:
    if not selectors:
        return list(sources)

    root = Path(data_path)
    search_root = root if root.is_dir() else root.parent
    selected: List[Tuple[str, Path, int]] = []
    seen = set()
    for selector in selectors:
        selector_path = Path(selector)
        resolved_matches = set()
        if selector_path.exists():
            resolved_matches.add(selector_path.resolve())
        for match in search_root.glob(selector):
            resolved_matches.add(match.resolve())
        for match in search_root.rglob(selector):
            resolved_matches.add(match.resolve())

        for source in sources:
            source_type, path, label = source
            path_resolved = path.resolve()
            path_text = str(path)
            rel_text = (
                str(path.relative_to(search_root))
                if path.is_relative_to(search_root)
                else path_text
            )
            matched = (
                path_resolved in resolved_matches
                or selector in path_text
                or selector in rel_text
                or selector in path.name
                or fnmatch.fnmatch(path_text, selector)
                or fnmatch.fnmatch(rel_text, selector)
            )
            if matched and path_resolved not in seen:
                selected.append((source_type, path, label))
                seen.add(path_resolved)

    if not selected:
        available = "\n".join(f"  - {p}" for _, p, _ in sources[:20])
        raise ValueError(
            "No sources matched --select. First available sources:\n" + available
        )
    return selected


def add_query_source_if_needed(
    sources: Sequence[Tuple[str, Path, int]],
    args: argparse.Namespace,
    label_dirs: Dict[str, int],
) -> List[Tuple[str, Path, int]]:
    query_token = args.query_path or args.query_episode
    if not query_token:
        return list(sources)

    query_path = Path(str(query_token)).expanduser()
    if not query_path.exists():
        if "/" in str(query_token) or "\\" in str(query_token):
            print(
                "[query] query path does not exist; "
                f"treating as a loaded-source substring: {query_path}"
            )
        return list(sources)

    query_sources = discover_sources(str(query_path), label_dirs)
    if len(query_sources) != 1:
        raise ValueError(
            f"--query_path/--query_episode={query_token!r} resolved to "
            f"{len(query_sources)} sources. Use a single episode directory, data.json, or .pkl file."
        )

    query_source = query_sources[0]
    query_resolved = query_source[1].resolve()
    out = list(sources)
    if any(path.resolve() == query_resolved for _, path, _ in out):
        return out

    print(f"[data] added query source outside selected data: {query_source[1]}")
    out.append(query_source)
    return out


def _load_brainco_frame_ids(ep_path: Path) -> List[int]:
    with open(ep_path / "data.json", "r", encoding="utf-8") as f:
        frames = json.load(f).get("data", [])
    return [int(frame.get("idx", i)) for i, frame in enumerate(frames)]


def convert_brainco_contact(
    joint_contact: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "all":
        return joint_contact
    if mode == "first":
        return joint_contact[..., :1]
    if mode == "first_to_42":
        first = joint_contact[..., :1]
        full = torch.zeros(
            first.shape[0],
            FULL_SKELETON_SIZE,
            1,
            dtype=first.dtype,
        )
        full[:, TACTILE_SENSOR_IDXS, :] = first
        return full
    raise ValueError(f"Unknown brainco_contact_mode: {mode}")


def add_brainco_source(
    dataset: SimpleWindowDataset,
    source_path: Path,
    label: int,
    cfg: OmegaConf,
    subtract_baseline: bool,
    contact_mode: str,
) -> None:
    ep_ds = BraincoAngleTactileDataset(
        config=cfg,
        data_path=str(source_path),
        object_class=label if label >= 0 else None,
        subtract_baseline=subtract_baseline,
    )
    frame_ids = _load_brainco_frame_ids(source_path)
    window_starts = [int(v) for v in ep_ds.data_idxs]
    dataset.episode_data.append(
        {
            "path": str(source_path),
            "window_starts": window_starts,
            "label": label,
            "source_type": "brainco",
        }
    )
    for idx, start in enumerate(window_starts):
        sample = ep_ds[idx]
        joint_contact = convert_brainco_contact(sample["joint_contact"], contact_mode)
        end = start + int(joint_contact.shape[0])
        dataset.windows.append(
            {
                "joint_contact": joint_contact,
                "finger_angles": sample["finger_angles"],
                "label": torch.tensor(label, dtype=torch.long),
                "episode_path": str(source_path),
                "window_start_frame": start,
                "frame_ids": frame_ids[start:end],
                "num_source_frames": int(len(frame_ids)),
                "source_type": "brainco",
                "contact_mode": contact_mode,
            }
        )


def _load_angle_pkl(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        raw = pickle.load(f)

    frame_indices = sorted(k for k in raw.keys() if isinstance(k, int))
    if not frame_indices:
        raise ValueError(f"No integer frame keys found in {path}")

    finger_angles = []
    joint_contact = []
    for t in frame_indices:
        frame = raw[t]
        finger_angles.append(
            np.concatenate(
                [
                    np.asarray(frame["lh_angles"], dtype=np.float32),
                    np.asarray(frame["rh_angles"], dtype=np.float32),
                ],
                axis=0,
            )
        )
        joint_contact.append(
            np.concatenate(
                [
                    np.asarray(frame["lh_contact"], dtype=np.float32),
                    np.asarray(frame["rh_contact"], dtype=np.float32),
                ],
                axis=0,
            )
        )

    return (
        np.asarray(frame_indices, dtype=np.int64),
        np.asarray(joint_contact, dtype=np.float32),
        np.asarray(finger_angles, dtype=np.float32),
    )


def add_angle_pkl_source(
    dataset: SimpleWindowDataset,
    source_path: Path,
    window_size: int,
    window_stride: int,
) -> None:
    frame_ids, joint_contact, finger_angles = _load_angle_pkl(source_path)
    if window_size <= 0 or window_stride <= 0:
        raise ValueError("--pkl_window_size and --pkl_window_stride must be >= 1")
    if len(frame_ids) < window_size:
        window_starts = [0]
    else:
        window_starts = list(range(0, len(frame_ids) - window_size + 1, window_stride))

    dataset.episode_data.append(
        {
            "path": str(source_path),
            "window_starts": window_starts,
            "label": -1,
            "source_type": "angle_pkl",
        }
    )
    for start in window_starts:
        end = min(start + window_size, len(frame_ids))
        dataset.windows.append(
            {
                "joint_contact": torch.from_numpy(joint_contact[start:end].copy()).float(),
                "finger_angles": torch.from_numpy(finger_angles[start:end].copy()).float(),
                "label": torch.tensor(-1, dtype=torch.long),
                "episode_path": str(source_path),
                "window_start_frame": int(start),
                "frame_ids": [int(v) for v in frame_ids[start:end]],
                "num_source_frames": int(len(frame_ids)),
                "source_type": "angle_pkl",
            }
        )


def validate_window_shapes(dataset: SimpleWindowDataset) -> Tuple[int, int]:
    shapes = {
        tuple(sample["joint_contact"].shape[-2:])
        for sample in dataset.windows
    }
    if len(shapes) != 1:
        raise ValueError(
            "Selected sources have incompatible contact shapes and cannot be "
            f"compared in one encoder run: {sorted(shapes)}"
        )
    in_dim, in_chans = next(iter(shapes))
    return int(in_dim), int(in_chans)


def build_dataset(args: argparse.Namespace) -> SimpleWindowDataset:
    label_dirs = parse_label_dirs(args.label_dirs)
    data_paths = args.data_path if isinstance(args.data_path, list) else [args.data_path]
    sources: List[Tuple[str, Path, int]] = []
    for data_path in data_paths:
        data_sources = discover_sources(data_path, label_dirs)
        if args.select:
            try:
                data_sources = filter_sources(data_sources, args.select, data_path)
            except ValueError:
                data_sources = []
        sources.extend(data_sources)
    if not sources:
        raise ValueError("No sources were selected from --data_path/--select.")

    sources = add_query_source_if_needed(sources, args, label_dirs)

    has_brainco = any(source_type == "brainco" for source_type, _, _ in sources)
    has_angle_pkl = any(source_type == "angle_pkl" for source_type, _, _ in sources)
    brainco_contact_mode = args.brainco_contact_mode
    if brainco_contact_mode == "auto":
        brainco_contact_mode = "first_to_42" if has_brainco and has_angle_pkl else "all"
    args.resolved_brainco_contact_mode = brainco_contact_mode
    print(f"[data] brainco_contact_mode={brainco_contact_mode}")

    cfg = OmegaConf.create(
        {
            "window_time": args.window_time,
            "window_overlap": args.window_overlap,
            "interpolating_freq": args.interpolating_freq,
            "subtract_baseline": bool(args.subtract_baseline),
        }
    )

    dataset = SimpleWindowDataset()
    for source_type, path, label in sources:
        if source_type == "brainco":
            add_brainco_source(
                dataset,
                path,
                label,
                cfg,
                bool(args.subtract_baseline),
                brainco_contact_mode,
            )
        elif source_type == "angle_pkl":
            add_angle_pkl_source(dataset, path, args.pkl_window_size, args.pkl_window_stride)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    if len(dataset) == 0:
        raise ValueError(
            f"No windows were loaded from {args.data_path}. "
            "Check --data_path, --select, and window settings."
        )
    return dataset


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    factory = angle_tiny if args.model_size == "tiny" else angle_small
    return factory(
        in_dim=args.in_dim,
        in_chans=args.in_chans,
        pos_in_dim=args.pos_in_dim,
        pos_in_chans=args.pos_in_chans,
        sequence_length=args.sequence_length,
        time_chunk_size=args.time_chunk_size,
        num_register_tokens=1,
        with_masktoken=args.with_masktoken,
        use_null_token=args.use_null_token,
        pos_embed_fn="learned",
    )


def _state_dict_candidates(ckpt) -> List[Tuple[str, dict]]:
    candidates: List[Tuple[str, dict]] = []

    def add(name: str, obj) -> None:
        if not isinstance(obj, dict):
            return
        if any(hasattr(v, "shape") for v in obj.values()):
            candidates.append((name, obj))

    add("checkpoint", ckpt)
    if isinstance(ckpt, dict):
        for key in (
            "state_dict",
            "model",
            "encoder",
            "model_encoder",
            "self.encoder",
            "self.model_encoder",
        ):
            if key in ckpt:
                add(key, ckpt[key])
    return candidates


def _candidate_state_keys(key: str) -> List[str]:
    prefixes = [
        "_forward_module.teacher_encoder.backbone.",
        "_forward_module.student_encoder.backbone.",
        "_forward_module.teacher_encoder_dict.backbone.",
        "_forward_module.student_encoder_dict.backbone.",
        "_forward_module.model_encoder.",
        "_forward_module.encoder.",
        "_forward_module.self.model_encoder.",
        "_forward_module.self.encoder.",
        "module.teacher_encoder.backbone.",
        "module.student_encoder.backbone.",
        "module.teacher_encoder_dict.backbone.",
        "module.student_encoder_dict.backbone.",
        "module.model_encoder.",
        "module.encoder.",
        "module.self.model_encoder.",
        "module.self.encoder.",
        "teacher_encoder.backbone.",
        "student_encoder.backbone.",
        "teacher_encoder_dict.backbone.",
        "student_encoder_dict.backbone.",
        "teacher_encoder.",
        "student_encoder.",
        "teacher_encoder_dict.",
        "student_encoder_dict.",
        "model.encoder.",
        "model.model_encoder.",
        "model_encoder.",
        "self.model_encoder.",
        "self.encoder.",
        "algorithm.encoder.",
        "encoder.",
        "backbone.",
        "module.",
        "_forward_module.",
    ]
    out = []
    pending = [key]
    seen = set()
    while pending:
        item = pending.pop(0)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        for prefix in prefixes:
            if item.startswith(prefix):
                pending.append(item[len(prefix) :])
    return out


def load_encoder(checkpoint_path: str, model: torch.nn.Module) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict_candidates = _state_dict_candidates(ckpt)
    model_state = model.state_dict()

    cleaned = {}
    skipped_shape = []
    matched_source = None
    for source_name, state_dict in state_dict_candidates:
        cleaned = {}
        skipped_shape = []
        for key, value in state_dict.items():
            if not hasattr(value, "shape"):
                continue
            for candidate in _candidate_state_keys(key):
                if candidate not in model_state:
                    continue
                if tuple(value.shape) != tuple(model_state[candidate].shape):
                    skipped_shape.append(
                        (
                            key,
                            candidate,
                            tuple(value.shape),
                            tuple(model_state[candidate].shape),
                        )
                    )
                    continue
                cleaned[candidate] = value
                break
        if cleaned:
            matched_source = source_name
            break

    if not cleaned:
        key_lines = []
        only_task_keys = False
        for source_name, state_dict in state_dict_candidates:
            tensor_keys = [k for k, v in state_dict.items() if hasattr(v, "shape")]
            only_task_keys = only_task_keys or (
                bool(tensor_keys)
                and all(k.startswith(("model_task.", "classifier.", "probe.")) for k in tensor_keys)
            )
            key_lines.append(
                f"{source_name}: {len(tensor_keys)} tensor keys, first 8={tensor_keys[:8]}"
            )
        hint = ""
        if only_task_keys:
            hint = (
                "\nThis checkpoint appears to contain only task/probe weights. "
                "Use a checkpoint saved with encoder weights, or set trainer.save_probe_weights_only=False."
            )
        raise RuntimeError(
            f"No compatible encoder parameters were found in checkpoint: {checkpoint_path}"
            f"{hint}\nChecked checkpoint/state_dict/model/encoder/model_encoder/self.encoder entries.\n"
            + "\n".join(key_lines)
        )

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(
        f"[load_encoder] loaded {len(cleaned)} tensors from {checkpoint_path} "
        f"(source={matched_source})"
    )
    if missing:
        print(f"[load_encoder] missing keys: {len(missing)} (first 8: {missing[:8]})")
    if unexpected:
        print(f"[load_encoder] unexpected keys: {len(unexpected)} (first 8: {unexpected[:8]})")
    if skipped_shape:
        print(f"[load_encoder] skipped shape mismatches: {len(skipped_shape)}")
    return model


def compute_contact_stats(dataset: SimpleWindowDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    num_chans = int(dataset.windows[0]["joint_contact"].shape[-1])
    sums = torch.zeros(num_chans, dtype=torch.float64)
    sq_sums = torch.zeros(num_chans, dtype=torch.float64)
    counts = torch.zeros(num_chans, dtype=torch.float64)

    for sample in dataset.windows:
        x = sample["joint_contact"].double()
        for c in range(num_chans):
            valid = x[..., c] >= 0
            vals = x[..., c][valid]
            if vals.numel() == 0:
                continue
            sums[c] += vals.sum()
            sq_sums[c] += (vals * vals).sum()
            counts[c] += vals.numel()

    mean = torch.zeros(num_chans, dtype=torch.float32)
    std = torch.ones(num_chans, dtype=torch.float32)
    ok = counts > 0
    mean[ok] = (sums[ok] / counts[ok]).float()
    var = (sq_sums[ok] / counts[ok]) - (sums[ok] / counts[ok]).pow(2)
    std[ok] = torch.sqrt(var.clamp(min=1e-12)).float().clamp(min=1e-6)
    return mean, std


def resolve_query_window_frame(
    dataset: SimpleWindowDataset,
    args: argparse.Namespace,
) -> Tuple[int, int]:
    if args.query_window_index is not None:
        if args.query_window_index < 0 or args.query_window_index >= len(dataset.windows):
            raise IndexError(
                f"--query_window_index {args.query_window_index} out of range [0, {len(dataset.windows) - 1}]"
            )
        sample = dataset.windows[args.query_window_index]
        w_len = int(sample["joint_contact"].shape[0])
        local = w_len // 2 if args.query_frame_in_window is None else args.query_frame_in_window
        return args.query_window_index, max(0, min(local, w_len - 1))

    if args.query_episode is not None or args.query_frame is not None:
        episode_token = args.query_path or args.query_episode
        query_frame = 0 if args.query_frame is None else int(args.query_frame)
        best = None
        best_dist = None
        for idx, sample in enumerate(dataset.windows):
            ep_path = str(sample["episode_path"])
            if episode_token is not None:
                ep = Path(ep_path)
                token = str(episode_token)
                if token not in ep_path and token not in ep.name and token != f"{ep.parent.name}/{ep.name}":
                    continue

            start = int(sample["window_start_frame"])
            w_len = int(sample["joint_contact"].shape[0])
            end = start + w_len - 1
            if start <= query_frame <= end:
                return idx, query_frame - start

            dist = min(abs(query_frame - start), abs(query_frame - end))
            if best_dist is None or dist < best_dist:
                local = max(0, min(query_frame - start, w_len - 1))
                best = (idx, local)
                best_dist = dist

        if best is None:
            raise ValueError(
                f"No window matched --query_path={args.query_path!r}, "
                f"--query_episode={args.query_episode!r} "
                f"and --query_frame={args.query_frame!r}."
            )
        print(
            "[query] exact frame was not inside a matched window; "
            f"using nearest window/frame with distance {best_dist}."
        )
        return best

    sample = dataset.windows[0]
    return 0, int(sample["joint_contact"].shape[0]) // 2


def build_frame_records(
    dataset: SimpleWindowDataset,
    frame_stride: int,
) -> List[FrameRecord]:
    if frame_stride <= 0:
        raise ValueError("--frame_stride must be >= 1")

    records: List[FrameRecord] = []
    for window_index, sample in enumerate(dataset.windows):
        window_len = int(sample["joint_contact"].shape[0])
        start = int(sample["window_start_frame"])
        label = int(sample["label"].item() if hasattr(sample["label"], "item") else sample["label"])
        frame_ids = sample.get("frame_ids")
        for local in range(0, window_len, frame_stride):
            frame_id = int(frame_ids[local]) if frame_ids and local < len(frame_ids) else None
            records.append(
                FrameRecord(
                    record_index=len(records),
                    window_index=window_index,
                    local_frame=local,
                    abs_frame=start + local,
                    episode_path=str(sample["episode_path"]),
                    label=label,
                    frame_id=frame_id,
                    source_type=str(sample.get("source_type", "brainco")),
                )
            )
    return records


def build_window_records(
    dataset: SimpleWindowDataset,
    query_window_index: int,
    query_local_frame: int,
) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    for window_index, sample in enumerate(dataset.windows):
        window_len = int(sample["joint_contact"].shape[0])
        if window_index == query_window_index:
            local = max(0, min(query_local_frame, window_len - 1))
        else:
            local = window_len // 2
        start = int(sample["window_start_frame"])
        label = int(sample["label"].item() if hasattr(sample["label"], "item") else sample["label"])
        frame_ids = sample.get("frame_ids")
        frame_id = int(frame_ids[local]) if frame_ids and local < len(frame_ids) else None
        records.append(
            FrameRecord(
                record_index=len(records),
                window_index=window_index,
                local_frame=local,
                abs_frame=start + local,
                episode_path=str(sample["episode_path"]),
                label=label,
                frame_id=frame_id,
                source_type=str(sample.get("source_type", "brainco")),
            )
        )
    return records


def find_record_index(
    records: Sequence[FrameRecord],
    query_window_index: int,
    query_local_frame: int,
) -> int:
    for i, record in enumerate(records):
        if record.window_index == query_window_index and record.local_frame == query_local_frame:
            return i
    raise ValueError(
        "Query frame is not present in candidate records. "
        "Use a smaller --frame_stride or remove --max_frames."
    )


def ensure_query_record(
    records: List[FrameRecord],
    dataset: SimpleWindowDataset,
    query_window_index: int,
    query_local_frame: int,
) -> List[FrameRecord]:
    for record in records:
        if record.window_index == query_window_index and record.local_frame == query_local_frame:
            return records

    sample = dataset.windows[query_window_index]
    label = int(sample["label"].item() if hasattr(sample["label"], "item") else sample["label"])
    frame_ids = sample.get("frame_ids")
    frame_id = (
        int(frame_ids[query_local_frame])
        if frame_ids and query_local_frame < len(frame_ids)
        else None
    )
    records.append(
        FrameRecord(
            record_index=len(records),
            window_index=query_window_index,
            local_frame=query_local_frame,
            abs_frame=int(sample["window_start_frame"]) + query_local_frame,
            episode_path=str(sample["episode_path"]),
            label=label,
            frame_id=frame_id,
            source_type=str(sample.get("source_type", "brainco")),
        )
    )
    return records


def filter_query_episode_records(
    records: List[FrameRecord],
    query_window_index: int,
    query_local_frame: int,
    exclude_same_episode: bool,
) -> List[FrameRecord]:
    if not exclude_same_episode:
        return records

    query_records = [
        r
        for r in records
        if r.window_index == query_window_index and r.local_frame == query_local_frame
    ]
    if not query_records:
        return records

    query_record = query_records[0]
    filtered = [
        r
        for r in records
        if r.episode_path != query_record.episode_path
        or (r.window_index == query_window_index and r.local_frame == query_local_frame)
    ]
    return [
        FrameRecord(
            record_index=i,
            window_index=r.window_index,
            local_frame=r.local_frame,
            abs_frame=r.abs_frame,
            episode_path=r.episode_path,
            label=r.label,
            frame_id=r.frame_id,
            source_type=r.source_type,
        )
        for i, r in enumerate(filtered)
    ]


def subsample_records(
    records: List[FrameRecord],
    query_window_index: int,
    query_local_frame: int,
    max_frames: Optional[int],
) -> List[FrameRecord]:
    if max_frames is None or len(records) <= max_frames:
        return records
    if max_frames < 1:
        raise ValueError("--max_frames must be positive when set.")

    idxs = set(np.linspace(0, len(records) - 1, num=max_frames, dtype=int).tolist())
    for i, record in enumerate(records):
        if record.window_index == query_window_index and record.local_frame == query_local_frame:
            idxs.add(i)
            break
    selected = [records[i] for i in sorted(idxs)]
    return [
        FrameRecord(
            record_index=i,
            window_index=r.window_index,
            local_frame=r.local_frame,
            abs_frame=r.abs_frame,
            episode_path=r.episode_path,
            label=r.label,
            frame_id=r.frame_id,
            source_type=r.source_type,
        )
        for i, r in enumerate(selected)
    ]


def _context_indices(center: int, length: int, seq_len: int) -> np.ndarray:
    if seq_len == 1:
        return np.asarray([center], dtype=np.int64)
    left = seq_len // 2
    raw = np.arange(center - left, center - left + seq_len)
    return np.clip(raw, 0, length - 1).astype(np.int64)


@torch.no_grad()
def embed_frame_records(
    model: torch.nn.Module,
    dataset: SimpleWindowDataset,
    records: Sequence[FrameRecord],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    model.eval()
    model.to(device)
    seq_len = int(model.sequence_length)

    embeddings: List[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        contacts = []
        angles = []
        for record in batch_records:
            sample = dataset.windows[record.window_index]
            window_len = int(sample["joint_contact"].shape[0])
            idxs = _context_indices(record.local_frame, window_len, seq_len)
            contacts.append(sample["joint_contact"][idxs])
            angles.append(sample["finger_angles"][idxs])

        x = torch.stack(contacts).float().to(device)
        pos = torch.stack(angles).float().to(device)
        out = model.forward_features(x, pos)
        embeddings.append(out["x_norm_regtokens"][:, 0, :].detach().cpu())

    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def embed_window_records(
    model: torch.nn.Module,
    dataset: SimpleWindowDataset,
    records: Sequence[FrameRecord],
    batch_size: int,
    device: str,
    pool_frame_stride: int,
) -> torch.Tensor:
    if pool_frame_stride <= 0:
        raise ValueError("--pool_frame_stride must be >= 1")

    model.eval()
    model.to(device)
    seq_len = int(model.sequence_length)

    record_embeddings: List[torch.Tensor] = []
    pending_contacts: List[torch.Tensor] = []
    pending_angles: List[torch.Tensor] = []
    pending_counts: List[int] = []

    def flush_pending() -> None:
        nonlocal pending_contacts, pending_angles, pending_counts
        if not pending_contacts:
            return

        x = torch.stack(pending_contacts).float().to(device)
        pos = torch.stack(pending_angles).float().to(device)
        out = model.forward_features(x, pos)
        ctx_emb = out["x_norm_regtokens"][:, 0, :].detach().cpu()

        cursor = 0
        for count in pending_counts:
            record_embeddings.append(ctx_emb[cursor : cursor + count].mean(dim=0))
            cursor += count

        pending_contacts = []
        pending_angles = []
        pending_counts = []

    for record in records:
        sample = dataset.windows[record.window_index]
        window_len = int(sample["joint_contact"].shape[0])
        local_frames = list(range(0, window_len, pool_frame_stride))
        if record.local_frame not in local_frames:
            local_frames.append(record.local_frame)
            local_frames.sort()

        if pending_contacts and len(pending_contacts) + len(local_frames) > batch_size:
            flush_pending()

        pending_counts.append(len(local_frames))
        for local in local_frames:
            idxs = _context_indices(local, window_len, seq_len)
            pending_contacts.append(sample["joint_contact"][idxs])
            pending_angles.append(sample["finger_angles"][idxs])

    flush_pending()
    return torch.stack(record_embeddings, dim=0)


def embed_records(
    model: torch.nn.Module,
    dataset: SimpleWindowDataset,
    records: Sequence[FrameRecord],
    batch_size: int,
    device: str,
    retrieval_unit: str,
    pool_frame_stride: int,
) -> torch.Tensor:
    if retrieval_unit == "frame":
        return embed_frame_records(model, dataset, records, batch_size, device)
    if retrieval_unit == "window":
        return embed_window_records(
            model,
            dataset,
            records,
            batch_size,
            device,
            pool_frame_stride,
        )
    raise ValueError(f"Unknown retrieval_unit: {retrieval_unit}")


def retrieve_topk(
    embeddings: torch.Tensor,
    records: Sequence[FrameRecord],
    query_record_index: int,
    top_k: int,
    metric: str,
    include_query: bool,
    exclude_same_episode: bool,
    one_per_episode: bool,
    min_frame_gap: int,
    rank_mode: str = "top",
) -> List[Tuple[int, float]]:
    query = embeddings[query_record_index : query_record_index + 1]
    if metric == "cosine":
        scores = (F.normalize(query, dim=1) @ F.normalize(embeddings, dim=1).T).squeeze(0)
        sorted_indices = torch.argsort(scores, descending=(rank_mode == "top"))
    else:
        scores = torch.cdist(query, embeddings, p=2).squeeze(0)
        sorted_indices = torch.argsort(scores, descending=(rank_mode == "bottom"))

    query_record = records[query_record_index]
    query_key = (query_record.episode_path, query_record.abs_frame)
    seen = set()
    used_episodes = set()
    selected: List[Tuple[int, float]] = []
    for idx_t in sorted_indices:
        idx = int(idx_t.item())
        record = records[idx]
        key = (record.episode_path, record.abs_frame)
        if key in seen:
            continue
        if not include_query and key == query_key:
            continue
        if exclude_same_episode and record.episode_path == query_record.episode_path:
            continue
        if min_frame_gap > 0 and abs(record.abs_frame - query_record.abs_frame) < min_frame_gap:
            continue
        if one_per_episode and record.episode_path in used_episodes:
            continue
        seen.add(key)
        used_episodes.add(record.episode_path)
        selected.append((idx, float(scores[idx].item())))
        if len(selected) >= top_k:
            break
    return selected


_JSON_CACHE: Dict[str, Optional[List[dict]]] = {}


def load_episode_frames(ep_path: str) -> Optional[List[dict]]:
    if ep_path in _JSON_CACHE:
        return _JSON_CACHE[ep_path]
    path = Path(ep_path) / "data.json"
    if not path.exists():
        _JSON_CACHE[ep_path] = None
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    frames = raw.get("data", [])
    _JSON_CACHE[ep_path] = frames
    return frames


def _collect_image_files(rgb_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not rgb_dir.exists() or not rgb_dir.is_dir():
        return []
    return sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in exts and p.is_file())


def load_rgb_frame(
    ep_path: str,
    abs_frame: int,
    camera_key: str,
) -> Tuple[Optional[np.ndarray], str]:
    ep = Path(ep_path)
    frames = load_episode_frames(ep_path)
    if frames:
        frame_idx = max(0, min(int(abs_frame), len(frames) - 1))
        colors = frames[frame_idx].get("colors", {})
        if isinstance(colors, dict) and colors:
            keys = [camera_key] if camera_key in colors else sorted(colors.keys())
            for key in keys:
                rel = colors.get(key)
                if not rel:
                    continue
                img_path = ep / rel
                if img_path.exists():
                    img = np.asarray(Image.open(img_path).convert("RGB"))
                    return img, str(img_path)

    for folder_name in ("colors", "rgb", "RGB", "images", "image"):
        files = _collect_image_files(ep / folder_name)
        if not files:
            continue
        denom = len(frames) if frames else len(files)
        img_idx = min(int(round(abs_frame * len(files) / max(denom, 1))), len(files) - 1)
        img = np.asarray(Image.open(files[img_idx]).convert("RGB"))
        return img, str(files[img_idx])

    return None, "RGB not found"


def taco_video_path_from_pkl(pkl_path: str, video_root: str) -> Path:
    p = Path(pkl_path)
    parts = p.stem.split("__")
    if len(parts) < 4:
        raise ValueError(f"Unsupported TACO pkl naming format: {p.name}")

    action, tool, obj = parts[0], parts[1], parts[2]
    timestamp = "__".join(parts[3:])
    root = Path(video_root)
    candidates = [
        (action, tool, obj),
        (action.replace("_", " "), tool.replace("_", " "), obj.replace("_", " ")),
    ]

    for action_name, tool_name, obj_name in candidates:
        video_path = root / f"({action_name}, {tool_name}, {obj_name})" / timestamp / "color.mp4"
        if video_path.exists():
            return video_path

    raise FileNotFoundError(
        "TACO RGB video not found. Tried: "
        + ", ".join(
            str(root / f"({a}, {t}, {o})" / timestamp / "color.mp4")
            for a, t, o in candidates
        )
    )


def map_pkl_frame_to_video_frame(
    pkl_frame_idx: int,
    num_pkl_frames: int,
    num_video_frames: int,
) -> int:
    if num_video_frames <= 0:
        return 0
    return min(
        int(round(pkl_frame_idx * num_video_frames / max(num_pkl_frames, 1))),
        num_video_frames - 1,
    )


def load_video_frame(video_path: Path, frame_idx: int) -> Tuple[np.ndarray, int]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to read TACO RGB videos.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(int(frame_idx), max(total - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()

    if not ok or bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, total


def load_taco_rgb_frame(
    pkl_path: str,
    pkl_frame_idx: int,
    num_pkl_frames: int,
    video_root: str,
) -> Tuple[Optional[np.ndarray], str]:
    try:
        video_path = taco_video_path_from_pkl(pkl_path, video_root)
        _, total_frames = load_video_frame(video_path, 0)
        video_frame_idx = map_pkl_frame_to_video_frame(
            pkl_frame_idx,
            num_pkl_frames,
            total_frames,
        )
        rgb, _ = load_video_frame(video_path, video_frame_idx)
        return rgb, f"{video_path}#frame={video_frame_idx}/{total_frames}"
    except Exception as exc:
        return None, f"TACO RGB not found: {exc}"


def data_json_frame_id(record: FrameRecord) -> Optional[int]:
    if record.frame_id is not None:
        return int(record.frame_id)
    frames = load_episode_frames(record.episode_path)
    if not frames or record.abs_frame < 0 or record.abs_frame >= len(frames):
        return None
    frame_id = frames[record.abs_frame].get("idx")
    return int(frame_id) if frame_id is not None else None


def record_to_dict(record: FrameRecord) -> dict:
    out = asdict(record)
    out["data_json_frame_id"] = data_json_frame_id(record)
    return out


def frame_title(record: FrameRecord, prefix: str, score: Optional[float], metric: str) -> str:
    ep = Path(record.episode_path)
    if record.label == 1:
        label = "success"
    elif record.label == 0:
        label = "fail"
    else:
        label = "unlabeled"
    score_line = "" if score is None else f"\n{metric}={score:.4f}"
    frame_id = data_json_frame_id(record)
    frame_text = f"row={record.abs_frame}" if frame_id is None else f"row={record.abs_frame}  id={frame_id}"
    return (
        f"{prefix}{score_line}\n"
        f"{ep.parent.name}/{ep.name}\n"
        f"{frame_text}  label={label}"
    )


def draw_record(
    ax,
    dataset: SimpleWindowDataset,
    record: FrameRecord,
    title: str,
    image_mode: str,
    camera_key: str,
    taco_video_root: str,
) -> str:
    sample = dataset.windows[record.window_index]
    if image_mode in ("auto", "rgb"):
        if sample.get("source_type") == "angle_pkl":
            rgb, source = load_taco_rgb_frame(
                record.episode_path,
                record.abs_frame,
                int(sample.get("num_source_frames", max(record.abs_frame + 1, 1))),
                taco_video_root,
            )
        else:
            rgb, source = load_rgb_frame(record.episode_path, record.abs_frame, camera_key)

        if rgb is not None:
            ax.imshow(rgb)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            return source
        if image_mode == "rgb":
            ax.text(
                0.5,
                0.5,
                "RGB not found",
                ha="center",
                va="center",
                fontsize=10,
                color="crimson",
                transform=ax.transAxes,
            )
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            return source

    contact = sample["joint_contact"][record.local_frame].detach().cpu().float().numpy()
    contact = np.where(contact < 0, np.nan, contact)
    ax.imshow(contact, aspect="auto", cmap="viridis")
    ax.set_xticks(range(contact.shape[1]))
    ax.set_yticks(range(contact.shape[0]))
    ax.set_xlabel("channel", fontsize=7)
    ax.set_ylabel("sensor", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)
    return "tactile_heatmap"


def save_visualization(
    dataset: SimpleWindowDataset,
    records: Sequence[FrameRecord],
    query_record_index: int,
    top_matches: Sequence[Tuple[int, float]],
    args: argparse.Namespace,
    rank_label: str = "Top",
    output_png: Optional[str] = None,
) -> Dict[int, str]:
    cols = 1 + len(top_matches)
    fig, axes = plt.subplots(1, cols, figsize=(4.0 * cols, 3.8))
    if cols == 1:
        axes = np.asarray([axes])

    image_sources: Dict[int, str] = {}
    query_record = records[query_record_index]
    image_sources[query_record_index] = draw_record(
        axes[0],
        dataset,
        query_record,
        frame_title(query_record, "Query", None, args.metric),
        args.image_mode,
        args.camera_key,
        args.taco_video_root,
    )

    for col, (record_index, score) in enumerate(top_matches, start=1):
        record = records[record_index]
        image_sources[record_index] = draw_record(
            axes[col],
            dataset,
            record,
            frame_title(record, f"{rank_label}-{col}", score, args.metric),
            args.image_mode,
            args.camera_key,
            args.taco_video_root,
        )

    fig.suptitle(
        f"BrainCo embedding retrieval ({rank_label.lower()}, {args.retrieval_unit}, {args.metric}, k={args.top_k})",
        fontsize=11,
    )
    plt.tight_layout()
    out = Path(output_png or args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return image_sources


def build_result_json(
    records: Sequence[FrameRecord],
    query_record_index: int,
    ranked_matches: Dict[str, Sequence[Tuple[int, float]]],
    image_sources_by_mode: Dict[str, Dict[int, str]],
    args: argparse.Namespace,
) -> dict:
    query = records[query_record_index]
    rankings = {}
    for mode, matches_for_mode in ranked_matches.items():
        mode_entries = []
        image_sources = image_sources_by_mode.get(mode, {})
        for rank, (record_index, score) in enumerate(matches_for_mode, start=1):
            record = records[record_index]
            entry = record_to_dict(record)
            entry.update(
                {
                    "rank": rank,
                    args.metric: score,
                    "image_source": image_sources.get(record_index),
                }
            )
            mode_entries.append(entry)
        rankings[mode] = mode_entries

    query_entry = record_to_dict(query)
    first_sources = next(iter(image_sources_by_mode.values()), {})
    query_entry["image_source"] = first_sources.get(query_record_index)
    matches = rankings.get(args.rank_mode, [])
    if args.rank_mode == "both":
        matches = rankings.get("top", [])
    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_path": [str(Path(p).resolve()) for p in args.data_path],
        "metric": args.metric,
        "retrieval_unit": args.retrieval_unit,
        "top_k": args.top_k,
        "rank_mode": args.rank_mode,
        "brainco_contact_mode": getattr(
            args,
            "resolved_brainco_contact_mode",
            args.brainco_contact_mode,
        ),
        "exclude_same_episode": args.exclude_same_episode,
        "one_per_episode": args.one_per_episode,
        "min_frame_gap": args.min_frame_gap,
        "query": query_entry,
        "matches": matches,
        "rankings": rankings,
    }


def main() -> None:
    args = parse_args()

    print(f"Loading angle dataset from {args.data_path} ...")
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

    query_window_index, query_local_frame = resolve_query_window_frame(dataset, args)
    query_sample = dataset.windows[query_window_index]
    query_frame_ids = query_sample.get("frame_ids")
    query_frame_id = (
        int(query_frame_ids[query_local_frame])
        if query_frame_ids and query_local_frame < len(query_frame_ids)
        else None
    )
    print(
        "[query] "
        f"window={query_window_index}, local_frame={query_local_frame}, "
        f"row={int(query_sample['window_start_frame']) + query_local_frame}, "
        f"id={query_frame_id}, "
        f"episode={query_sample['episode_path']}"
    )

    print(
        f"Building AngleTransformer ({args.model_size}) "
        f"seq={args.sequence_length}, chunk={args.time_chunk_size}, "
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

    if args.retrieval_unit == "window":
        records = build_window_records(dataset, query_window_index, query_local_frame)
    else:
        records = build_frame_records(dataset, args.frame_stride)
    records = ensure_query_record(records, dataset, query_window_index, query_local_frame)
    records = filter_query_episode_records(
        records,
        query_window_index,
        query_local_frame,
        args.exclude_same_episode,
    )
    records = subsample_records(records, query_window_index, query_local_frame, args.max_frames)
    query_record_index = find_record_index(records, query_window_index, query_local_frame)
    print(f"Candidate {args.retrieval_unit}s: {len(records)}")

    print(f"Embedding {args.retrieval_unit}s on {args.device} ...")
    embeddings = embed_records(
        model=model,
        dataset=dataset,
        records=records,
        batch_size=args.batch_size,
        device=args.device,
        retrieval_unit=args.retrieval_unit,
        pool_frame_stride=args.pool_frame_stride,
    )

    rank_modes = ["top", "bottom"] if args.rank_mode == "both" else [args.rank_mode]
    ranked_matches: Dict[str, List[Tuple[int, float]]] = {}
    image_sources_by_mode: Dict[str, Dict[int, str]] = {}
    output_pngs: Dict[str, str] = {}
    output_base = Path(args.output_png)

    for mode in rank_modes:
        label = "Top" if mode == "top" else "Bottom"
        print(f"Retrieving {label}-{args.top_k} frames with metric={args.metric} ...")
        matches_for_mode = retrieve_topk(
            embeddings=embeddings,
            records=records,
            query_record_index=query_record_index,
            top_k=args.top_k,
            metric=args.metric,
            include_query=args.include_query,
            exclude_same_episode=args.exclude_same_episode,
            one_per_episode=args.one_per_episode,
            min_frame_gap=args.min_frame_gap,
            rank_mode=mode,
        )
        if len(matches_for_mode) < args.top_k:
            print(
                f"[retrieve] Only {len(matches_for_mode)} {mode} matches found for top_k={args.top_k}. "
                "This can happen when one_per_episode/exclude_same_episode filters leave too few episodes."
            )

        if args.rank_mode == "both":
            output_png = str(output_base.with_name(f"{output_base.stem}_{mode}{output_base.suffix}"))
        else:
            output_png = str(output_base)
        output_pngs[mode] = output_png
        image_sources_by_mode[mode] = save_visualization(
            dataset=dataset,
            records=records,
            query_record_index=query_record_index,
            top_matches=matches_for_mode,
            args=args,
            rank_label=label,
            output_png=output_png,
        )
        ranked_matches[mode] = matches_for_mode

    result = build_result_json(
        records=records,
        query_record_index=query_record_index,
        ranked_matches=ranked_matches,
        image_sources_by_mode=image_sources_by_mode,
        args=args,
    )
    result["output_pngs"] = output_pngs

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    for mode in rank_modes:
        label = "Top" if mode == "top" else "Bottom"
        print(f"\n{label} matches:")
        for match in result["rankings"].get(mode, []):
            score = match[args.metric]
            ep = Path(match["episode_path"])
            print(
                f"  {label}-{match['rank']}: {args.metric}={score:.4f} | "
                f"{ep.parent.name}/{ep.name} row={match['abs_frame']} "
                f"id={match['data_json_frame_id']} "
                f"label={match['label']}"
            )
        print(f"Saved {label.lower()} visualization: {Path(output_pngs[mode]).resolve()}")
    print(f"Saved metadata:      {out_json.resolve()}")


if __name__ == "__main__":
    main()
