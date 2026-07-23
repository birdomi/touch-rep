#!/usr/bin/env python3
"""Retrieve downstream grasp frames nearest to a random HOI force frame.

The query is sampled from a pseudo-force PKL (or compact NPZ fallback). Both
query and downstream frames are passed through the same 10-fingertip input
path of the configured XYZ encoder. Retrieval uses cosine similarity between
L2-normalized encoder register-token embeddings.

Example:
    XFORMERS_DISABLED=TRUE python scripts/visualize_hoi_downstream_neighbors.py \
        --hoi-root /path/to/pseudo_force_pickles \
        --top-k 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tactile_ssl.data.pseudo_force_tactile import (  # noqa: E402
    FORCE_CHANNELS,
    _load_compact_pseudo_force_sequence,
    _load_pseudo_force_sequence,
)


OmegaConf.register_new_resolver(
    "int_multiply", lambda a, b: int(a * b), replace=True
)
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b, replace=True)
OmegaConf.register_new_resolver("capitalize", lambda s: s.title(), replace=True)


FINGERTIP_JOINTS_PER_HAND = (4, 8, 12, 16, 20)
FINGERTIP_JOINTS = FINGERTIP_JOINTS_PER_HAND + tuple(
    21 + index for index in FINGERTIP_JOINTS_PER_HAND
)
FINGER_NAMES = ("L-thumb", "L-index", "L-middle", "L-ring", "L-pinky",
                "R-thumb", "R-index", "R-middle", "R-ring", "R-pinky")
DEFAULT_EXPERIMENT = (
    "brainco/ours_3d/task/grasp_prediction/"
    "dinov2_all_rope_temporal_rope_probe"
)


@dataclass(frozen=True)
class CandidateRecord:
    dataset_index: int
    local_frame: int
    absolute_frame: int
    episode_path: str
    label: int
    similarity: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hoi-root",
        type=Path,
        default=Path("pretraining_dataset/pseudo_force_dataset"),
        help="Pseudo-force PKL file/directory. If it has no PKLs, NPZ is used.",
    )
    parser.add_argument(
        "--experiment", default=DEFAULT_EXPERIMENT,
        help="Hydra experiment used to construct and load the encoder.",
    )
    parser.add_argument(
        "--task-checkpoint",
        type=Path,
        default=None,
        help="Optional downstream checkpoint to apply after config initialization.",
    )
    parser.add_argument(
        "--objects", nargs="+", default=None,
        help="Optional downstream objects, e.g. --objects box case.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-frame", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--one-per-episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep at most one retrieved frame from each downstream episode.",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=0,
        help="Random candidate-pool cap; default 0 searches every downstream frame.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/hoi_downstream_nearest.png"),
    )
    args = parser.parse_args()
    if args.top_k <= 0 or args.batch_size <= 0 or args.max_candidates < 0:
        parser.error("top-k/batch-size must be positive and max-candidates must be >= 0")
    return args


def collect_hoi_files(root: Path) -> list[Path]:
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(root.rglob("*.pkl"))
        if not files:
            files = sorted(root.rglob("*.npz"))
            if files:
                print(f"[query] No PKLs under {root}; using compact NPZ files.")
    else:
        raise FileNotFoundError(f"HOI path does not exist: {root}")
    if not files:
        raise FileNotFoundError(f"No pseudo-force PKL/NPZ files found under {root}")
    return files


def load_query(path: Path, frame: int | None, rng: random.Random):
    loader = (
        _load_pseudo_force_sequence if path.suffix.lower() == ".pkl"
        else _load_compact_pseudo_force_sequence
    )
    try:
        sequence = loader(path)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} is not a pseudo-force file with 42x4 joint force and "
            "10x3 fingertip XYZ data"
        ) from exc
    if frame is None:
        frame = rng.randrange(sequence.num_frames)
    if not 0 <= frame < sequence.num_frames:
        raise IndexError(f"query-frame {frame} outside [0, {sequence.num_frames - 1}]")
    contact = sequence.joint_contact[frame, list(FINGERTIP_JOINTS)].float()
    xyz = sequence.finger_xyz[frame].float()
    return contact, xyz, frame, sequence.num_frames


def compose_config(experiment: str, objects: list[str] | None):
    config_dir = PROJECT_ROOT / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="default_task.yaml", overrides=[f"+experiment={experiment}"]
        )
    if objects:
        available = dict(cfg.data.dataset.config.data_roots[0].label_dirs)
        selected = {}
        for obj in objects:
            for suffix in ("succ", "fail"):
                name = f"{obj}_{suffix}"
                if name not in available:
                    raise KeyError(f"Unknown downstream class {name!r}")
                selected[name] = int(available[name])
        cfg.data.dataset.config.data_roots[0].label_dirs = OmegaConf.create(selected)
    return cfg


def load_compatible_checkpoint(module: torch.nn.Module, path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    target = module.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in target and hasattr(value, "shape") and value.shape == target[key].shape
    }
    if not compatible and all(not key.startswith("model_encoder.") for key in state):
        # Probe-only checkpoints are occasionally stored without model_task prefix.
        compatible = {
            f"classifier.{key}": value for key, value in state.items()
            if f"classifier.{key}" in target
            and hasattr(value, "shape")
            and value.shape == target[f"classifier.{key}"].shape
        }
    module.load_state_dict(compatible, strict=False)
    return len(compatible)


def candidate_records(dataset) -> list[CandidateRecord]:
    records = []
    for dataset_index, window in enumerate(dataset.windows):
        start = int(window["window_start_frame"])
        for local_frame in range(window["joint_contact"].shape[0]):
            records.append(CandidateRecord(
                dataset_index=dataset_index,
                local_frame=local_frame,
                absolute_frame=start + local_frame,
                episode_path=str(window["episode_path"]),
                label=int(window["label"]),
            ))
    return records


def sample_records(
    records: list[CandidateRecord], maximum: int, rng: random.Random
) -> list[CandidateRecord]:
    if maximum and len(records) > maximum:
        return [records[index] for index in rng.sample(range(len(records)), maximum)]
    return records


def select_top_indices(
    similarities: torch.Tensor,
    records: list[CandidateRecord],
    top_k: int,
    one_per_episode: bool,
) -> list[int]:
    ranked_indices = similarities.argsort(descending=True).tolist()
    selected = []
    selected_episodes = set()
    for index in ranked_indices:
        episode_path = records[index].episode_path
        if one_per_episode and episode_path in selected_episodes:
            continue
        selected.append(index)
        selected_episodes.add(episode_path)
        if len(selected) == top_k:
            break
    return selected


@torch.inference_mode()
def embed_frames(
    encoder: torch.nn.Module,
    contacts: torch.Tensor,
    xyz: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(contacts), batch_size):
        contact_batch = contacts[start:start + batch_size].to(device).unsqueeze(1)
        xyz_batch = xyz[start:start + batch_size].to(device).unsqueeze(1)
        features = encoder.forward_features(contact_batch, xyz_batch)
        embedding = features["x_norm_regtokens"].mean(dim=1)
        outputs.append(F.normalize(embedding.float(), dim=-1).cpu())
    return torch.cat(outputs)


def collect_candidate_tensors(dataset, records: Iterable[CandidateRecord]):
    contacts, xyz = [], []
    for record in records:
        window = dataset.windows[record.dataset_index]
        contacts.append(window["joint_contact"][record.local_frame].float())
        xyz.append(window["finger_xyz"][record.local_frame].float())
    return torch.stack(contacts), torch.stack(xyz)


def read_episode_length(episode_path: Path) -> int | None:
    try:
        raw = json.loads((episode_path / "data.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    frames = raw.get("data") if isinstance(raw, dict) else None
    return len(frames) if isinstance(frames, list) else None


def load_video_frame(episode_path: Path, absolute_frame: int):
    try:
        import cv2
    except ImportError:
        return None
    video_path = next(
        (episode_path / name for name in ("result.mp4", "colors.mp4", "prepare.mp4")
        if (episode_path / name).exists()),
        None,
    )
    if video_path is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    data_length = read_episode_length(episode_path)
    if video_length <= 0:
        cap.release()
        return None
    if data_length and data_length > 1:
        video_frame = round(absolute_frame * (video_length - 1) / (data_length - 1))
    else:
        video_frame = min(absolute_frame, video_length - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
    ok, image = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def approximate_hand_skeleton(fingertips: np.ndarray) -> np.ndarray:
    """Estimate two 21-joint hands from wrist-local fingertip positions.

    NPZ stores no intermediate joints, so each finger chain is interpolated
    from the wrist origin to its measured fingertip. The returned ordering is
    wrist followed by four joints for thumb, index, middle, ring, and pinky.
    """
    skeletons = []
    fractions = np.asarray((0.30, 0.55, 0.78, 1.0), dtype=np.float32)
    for hand_tips in (fingertips[:5], fingertips[5:]):
        wrist = np.zeros((1, 3), dtype=np.float32)
        finger_chains = [tip[None, :] * fractions[:, None] for tip in hand_tips]
        skeletons.append(np.concatenate((wrist, *finger_chains), axis=0))
    return np.stack(skeletons)


def draw_xyz(axis, xyz: np.ndarray, title: str, contact: np.ndarray | None = None):
    skeletons = approximate_hand_skeleton(xyz)
    hand_colors = ("tab:blue", "tab:orange")
    for hand_index, (skeleton, color) in enumerate(zip(skeletons, hand_colors)):
        wrist = skeleton[0]
        axis.scatter(*wrist, c=color, s=45, marker="s")
        mcp_indices = []
        for finger_index in range(5):
            start = 1 + 4 * finger_index
            chain = np.concatenate((wrist[None, :], skeleton[start:start + 4]), axis=0)
            axis.plot(chain[:, 0], chain[:, 1], chain[:, 2], color=color, linewidth=2)
            axis.scatter(chain[1:-1, 0], chain[1:-1, 1], chain[1:-1, 2],
                         c=color, s=16)
            tip_size = 42.0
            if contact is not None:
                force = max(float(contact[hand_index * 5 + finger_index, 0]), 0.0)
                tip_size += 100.0 * min(force, 1.0)
            axis.scatter(*chain[-1], c=color, edgecolors="black", s=tip_size)
            axis.text(*chain[-1], str(finger_index), fontsize=7)
            mcp_indices.append(start)
        palm = skeleton[mcp_indices]
        axis.plot(palm[:, 0], palm[:, 1], palm[:, 2], color=color,
                  linewidth=1.5, alpha=0.8)
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("x", fontsize=7)
    axis.set_ylabel("y", fontsize=7)
    axis.set_zlabel("z", fontsize=7)
    axis.tick_params(labelsize=6)


def draw_heatmap(axis, contact: np.ndarray):
    image = axis.imshow(contact, aspect="auto", cmap="coolwarm")
    axis.set_yticks(range(10), labels=FINGER_NAMES, fontsize=6)
    axis.set_xticks(range(len(FORCE_CHANNELS)), labels=FORCE_CHANNELS,
                    rotation=35, ha="right", fontsize=6)
    plt.colorbar(image, ax=axis, fraction=0.025, pad=0.02)


def visualize(
    output: Path,
    query_path: Path,
    query_frame: int,
    query_contact: torch.Tensor,
    query_xyz: torch.Tensor,
    results: list[tuple[CandidateRecord, torch.Tensor, torch.Tensor]],
):
    rows = 1 + len(results)
    figure = plt.figure(figsize=(16, 3.5 * rows), constrained_layout=True)
    grid = figure.add_gridspec(rows, 3, width_ratios=(1.25, 1.0, 1.5))

    ax = figure.add_subplot(grid[0, 0])
    ax.axis("off")
    ax.text(0.02, 0.95, "Random HOI query", va="top", weight="bold")
    ax.text(0.02, 0.80, f"file: {query_path}\nframe: {query_frame}", va="top", wrap=True)
    draw_xyz(figure.add_subplot(grid[0, 1], projection="3d"),
             query_xyz.numpy(), "HOI approximate hand skeleton", query_contact.numpy())
    draw_heatmap(figure.add_subplot(grid[0, 2]), query_contact.numpy())

    for row, (record, contact, xyz) in enumerate(results, start=1):
        episode_path = Path(record.episode_path)
        ax_rgb = figure.add_subplot(grid[row, 0])
        rgb = load_video_frame(episode_path, record.absolute_frame)
        if rgb is None:
            ax_rgb.axis("off")
            ax_rgb.text(0.02, 0.95, "RGB unavailable", va="top")
        else:
            ax_rgb.imshow(rgb)
            ax_rgb.axis("off")
        ax_rgb.set_title(
            f"#{row} cosine={record.similarity:.4f}  "
            f"label={'success' if record.label else 'fail'}\n"
            f"{episode_path.parent.name}/{episode_path.name}  frame={record.absolute_frame}",
            fontsize=9,
        )
        draw_xyz(figure.add_subplot(grid[row, 1], projection="3d"),
                 xyz.numpy(), "Downstream approximate hand skeleton", contact.numpy())
        draw_heatmap(figure.add_subplot(grid[row, 2]), contact.numpy())

    figure.suptitle("Nearest downstream frames to a random HOI frame", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    hoi_files = collect_hoi_files(args.hoi_root)
    query_path = rng.choice(hoi_files)
    query_contact, query_xyz, query_frame, num_query_frames = load_query(
        query_path, args.query_frame, rng
    )
    print(f"[query] {query_path} frame={query_frame}/{num_query_frames - 1}")

    cfg = compose_config(args.experiment, args.objects)
    print(f"[data] Instantiating {cfg.data.dataset._target_}")
    dataset = hydra.utils.instantiate(cfg.data.dataset)
    records = sample_records(candidate_records(dataset), args.max_candidates, rng)
    if len(records) < args.top_k:
        raise ValueError(f"Only {len(records)} candidates available for top-k={args.top_k}")
    print(f"[data] Retrieval pool: {len(records)} downstream frames")

    model = hydra.utils.instantiate(cfg.task)
    if args.task_checkpoint is not None:
        loaded = load_compatible_checkpoint(model, args.task_checkpoint)
        print(f"[model] Loaded {loaded} compatible tensors from {args.task_checkpoint}")
    encoder = model.model_encoder.to(args.device).eval()
    if encoder.in_chans != 4 or encoder.sequence_length != 1:
        raise ValueError(
            "This visualization currently requires a 4-channel, single-frame encoder; "
            f"got in_chans={encoder.in_chans}, T={encoder.sequence_length}"
        )

    candidate_contact, candidate_xyz = collect_candidate_tensors(dataset, records)
    candidate_embeddings = embed_frames(
        encoder, candidate_contact, candidate_xyz, args.batch_size, args.device
    )
    query_embedding = embed_frames(
        encoder, query_contact.unsqueeze(0), query_xyz.unsqueeze(0), 1, args.device
    )[0]
    similarities = candidate_embeddings @ query_embedding
    top_indices = select_top_indices(
        similarities, records, args.top_k, args.one_per_episode
    )
    if len(top_indices) < args.top_k:
        print(
            f"[warning] Requested top-{args.top_k}, but only {len(top_indices)} "
            "eligible episodes were available."
        )

    results = []
    json_results = []
    for index in top_indices:
        record = CandidateRecord(
            **{**asdict(records[index]), "similarity": float(similarities[index])}
        )
        results.append((record, candidate_contact[index], candidate_xyz[index]))
        json_results.append(asdict(record))
        print(
            f"[top] cosine={record.similarity:.5f} label={record.label} "
            f"frame={record.absolute_frame} {record.episode_path}"
        )

    visualize(
        args.output, query_path, query_frame, query_contact, query_xyz, results
    )
    metadata = {
        "experiment": args.experiment,
        "task_checkpoint": str(args.task_checkpoint) if args.task_checkpoint else None,
        "seed": args.seed,
        "query": {
            "path": str(query_path),
            "frame": query_frame,
            "num_frames": num_query_frames,
        },
        "candidate_pool_size": len(records),
        "one_per_episode": args.one_per_episode,
        "results": json_results,
    }
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"[saved] {args.output}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
