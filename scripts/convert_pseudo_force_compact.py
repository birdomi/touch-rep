#!/usr/bin/env python3
"""Convert verbose pseudo-force pickles into compact training arrays.

Each source pickle becomes one NPZ containing only:

* joint_contact: float32 (N, 42, 4)
* finger_xyz: float32 (N, 10, 3)

The output directory mirrors the source dataset directories. Conversion is
atomic and resumable: an existing, valid NPZ is skipped unless ``--overwrite``
is passed. Global channel statistics are written to ``metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tactile_ssl.data.pseudo_force_tactile import (  # noqa: E402
    FORCE_CHANNELS,
    _load_pseudo_force_sequence,
)

DEFAULT_DATASETS = ("arctic", "hot3d", "oakinkv2", "taco")


@dataclass
class ConversionResult:
    source: str
    destination: str
    frames: int
    raw_bytes: int
    compact_bytes: int
    channel_sum: list[float]
    channel_sq_sum: list[float]
    channel_count: int
    skipped: bool


def _array_stats(joint_contact: np.ndarray):
    values = joint_contact.reshape(-1, len(FORCE_CHANNELS)).astype(
        np.float64, copy=False
    )
    return (
        values.sum(axis=0).tolist(),
        np.square(values).sum(axis=0).tolist(),
        values.shape[0],
    )


def _load_compact(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "joint_force" in data:
            force_key = "joint_force"
        elif "joint_contact" in data:
            force_key = "joint_contact"
        else:
            raise KeyError(f"{path}: expected 'joint_force' or 'joint_contact'")
        joint_contact = np.asarray(data[force_key], dtype=np.float32)
        finger_xyz = np.asarray(data["finger_xyz"], dtype=np.float32)
    if joint_contact.ndim != 3 or joint_contact.shape[1:] != (42, 4):
        raise ValueError(f"Invalid joint_contact shape in {path}: {joint_contact.shape}")
    if finger_xyz.shape != (joint_contact.shape[0], 10, 3):
        raise ValueError(f"Invalid finger_xyz shape in {path}: {finger_xyz.shape}")
    return joint_contact, finger_xyz


def _write_atomic(
    destination: Path,
    joint_contact: np.ndarray,
    finger_xyz: np.ndarray,
    compressed: bool,
):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.",
            suffix=".npz",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        save = np.savez_compressed if compressed else np.savez
        save(
            temporary_path,
            joint_contact=np.ascontiguousarray(joint_contact, dtype=np.float32),
            finger_xyz=np.ascontiguousarray(finger_xyz, dtype=np.float32),
        )
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _convert_one(
    source_string: str,
    destination_string: str,
    compressed: bool,
    overwrite: bool,
) -> ConversionResult:
    source = Path(source_string)
    destination = Path(destination_string)
    skipped = False

    if destination.exists() and not overwrite:
        try:
            joint_contact, finger_xyz = _load_compact(destination)
            skipped = True
        except (KeyError, OSError, ValueError):
            joint_contact = finger_xyz = None
    else:
        joint_contact = finger_xyz = None

    if joint_contact is None or finger_xyz is None:
        sequence = _load_pseudo_force_sequence(source)
        joint_contact = sequence.joint_contact.numpy()
        finger_xyz = sequence.finger_xyz.numpy()
        _write_atomic(destination, joint_contact, finger_xyz, compressed)

    channel_sum, channel_sq_sum, channel_count = _array_stats(joint_contact)
    return ConversionResult(
        source=str(source),
        destination=str(destination),
        frames=joint_contact.shape[0],
        raw_bytes=source.stat().st_size,
        compact_bytes=destination.stat().st_size,
        channel_sum=channel_sum,
        channel_sq_sum=channel_sq_sum,
        channel_count=channel_count,
        skipped=skipped,
    )


def _collect_jobs(
    input_root: Path,
    output_root: Path,
    datasets: Iterable[str],
    limit_per_dataset: int | None,
):
    jobs = []
    for dataset_name in datasets:
        source_dir = input_root / dataset_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {source_dir}")
        sources = sorted(source_dir.glob("*.pkl"))
        if limit_per_dataset is not None:
            sources = sources[:limit_per_dataset]
        if not sources:
            raise FileNotFoundError(f"No pickle files found in {source_dir}")
        jobs.extend(
            (source, output_root / dataset_name / f"{source.stem}.npz")
            for source in sources
        )
    return jobs


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _write_metadata(
    output_root: Path,
    input_root: Path,
    datasets: list[str],
    results: list[ConversionResult],
    failures: list[dict],
    compressed: bool,
):
    channel_sum = np.sum([result.channel_sum for result in results], axis=0)
    channel_sq_sum = np.sum([result.channel_sq_sum for result in results], axis=0)
    channel_count = sum(result.channel_count for result in results)
    if channel_count:
        mean = channel_sum / channel_count
        variance = np.maximum(channel_sq_sum / channel_count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std[std <= 1e-6] = 1.0
    else:
        mean = np.zeros(len(FORCE_CHANNELS), dtype=np.float64)
        std = np.ones(len(FORCE_CHANNELS), dtype=np.float64)

    metadata = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(input_root.resolve()),
        "datasets": datasets,
        "compressed": compressed,
        "force_channels": list(FORCE_CHANNELS),
        "joint_contact_shape": ["N", 42, 4],
        "finger_xyz_shape": ["N", 10, 3],
        "normalization": {
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "sample_count_per_channel": channel_count,
        },
        "converted_files": len(results),
        "skipped_files": sum(result.skipped for result in results),
        "total_frames": sum(result.frames for result in results),
        "raw_bytes": sum(result.raw_bytes for result in results),
        "compact_bytes": sum(result.compact_bytes for result in results),
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.json"
    temporary_path = output_root / ".metadata.json.tmp"
    temporary_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, metadata_path)
    return metadata_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("pretraining_dataset/pseudo_force_dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pretraining_dataset/pseudo_force_compact"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent files. Keep at 1 unless the machine has abundant RAM.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Use faster, larger uncompressed NPZ output.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Convert only the first N files of each dataset (for testing).",
    )
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.limit_per_dataset is not None and args.limit_per_dataset <= 0:
        parser.error("--limit-per-dataset must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    return args


def main():
    args = parse_args()
    jobs = _collect_jobs(
        args.input_root,
        args.output_root,
        args.datasets,
        args.limit_per_dataset,
    )
    compressed = not args.uncompressed
    print(f"Input:       {args.input_root}", flush=True)
    print(f"Output:      {args.output_root}", flush=True)
    print(f"Datasets:    {', '.join(args.datasets)}", flush=True)
    print(f"Files:       {len(jobs)}", flush=True)
    print(f"Workers:     {args.workers}", flush=True)
    print(f"Compression: {'deflate' if compressed else 'none'}", flush=True)

    start_time = time.monotonic()
    results: list[ConversionResult] = []
    failures: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _convert_one,
                str(source),
                str(destination),
                compressed,
                args.overwrite,
            ): source
            for source, destination in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                failures.append({"source": str(source), "error": repr(exc)})
                print(f"ERROR [{completed}/{len(jobs)}] {source}: {exc!r}", flush=True)

            if completed % args.log_every == 0 or completed == len(jobs):
                elapsed = time.monotonic() - start_time
                frames = sum(result.frames for result in results)
                print(
                    f"[{completed}/{len(jobs)}] elapsed={elapsed / 60:.1f} min "
                    f"frames={frames:,} failures={len(failures)}",
                    flush=True,
                )

    metadata_path = _write_metadata(
        output_root=args.output_root,
        input_root=args.input_root,
        datasets=args.datasets,
        results=results,
        failures=failures,
        compressed=compressed,
    )
    raw_bytes = sum(result.raw_bytes for result in results)
    compact_bytes = sum(result.compact_bytes for result in results)
    ratio = raw_bytes / compact_bytes if compact_bytes else float("inf")
    print(f"Raw size:     {_format_bytes(raw_bytes)}", flush=True)
    print(f"Compact size: {_format_bytes(compact_bytes)}", flush=True)
    print(f"Size ratio:   {ratio:.1f}x", flush=True)
    print(f"Metadata:     {metadata_path}", flush=True)

    if failures:
        raise SystemExit(f"Conversion completed with {len(failures)} failed files")


if __name__ == "__main__":
    main()
