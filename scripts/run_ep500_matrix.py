#!/usr/bin/env python
"""Grasp + slip at the epoch-500 (completed) snapshot of the prox pretraining runs.

The epoch-500 counterpart of run_ep100_matrix. Same three arms, same downstream
stack (temporal_w15_cls_d4_fe4, 15-frame window), so the ep100 and ep500 tables
are directly comparable:
    fdino_lr4e4  : 42 joints, lr 4e-4     <- reference arm
    fdino_lr1e4  : 42 joints, lr 1e-4     <- lr axis
    tip_lr4e4    : 10 fingertips, lr 4e-4 <- sensor-extent axis

The joint-only arm is NOT here: it needs the *_jointonly downstream configs
(input_streams: pos) and is covered by run_jointonly_matrix.

Each label waits for its own run to finish, so ``--only`` lets the arms that
have already completed be evaluated without blocking on the slower ones. As of
2026-07-30 09:00 only tip_lr4e4 had reached 500; the two 42-joint runs were at
epoch ~362 and ~338.

Usage:
    python scripts/run_ep500_matrix.py --only tip_lr4e4_ep500 --gpus 0,2 --seeds 0,1
    python scripts/run_ep500_matrix.py --gpus 0,2 --seeds 0,1 --protocols id
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = ROOT / "checkpoints" / "queued_w15_snapshots"

RUNS = {
    "fdino_lr4e4_ep500": (
        "experiments/dinov2_temporal_all_pseudo_force_tiny_w15_prox_cls_d4_fe4_fdino_lr4e4"
        "/2026.07.29-16-48/checkpoints/epoch-0500.ckpt",
        "temporal_d4fe4cls_fdino_lr4e4_ep0500.ckpt",
    ),
    "fdino_lr1e4_ep500": (
        "experiments/dinov2_temporal_all_pseudo_force_tiny_w15_prox_cls_d4_fe4_fdino"
        "/2026.07.29-15-20/checkpoints/epoch-0500.ckpt",
        "temporal_d4fe4cls_fdino_lr1e4_ep0500.ckpt",
    ),
    "tip_lr4e4_ep500": (
        "experiments/dinov2_temporal_all_pseudo_force_tiny_tip_w15_prox_cls_d4_fe4_fdino_lr4e4"
        "/2026.07.29-17-47/checkpoints/epoch-0500.ckpt",
        "temporal_d4fe4cls_tip_fdino_lr4e4_ep0500.ckpt",
    ),
}

POLL_SECONDS = 300
STABLE_SECONDS = 30


def pop_only_flag() -> list:
    """Read and remove --only before run_seed_ood_matrix's argparse sees it."""
    if "--only" not in sys.argv:
        return list(RUNS)
    i = sys.argv.index("--only")
    labels = [s.strip() for s in sys.argv[i + 1].split(",") if s.strip()]
    del sys.argv[i:i + 2]
    unknown = [l for l in labels if l not in RUNS]
    if unknown:
        raise SystemExit(f"Unknown --only labels {unknown}; valid: {list(RUNS)}")
    return labels


def wait_and_snapshot(labels) -> dict:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    pending = {k: v for k, v in RUNS.items() if k in labels}
    ready = {}
    while pending:
        for label, (src_rel, dst_name) in list(pending.items()):
            src = ROOT / src_rel
            if not src.exists():
                continue
            size = src.stat().st_size
            time.sleep(STABLE_SECONDS)
            if not src.exists() or src.stat().st_size != size:
                print(f"[wait] {label}: still being written, retrying", flush=True)
                continue
            dst = SNAP_DIR / dst_name
            shutil.copy2(src, dst)
            ready[label] = str(dst.relative_to(ROOT))
            del pending[label]
            print(f"[ready] {label} -> {ready[label]} ({size/1e6:.0f} MB)", flush=True)
        if pending:
            print(f"[wait] still waiting for: {', '.join(sorted(pending))}", flush=True)
            time.sleep(POLL_SECONDS)
    return ready


if __name__ == "__main__":
    labels = pop_only_flag()
    print(f"Waiting for epoch-0500 checkpoints of: {', '.join(labels)} ...", flush=True)
    encoders = wait_and_snapshot(labels)
    print("All checkpoints ready; starting the downstream matrix.\n", flush=True)

    for task in ("grasp_prediction", "slip_detection"):
        base.TASKS[task]["experiment"] = (
            f"brainco/ours_3d/task/{task}/temporal_w15_cls_d4_fe4"
        )
    order = ("fdino_lr4e4_ep500", "fdino_lr1e4_ep500", "tip_lr4e4_ep500")
    base.ENCODERS = {k: encoders[k] for k in order if k in encoders}
    base.main()
