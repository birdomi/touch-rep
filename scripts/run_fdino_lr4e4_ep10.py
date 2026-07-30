#!/usr/bin/env python
"""Grasp + slip ID runs for the fdino lr4e-4 pretraining, epoch 10 snapshot.

Thin wrapper over run_seed_ood_matrix -- same protocol, parsing and report
format -- with both tasks pointed at the temporal_w15_cls_d4_fe4 task config so
the former (depth 4) and frame encoder (depth 4) match the pretraining stack.
Anything else silently drops blocks on load_encoder.

The snapshot comes from dinov2_temporal_w15_prox_cls_d4_fe4_fdino_lr4e4, the
first run carrying the frame-branch pose-crop and cross-view DINO fixes, at
only 10 of 500 epochs and still inside the 10-epoch LR warmup. Treat the
numbers as an early sanity check, not a converged comparison.

Usage:
    python scripts/run_fdino_lr4e4_ep10.py --gpus 0,2 --seeds 0 --protocols id
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

base.TASKS["grasp_prediction"]["experiment"] = (
    "brainco/ours_3d/task/grasp_prediction/temporal_w15_cls_d4_fe4"
)
base.TASKS["slip_detection"]["experiment"] = (
    "brainco/ours_3d/task/slip_detection/temporal_w15_cls_d4_fe4"
)
base.ENCODERS = {
    "scratch": "null",
    "fdino_lr4e4_ep10": (
        "checkpoints/queued_w15_snapshots/temporal_d4fe4cls_fdino_lr4e4_ep0010.ckpt"
    ),
}

if __name__ == "__main__":
    base.main()
