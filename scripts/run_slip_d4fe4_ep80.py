#!/usr/bin/env python
"""Slip detection: scratch vs. the d4_fe4 prox pretraining ep80 snapshot.

Companion to run_grasp_d4fe4_ep80 — same protocol and report format, same
encoders, restricted to slip_detection and pointed at the
temporal_w15_cls_d4_fe4 task config so both former (depth 4) and frame encoder
(depth 4) match the pretraining stack.

Usage:
    python scripts/run_slip_d4fe4_ep80.py --gpus 0,1,2,3 --seeds 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

base.TASKS.pop("grasp_prediction", None)
base.TASKS["slip_detection"]["experiment"] = (
    "brainco/ours_3d/task/slip_detection/temporal_w15_cls_d4_fe4"
)
base.ENCODERS = {
    "scratch": "null",
    "d4fe4_prox_ep80": (
        "checkpoints/queued_w15_snapshots/temporal_d4fe4cls_prox_ep0080.ckpt"
    ),
}

if __name__ == "__main__":
    base.main()
