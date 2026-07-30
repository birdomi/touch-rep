#!/usr/bin/env python
"""Grasp prediction: scratch vs. the d4_fe4 prox pretraining ep80 snapshot.

Thin wrapper over run_seed_ood_matrix — same protocol, parsing and report
format — restricted to grasp_prediction and pointed at the
temporal_w15_cls_d4_fe4 task config so both former (depth 4) and frame encoder
(depth 4) match the pretraining stack. Anything else silently drops blocks on
load_encoder.

Usage:
    python scripts/run_grasp_d4fe4_ep80.py --gpus 0,1,2,3 --seeds 0,1,2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

base.TASKS.pop("slip_detection", None)
base.TASKS["grasp_prediction"]["experiment"] = (
    "brainco/ours_3d/task/grasp_prediction/temporal_w15_cls_d4_fe4"
)
base.ENCODERS = {
    "scratch": "null",
    "d4fe4_prox_ep80": (
        "checkpoints/queued_w15_snapshots/temporal_d4fe4cls_prox_ep0080.ckpt"
    ),
}

if __name__ == "__main__":
    base.main()
