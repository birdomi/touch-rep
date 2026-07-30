#!/usr/bin/env python
"""Leave-one-object-out (OOD) runs for the temporal d4_fe4 CLS stack.

Thin wrapper over run_seed_ood_matrix: same protocol, parsing and report
format, with the experiments pointed at the temporal_w15_cls_d4_fe4 task
configs and the encoders at {scratch, d4fe4 prox ep50 snapshot}.

Usage:
    python scripts/run_ood_temporal_d4fe4.py --gpus 0,1 --seeds 0 --protocols ood
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
    "d4fe4_prox_ep50": (
        "checkpoints/queued_w15_snapshots/temporal_d4fe4cls_prox_ep0050.ckpt"
    ),
}

if __name__ == "__main__":
    base.main()
