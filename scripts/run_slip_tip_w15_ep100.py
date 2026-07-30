#!/usr/bin/env python
"""Slip detection: scratch vs. the tip_w15 pretraining ep100 snapshot.

The tip_w15 stack (config/algorithm/temporal_dinov2.yaml) differs from the
prox_cls_d4_fe4 one evaluated earlier today: 4 input channels, fingertip-only
in_dim 10, ``frame_token_set: all`` and the default frame-encoder depth. The
matching downstream config is therefore ``slip_detection/temporal_w15`` — the
same pairing queue_w15_downstream.py uses for this experiment. Using the
d4_fe4 config here would make load_encoder silently drop blocks.

Usage:
    python scripts/run_slip_tip_w15_ep100.py --gpus 0,1,2 --seeds 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

base.TASKS.pop("grasp_prediction", None)
base.TASKS["slip_detection"]["experiment"] = (
    "brainco/ours_3d/task/slip_detection/temporal_w15"
)
base.ENCODERS = {
    "scratch": "null",
    "tip_w15_ep100": "checkpoints/queued_w15_snapshots/temporal_w15_ep0100.ckpt",
}

if __name__ == "__main__":
    base.main()
