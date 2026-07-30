#!/usr/bin/env python
"""Slip detection at a 30-frame downstream window: scratch vs. fdino lr4e-4 ep40.

Same encoders and protocol as run_slip_fdino_lr4e4_ep40, pointed at
slip_detection/temporal_w30_cls_d4_fe4 instead of the w15 config. That config
sets input_window_frames/stride to 30 and model_encoder.sequence_length to 30
together -- changing only the data side would leave encode() splitting each
window into two encoder passes.

The pretraining ran at W=15, so the former sees twice its training sequence
length here. It is RoPE over frame index, so that is extrapolation, not a shape
mismatch; read any w15-vs-w30 difference as a window-length effect.

Usage:
    python scripts/run_slip_w30_fdino_lr4e4_ep40.py --gpus 2 --seeds 0 --protocols id
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

base.TASKS.pop("grasp_prediction", None)
base.TASKS["slip_detection"]["experiment"] = (
    "brainco/ours_3d/task/slip_detection/temporal_w30_cls_d4_fe4"
)
base.ENCODERS = {
    "scratch": "null",
    "fdino_lr4e4_ep40": (
        "checkpoints/queued_w15_snapshots/temporal_d4fe4cls_fdino_lr4e4_ep0040.ckpt"
    ),
}

if __name__ == "__main__":
    base.main()
