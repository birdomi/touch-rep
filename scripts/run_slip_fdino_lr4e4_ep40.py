#!/usr/bin/env python
"""Slip detection: scratch vs. the fdino lr4e-4 pretraining, epoch 40 snapshot.

Same pairing as run_fdino_lr4e4_ep10 -- the snapshot comes from
dinov2_temporal_w15_prox_cls_d4_fe4_fdino_lr4e4, so the downstream stack must be
slip_detection/temporal_w15_cls_d4_fe4 (former depth 4, frame encoder depth 4,
proximity-only). Any other task config silently drops blocks on load_encoder.

Restricted to slip_detection, unlike the ep10 wrapper. Epoch 40 of 500 is past
the 10-epoch LR warmup but nowhere near converged; the ep10 run put slip at
0.678 F1macro vs. 0.662 scratch on a single seed, which is inside seed noise,
so this one runs 3 seeds to get an actual error bar on the gap.

Usage:
    python scripts/run_slip_fdino_lr4e4_ep40.py --gpus 2 --seeds 0,1,2 --protocols id
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
    "fdino_lr4e4_ep40": (
        "checkpoints/queued_w15_snapshots/temporal_d4fe4cls_fdino_lr4e4_ep0040.ckpt"
    ),
}

if __name__ == "__main__":
    base.main()
