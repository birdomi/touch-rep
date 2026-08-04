#!/usr/bin/env python
"""Slip detection on slip_data_v3 for the joint-only arm (no force input).

Separate from run_slip_v3.py because this arm needs the *_v3_jointonly task
config, which sets frame_encoder.input_streams: pos so the downstream encoder has
no sensor stream either. Evaluating this checkpoint with the plain v3 config
would load its untrained sensor_embed/sensor_block and route real force through
them.

On v2 the arm sat exactly at chance (0.5007 BalAcc at both ep100 and ep500) with
a degenerate single-class head. v3's episodes are 6-10x longer, so this checks
whether any pose-only slip signal appears there.

Usage:
    python scripts/run_slip_v3_jointonly.py --gpus 1,3 --seeds 0,1,2 --num-folds 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

base.TASKS.pop("grasp_prediction", None)
base.TASKS["slip_detection"]["experiment"] = (
    "brainco/ours_3d/task/slip_detection/temporal_w15_cls_d4_fe4_v3_jointonly"
)

SNAP = "checkpoints/queued_w15_snapshots"
base.ENCODERS = {
    "jointonly_ep500": f"{SNAP}/temporal_d4fe4cls_jointonly_lr4e4_ep0500.ckpt",
}

if __name__ == "__main__":
    base.main()
