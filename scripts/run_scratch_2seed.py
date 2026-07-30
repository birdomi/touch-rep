#!/usr/bin/env python
"""The scratch baseline at 2 seeds, for both tasks, on the temporal_w15_cls_d4_fe4 stack.

Every pretrained arm in the ep100/ep500 tables has 2 seeds, but the scratch row
they are all compared against had only seed 0. On slip that matters: the arms sit
within +-0.01 BalAcc of scratch while single-arm seed spread reaches 0.035, so a
one-run baseline cannot support any claim either way.

Same downstream config and protocol as run_ep100_matrix / run_ep500_matrix
(15-frame window, ID 4-fold, backbone lr 1e-4), with checkpoint_encoder=null.
in_dim is inert under RoPE, so this single scratch baseline is the correct
reference for both the 42-joint and the fingertip arms.

Usage:
    python scripts/run_scratch_2seed.py --gpus 0,2 --seeds 0,1 --protocols id
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

for task in ("grasp_prediction", "slip_detection"):
    base.TASKS[task]["experiment"] = (
        f"brainco/ours_3d/task/{task}/temporal_w15_cls_d4_fe4"
    )
base.ENCODERS = {"scratch": "null"}

if __name__ == "__main__":
    base.main()
