#!/usr/bin/env python
"""Slip detection on slip_data_v3, scratch only, 3 seeds.

Re-runs the w15 scratch baseline on the new v3 collection. Every slip number so
far came from v2, where scratch sat at ~0.689 BalAcc and no pretrained arm beat
it; this establishes what scratch does on v3 before anything else is compared
against it.

v3 differs from v2 in more than size: 18 episodes (3 per class) across 6 classes
instead of 97 across 4, and episodes run 1795-3775 frames instead of 220-446.
Episode-level 4-fold CV therefore leaves only 4-5 validation episodes per fold,
so the per-seed spread matters more than the mean here.

Usage:
    python scripts/run_slip_v3_scratch.py --gpus 1 --seeds 0,1,2 --protocols id
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

base.TASKS.pop("grasp_prediction", None)
base.TASKS["slip_detection"]["experiment"] = (
    "brainco/ours_3d/task/slip_detection/temporal_w15_cls_d4_fe4_v3"
)
base.ENCODERS = {"scratch": "null"}

if __name__ == "__main__":
    base.main()
