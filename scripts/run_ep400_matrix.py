#!/usr/bin/env python
"""Grasp + slip at the epoch-400 snapshot of the two 42-joint prox runs.

Fills in the middle of the epoch sweep for the lr axis, between the ep100 table
(results_seed_ood_20260729_204402 / _213833) and the ep500 runs still finishing:

    fdino_lr4e4  : 42 joints, lr 4e-4   -- grasp 0.9253 at ep100
    fdino_lr1e4  : 42 joints, lr 1e-4   -- grasp 0.8770 at ep100

Both snapshots were copied out of the live runs while they continue toward 500,
so they are frozen files, not paths into a directory being written.

Same downstream stack and protocol as every other entry in that sweep:
temporal_w15_cls_d4_fe4, 15-frame window, ID 4-fold, backbone lr 1e-4, 2 seeds.
Scratch reference at 2 seeds: grasp 0.8480 +- 0.0120, slip 0.6859 +- 0.0041.

Usage:
    python scripts/run_ep400_matrix.py --gpus 0,2 --seeds 0,1 --protocols id
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

for task in ("grasp_prediction", "slip_detection"):
    base.TASKS[task]["experiment"] = (
        f"brainco/ours_3d/task/{task}/temporal_w15_cls_d4_fe4"
    )

SNAP = "checkpoints/queued_w15_snapshots"
base.ENCODERS = {
    "fdino_lr4e4_ep400": f"{SNAP}/temporal_d4fe4cls_fdino_lr4e4_ep0400.ckpt",
    "fdino_lr1e4_ep400": f"{SNAP}/temporal_d4fe4cls_fdino_lr1e4_ep0400.ckpt",
}

if __name__ == "__main__":
    base.main()
