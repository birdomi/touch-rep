#!/usr/bin/env python
"""Slip detection on slip_data_v3: scratch and the temporal_w15_42j_lr4e4 arms.

Same protocol as the v2 runs so the two collections can be compared, except the
folds are stratified by episode index within class and num_folds must be 3 --
v3 has exactly 3 episodes per class, so fold k validates episode k of all six
classes. Pass --num-folds 3.

Encoders come from the temporal_w15_42j_lr4e4 pretraining (42 joints,
proximity-only, lr 4e-4, the reference arm). On v2 this arm scored 0.6843 BalAcc
at epoch 100 and 0.6836 at epoch 500, i.e. indistinguishable from scratch's
0.6888 -- v3 is a chance to see whether that holds on the new collection.

Select an arm with --encoders, e.g.

    python scripts/run_slip_v3.py --encoders scratch --gpus 1 --seeds 0,1,2 --num-folds 3
    python scripts/run_slip_v3.py --encoders fdino_lr4e4_ep500 --gpus 3 --seeds 0,1,2 --num-folds 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

base.TASKS.pop("grasp_prediction", None)
base.TASKS["slip_detection"]["experiment"] = (
    "brainco/ours_3d/task/slip_detection/temporal_w15_cls_d4_fe4_v3"
)

SNAP = "checkpoints/queued_w15_snapshots"
base.ENCODERS = {
    "scratch": "null",
    "fdino_lr4e4_ep500": f"{SNAP}/temporal_d4fe4cls_fdino_lr4e4_ep0500.ckpt",
    "fdino_lr4e4_ep100": f"{SNAP}/temporal_d4fe4cls_fdino_lr4e4_ep0100.ckpt",
    "fdino_lr1e4_ep500": f"{SNAP}/temporal_d4fe4cls_fdino_lr1e4_ep0500.ckpt",
    "tip_lr4e4_ep500": f"{SNAP}/temporal_d4fe4cls_tip_fdino_lr4e4_ep0500.ckpt",
}

# The joint-only checkpoint is deliberately absent: it needs the
# *_v3_jointonly task config (input_streams: pos), so it goes through
# run_slip_v3_jointonly.py instead. Listing it here would silently evaluate it
# with a sensor stream it never trained.

if __name__ == "__main__":
    base.main()
