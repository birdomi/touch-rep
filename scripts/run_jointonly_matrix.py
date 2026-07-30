#!/usr/bin/env python
"""Grasp + slip for the joint-only pretraining (no force input at all).

The control arm for the epoch-100 comparison in
results_seed_ood_20260729_204402 / _213833: those three arms all had proximity,
and grasp came out at 0.92-0.93 BalAcc versus 0.84 scratch. This run answers
whether that gain needs the tactile channel or whether hand kinematics alone
reach it.

Two snapshots:
    jointonly_ep100  -- epoch-matched to the three prox arms
    jointonly_ep500  -- the completed run (finished 2026-07-30 03:56)

Both use the *_jointonly downstream configs, which set
frame_encoder.input_streams: pos. That is mandatory, not cosmetic. The
checkpoints still contain sensor_embed/sensor_block tensors that were never fed
and so never trained; with the default input_streams='both' the downstream loads
those untrained weights and routes real BrainCo force through them. Measured on
epoch-0500: swapping the force input changes the encoder output by 1.39 under the
default config and by exactly 0.0 under the jointonly config.

load_encoder reports 126/129 tensors loaded here. The 3 skipped
(sensor_embed.proj.weight, signal_mean, signal_std) are the 1-channel vs
4-channel mismatch every prox evaluation also hits -- but for this arm they are
tensors the model does not use at all.

Usage:
    python scripts/run_jointonly_matrix.py --gpus 0,2 --seeds 0,1 --protocols id
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

for task in ("grasp_prediction", "slip_detection"):
    base.TASKS[task]["experiment"] = (
        f"brainco/ours_3d/task/{task}/temporal_w15_cls_d4_fe4_jointonly"
    )

SNAP = "checkpoints/queued_w15_snapshots"
base.ENCODERS = {
    "jointonly_ep100": f"{SNAP}/temporal_d4fe4cls_jointonly_lr4e4_ep0100.ckpt",
    "jointonly_ep500": f"{SNAP}/temporal_d4fe4cls_jointonly_lr4e4_ep0500.ckpt",
}

if __name__ == "__main__":
    base.main()
