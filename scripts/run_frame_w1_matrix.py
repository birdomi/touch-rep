#!/usr/bin/env python
"""Grasp + slip for the frame_w1_* (single-frame, 4D-RoPE) pretraining line.

These checkpoints come from a different stack than everything evaluated so far:
angle_tiny at depth 8 with no temporal former, one frame in and one
representation out. The matching downstream config is therefore
``dinov2_all_rope`` (grasp) / ``dinov2_all_rope_v3`` (slip), NOT the
temporal_w15_* configs -- those wrap the encoder in a temporal former the
frame_w1 checkpoints never had.

Neither pretraining reached its 500-epoch target; both died on 2026-07-30/31
with tip at epoch 240 and 42j at epoch 130. Three encoders are evaluated so the
sensor-extent axis stays epoch-matched:

    tip_ep130   10 fingertips, epoch 130   <- compare against 42j_ep130
    42j_ep130   42 joints,     epoch 130
    tip_ep240   10 fingertips, epoch 240   <- best tip checkpoint available

Loading is verified: 121/125 checkpoint tensors land, the four that do not are
``mask_token`` (absent downstream, with_masktoken is false there) and the usual
1-channel-vs-4-channel trio (``sensor_embed.proj.weight``, ``signal_mean``,
``signal_std``) that every proximity-pretrained evaluation hits.

Usage:
    python scripts/run_frame_w1_matrix.py --task grasp --gpus 0,1,2,3 --seeds 1000,1001,1002 --num-folds 4
    python scripts/run_frame_w1_matrix.py --task slip  --gpus 0,1,2,3 --seeds 1000,1001,1002 --num-folds 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

SNAP = "checkpoints/queued_w15_snapshots"
ENCODERS = {
    "scratch": "null",
    "frame_w1_tip_ep130": f"{SNAP}/frame_w1_tip_ep0130.ckpt",
    "frame_w1_42j_ep130": f"{SNAP}/frame_w1_42j_ep0130.ckpt",
    "frame_w1_tip_ep240": f"{SNAP}/frame_w1_tip_ep0240.ckpt",
}

CONFIGS = {
    "grasp": ("grasp_prediction",
              "brainco/ours_3d/task/grasp_prediction/dinov2_all_rope"),
    "slip": ("slip_detection",
             "brainco/ours_3d/task/slip_detection/dinov2_all_rope_v3"),
}


def pop(flag, valid):
    if flag not in sys.argv:
        raise SystemExit(f"{flag} is required (one of {valid})")
    i = sys.argv.index(flag)
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    if v not in valid:
        raise SystemExit(f"Unknown {flag} {v!r}; valid: {valid}")
    return v


if __name__ == "__main__":
    task = pop("--task", list(CONFIGS))
    key, experiment = CONFIGS[task]
    for other in [k for k in base.TASKS if k != key]:
        base.TASKS.pop(other)
    base.TASKS[key]["experiment"] = experiment
    base.ENCODERS = ENCODERS
    print(f"task={key}  experiment={experiment}")
    base.main()
