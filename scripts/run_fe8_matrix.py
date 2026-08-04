#!/usr/bin/env python
"""Grasp + slip for the depth-8 frame-encoder pretraining runs.

Uses the *_d4_fe8 downstream configs, which set
model_encoder.frame_encoder.depth: 8. That is mandatory: load_encoder filters
checkpoint tensors by name and shape and then loads with strict=False, so a
depth-8 checkpoint against the depth-4 downstream contributes only its first 4
fusion blocks and leaves the rest random -- silently. Measured on the fe8 tip
epoch-0010 checkpoint: 174/177 tensors load under the fe8 config, only 126/177
under the fe4 one (48 tensors have no slot at all).

Checkpoints are picked up from checkpoints/queued_w15_snapshots via the
FE8_CKPTS env var, as ``label=path`` pairs separated by commas, so the same
script serves whichever epoch snapshot is current:

    FE8_CKPTS="fe8_tip_ep0010=checkpoints/queued_w15_snapshots/temporal_d4fe8cls_tip_fdino_lr4e4_ep0010.ckpt" \
        python scripts/run_fe8_matrix.py --gpus 0 --seeds 0,1 --protocols id

Comparison targets at frame depth 4: tip_lr4e4_ep100 (grasp 0.9308, slip 0.6786)
and tip_lr4e4_ep500 (grasp 0.9333, slip 0.6919), plus the 2-seed scratch
baseline (grasp 0.8480, slip 0.6859).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

raw = os.environ.get("FE8_CKPTS", "").strip()
if not raw:
    raise SystemExit("Set FE8_CKPTS to comma-separated label=path pairs")

encoders = {}
for pair in raw.split(","):
    pair = pair.strip()
    if not pair:
        continue
    label, _, path = pair.partition("=")
    if not path:
        raise SystemExit(f"Malformed FE8_CKPTS entry {pair!r}; expected label=path")
    if not (Path(__file__).resolve().parent.parent / path).exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    encoders[label.strip()] = path.strip()

for task in ("grasp_prediction", "slip_detection"):
    base.TASKS[task]["experiment"] = (
        f"brainco/ours_3d/task/{task}/temporal_w15_cls_d4_fe8"
    )
base.ENCODERS = encoders

if __name__ == "__main__":
    print(f"fe8 encoders: {encoders}", flush=True)
    base.main()
