#!/usr/bin/env python
"""Re-run a single slip OOD job that died on a transient CUDA failure.

Schedules exactly one job through run_seed_ood_matrix.run_job, so the protocol,
parsing and logging are identical to the batch it belongs to.

TASKS["slip_detection"]["objects"] must keep the FULL object list — ood_overrides
derives the training classes from it (train = objects - held_out). Narrowing it
to the held-out object alone leaves the training set with `classes: []`.

Usage:
    python scripts/rerun_slip_scratch_plastic.py --gpu 0 --held-out slip_plastic
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

base.TASKS["slip_detection"]["experiment"] = (
    "brainco/ours_3d/task/slip_detection/temporal_w15_cls_d4_fe4"
)
base.ENCODERS = {
    "scratch": "null",
    "d4fe4_prox_ep80": (
        "checkpoints/queued_w15_snapshots/temporal_d4fe4cls_prox_ep0080.ckpt"
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--encoder", default="scratch", choices=list(base.ENCODERS))
    ap.add_argument("--held-out", default="slip_plastic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone-lr", default="1e-4")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--num-folds", type=int, default=4)
    args = ap.parse_args()

    assert args.held_out in base.TASKS["slip_detection"]["objects"], args.held_out
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = base.PROJECT_ROOT / "scripts" / "logs" / f"slip_rerun_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    job = ("ood", "slip_detection", args.encoder, args.seed, args.held_out)
    row = base.run_job(job, args.gpu, log_dir, args)
    print(f"\nbalacc={row.get('last_balacc')} f1macro={row.get('last_f1macro')} "
          f"f1={row.get('last_f1')} status={row['status']}\nlog: {row['log']}")


if __name__ == "__main__":
    main()
