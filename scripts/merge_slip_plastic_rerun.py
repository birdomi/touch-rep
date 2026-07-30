#!/usr/bin/env python
"""Fold the re-run slip_plastic scratch row into the d4_fe4 slip results.

The original job died on a transient CUDA failure, so the batch CSV/MD carry it
as FAILED / n/a. This replaces that row with the successful re-run and
regenerates the markdown through run_seed_ood_matrix.write_md, so the report
stays byte-identical in format to every other one.

Usage:
    python scripts/merge_slip_plastic_rerun.py
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

BATCH = base.PROJECT_ROOT / "scripts" / "results_seed_ood_20260729_170423"
RERUN_LOG_DIR = "slip_rerun_"

base.TASKS.pop("grasp_prediction", None)
base.ENCODERS = {
    "scratch": "null",
    "d4fe4_prox_ep80": (
        "checkpoints/queued_w15_snapshots/temporal_d4fe4cls_prox_ep0080.ckpt"
    ),
}


def main():
    csv_path = BATCH.with_suffix(".csv")
    rows = list(csv.DictReader(csv_path.open()))

    rerun_dirs = sorted((base.PROJECT_ROOT / "scripts" / "logs").glob(f"{RERUN_LOG_DIR}*"))
    if not rerun_dirs:
        raise SystemExit("no slip_rerun_* log directory found")
    log = next(rerun_dirs[-1].glob("ood__slip_detection__scratch__seed0__ho_slip_plastic.log"))
    parsed = base.parse_log(log.read_text(errors="ignore"), "ood")
    if not parsed.get("last_balacc"):
        raise SystemExit(f"could not parse metrics from {log}")

    target = [r for r in rows if r["protocol"] == "ood" and r["encoder"] == "scratch"
              and r["held_out"] == "slip_plastic"]
    if len(target) != 1:
        raise SystemExit(f"expected exactly one slip_plastic scratch row, got {len(target)}")
    row = target[0]
    row.update(parsed)
    row["status"] = "OK"
    row["log"] = str(log)

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=base.FIELDS)
        w.writeheader()
        w.writerows(rows)

    args = argparse.Namespace(backbone_lr="1e-4", num_folds=4, split_seed=42)
    base.write_md(rows, BATCH.with_suffix(".md"), args, [0], list(base.ENCODERS))
    print(f"merged {log}\n  balacc={row['last_balacc']} f1macro={row['last_f1macro']}")
    print(f"updated {csv_path}\n        {BATCH.with_suffix('.md')}")


if __name__ == "__main__":
    main()
