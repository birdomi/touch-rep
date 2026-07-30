#!/usr/bin/env python
"""Run grasp + slip downstream on every new epoch-*.ckpt of a pretraining run.

Polls the checkpoint directory and, whenever an epoch snapshot appears that has
not been evaluated yet, launches both downstream tasks (one GPU each) and
appends the result to a CSV. Exits when the pretraining process is gone and no
unevaluated checkpoints remain.

Usage:
    python scripts/watch_recon_epochs.py \
        --ckpt-dir experiments/<run>/checkpoints --gpus 5,6
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS = {
    "grasp_prediction": ("brainco/ours_3d/task/grasp_prediction/dinov2_all_rope", []),
    "slip_detection": ("brainco/ours_3d/task/slip_detection/dinov2_all_rope", [
        "data.dataset.config.input_window_frames=3",
        "data.dataset.config.input_window_stride=3",
    ]),
}
PROBE = [
    "task.model_task._target_=tactile_ssl.downstream_task.BraincoGraspRoPEProbe",
    "+task.model_task.depth=2",
    "+task.model_task.num_heads=3",
]
SUMMARY_RE = re.compile(
    r"K-FOLD SUMMARY \(Last Epoch\).*?^\s*Mean\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)"
    r".*?^\s*Std\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)",
    re.DOTALL | re.MULTILINE,
)
FIELDS = ["epoch", "task", "balacc", "f1", "f1macro",
          "balacc_foldstd", "f1macro_foldstd", "status", "log"]


def run_one(epoch, ckpt, task, gpu, log_dir, args, out):
    experiment, extra = TASKS[task]
    log_path = log_dir / f"e{epoch:04d}__{task}.log"
    cmd = [
        "python", "train_task_brainco_angle.py",
        f"+experiment={experiment}",
        f"task.checkpoint_encoder={ckpt}",
        f"task.encoder_lr={args.backbone_lr}", f"task.task_lr={args.task_lr}",
        *PROBE, *extra,
        f"seed={args.seed}", f"++split_seed={args.split_seed}",
        f"experiment_name=reconwatch_e{epoch:04d}_{task}",
        "--all_split", "--num_folds", str(args.num_folds),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XFORMERS_DISABLED": "TRUE"}
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=fh,
                              stderr=subprocess.STDOUT)
    m = SUMMARY_RE.search(log_path.read_text(errors="ignore"))
    row = {f: "" for f in FIELDS}
    row.update(epoch=str(epoch), task=task, log=str(log_path))
    if m and proc.returncode == 0:
        ba, f1, f1m, ba_s, _f1_s, f1m_s = m.groups()
        row.update(balacc=ba, f1=f1, f1macro=f1m,
                   balacc_foldstd=ba_s, f1macro_foldstd=f1m_s, status="OK")
    else:
        row["status"] = "FAILED"
    out.append(row)
    print(f"[{row['status']}] epoch {epoch:>4} {task:<17} "
          f"balacc={row['balacc'] or '-'} f1macro={row['f1macro'] or '-'}", flush=True)


def pretraining_alive(pattern):
    if not pattern:
        return False
    return subprocess.run(["pgrep", "-f", pattern],
                          stdout=subprocess.DEVNULL).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--gpus", default="5,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone-lr", default="1e-4")
    ap.add_argument("--task-lr", default="1e-4")
    ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--skip", default="10,20,30", help="epochs already measured")
    ap.add_argument("--alive-pattern", default="dinov2_recon_all_rope_v2_hp1",
                    help="pgrep pattern for the pretraining job; empty disables the check")
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--max-idle-minutes", type=int, default=90)
    args = ap.parse_args()

    ckpt_dir = (PROJECT_ROOT / args.ckpt_dir).resolve()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()][:2]
    done = {int(e) for e in args.skip.split(",") if e.strip().isdigit()}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "scripts" / "logs" / f"reconwatch_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = PROJECT_ROOT / "scripts" / f"results_reconwatch_{stamp}.csv"
    rows: list = []
    print(f"watching {ckpt_dir}; already measured {sorted(done)}; gpus {gpus}", flush=True)

    last_progress = time.time()
    while True:
        pending = sorted(
            int(m.group(1))
            for p in ckpt_dir.glob("epoch-*.ckpt")
            if (m := re.search(r"epoch-(\d+)\.ckpt$", p.name)) and int(m.group(1)) not in done
        )
        if pending:
            epoch = pending[0]
            ckpt = str((ckpt_dir / f"epoch-{epoch:04d}.ckpt").relative_to(PROJECT_ROOT))
            print(f"--- epoch {epoch} ---", flush=True)
            threads = [
                threading.Thread(target=run_one,
                                 args=(epoch, ckpt, task, gpu, log_dir, args, rows))
                for task, gpu in zip(TASKS, gpus)
            ]
            for t in threads:
                t.start()
                time.sleep(2)
            for t in threads:
                t.join()
            done.add(epoch)
            last_progress = time.time()
            with csv_path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(sorted(rows, key=lambda r: (int(r["epoch"]), r["task"])))
            continue

        if not pretraining_alive(args.alive_pattern):
            print("pretraining process gone and nothing pending — exiting", flush=True)
            break
        if (time.time() - last_progress) / 60 > args.max_idle_minutes:
            print(f"no new checkpoint for {args.max_idle_minutes} min — exiting", flush=True)
            break
        time.sleep(args.poll_seconds)

    print(f"\nCSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
