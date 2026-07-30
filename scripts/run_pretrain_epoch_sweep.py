#!/usr/bin/env python
"""Downstream performance vs. pretraining epoch.

Runs grasp prediction and slip detection (ID, K-fold) once per `epoch-*.ckpt`
found in a pretraining run directory, so the transfer curve over pretraining
duration can be read off directly.

Usage:
    python scripts/run_pretrain_epoch_sweep.py \\
        --run-dir experiments/<pretrain_experiment>/<timestamp> --gpus 4,5,6,7
"""
from __future__ import annotations

import argparse
import csv
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_lr_checkpoint_matrix import TASKS, parse_log  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIELDS = [
    "task", "epoch", "checkpoint", "status",
    "last_balacc", "last_f1", "last_f1macro",
    "best_balacc", "best_f1", "best_f1macro",
    "epochavg_balacc", "epochavg_f1", "epochavg_f1macro",
    "last_balacc_std", "last_f1macro_std", "runtime_s", "log",
]


def discover(run_dir: Path):
    """Return [(epoch:int, path:str)] for stable epoch-*.ckpt files."""
    ck = run_dir / "checkpoints"
    found = []
    for p in sorted(ck.glob("epoch-*.ckpt")):
        m = re.search(r"epoch-(\d+)\.ckpt$", p.name)
        if m:
            found.append((int(m.group(1)), str(p.relative_to(PROJECT_ROOT))))
    return sorted(found)


def run_job(job, gpu, log_dir, args):
    task, epoch, ckpt = job
    name = f"{task}__e{epoch:04d}"
    log_path = log_dir / f"{name}.log"
    cmd = [
        "python", "train_task_brainco_angle.py",
        f"+experiment={TASKS[task]}",
        f"task.checkpoint_encoder={ckpt}",
        f"task.encoder_lr={args.backbone_lr}",
        f"seed={args.seed}", f"++split_seed={args.split_seed}",
        f"experiment_name=epochsweep_{name}",
        f"wandb.group=epochsweep_{task}",
        "--all_split", "--num_folds", str(args.num_folds),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XFORMERS_DISABLED": "TRUE"}
    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    runtime = time.time() - t0
    row = {f: "" for f in FIELDS}
    row.update(task=task, epoch=str(epoch), checkpoint=ckpt,
               runtime_s=f"{runtime:.0f}", log=str(log_path))
    row.update(parse_log(log_path.read_text(errors="ignore")))
    row["status"] = "OK" if (proc.returncode == 0 and row.get("last_balacc")) else "FAILED"
    print(f"[{row['status']}] gpu{gpu} {name}  balacc={row.get('last_balacc','-')} "
          f"f1macro={row.get('last_f1macro','-')} ({runtime/60:.1f}m)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone-lr", default="1e-4")
    ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--tasks", default=",".join(TASKS))
    args = ap.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    ckpts = discover(run_dir)
    if not ckpts:
        raise SystemExit(f"No epoch-*.ckpt under {run_dir}/checkpoints")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "scripts" / "logs" / f"epoch_sweep_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = PROJECT_ROOT / "scripts" / f"results_epoch_sweep_{stamp}.csv"
    md_path = PROJECT_ROOT / "scripts" / f"results_epoch_sweep_{stamp}.md"

    jobs = [(t, e, c) for t in tasks for e, c in ckpts]
    jobs.sort(key=lambda j: j[0] != "slip_detection")   # long jobs first
    print(f"{len(jobs)} jobs ({len(ckpts)} checkpoints x {len(tasks)} tasks) "
          f"over GPUs {gpus}\nepochs: {[e for e, _ in ckpts]}\nlogs -> {log_dir}", flush=True)

    pool: queue.Queue = queue.Queue()
    for g in gpus:
        pool.put(g)
    rows, lock = [], threading.Lock()

    def worker(job):
        gpu = pool.get()
        try:
            row = run_job(job, gpu, log_dir, args)
        finally:
            pool.put(gpu)
        with lock:
            rows.append(row)

    threads = []
    for job in jobs:
        th = threading.Thread(target=worker, args=(job,))
        th.start(); threads.append(th); time.sleep(2)
    for th in threads:
        th.join()

    rows.sort(key=lambda r: (r["task"], int(r["epoch"])))
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    write_md(rows, md_path, args, run_dir, tasks)
    print(f"\nCSV: {csv_path}\nMD : {md_path}", flush=True)


def write_md(rows, md_path, args, run_dir, tasks):
    def f(v):
        return v if v else "n/a"

    L = [
        "# Downstream performance vs. pretraining epoch", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}", "",
        "## Protocol", "",
        f"- Pretraining run: `{run_dir.relative_to(PROJECT_ROOT)}`",
        f"- ID protocol: episode-level {args.num_folds}-fold CV, seed {args.seed}, "
        f"split seed {args.split_seed}",
        f"- Backbone LR {args.backbone_lr}; probe LR 1e-4; batch 256",
        "- Metrics: balanced accuracy and macro F1 (last epoch of downstream training)", "",
    ]
    for task in tasks:
        trs = [r for r in rows if r["task"] == task]
        if not trs:
            continue
        L += [f"## {task}", "",
              "| Pretrain epoch | Bal Acc | ± (fold) | F1 macro | ± (fold) | F1 (bin) | Status |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for r in trs:
            L.append(f"| {r['epoch']} | {f(r['last_balacc'])} | {f(r['last_balacc_std'])} | "
                     f"{f(r['last_f1macro'])} | {f(r['last_f1macro_std'])} | "
                     f"{f(r['last_f1'])} | {r['status']} |")
        ok = [r for r in trs if r["status"] == "OK" and r["last_f1macro"]]
        if ok:
            b = max(ok, key=lambda r: float(r["last_f1macro"]))
            L += ["", f"Best: **epoch {b['epoch']}** (macro F1 {b['last_f1macro']}, "
                      f"bal acc {b['last_balacc']}).", ""]
    L += ["## Caveats", "",
          "- Single seed; `±` is fold-to-fold spread within each run.",
          "- One backbone LR only.", ""]
    md_path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
