#!/usr/bin/env python
"""Backbone-LR x checkpoint matrix for grasp prediction and slip detection.

Runs every (task, encoder init, backbone LR) combination as its own 4-fold
process, spreading jobs over the GPUs given in --gpus, then writes a CSV and a
Markdown summary.

Usage:
    python scripts/run_lr_checkpoint_matrix.py --gpus 4,5,6,7
"""
from __future__ import annotations

import argparse
import csv
import os
import queue
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = "checkpoints/dinov2_xyz_temp"

TASKS = {
    "grasp_prediction": "brainco/ours_3d/task/grasp_prediction/dinov2_all_rope",
    "slip_detection": "brainco/ours_3d/task/slip_detection/dinov2_all_rope",
}
ENCODERS = {
    "scratch": "null",
    "local_e80": f"{CKPT_DIR}/epoch-0080-local.ckpt",
    "base_e100": f"{CKPT_DIR}/epoch-0100-base.ckpt",
}
BACKBONE_LRS = ["1e-4", "1e-5", "1e-6"]

# "Mean" row of each K-FOLD SUMMARY block: Bal Acc, F1, F1 macro.
SUMMARY_RE = {
    tag: re.compile(
        rf"K-FOLD SUMMARY \({label}\).*?^\s*Mean\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)"
        rf".*?^\s*Std\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)",
        re.DOTALL | re.MULTILINE,
    )
    for tag, label in [("last", "Last Epoch"), ("best", "Best Epoch"), ("epoch_avg", "Epoch Average")]
}

FIELDS = [
    "task", "encoder", "backbone_lr", "status",
    "last_balacc", "last_f1", "last_f1macro",
    "best_balacc", "best_f1", "best_f1macro",
    "epochavg_balacc", "epochavg_f1", "epochavg_f1macro",
    "last_balacc_std", "last_f1macro_std",
    "runtime_s", "log",
]


def parse_log(text: str) -> dict:
    out = {}
    for tag, rx in SUMMARY_RE.items():
        m = rx.search(text)
        if not m:
            continue
        acc, f1, f1m, acc_s, _f1_s, f1m_s = m.groups()
        key = {"last": "last", "best": "best", "epoch_avg": "epochavg"}[tag]
        out[f"{key}_balacc"] = acc
        out[f"{key}_f1"] = f1
        out[f"{key}_f1macro"] = f1m
        if tag == "last":
            out["last_balacc_std"] = acc_s
            out["last_f1macro_std"] = f1m_s
    return out


def run_job(job, gpu, log_dir, seed, num_folds, split_seed, extra):
    task, encoder, lr = job
    name = f"{task}__{encoder}__blr{lr}"
    log_path = log_dir / f"{name}.log"
    cmd = [
        "python", "train_task_brainco_angle.py",
        f"+experiment={TASKS[task]}",
        f"task.checkpoint_encoder={ENCODERS[encoder]}",
        f"task.encoder_lr={lr}",
        f"seed={seed}", f"++split_seed={split_seed}",
        f"experiment_name=lrmatrix_{name}",
        f"wandb.group=lrmatrix_{task}",
        *extra,
        "--all_split", "--num_folds", str(num_folds),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XFORMERS_DISABLED": "TRUE"}
    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=fh,
                              stderr=subprocess.STDOUT)
    runtime = time.time() - t0
    text = log_path.read_text(errors="ignore")
    row = {f: "" for f in FIELDS}
    row.update(task=task, encoder=encoder, backbone_lr=lr,
               runtime_s=f"{runtime:.0f}", log=str(log_path))
    row.update(parse_log(text))
    row["status"] = "OK" if (proc.returncode == 0 and row.get("last_balacc")) else "FAILED"
    print(f"[{row['status']}] gpu{gpu} {name}  "
          f"balacc={row.get('last_balacc','-')} f1macro={row.get('last_f1macro','-')} "
          f"({runtime/60:.1f} min)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("extra", nargs="*", help="extra hydra overrides")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "scripts" / "logs" / f"lr_matrix_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = PROJECT_ROOT / "scripts" / f"results_lr_matrix_{stamp}.csv"
    md_path = PROJECT_ROOT / "scripts" / f"results_lr_matrix_{stamp}.md"

    # Slip runs are the long pole; start them first so they overlap the short ones.
    jobs = [(t, e, lr) for t in tasks for e in ENCODERS for lr in BACKBONE_LRS]
    jobs.sort(key=lambda j: 0 if j[0] == "slip_detection" else 1)
    print(f"{len(jobs)} jobs over GPUs {gpus}; logs -> {log_dir}", flush=True)

    gpu_pool: queue.Queue = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)
    rows, lock = [], threading.Lock()

    def worker(job):
        gpu = gpu_pool.get()
        try:
            row = run_job(job, gpu, log_dir, args.seed, args.num_folds,
                          args.split_seed, args.extra)
        finally:
            gpu_pool.put(gpu)
        with lock:
            rows.append(row)

    threads = []
    for job in jobs:
        th = threading.Thread(target=worker, args=(job,))
        th.start()
        threads.append(th)
        time.sleep(2)          # stagger so log dirs/timestamps do not collide
    for th in threads:
        th.join()

    order = {(t, e, lr): i for i, (t, e, lr) in enumerate(
        [(t, e, lr) for t in tasks for e in ENCODERS for lr in BACKBONE_LRS])}
    rows.sort(key=lambda r: order[(r["task"], r["encoder"], r["backbone_lr"])])

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_md(rows, md_path, args, tasks)
    print(f"\nCSV: {csv_path}\nMD : {md_path}", flush=True)


def write_md(rows, md_path, args, tasks):
    def fmt(v):
        return v if v else "n/a"

    L = [
        "# Backbone LR x Checkpoint Matrix",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Protocol",
        "",
        f"- Episode-level {args.num_folds}-fold CV, seed {args.seed}, split seed {args.split_seed}",
        "- Probe LR fixed at 1e-4; only the backbone (encoder) LR varies",
        "- Metrics are **balanced accuracy** (mean per-class recall) and **macro F1**",
        "- `F1` is the binary/positive-class F1 kept for continuity with earlier reports",
        "- Encoders: `scratch` (random init), "
        f"`local_e80` (`{ENCODERS['local_e80']}`), `base_e100` (`{ENCODERS['base_e100']}`)",
        "",
    ]
    for task in tasks:
        trs = [r for r in rows if r["task"] == task]
        if not trs:
            continue
        L += [f"## {task}", "",
              "| Encoder | Backbone LR | Bal Acc (last) | F1 macro (last) | Bal Acc (best) | "
              "F1 macro (best) | Bal Acc (epoch avg) | F1 macro (epoch avg) | Status |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for r in trs:
            L.append(
                f"| {r['encoder']} | {r['backbone_lr']} | {fmt(r['last_balacc'])} | "
                f"{fmt(r['last_f1macro'])} | {fmt(r['best_balacc'])} | {fmt(r['best_f1macro'])} | "
                f"{fmt(r['epochavg_balacc'])} | {fmt(r['epochavg_f1macro'])} | {r['status']} |")
        ok = [r for r in trs if r["status"] == "OK" and r["last_balacc"]]
        if ok:
            best = max(ok, key=lambda r: float(r["last_f1macro"] or 0))
            L += ["", f"Best by last-epoch macro F1: **{best['encoder']} @ backbone LR "
                      f"{best['backbone_lr']}** (macro F1 {best['last_f1macro']}, "
                      f"bal acc {best['last_balacc']}).", ""]
    L += ["## Notes", "",
          "- `Bal Acc (last)` std across folds is in the CSV (`last_balacc_std`).",
          "- Single seed: the fold-to-fold std is the only spread reported here.",
          ""]
    md_path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
