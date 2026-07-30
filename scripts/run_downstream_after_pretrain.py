#!/usr/bin/env python
"""Wait for pretraining runs to finish, then evaluate their final checkpoints.

Blocks until every PID in --wait-pids has exited, resolves each experiment's
newest run directory and highest `epoch-*.ckpt`, then runs grasp prediction and
slip detection (ID, K-fold) for each checkpoint and writes a CSV + Markdown
summary.

Usage:
    python scripts/run_downstream_after_pretrain.py \\
        --experiments dinov2_prediction_local_all_pseudo_force_tiny_rope_v2_hp1,... \\
        --wait-pids 1731320,1731322,1731324 --gpus 4,5,6
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

# Same settings as results_seed_ood_20260727_185607.md, for direct comparison.
SCRATCH_REF = {"grasp_prediction": "0.8467 ± 0.0058", "slip_detection": "0.6463 ± 0.0052"}

FIELDS = [
    "experiment", "task", "epoch", "checkpoint", "status",
    "last_balacc", "last_f1", "last_f1macro",
    "best_balacc", "best_f1", "best_f1macro",
    "epochavg_balacc", "epochavg_f1", "epochavg_f1macro",
    "last_balacc_std", "last_f1macro_std", "runtime_s", "log",
]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for(pids, poll=60):
    if not pids:
        return
    print(f"Waiting for pretraining PIDs {pids} ...", flush=True)
    while True:
        running = [p for p in pids if alive(p)]
        if not running:
            print("All pretraining runs finished.", flush=True)
            return
        print(f"  [{datetime.now():%H:%M}] still running: {running}", flush=True)
        time.sleep(poll)


def latest_checkpoint(experiment: str):
    """Newest run dir for `experiment`, and its highest-numbered epoch ckpt."""
    root = PROJECT_ROOT / "experiments" / experiment
    runs = sorted((d for d in root.iterdir() if (d / "checkpoints").is_dir()),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    for run in runs:
        cks = []
        for p in (run / "checkpoints").glob("epoch-*.ckpt"):
            m = re.search(r"epoch-(\d+)\.ckpt$", p.name)
            if m:
                cks.append((int(m.group(1)), p))
        if cks:
            epoch, path = max(cks)
            return epoch, str(path.relative_to(PROJECT_ROOT))
    return None, None


def run_job(job, gpu, log_dir, args):
    experiment, task, epoch, ckpt = job
    short = experiment.replace("dinov2_", "").replace("_all_pseudo_force_tiny", "")
    name = f"{short}__{task}"
    log_path = log_dir / f"{name}.log"
    cmd = [
        "python", "train_task_brainco_angle.py",
        f"+experiment={TASKS[task]}",
        f"task.checkpoint_encoder={ckpt}",
        f"task.encoder_lr={args.backbone_lr}",
        f"seed={args.seed}", f"++split_seed={args.split_seed}",
        f"experiment_name=afterpre_{name}",
        f"wandb.group=afterpre_{task}",
        "--all_split", "--num_folds", str(args.num_folds),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XFORMERS_DISABLED": "TRUE"}
    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    runtime = time.time() - t0
    row = {f: "" for f in FIELDS}
    row.update(experiment=experiment, task=task, epoch=str(epoch), checkpoint=ckpt,
               runtime_s=f"{runtime:.0f}", log=str(log_path))
    row.update(parse_log(log_path.read_text(errors="ignore")))
    row["status"] = "OK" if (proc.returncode == 0 and row.get("last_balacc")) else "FAILED"
    print(f"[{row['status']}] gpu{gpu} {name}  balacc={row.get('last_balacc','-')} "
          f"f1macro={row.get('last_f1macro','-')} ({runtime/60:.1f}m)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", required=True, help="comma-separated experiment_name values")
    ap.add_argument("--wait-pids", default="", help="comma-separated PIDs to wait on")
    ap.add_argument("--gpus", default="4,5,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone-lr", default="1e-4")
    ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--tasks", default=",".join(TASKS))
    args = ap.parse_args()

    pids = [int(p) for p in args.wait_pids.split(",") if p.strip()]
    wait_for(pids)

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    resolved, missing = [], []
    for e in experiments:
        epoch, ck = latest_checkpoint(e)
        if ck is None:
            missing.append(e)
        else:
            resolved.append((e, epoch, ck))
            print(f"  {e}: epoch {epoch} -> {ck}", flush=True)
    for e in missing:
        print(f"  [WARN] no epoch-*.ckpt found for {e}; skipping", flush=True)
    if not resolved:
        raise SystemExit("No checkpoints resolved; nothing to evaluate.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "scripts" / "logs" / f"after_pretrain_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = PROJECT_ROOT / "scripts" / f"results_after_pretrain_{stamp}.csv"
    md_path = PROJECT_ROOT / "scripts" / f"results_after_pretrain_{stamp}.md"

    jobs = [(e, t, ep, ck) for e, ep, ck in resolved for t in tasks]
    jobs.sort(key=lambda j: j[1] != "slip_detection")   # long jobs first
    print(f"\n{len(jobs)} downstream jobs over GPUs {gpus}; logs -> {log_dir}", flush=True)

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

    rows.sort(key=lambda r: (r["task"], r["experiment"]))
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    write_md(rows, md_path, args, tasks, missing)
    print(f"\nCSV: {csv_path}\nMD : {md_path}", flush=True)


def write_md(rows, md_path, args, tasks, missing):
    def f(v):
        return v if v else "n/a"

    L = ["# Downstream results for the hp1 pretraining variants", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M}", "",
         "## Protocol", "",
         f"- ID protocol: episode-level {args.num_folds}-fold CV, seed {args.seed}, "
         f"split seed {args.split_seed}",
         f"- Backbone LR {args.backbone_lr}; probe LR 1e-4; batch 256",
         "- Final `epoch-*.ckpt` of each pretraining run",
         "- Metrics: balanced accuracy and macro F1 (last downstream epoch); "
         "`±` is fold-to-fold spread", ""]
    for task in tasks:
        trs = [r for r in rows if r["task"] == task]
        if not trs:
            continue
        L += [f"## {task}", "",
              f"Scratch reference (3 seeds, same settings): **{SCRATCH_REF.get(task,'n/a')}** macro F1.", "",
              "| Pretraining variant | Epoch | Bal Acc | ± | F1 macro | ± | F1 (bin) | Status |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for r in trs:
            short = r["experiment"].replace("dinov2_", "").replace("_all_pseudo_force_tiny", "")
            L.append(f"| {short} | {r['epoch']} | {f(r['last_balacc'])} | {f(r['last_balacc_std'])} | "
                     f"{f(r['last_f1macro'])} | {f(r['last_f1macro_std'])} | {f(r['last_f1'])} | "
                     f"{r['status']} |")
        ok = [r for r in trs if r["status"] == "OK" and r["last_f1macro"]]
        if ok:
            b = max(ok, key=lambda r: float(r["last_f1macro"]))
            L += ["", f"Best: **{b['experiment']}** (macro F1 {b['last_f1macro']}, "
                      f"bal acc {b['last_balacc']}).", ""]
    if missing:
        L += ["## Skipped", ""] + [f"- `{e}` — no `epoch-*.ckpt` found" for e in missing] + [""]
    L += ["## Caveats", "",
          "- Single seed; `±` is fold-to-fold spread within each run.",
          "- One backbone LR (1e-4); ID protocol only (no OOD).",
          "- hp1 changes several hyperparameters at once, so a win here does not "
          "attribute to any single one.", ""]
    md_path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
