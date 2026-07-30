#!/usr/bin/env python
"""scratch vs `rope` pretraining with BraincoGraspRoPEProbe, over 3 seeds.

Grasp prediction keeps its 30-frame window; slip detection uses 3-frame windows.
Each (task, encoder, seed) is one 4-fold process; jobs are spread over --gpus.

Usage:
    python scripts/run_ropeprobe_seed_matrix.py --gpus 5,6,7
"""
from __future__ import annotations

import argparse
import csv
import os
import queue
import re
import statistics as st
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROPE_CKPT = (
    "experiments/dinov2_all_pseudo_force_tiny_rope_v2_hp1/"
    "2026.07.27-22-14/checkpoints/epoch-0100.ckpt"
)

TASKS = {
    "grasp_prediction": {
        "experiment": "brainco/ours_3d/task/grasp_prediction/dinov2_all_rope",
        "extra": [],                       # 30-frame window from the data config
    },
    "slip_detection": {
        "experiment": "brainco/ours_3d/task/slip_detection/dinov2_all_rope",
        "extra": [
            "data.dataset.config.input_window_frames=3",
            "data.dataset.config.input_window_stride=3",
        ],
    },
}
ENCODERS = {"scratch": "null", "rope": ROPE_CKPT}
PROBE = [
    "task.model_task._target_=tactile_ssl.downstream_task.BraincoGraspRoPEProbe",
    "+task.model_task.depth=2",
    "+task.model_task.num_heads=3",
]

SUMMARY_RE = {
    tag: re.compile(
        rf"K-FOLD SUMMARY \({label}\).*?^\s*Mean\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)"
        rf".*?^\s*Std\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)",
        re.DOTALL | re.MULTILINE,
    )
    for tag, label in [("last", "Last Epoch"), ("best", "Best Epoch"),
                       ("epoch_avg", "Epoch Average")]
}
FIELDS = [
    "task", "encoder", "seed", "status",
    "last_balacc", "last_f1", "last_f1macro",
    "best_balacc", "best_f1", "best_f1macro",
    "epochavg_balacc", "epochavg_f1", "epochavg_f1macro",
    "last_balacc_foldstd", "last_f1macro_foldstd",
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
        out[f"{key}_balacc"], out[f"{key}_f1"], out[f"{key}_f1macro"] = acc, f1, f1m
        if tag == "last":
            out["last_balacc_foldstd"], out["last_f1macro_foldstd"] = acc_s, f1m_s
    return out


def run_job(job, gpu, log_dir, args):
    task, encoder, seed = job
    name = f"{task}__{encoder}__seed{seed}"
    log_path = log_dir / f"{name}.log"
    cmd = [
        "python", "train_task_brainco_angle.py",
        f"+experiment={TASKS[task]['experiment']}",
        f"task.checkpoint_encoder={ENCODERS[encoder]}",
        f"task.encoder_lr={args.backbone_lr}",
        f"task.task_lr={args.task_lr}",
        *PROBE,
        *TASKS[task]["extra"],
        f"seed={seed}", f"++split_seed={args.split_seed}",
        f"experiment_name=ropeprobe_{name}",
        f"wandb.group=ropeprobe_{task}",
        "--all_split", "--num_folds", str(args.num_folds),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XFORMERS_DISABLED": "TRUE"}
    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=fh,
                              stderr=subprocess.STDOUT)
    runtime = time.time() - t0
    row = {f: "" for f in FIELDS}
    row.update(task=task, encoder=encoder, seed=str(seed),
               runtime_s=f"{runtime:.0f}", log=str(log_path))
    row.update(parse_log(log_path.read_text(errors="ignore")))
    row["status"] = "OK" if (proc.returncode == 0 and row.get("last_balacc")) else "FAILED"
    print(f"[{row['status']}] gpu{gpu} {name}  balacc={row.get('last_balacc','-')} "
          f"f1macro={row.get('last_f1macro','-')} ({runtime/60:.1f}m)", flush=True)
    return row


def agg(rows, task, encoder, key):
    vals = [float(r[key]) for r in rows
            if r["task"] == task and r["encoder"] == encoder
            and r["status"] == "OK" and r[key] not in ("", "nan")]
    if not vals:
        return None, None, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def write_md(rows, md_path, args):
    def cell(task, enc, key):
        m, s, n = agg(rows, task, enc, key)
        return "n/a" if m is None else f"{m:.4f} ± {s:.4f}"

    L = [
        "# BraincoGraspRoPEProbe: scratch vs `rope` pretraining, 3 seeds",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
        f"{sum(r['status']=='OK' for r in rows)}/{len(rows)} runs OK.",
        "",
        "## Protocol",
        "",
        f"- Probe: `BraincoGraspRoPEProbe`, depth 2, 3 heads (causal RoPE attention, "
        "classifies from the last time step) — replaces `MeanPoolProbe`",
        f"- Episode-level {args.num_folds}-fold CV, seeds {args.seeds}, split seed {args.split_seed}",
        f"- Backbone LR {args.backbone_lr}, probe LR {args.task_lr}, 50 epochs, train batch 256",
        "- Encoders: `scratch` (random init) vs `rope` "
        f"(`{ROPE_CKPT}`)",
        "- Grasp prediction: 30-frame windows (data-config default). "
        "Slip detection: 3-frame windows.",
        "- Metrics: **balanced accuracy** (mean per-class recall) and **macro F1**, last epoch.",
        "- `±` in the summary tables is the **seed-to-seed** std; per-run fold std is in the CSV.",
        "",
    ]
    for task in TASKS:
        L += [f"## {task}", "",
              "| Encoder | Bal Acc | Macro F1 | F1 (bin) | n seeds |",
              "| --- | ---: | ---: | ---: | ---: |"]
        for enc in ENCODERS:
            _, _, n = agg(rows, task, enc, "last_balacc")
            L.append(f"| {enc} | {cell(task, enc, 'last_balacc')} | "
                     f"{cell(task, enc, 'last_f1macro')} | {cell(task, enc, 'last_f1')} | {n} |")
        gm, _, _ = agg(rows, task, "rope", "last_f1macro")
        sm, ss, _ = agg(rows, task, "scratch", "last_f1macro")
        if gm is not None and sm is not None:
            d = gm - sm
            verdict = ("larger than" if abs(d) > ss else "within") + " the scratch seed spread"
            L += ["", f"Δ macro F1 (`rope` − scratch): **{d:+.4f}** — {verdict} "
                      f"(±{ss:.4f}).", ""]
        L += ["### Per-seed (last epoch)", "",
              "| Encoder | Seed | Bal Acc | Macro F1 | fold std (macro F1) | Status |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
        for enc in ENCODERS:
            for r in [r for r in rows if r["task"] == task and r["encoder"] == enc]:
                L.append(f"| {enc} | {r['seed']} | {r['last_balacc'] or 'n/a'} | "
                         f"{r['last_f1macro'] or 'n/a'} | "
                         f"{r['last_f1macro_foldstd'] or 'n/a'} | {r['status']} |")
        L.append("")
    L += ["## Caveats", "",
          "- One backbone LR and one probe LR; neither was swept for the larger probe "
          "(890k params vs 386 for `MeanPoolProbe`).",
          "- ID protocol only (episode-level K-fold over all objects); no leave-one-object-out.",
          "- Only the `rope` checkpoint is compared; `temp3` and `prediction_local` are not in "
          "this run.",
          ""]
    md_path.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="5,6,7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--backbone-lr", default="1e-4")
    ap.add_argument("--task-lr", default="1e-4")
    ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "scripts" / "logs" / f"ropeprobe_seeds_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = PROJECT_ROOT / "scripts" / f"results_ropeprobe_seeds_{stamp}.csv"
    md_path = PROJECT_ROOT / "scripts" / f"results_ropeprobe_seeds_{stamp}.md"

    jobs = [(t, e, s) for t in TASKS for e in ENCODERS for s in seeds]
    # Slip runs are ~3x longer; start them first so the pool drains evenly.
    jobs.sort(key=lambda j: 0 if j[0] == "slip_detection" else 1)
    print(f"{len(jobs)} jobs over GPUs {gpus}; logs -> {log_dir}", flush=True)

    pool: queue.Queue = queue.Queue()
    for g in gpus:
        pool.put(g)
    rows, lock = [], threading.Lock()

    def worker(job):
        gpu = pool.get()
        try:
            row = run_job(job, gpu, log_dir, args)
        except Exception as exc:                       # keep the sweep alive
            row = {f: "" for f in FIELDS}
            row.update(task=job[0], encoder=job[1], seed=str(job[2]),
                       status="FAILED", log=f"exception: {exc}")
            print(f"[FAILED] {job} raised {exc}", flush=True)
        finally:
            pool.put(gpu)
        with lock:
            rows.append(row)

    threads = []
    for job in jobs:
        th = threading.Thread(target=worker, args=(job,))
        th.start()
        threads.append(th)
        time.sleep(2)
    for th in threads:
        th.join()

    order = {(t, e, s): i for i, (t, e, s) in enumerate(
        [(t, e, s) for t in TASKS for e in ENCODERS for s in seeds])}
    rows.sort(key=lambda r: order[(r["task"], r["encoder"], int(r["seed"]))])

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_md(rows, md_path, args)
    print(f"\nCSV: {csv_path}\nMD : {md_path}", flush=True)


if __name__ == "__main__":
    main()
