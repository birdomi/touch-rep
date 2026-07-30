#!/usr/bin/env python
"""Multi-seed ID + leave-one-object-out (OOD) runs at a fixed backbone LR.

ID  : episode-level K-fold over all objects (`--all_split`).
OOD : one run per held-out object; the remaining objects train, the held-out
      one is the entire validation set.

Usage:
    python scripts/run_seed_ood_matrix.py --gpus 4,5,6,7
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

ENCODERS = {
    "scratch": "null",
    "local_e80": f"{CKPT_DIR}/epoch-0080-local.ckpt",
}

GRASP_OBJECTS = ["box", "tumbler", "eraser", "driver", "case"]
SLIP_OBJECTS = ["slip_box", "slip_doll", "slip_evacase", "slip_plastic"]

TASKS = {
    "grasp_prediction": {
        "experiment": "brainco/ours_3d/task/grasp_prediction/dinov2_all_rope",
        "objects": GRASP_OBJECTS,
        "target": "tactile_ssl.data.brainco_xyz_grasp_dataset.BraincoXYZGraspDataset",
        "data_path": "${paths.data_root}/brainco/downstream/grasp_prediction_0611",
    },
    "slip_detection": {
        "experiment": "brainco/ours_3d/task/slip_detection/dinov2_all_rope",
        "objects": SLIP_OBJECTS,
        "target": "tactile_ssl.data.brainco_xyz_slip_detection_dataset.BraincoXYZSlipDetectionDataset",
        "data_path": "${paths.data_root}/brainco/downstream/slip_data_v2",
    },
}

# K-fold summary (ID runs): Mean row -> BalAcc, F1, F1 macro.
KFOLD_RE = {
    tag: re.compile(
        rf"K-FOLD SUMMARY \({label}\).*?^\s*Mean\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)"
        rf".*?^\s*Std\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)",
        re.DOTALL | re.MULTILINE,
    )
    for tag, label in [("last", "Last Epoch"), ("best", "Best Epoch"),
                       ("epoch_avg", "Epoch Average")]
}
# Single-split summary (OOD runs).
SINGLE_RE = {
    tag: re.compile(
        rf"{label}\s+— BalAcc:\s+([\d.]+|nan)\s+F1:\s+([\d.]+|nan)\s+F1macro:\s+([\d.]+|nan)")
    for tag, label in [("last", "Last"), ("best", "Best"), ("epoch_avg", "EpochAvg")]
}

FIELDS = [
    "protocol", "task", "encoder", "seed", "held_out", "status",
    "last_balacc", "last_f1", "last_f1macro",
    "best_balacc", "best_f1", "best_f1macro",
    "epochavg_balacc", "epochavg_f1", "epochavg_f1macro",
    "last_balacc_std", "last_f1macro_std", "runtime_s", "log",
]


def parse_log(text: str, protocol: str) -> dict:
    out, table = {}, KFOLD_RE if protocol == "id" else SINGLE_RE
    for tag, rx in table.items():
        m = rx.search(text)
        if not m:
            continue
        key = {"last": "last", "best": "best", "epoch_avg": "epochavg"}[tag]
        out[f"{key}_balacc"], out[f"{key}_f1"], out[f"{key}_f1macro"] = m.group(1), m.group(2), m.group(3)
        if protocol == "id" and tag == "last":
            out["last_balacc_std"], out["last_f1macro_std"] = m.group(4), m.group(6)
    return out


def grasp_label_dirs(objects):
    return ",".join(f"{o}_succ:1,{o}_fail:0" for o in objects)


def ood_overrides(task: str, held_out: str):
    """Train on every object except held_out; validate on held_out only."""
    spec = TASKS[task]
    train_objs = [o for o in spec["objects"] if o != held_out]
    dp, tgt = spec["data_path"], spec["target"]
    common = ("window_time:${data.window_time},window_overlap:${data.window_overlap},"
              "interpolating_freq:${data.interpolating_freq},"
              "subtract_baseline:${data.dataset.config.subtract_baseline},"
              "align_xyz_to_npz:${data.dataset.config.align_xyz_to_npz}")
    if task == "grasp_prediction":
        train = (f"data.dataset.config.data_roots="
                 f"[{{data_path:{dp},label_dirs:{{{grasp_label_dirs(train_objs)}}}}}]")
        val = (f"data.val_dataset={{_target_:{tgt},config:{{{common},"
               f"data_roots:[{{data_path:{dp},label_dirs:{{{grasp_label_dirs([held_out])}}}}}]}},"
               f"data_path:{dp},brainco_urdf_path:${{data.dataset.brainco_urdf_path}}}}")
    else:
        win = ("input_window_frames:${data.dataset.config.input_window_frames},"
               "input_window_stride:${data.dataset.config.input_window_stride},"
               "exclude_before_slip_start_frames:${data.dataset.config.exclude_before_slip_start_frames},"
               "exclude_after_slip_end_frames:${data.dataset.config.exclude_after_slip_end_frames}")
        train = f"data.dataset.config.classes=[{','.join(train_objs)}]"
        val = (f"data.val_dataset={{_target_:{tgt},config:{{{common},{win},"
               f"classes:[{held_out}]}},data_path:{dp},"
               f"brainco_urdf_path:${{data.dataset.brainco_urdf_path}}}}")
    return [train, val]


def run_job(job, gpu, log_dir, args):
    protocol, task, encoder, seed, held_out = job
    name = (f"{protocol}__{task}__{encoder}__seed{seed}"
            + (f"__ho_{held_out}" if held_out else ""))
    log_path = log_dir / f"{name}.log"
    cmd = [
        "python", "train_task_brainco_angle.py",
        f"+experiment={TASKS[task]['experiment']}",
        f"task.checkpoint_encoder={ENCODERS[encoder]}",
        f"task.encoder_lr={args.backbone_lr}",
        f"seed={seed}", f"++split_seed={args.split_seed}",
        f"experiment_name=seedood_{name}",
        f"wandb.group=seedood_{protocol}_{task}",
    ]
    if protocol == "ood":
        cmd += ood_overrides(task, held_out) + ["all_split=false"]
    else:
        cmd += ["--all_split", "--num_folds", str(args.num_folds)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XFORMERS_DISABLED": "TRUE"}
    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    runtime = time.time() - t0
    row = {f: "" for f in FIELDS}
    row.update(protocol=protocol, task=task, encoder=encoder, seed=str(seed),
               held_out=held_out or "", runtime_s=f"{runtime:.0f}", log=str(log_path))
    row.update(parse_log(log_path.read_text(errors="ignore"), protocol))
    row["status"] = "OK" if (proc.returncode == 0 and row.get("last_balacc")) else "FAILED"
    print(f"[{row['status']}] gpu{gpu} {name}  balacc={row.get('last_balacc','-')} "
          f"f1macro={row.get('last_f1macro','-')} ({runtime/60:.1f}m)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--encoders", default=",".join(ENCODERS))
    ap.add_argument("--backbone-lr", default="1e-4")
    ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--protocols", default="id,ood")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "scripts" / "logs" / f"seed_ood_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = PROJECT_ROOT / "scripts" / f"results_seed_ood_{stamp}.csv"
    md_path = PROJECT_ROOT / "scripts" / f"results_seed_ood_{stamp}.md"

    jobs = []
    for task in TASKS:
        for enc in encoders:
            for seed in seeds:
                if "id" in protocols:
                    jobs.append(("id", task, enc, seed, None))
                if "ood" in protocols:
                    for obj in TASKS[task]["objects"]:
                        jobs.append(("ood", task, enc, seed, obj))
    # Long jobs first so the tail is short ones.
    jobs.sort(key=lambda j: (j[0] != "id", j[1] != "slip_detection"))
    print(f"{len(jobs)} jobs over GPUs {gpus}; logs -> {log_dir}", flush=True)

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

    rows.sort(key=lambda r: (r["protocol"], r["task"], r["encoder"], r["seed"], r["held_out"]))
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    write_md(rows, md_path, args, seeds, encoders)
    print(f"\nCSV: {csv_path}\nMD : {md_path}", flush=True)


def agg(vals):
    import statistics
    v = [float(x) for x in vals if x not in ("", "nan", None)]
    if not v:
        return "n/a"
    return f"{statistics.mean(v):.4f} ± {(statistics.stdev(v) if len(v) > 1 else 0.0):.4f}"


def write_md(rows, md_path, args, seeds, encoders):
    L = [
        "# Multi-seed ID + OOD (leave-one-object-out)", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}", "",
        "## Protocol", "",
        f"- Seeds {seeds}; backbone LR fixed at **{args.backbone_lr}**; probe LR 1e-4",
        f"- ID: episode-level {args.num_folds}-fold CV over all objects (split seed {args.split_seed})",
        "- OOD: leave-one-object-out — train on the remaining objects, validate on the held-out one",
        f"- Encoders: {', '.join(f'`{e}`' for e in encoders)}"
        f" (`local_e80` = `{ENCODERS.get('local_e80','')}`)",
        "- Metrics: balanced accuracy and macro F1. `±` is across seeds.", "",
    ]
    for protocol in ("id", "ood"):
        prs = [r for r in rows if r["protocol"] == protocol and r["status"] == "OK"]
        if not prs:
            continue
        L += [f"## {protocol.upper()}", "",
              "| Task | Encoder | Bal Acc | F1 macro | F1 (bin) | n runs |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
        for task in sorted({r["task"] for r in prs}):
            for enc in encoders:
                sel = [r for r in prs if r["task"] == task and r["encoder"] == enc]
                if not sel:
                    continue
                L.append(f"| {task} | {enc} | {agg([r['last_balacc'] for r in sel])} | "
                         f"{agg([r['last_f1macro'] for r in sel])} | "
                         f"{agg([r['last_f1'] for r in sel])} | {len(sel)} |")
        L.append("")
        if protocol == "ood":
            L += ["### OOD per held-out object (macro F1, mean ± std over seeds)", "",
                  "| Task | Held-out | " + " | ".join(encoders) + " |",
                  "| --- | --- | " + " | ".join("---:" for _ in encoders) + " |"]
            for task in sorted({r["task"] for r in prs}):
                for obj in TASKS[task]["objects"]:
                    cells = []
                    for enc in encoders:
                        sel = [r for r in prs if r["task"] == task and r["encoder"] == enc
                               and r["held_out"] == obj]
                        cells.append(agg([r["last_f1macro"] for r in sel]))
                    L.append(f"| {task} | {obj} | " + " | ".join(cells) + " |")
            L.append("")
    failed = [r for r in rows if r["status"] != "OK"]
    if failed:
        L += ["## Failed runs", ""] + [f"- {r['protocol']}/{r['task']}/{r['encoder']}/"
                                       f"seed{r['seed']}/{r['held_out']} — `{r['log']}`" for r in failed] + [""]
    L += ["## Caveats", "",
          "- `±` is the spread across seeds (per-fold spread for ID runs is in the CSV).",
          "- OOD validates on a single held-out object, so each cell is one split, not a K-fold mean.",
          "- Batch 256, `max_values = [1000, 1000, 1000, 100000]`, train-split signal statistics.", ""]
    md_path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
