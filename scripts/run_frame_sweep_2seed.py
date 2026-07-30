#!/usr/bin/env python
"""Frame-length sweep over 4 encoders x 2 seeds, for slip detection and grasp prediction.

slip_detection : window lengths 1, 5, 15, 30 (input_window_frames == stride)
grasp_prediction: window lengths 15, 30 (via data.window_time at 100 Hz)

`require_uniform_label` stays off, balanced sampling stays off, probe is
BraincoGraspRoPEProbe(depth 2). Writes a CSV and a Markdown summary reporting
both Last-epoch and EpochAvg macro F1.

Usage:
    python scripts/run_frame_sweep_2seed.py --gpus 5,6,7
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
B = "experiments"
ENCODERS = {
    "scratch": "null",
    "tip_s3": f"{B}/dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip/"
              f"2026.07.28-13-23/checkpoints/epoch-0100.ckpt",
    "tip_s1b2048": f"{B}/dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048/"
                   f"2026.07.28-14-48/checkpoints/epoch-0100.ckpt",
    "recon_tip_s1b2048": f"{B}/dinov2_recon_all_pseudo_force_tiny_rope_v2_hp1_tip_s1_b2048/"
                         f"2026.07.28-14-49/checkpoints/epoch-0100.ckpt",
}
FRAMES = {"slip_detection": [1, 5, 15, 30], "grasp_prediction": [15, 30]}
PROBE = [
    "task.model_task._target_=tactile_ssl.downstream_task.BraincoGraspRoPEProbe",
    "+task.model_task.depth=2",
    "+task.model_task.num_heads=3",
]
SUMMARY_RE = {
    tag: re.compile(
        rf"K-FOLD SUMMARY \({label}\).*?^\s*Mean\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)"
        rf".*?^\s*Std\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)",
        re.DOTALL | re.MULTILINE)
    for tag, label in [("last", "Last Epoch"), ("epochavg", "Epoch Average")]
}
FIELDS = ["task", "frames", "encoder", "seed", "status",
          "last_balacc", "last_f1macro", "last_f1macro_foldstd",
          "epochavg_balacc", "epochavg_f1macro", "epochavg_f1macro_foldstd",
          "windows", "runtime_s", "log"]


def parse_log(text):
    out = {}
    for tag, rx in SUMMARY_RE.items():
        m = rx.search(text)
        if m:
            ba, _f1, f1m, _bas, _f1s, f1ms = m.groups()
            out[f"{tag}_balacc"] = ba
            out[f"{tag}_f1macro"] = f1m
            out[f"{tag}_f1macro_foldstd"] = f1ms
    m = re.search(r"Total windows: (\d+) train", text)
    if m:
        out["windows"] = m.group(1)
    return out


def task_overrides(task, frames):
    if task == "slip_detection":
        # The grasp data config has no `balanced_sampling` key, so this override
        # only applies to slip; grasp already defaults to unbalanced sampling.
        return [f"data.dataset.config.input_window_frames={frames}",
                f"data.dataset.config.input_window_stride={frames}",
                "data.balanced_sampling=false"]
    # grasp: the dataset derives its window from window_time * interpolating_freq (100 Hz)
    return [f"data.window_time={frames / 100:.2f}"]


def run_job(job, gpu, log_dir, args):
    task, frames, encoder, seed = job
    name = f"{task}__w{frames:02d}__{encoder}__s{seed}"
    log_path = log_dir / f"{name}.log"
    cmd = [
        "python", "train_task_brainco_angle.py",
        f"+experiment=brainco/ours_3d/task/{task}/dinov2_all_rope",
        f"task.checkpoint_encoder={ENCODERS[encoder]}",
        "task.encoder_lr=1e-4", "task.task_lr=1e-4",
        *PROBE, *task_overrides(task, frames),
        f"seed={seed}", f"++split_seed={args.split_seed}",
        f"experiment_name=fsweep_{name}",
        "--all_split", "--num_folds", str(args.num_folds),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XFORMERS_DISABLED": "TRUE"}
    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=fh,
                              stderr=subprocess.STDOUT)
    runtime = time.time() - t0
    row = {f: "" for f in FIELDS}
    row.update(task=task, frames=str(frames), encoder=encoder, seed=str(seed),
               runtime_s=f"{runtime:.0f}", log=str(log_path))
    row.update(parse_log(log_path.read_text(errors="ignore")))
    row["status"] = "OK" if (proc.returncode == 0 and row.get("last_f1macro")) else "FAILED"
    print(f"[{row['status']}] gpu{gpu} {name}  last={row.get('last_f1macro','-')} "
          f"avg={row.get('epochavg_f1macro','-')} ({runtime/60:.1f}m)", flush=True)
    return row


def agg(rows, task, frames, enc, key):
    vals = [float(r[key]) for r in rows
            if r["task"] == task and r["frames"] == str(frames) and r["encoder"] == enc
            and r["status"] == "OK" and r[key] not in ("", "nan")]
    if not vals:
        return None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def write_md(rows, md_path, args):
    def cell(task, f, enc, key):
        a = agg(rows, task, f, enc, key)
        return "n/a" if a is None else f"{a[0]:.4f} ± {a[1]:.4f}"

    L = [
        "# Frame-length sweep — slip detection & grasp prediction",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
        f"{sum(r['status']=='OK' for r in rows)}/{len(rows)} runs OK.",
        "",
        "## Protocol",
        "",
        f"- Episode-level {args.num_folds}-fold CV, seeds {args.seeds}, split seed {args.split_seed}",
        "- Probe `BraincoGraspRoPEProbe` (depth 2, 3 heads); backbone LR 1e-4, probe LR 1e-4",
        "- **`balanced_sampling` off, `require_uniform_label` off** (windows may straddle a "
        "slip boundary; the label is the last frame's)",
        "- slip: `input_window_frames` = `input_window_stride` = the frame count; "
        "grasp: `window_time` = frames / 100 Hz",
        "- Encoders:",
        f"  - `tip_s3` — `{ENCODERS['tip_s3']}`",
        f"  - `tip_s1b2048` — `{ENCODERS['tip_s1b2048']}`",
        f"  - `recon_tip_s1b2048` — `{ENCODERS['recon_tip_s1b2048']}`",
        "- Metrics: macro F1 at the **last** downstream epoch and the **EpochAvg** over "
        "epochs 10/20/30/40/50. `±` is the seed-to-seed spread; per-run fold std is in the CSV.",
        "",
    ]
    for task in FRAMES:
        trs = [r for r in rows if r["task"] == task]
        if not trs:
            continue
        L += [f"## {task}", ""]
        for metric, key in (("Last epoch", "last_f1macro"), ("EpochAvg", "epochavg_f1macro")):
            L += [f"### {metric} — macro F1", "",
                  "| Encoder | " + " | ".join(f"{f} frames" for f in FRAMES[task]) + " |",
                  "| --- | " + " | ".join("---:" for _ in FRAMES[task]) + " |"]
            for enc in ENCODERS:
                L.append(f"| {enc} | " +
                         " | ".join(cell(task, f, enc, key) for f in FRAMES[task]) + " |")
            L.append("")
            # gap vs scratch
            L += ["Δ vs scratch:", "",
                  "| Encoder | " + " | ".join(f"{f} frames" for f in FRAMES[task]) + " |",
                  "| --- | " + " | ".join("---:" for _ in FRAMES[task]) + " |"]
            for enc in ENCODERS:
                if enc == "scratch":
                    continue
                cells = []
                for f in FRAMES[task]:
                    a, b = agg(rows, task, f, enc, key), agg(rows, task, f, "scratch", key)
                    cells.append("n/a" if not a or not b else f"{a[0]-b[0]:+.4f}")
                L.append(f"| {enc} | " + " | ".join(cells) + " |")
            L.append("")
    L += ["## Caveats", "",
          "- Two seeds only; `±` is the spread between them, not a confidence interval.",
          "- Window count falls sharply with frame length (slip: ~49k windows at 1 frame vs "
          "~1.7k at 30), so frame length and sample count move together.",
          "- With `require_uniform_label` off, the fraction of windows that straddle a slip "
          "boundary grows with the window: 3.0% at 3 frames, 17.1% at 15, 29.3% at 30. The "
          "label is taken from the last frame, so longer windows carry more label noise.",
          "- ID protocol only; no leave-one-object-out.", ""]
    md_path.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="5,6,7")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "scripts" / "logs" / f"frame_sweep_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = PROJECT_ROOT / "scripts" / f"results_frame_sweep_{stamp}.csv"
    md_path = PROJECT_ROOT / "scripts" / f"results_frame_sweep_{stamp}.md"

    jobs = [(t, f, e, s) for t in FRAMES for f in FRAMES[t] for e in ENCODERS for s in seeds]
    # Shortest window == fewest samples == fastest; run the long ones first.
    jobs.sort(key=lambda j: (j[0] != "slip_detection", j[1]))
    print(f"{len(jobs)} jobs over GPUs {gpus}; logs -> {log_dir}", flush=True)

    pool: queue.Queue = queue.Queue()
    for g in gpus:
        pool.put(g)
    rows, lock = [], threading.Lock()

    def worker(job):
        gpu = pool.get()
        try:
            row = run_job(job, gpu, log_dir, args)
        except Exception as exc:
            row = {f: "" for f in FIELDS}
            row.update(task=job[0], frames=str(job[1]), encoder=job[2], seed=str(job[3]),
                       status="FAILED", log=f"exception: {exc}")
            print(f"[FAILED] {job}: {exc}", flush=True)
        finally:
            pool.put(gpu)
        with lock:
            rows.append(row)
            with csv_path.open("w", newline="") as fh:      # checkpoint after each job
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)

    threads = []
    for job in jobs:
        th = threading.Thread(target=worker, args=(job,))
        th.start()
        threads.append(th)
        time.sleep(2)
    for th in threads:
        th.join()

    order = {j: i for i, j in enumerate(
        [(t, f, e, s) for t in FRAMES for f in FRAMES[t] for e in ENCODERS for s in seeds])}
    rows.sort(key=lambda r: order[(r["task"], int(r["frames"]), r["encoder"], int(r["seed"]))])
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_md(rows, md_path, args)
    print(f"\nCSV: {csv_path}\nMD : {md_path}", flush=True)


if __name__ == "__main__":
    main()
