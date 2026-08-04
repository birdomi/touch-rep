#!/usr/bin/env python3
"""Second-stage driver: waits for orchestrate_xa.py, then finishes the study.

Three steps, all unattended:
  1. wait for the orchestrator's state to reach every job
  2. evaluate the two frame-wise (window-1) checkpoints, which are not part of
     the 8-row table and so were never queued there
  3. aggregate everything into scripts/results_xa_final_<date>.md -- the HOI
     data-scaling curve and the 8-row table with pretraining-seed variance

Launch detached in tmux; it does not need the interactive session.
"""

import json
import os
import re
import shutil
import statistics as st
import subprocess
import time
from collections import defaultdict
from pathlib import Path

REPO = Path("/raid/ygyu/workspace/touch-rep")
BASE = Path("/tmp/claude-1203/-raid-ygyu-workspace-touch-rep/"
            "4d80c27e-1b0b-464c-ab82-54ceeb31b81a/scratchpad/orch")
FIN = BASE.parent / "finish"
PY = "/raid/ygyu/miniconda3/envs/tactile/bin/python"
SNAP = REPO / "checkpoints/queued_w15_snapshots"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
READOUTS = ["Epoch Average", "Last Epoch", "Best Epoch"]
TOTAL_JOBS = 160

FRAME_W1 = [
    ("framew1_tip", "xyznorm_frame_w1_tip"),
    ("framew1_42j", "xyznorm_frame_w1_42j"),
]
GRASP_FW = "brainco/ours_3d/task/xyznorm/grasp_framew1"
SLIP_FW = "brainco/ours_3d/task/xyznorm/slip_framew1_v3"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    FIN.mkdir(parents=True, exist_ok=True)
    with open(FIN / "finish.log", "a") as f:
        f.write(line + "\n")


def wait_for_orchestrator():
    log("waiting for orchestrate_xa.py to drain")
    while True:
        try:
            state = json.loads((BASE / "state.json").read_text())
        except Exception:
            state = {}
        settled = len(state.get("done", [])) + len(state.get("failed", []))
        alive = subprocess.run(["tmux", "has-session", "-t", "orch_xa"],
                               capture_output=True).returncode == 0
        if settled >= TOTAL_JOBS or not alive:
            log(f"orchestrator finished: {len(state.get('done', []))} done, "
                f"{len(state.get('failed', []))} failed")
            return
        time.sleep(120)


def snapshot_frame_w1():
    """Copy the frame-wise checkpoints aside; they predate the orchestrator."""
    out = {}
    for arm, run_name in FRAME_W1:
        runs = sorted((REPO / "experiments").glob(f"*/{run_name}"),
                      key=lambda p: p.stat().st_mtime)
        if not runs:
            log(f"  {arm}: no run dir, skipping")
            continue
        ckpts = sorted((runs[-1] / "checkpoints").glob("epoch-*.ckpt"))
        if not ckpts:
            log(f"  {arm}: no checkpoint, skipping")
            continue
        dst = SNAP / f"xa_{arm}_ep0200.ckpt"
        shutil.copy2(ckpts[-1], dst)
        out[arm] = dst
        log(f"  {arm}: {ckpts[-1].name} -> {dst.name}")
    return out


def run_frame_w1_evals(ckpts):
    """4 GPUs, one job each, until the frame-wise eval list is done."""
    jobs = []
    for arm, ckpt in ckpts.items():
        for task, cfg, folds in (("slip", SLIP_FW, 3), ("grasp", GRASP_FW, 4)):
            for seed in (0, 1):
                jobs.append((task, arm, seed, cfg, folds, ckpt))
    log(f"frame-wise evals: {len(jobs)} runs")
    (FIN / "logs").mkdir(parents=True, exist_ok=True)

    running, queue = {}, list(jobs)
    while queue or running:
        for gpu in list(running):
            proc, desc = running[gpu]
            if proc.poll() is not None:
                log(f"  {'OK  ' if proc.returncode == 0 else 'FAIL'} {desc}")
                del running[gpu]
        for gpu in range(4):
            if gpu in running or not queue:
                continue
            task, arm, seed, cfg, folds, ckpt = queue.pop(0)
            desc = f"{task} {arm} seed{seed}"
            logfile = FIN / "logs" / f"{task}__{arm}__seed{seed}.log"
            env = {**os.environ, "XFORMERS_DISABLED": "TRUE",
                   "CUDA_VISIBLE_DEVICES": str(gpu)}
            with open(logfile, "w") as fh:
                proc = subprocess.Popen(
                    [PY, "train_task_brainco_angle.py", f"+experiment={cfg}",
                     f"task.checkpoint_encoder={ckpt}", "task.encoder_lr=1e-4",
                     f"seed={seed}", "++split_seed=42",
                     f"experiment_name=xa2__{task}__{arm}__seed{seed}",
                     f"wandb.group=xa2_{task}",
                     "--all_split", f"--num_folds={folds}"],
                    cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (proc, desc)
            log(f"  START {desc} on gpu{gpu}")
        time.sleep(20)


def collect(log_dirs):
    """arm -> task -> readout -> list of per-run means, across every log dir."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d in log_dirs:
        for path in Path(d).glob("*.log"):
            stem = path.name[:-4]
            # orchestrator: ev_<task>_<arm>_s<seed>;  finisher: <task>__<arm>__seed<n>
            m = re.fullmatch(r"ev_(grasp|slip)_(.+)_s(\d+)", stem)
            if m:
                task, arm = m.group(1), m.group(2)
            else:
                m = re.fullmatch(r"(grasp|slip)__(.+)__seed(\d+)", stem)
                if not m:
                    continue
                task, arm = m.group(1), m.group(2)
            text = ANSI.sub("", path.read_text(errors="ignore"))
            for r in READOUTS:
                m = re.search(
                    r"K-FOLD SUMMARY \(" + re.escape(r) +
                    r"\).*?Mean\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text, re.S)
                if m:
                    out[arm][task][r].append(float(m.group(1)))
    return out


def cell(values):
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{st.mean(values):.4f} ± {st.stdev(values):.4f}"


def write_report(data):
    date = time.strftime("%Y%m%d")
    path = REPO / "scripts" / f"results_xa_final_{date}.md"
    L = []
    L.append("# HOI data scaling and seed variance, on the BrainCo-frame pipeline\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')}\n")
    L.append("## Protocol\n")
    L.append("| | |")
    L.append("| --- | --- |")
    L.append("| Alignment | HOI converted to BrainCo's fingertip frame; no per-axis whitening |")
    L.append("| grasp | `task/xyznorm/grasp_*`, 4-fold CV |")
    L.append("| slip | `task/xyznorm/slip_*_v3` on `slip_data_v3`, 3-fold CV |")
    L.append("| Overrides | `task.encoder_lr=1e-4`, `++split_seed=42`, 100 downstream epochs |")
    L.append("| Pretraining | 200 epochs, lr 4e-4, batch 2048 (BrainCo arms: 512 / 1e-4) |")
    L.append("| Data scaling | constant 49,200 optimizer steps at every fraction |")
    L.append("| Aggregation | mean ± sd over pretraining seeds x downstream seeds |")
    L.append("")

    for readout in READOUTS:
        L.append(f"## HOI data scaling -- {readout}\n")
        L.append("| HOI data | grasp | slip | runs |")
        L.append("| ---: | ---: | ---: | ---: |")
        for tag, label in [("01", "1%"), ("02", "2%"), ("05", "5%"),
                           ("10", "10%"), ("20", "20%"), ("50", "50%")]:
            g, s = [], []
            for arm in data:
                if arm.startswith(f"tipf{tag}_s"):
                    g += data[arm]["grasp"][readout]
                    s += data[arm]["slip"][readout]
            L.append(f"| {label} | {cell(g)} | {cell(s)} | {len(g)}/{len(s)} |")
        g = data.get("tip_s43", {}).get("grasp", {}).get(readout, [])
        s = data.get("tip_s43", {}).get("slip", {}).get(readout, [])
        L.append(f"| 100% (seed 43) | {cell(g)} | {cell(s)} | {len(g)}/{len(s)} |")
        L.append("")

    ARMS = [("nopretrain", "no pretraining"), ("tip_s43", "HOI tip"),
            ("jointonly_s43", "HOI jointonly"), ("bconly_s43", "brainco-only SSL"),
            ("bconly_jo_s43", "brainco-only jointonly"),
            ("hoiinit_s43", "hoi-init SSL"), ("gentle_s43", "hoi-init gentle"),
            ("framew1_tip", "frame-w1 tip"), ("framew1_42j", "frame-w1 42j")]
    for readout in READOUTS:
        L.append(f"## Arms at pretraining seed 43 -- {readout}\n")
        L.append("| Model | grasp | slip | runs |")
        L.append("| --- | ---: | ---: | ---: |")
        for arm, label in ARMS:
            g = data.get(arm, {}).get("grasp", {}).get(readout, [])
            s = data.get(arm, {}).get("slip", {}).get(readout, [])
            if not g and not s:
                continue
            L.append(f"| {label} | {cell(g)} | {cell(s)} | {len(g)}/{len(s)} |")
        L.append("")

    L.append("## Caveats\n")
    L.append("- `±` spans pretraining seeds and downstream seeds together, so it "
             "mixes both sources of variance; per-fold spread is larger still.")
    L.append("- The seed-42 numbers for the 8-row table live in "
             "`results_xyz_alignment_20260731.md`; compare against those for the "
             "pretraining-seed effect.")
    L.append("- The frame-w1 arms use the frame-wise task stack "
             "(`task/xyznorm/*_framew1`), not the temporal former, so they are "
             "not directly comparable to the other rows.")
    path.write_text("\n".join(L) + "\n")
    log(f"report written: {path}")
    return path


def main():
    FIN.mkdir(parents=True, exist_ok=True)
    wait_for_orchestrator()
    log("snapshotting frame-wise checkpoints")
    ckpts = snapshot_frame_w1()
    if ckpts:
        run_frame_w1_evals(ckpts)
    log("aggregating")
    data = collect([BASE / "logs", FIN / "logs"])
    write_report(data)
    log("ALL DONE")


if __name__ == "__main__":
    main()
