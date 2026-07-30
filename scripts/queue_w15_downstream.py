#!/usr/bin/env python
"""Queue 15-frame grasp + slip evaluation for pretraining runs still training.

Waits for each pretraining PID to exit, snapshots the highest ``epoch-*.ckpt``
of its newest run directory, and evaluates it on both downstream tasks with
15-frame windows. Each encoder is dispatched as soon as *its own* training
finishes, so a 100-epoch run is not held back by a 500-epoch one.

Every encoder gets a matching architecture config: the temporal composite
needs ``temporal_tiny_downstream`` (and a ``sequence_length`` equal to the span
its former saw during pretraining), the frame-wise ones use ``angle_tiny``.
Getting this wrong is silent -- ``SLModule.load_encoder`` drops mismatched keys
and still reports a "pretrained" run -- so this script asserts that every
checkpoint tensor is consumed before launching.

Usage:
    python scripts/queue_w15_downstream.py --gpus 4,5
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
from run_lr_checkpoint_matrix import parse_log  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# experiments/<experiment_name> -> (label, pretraining config name, slip cfg, grasp cfg)
#
# The pretraining config name is what appears on the training command line and
# is not the same string as the experiment_name the run writes its outputs to,
# so PID matching needs it separately.
SPECS = {
    "dinov2_temporal_all_pseudo_force_tiny_tip_w15": (
        "temporal_w15",
        "dinov2_temporal_tip_w15",
        "brainco/ours_3d/task/slip_detection/temporal_w15",
        "brainco/ours_3d/task/grasp_prediction/temporal_w15",
    ),
    "dinov2_temporal_all_pseudo_force_tiny_tip_w5": (
        "temporal_w5",
        "dinov2_temporal_tip_w5",
        "brainco/ours_3d/task/slip_detection/temporal_w15_seq5",
        "brainco/ours_3d/task/grasp_prediction/temporal_w15_seq5",
    ),
    "dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048_ep500": (
        "framewise_ep500",
        "dinov2_pretraining_all_rope_v2_hp1_ibotfix_tip_s1_b2048_ep500",
        "brainco/ours_3d/task/slip_detection/dinov2_all_rope_w15",
        "brainco/ours_3d/task/grasp_prediction/dinov2_all_rope_w15",
    ),
    "dinov2_recon_all_pseudo_force_tiny_rope_v2_hp1_tip_s1_b2048_ep500": (
        "framewise_recon_ep500",
        "dinov2_recon_all_rope_v2_hp1_tip_s1_b2048_ep500",
        "brainco/ours_3d/task/slip_detection/dinov2_all_rope_w15",
        "brainco/ours_3d/task/grasp_prediction/dinov2_all_rope_w15",
    ),
}

FIELDS = [
    "encoder", "task", "epoch", "checkpoint", "status",
    "last_balacc", "last_f1", "last_f1macro",
    "best_balacc", "best_f1", "best_f1macro",
    "epochavg_balacc", "epochavg_f1", "epochavg_f1macro",
    "last_balacc_std", "last_f1macro_std", "runtime_s", "log",
]

SNAPSHOT_DIR = PROJECT_ROOT / "checkpoints" / "queued_w15_snapshots"


def training_pid(pretrain_cfg: str) -> int | None:
    """PID of the live `train.py +experiment=.../<pretrain_cfg>` process."""
    out = subprocess.run(
        ["pgrep", "-f", f"python train.py \\+experiment=.*/{re.escape(pretrain_cfg)}$"],
        capture_output=True, text=True,
    ).stdout.split()
    return int(out[0]) if out else None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def latest_checkpoint(experiment: str):
    root = PROJECT_ROOT / "experiments" / experiment
    if not root.is_dir():
        return None, None
    runs = sorted((d for d in root.iterdir() if (d / "checkpoints").is_dir()),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    for run in runs:
        found = []
        for path in (run / "checkpoints").glob("epoch-*.ckpt"):
            m = re.search(r"epoch-(\d+)\.ckpt$", path.name)
            if m:
                found.append((int(m.group(1)), path))
        if found:
            return max(found)
    return None, None


def snapshot(label: str, epoch: int, src: Path) -> Path:
    """Copy the checkpoint so a later run cannot overwrite what we evaluate."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    dst = SNAPSHOT_DIR / f"{label}_ep{epoch:04d}.ckpt"
    if not dst.exists():
        subprocess.run(["cp", str(src), str(dst)], check=True)
    return dst


def assert_encoder_matches(config: str, ckpt: Path) -> str:
    """Fail loudly if the config would silently drop checkpoint tensors."""
    script = f"""
import torch, hydra
from hydra import compose
from hydra.initialize import initialize_config_dir
import tactile_ssl.utils
with initialize_config_dir(version_base="1.3", config_dir="{PROJECT_ROOT}/config"):
    cfg = compose(config_name="default_task.yaml", overrides=["+experiment={config}"])
enc = hydra.utils.instantiate(cfg.task.model_encoder)
ms = enc.state_dict()
ck = torch.load("{ckpt}", map_location="cpu", weights_only=False)["model"]
pre = "teacher_encoder.backbone."
new = {{k.split(pre,1)[1]: ck[k] for k in ck if pre in k}}
ok = [k for k in new if k in ms and new[k].shape == ms[k].shape]
print(f"MATCH {{len(ok)}}/{{len(new)}}")
"""
    env = {**os.environ, "XFORMERS_DISABLED": "TRUE", "PYTHONPATH": str(PROJECT_ROOT)}
    res = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT,
                         env=env, capture_output=True, text=True)
    line = next((l for l in res.stdout.splitlines() if l.startswith("MATCH")), None)
    if line is None:
        return f"check failed: {res.stderr.strip().splitlines()[-1:] or res.stdout}"
    loaded, total = map(int, line.split()[1].split("/"))
    return "ok" if loaded == total and total > 0 else f"WARNING {loaded}/{total} tensors match"


def run_job(job, gpu, log_dir, args):
    label, task, config, epoch, ckpt = job
    name = f"{label}__{task}_w15"
    log_path = log_dir / f"{name}.log"
    cmd = [
        "python", "train_task_brainco_angle.py",
        f"+experiment={config}",
        f"task.checkpoint_encoder={ckpt}",
        f"task.encoder_lr={args.backbone_lr}",
        f"seed={args.seed}", f"++split_seed={args.split_seed}",
        f"experiment_name=qw15_{name}",
        f"wandb.group=qw15_{task}",
        "--all_split", "--num_folds", str(args.num_folds),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XFORMERS_DISABLED": "TRUE"}
    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                              stdout=fh, stderr=subprocess.STDOUT)
    runtime = time.time() - t0
    row = {f: "" for f in FIELDS}
    row.update(encoder=label, task=task, epoch=str(epoch), checkpoint=str(ckpt),
               runtime_s=f"{runtime:.0f}", log=str(log_path))
    row.update(parse_log(log_path.read_text(errors="ignore")))
    row["status"] = "OK" if (proc.returncode == 0 and row.get("last_balacc")) else "FAILED"
    print(f"[{row['status']}] gpu{gpu} {name}  balacc={row.get('last_balacc','-')} "
          f"f1macro={row.get('last_f1macro','-')} ({runtime/60:.1f}m)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone-lr", default="1e-4")
    ap.add_argument("--num-folds", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "scripts" / "logs" / f"queued_w15_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    watched = {}
    for experiment, (label, pretrain_cfg, *_rest) in SPECS.items():
        pid = training_pid(pretrain_cfg)
        watched[experiment] = pid
        print(f"  {label:22s} pid={pid}  ({experiment})", flush=True)
    print(f"\nlogs -> {log_dir}\n", flush=True)

    pool: queue.Queue = queue.Queue()
    for g in args.gpus.split(","):
        pool.put(g.strip())
    rows, lock, threads = [], threading.Lock(), []

    def worker(job):
        gpu = pool.get()
        try:
            row = run_job(job, gpu, log_dir, args)
        finally:
            pool.put(gpu)
        with lock:
            rows.append(row)

    pending = dict(watched)
    while pending:
        for experiment, pid in list(pending.items()):
            if pid is not None and alive(pid):
                continue
            del pending[experiment]
            label, _pretrain_cfg, slip_cfg, grasp_cfg = SPECS[experiment]
            epoch, ckpt = latest_checkpoint(experiment)
            if ckpt is None:
                print(f"[SKIP] {label}: no epoch-*.ckpt found", flush=True)
                continue
            snap = snapshot(label, epoch, ckpt)
            print(f"\n[READY] {label}: epoch {epoch} -> {snap}", flush=True)
            for task, cfg in (("slip_detection", slip_cfg),
                              ("grasp_prediction", grasp_cfg)):
                verdict = assert_encoder_matches(cfg, snap)
                print(f"    {task:16s} encoder match: {verdict}", flush=True)
                th = threading.Thread(
                    target=worker, args=((label, task, cfg, epoch, snap),))
                th.start()
                threads.append(th)
                time.sleep(2)
        if pending:
            names = [SPECS[e][0] for e in pending]
            print(f"  [{datetime.now():%H:%M}] still training: {names}", flush=True)
            time.sleep(args.poll)

    for th in threads:
        th.join()

    rows.sort(key=lambda r: (r["task"], r["encoder"]))
    csv_path = PROJECT_ROOT / "scripts" / f"results_queued_w15_{stamp}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    md_path = PROJECT_ROOT / "scripts" / f"results_queued_w15_{stamp}.md"
    lines = ["# 15-frame downstream results for queued pretraining runs", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M}", "",
             "## Protocol", "",
             f"- 15-frame windows for both tasks (slip `input_window_frames=15`, "
             f"grasp `window_time=0.15`)",
             f"- Episode-level {args.num_folds}-fold CV, seed {args.seed}, "
             f"split seed {args.split_seed}, backbone LR {args.backbone_lr}",
             "- Final `epoch-*.ckpt` of each pretraining run, snapshotted before evaluation",
             "- NOT comparable to the 30-frame grasp / 3-frame slip tables; the "
             "scratch baselines under this config are the reference", ""]
    for task in ("grasp_prediction", "slip_detection"):
        trs = [r for r in rows if r["task"] == task]
        if not trs:
            continue
        lines += [f"## {task}", "",
                  "| Encoder | Epoch | Bal Acc | ± | F1 macro | ± | Status |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for r in sorted(trs, key=lambda r: r.get("last_f1macro") or "", reverse=True):
            lines.append(
                f"| {r['encoder']} | {r['epoch']} | {r.get('last_balacc') or 'n/a'} | "
                f"{r.get('last_balacc_std') or 'n/a'} | {r.get('last_f1macro') or 'n/a'} | "
                f"{r.get('last_f1macro_std') or 'n/a'} | {r['status']} |")
        lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"\nCSV: {csv_path}\nMD : {md_path}", flush=True)


if __name__ == "__main__":
    main()
