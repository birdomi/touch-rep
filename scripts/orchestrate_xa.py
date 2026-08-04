#!/usr/bin/env python3
"""Unattended driver for the xyz-alignment study.

Runs a dependency-aware job DAG across the GPUs: pretrainings first, each one
snapshotting its checkpoint on success, then the downstream evaluations that
depend on it. Survives losing the interactive session -- launch it detached in
tmux and it drives every remaining run to completion on its own.

State lives in <base>/state.json, so an interrupted driver can be relaunched and
will skip whatever already finished.

Two tracks:
  A  seed repeat -- the same arms at pretraining seed 43, to size seed variance
     against the seed-42 runs already measured.
  B  HOI data scaling -- temporal_w15_tip on 1/2/5/10/20/50% of the corpus at a
     *constant 49,200 optimizer steps*, so the curve isolates data quantity from
     optimization budget. Epoch counts are set per fraction to hit that budget.
"""

import json
import subprocess
import shutil
import sys
import time
from pathlib import Path

REPO = Path("/raid/ygyu/workspace/touch-rep")
BASE = Path("/tmp/claude-1203/-raid-ygyu-workspace-touch-rep/"
            "4d80c27e-1b0b-464c-ab82-54ceeb31b81a/scratchpad/orch")
PY = "/raid/ygyu/miniconda3/envs/tactile/bin/python"
SNAP = REPO / "checkpoints/queued_w15_snapshots"
GPUS = [0, 1, 2, 3]
TARGET_STEPS = 49_200          # the 100% baseline's budget: 246 steps x 200 ep
TOTAL_WINDOWS = 504_837
BATCH = 2048

# fraction -> epochs that land on TARGET_STEPS
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
PT_SEEDS = [42, 43]
# A third repetition of the data-scaling sweep, queued after everything else.
EXTRA_FRACTION_SEEDS = [44]
# A third pretraining seed for the arm table too, queued last.
EXTRA_ARM_SEEDS = [44]
DS_SEEDS = [0, 1]

GRASP = "brainco/ours_3d/task/xyznorm/grasp_temporal_w15_cls_d4_fe4"
SLIP = "brainco/ours_3d/task/xyznorm/slip_temporal_w15_cls_d4_fe4_v3"
GRASP_JO = "brainco/ours_3d/task/xyznorm/grasp_jointonly"
SLIP_JO = "brainco/ours_3d/task/xyznorm/slip_jointonly"


def epochs_for(fraction):
    """Epochs to reach TARGET_STEPS.

    train.py repeats a fractional subset x round(1/fraction) inside one epoch,
    so every fraction sees roughly the full-corpus number of windows per epoch
    and 200 epochs lands on the same step budget as the 100% baseline.
    """
    repeat = max(1, round(1.0 / fraction)) if fraction < 1.0 else 1
    windows = int(TOTAL_WINDOWS * fraction) * repeat
    steps_per_epoch = max(1, windows // BATCH)
    return max(1, round(TARGET_STEPS / steps_per_epoch))


def build_jobs():
    jobs = []

    def pretrain(jid, cfg, name, epochs, overrides, snap, deps=()):
        # Only the final epoch is kept: checkpoint_frequency == max_epochs writes
        # exactly one epoch-*.ckpt, and last.ckpt is refreshed every epoch anyway.
        # The default frequency would have written 2,460 files (629 GB) for the
        # 1% run, which needs 24,600 epochs to reach the shared step budget.
        # Rewrite last.ckpt at most ~20 times per run. Without this the 1% run
        # writes 262 MB x 24,600 epochs and goes disk-bound at 0% GPU.
        full = [f"trainer.max_epochs={epochs}",
                f"trainer.checkpoint_frequency={epochs}",
                f"trainer.latest_checkpoint_frequency={max(1, epochs // 20)}",
                *overrides]
        jobs.append(dict(id=jid, kind="pretrain", cfg=cfg, name=name, epochs=epochs,
                         overrides=full, snap=snap, deps=list(deps)))

    def evals(arm, ckpt, dep, grasp_cfg=GRASP, slip_cfg=SLIP):
        for task, cfg, folds in (("slip", slip_cfg, 3), ("grasp", grasp_cfg, 4)):
            for s in DS_SEEDS:
                jobs.append(dict(id=f"ev_{task}_{arm}_s{s}", kind="eval", arm=arm,
                                 task=task, cfg=cfg, folds=folds, seed=s,
                                 ckpt=ckpt, deps=[dep] if dep else []))

    # ---- Track B: HOI data scaling, both pretraining seeds -------------------
    for frac in FRACTIONS:
        tag = f"{int(frac * 100):02d}"
        for ps in PT_SEEDS:
            jid = f"pt_tipf{tag}_s{ps}"
            snap = f"xa_hoi_tip_f{tag}_s{ps}_ep.ckpt"
            pretrain(jid, "brainco/ours_3d/xyznorm/temporal_w15_tip",
                     f"xa_tip_f{tag}_s{ps}", epochs_for(frac),
                     [f"data.data_fraction={frac}", f"seed={ps}",
                      f"experiment_name=xa_tip_f{tag}_s{ps}"],
                     snap)
            evals(f"tipf{tag}_s{ps}", str(SNAP / snap), jid)

    # ---- Track A: seed 43 repeat of every arm --------------------------------
    ps = 43
    pretrain("pt_tip_s43", "brainco/ours_3d/xyznorm/temporal_w15_tip",
             "xa_tip_s43", 200,
             [f"seed={ps}", "experiment_name=xa_tip_s43"],
             "xa_hoi_tip_s43_ep.ckpt")
    evals("tip_s43", str(SNAP / "xa_hoi_tip_s43_ep.ckpt"), "pt_tip_s43")

    pretrain("pt_jo_s43", "brainco/ours_3d/xyznorm/temporal_w15_jointonly",
             "xa_jo_s43", 200,
             [f"seed={ps}", "experiment_name=xa_jo_s43"],
             "xa_hoi_jointonly_s43_ep.ckpt")
    evals("jointonly_s43", str(SNAP / "xa_hoi_jointonly_s43_ep.ckpt"),
          "pt_jo_s43", GRASP_JO, SLIP_JO)

    pretrain("pt_bconly_s43", "brainco/ours_3d/xyznorm/brainco_only",
             "xa_bconly_s43", 1000,
             [f"seed={ps}", "experiment_name=xa_bconly_s43"],
             "xa_brainco_only_s43_ep.ckpt")
    evals("bconly_s43", str(SNAP / "xa_brainco_only_s43_ep.ckpt"), "pt_bconly_s43")

    pretrain("pt_bcjo_s43", "brainco/ours_3d/xyznorm/brainco_only_jointonly",
             "xa_bcjo_s43", 1000,
             [f"seed={ps}", "experiment_name=xa_bcjo_s43"],
             "xa_brainco_only_jo_s43_ep.ckpt")
    evals("bconly_jo_s43", str(SNAP / "xa_brainco_only_jo_s43_ep.ckpt"),
          "pt_bcjo_s43", GRASP_JO, SLIP_JO)

    # These two initialise from the seed-43 tip checkpoint, so they wait on it.
    pretrain("pt_hoiinit_s43", "brainco/ours_3d/xyznorm/brainco_hoi_init",
             "xa_hoiinit_s43", 1000,
             [f"seed={ps}", "experiment_name=xa_hoiinit_s43",
              f"init_from_ckpt={SNAP / 'xa_hoi_tip_s43_ep.ckpt'}"],
             "xa_brainco_hoi_init_s43_ep.ckpt", deps=["pt_tip_s43"])
    evals("hoiinit_s43", str(SNAP / "xa_brainco_hoi_init_s43_ep.ckpt"),
          "pt_hoiinit_s43")

    pretrain("pt_gentle_s43", "brainco/ours_3d/xyznorm/brainco_hoi_init_gentle",
             "xa_gentle_s43", 200,
             [f"seed={ps}", "experiment_name=xa_gentle_s43",
              f"init_from_ckpt={SNAP / 'xa_hoi_tip_s43_ep.ckpt'}"],
             "xa_brainco_gentle_s43_ep.ckpt", deps=["pt_tip_s43"])
    evals("gentle_s43", str(SNAP / "xa_brainco_gentle_s43_ep.ckpt"),
          "pt_gentle_s43")

    # ---- Track A, third pretraining seed -------------------------------------
    for ps in EXTRA_ARM_SEEDS:
        sfx = f"s{ps}"
        tip_snap = f"xa_hoi_tip_{sfx}_ep.ckpt"
        for jid, cfg, name, eps, snap, extra, dep, gc, sc in [
            (f"pt_tip_{sfx}", "temporal_w15_tip", f"xa_tip_{sfx}", 200,
             tip_snap, [], None, GRASP, SLIP),
            (f"pt_jo_{sfx}", "temporal_w15_jointonly", f"xa_jo_{sfx}", 200,
             f"xa_hoi_jointonly_{sfx}_ep.ckpt", [], None, GRASP_JO, SLIP_JO),
            (f"pt_bconly_{sfx}", "brainco_only", f"xa_bconly_{sfx}", 1000,
             f"xa_brainco_only_{sfx}_ep.ckpt", [], None, GRASP, SLIP),
            (f"pt_bcjo_{sfx}", "brainco_only_jointonly", f"xa_bcjo_{sfx}", 1000,
             f"xa_brainco_only_jo_{sfx}_ep.ckpt", [], None, GRASP_JO, SLIP_JO),
            (f"pt_hoiinit_{sfx}", "brainco_hoi_init", f"xa_hoiinit_{sfx}", 1000,
             f"xa_brainco_hoi_init_{sfx}_ep.ckpt",
             [f"init_from_ckpt={SNAP / tip_snap}"], f"pt_tip_{sfx}", GRASP, SLIP),
            (f"pt_gentle_{sfx}", "brainco_hoi_init_gentle", f"xa_gentle_{sfx}", 200,
             f"xa_brainco_gentle_{sfx}_ep.ckpt",
             [f"init_from_ckpt={SNAP / tip_snap}"], f"pt_tip_{sfx}", GRASP, SLIP),
        ]:
            pretrain(jid, f"brainco/ours_3d/xyznorm/{cfg}", name, eps,
                     [f"seed={ps}", f"experiment_name={name}", *extra],
                     snap, deps=[dep] if dep else ())
            arm = jid.replace("pt_", "") .replace("tip_", "tip_")
            evals(arm, str(SNAP / snap), jid, gc, sc)

    # ---- brainco-only at the HOI step budget ---------------------------------
    # BrainCo carries 15,921 windows against HOI's 504,837 -- 3.15%. The arm
    # table trains it for 31,000 steps; this matches HOI's 49,200 exactly, so
    # the two differ in the data alone and not in optimization budget.
    # 15,921 / 512 = 31 steps per epoch, hence 1,587 epochs.
    for ps in (42, 43):
        jid = f"pt_bconly49k_s{ps}"
        snap = f"xa_brainco_only_49k_s{ps}_ep.ckpt"
        pretrain(jid, "brainco/ours_3d/xyznorm/brainco_only",
                 f"xa_bconly49k_s{ps}", 1587,
                 [f"seed={ps}", f"experiment_name=xa_bconly49k_s{ps}"], snap)
        evals(f"bconly49k_s{ps}", str(SNAP / snap), jid)

    # ---- brainco-only, longer and hotter -------------------------------------
    # The arm table showed brainco-only still climbing with steps: 0.8624 grasp
    # at 31k, 0.8734 at 49k. This pushes to 100k steps at lr 4e-4 -- four times
    # the batch-512 linear-scale value, i.e. deliberately hotter than the rest of
    # the BrainCo arms -- to find where the curve flattens. 15,921 / 512 = 31
    # steps per epoch, so 3,226 epochs.
    # Both learning rates at the same 100k steps, so the pair also isolates lr:
    # 1e-4 is the batch-512 linear scale of the HOI setting, 4e-4 is four times
    # hotter. Two pretraining seeds each, matching the rest of the table.
    for lr_tag, lr in (("lr4e4", "4e-4"), ("lr1e4", "1e-4")):
        for ps in (42, 43):
            jid = f"pt_bconly100k_{lr_tag}_s{ps}"
            snap = f"xa_brainco_only_100k_{lr_tag}_s{ps}_ep.ckpt"
            pretrain(jid, "brainco/ours_3d/xyznorm/brainco_only",
                     f"xa_bconly100k_{lr_tag}_s{ps}", 3226,
                     [f"seed={ps}", f"algorithm.optim_cfg.lr={lr}",
                      f"experiment_name=xa_bconly100k_{lr_tag}_s{ps}"], snap)
            evals(f"bconly100k_{lr_tag}_s{ps}", str(SNAP / snap), jid)

    # ---- Track B, third repetition -------------------------------------------
    # Appended last so the two-seed table completes first and this only adds
    # confidence to the scaling curve if there is time for it.
    for frac in FRACTIONS:
        tag = f"{int(frac * 100):02d}"
        for ps in EXTRA_FRACTION_SEEDS:
            jid = f"pt_tipf{tag}_s{ps}"
            snap = f"xa_hoi_tip_f{tag}_s{ps}_ep.ckpt"
            pretrain(jid, "brainco/ours_3d/xyznorm/temporal_w15_tip",
                     f"xa_tip_f{tag}_s{ps}", epochs_for(frac),
                     [f"data.data_fraction={frac}", f"seed={ps}",
                      f"experiment_name=xa_tip_f{tag}_s{ps}"],
                     snap)
            evals(f"tipf{tag}_s{ps}", str(SNAP / snap), jid)

    return jobs


def command(job, gpu):
    log = BASE / "logs" / f"{job['id']}.log"
    if job["kind"] == "pretrain":
        args = [PY, "train.py", f"+experiment={job['cfg']}.yaml", *job["overrides"]]
    else:
        args = [PY, "train_task_brainco_angle.py", f"+experiment={job['cfg']}",
                f"task.checkpoint_encoder={job['ckpt']}", "task.encoder_lr=1e-4",
                f"seed={job['seed']}", "++split_seed=42",
                f"experiment_name=xa2__{job['task']}__{job['arm']}__seed{job['seed']}",
                f"wandb.group=xa2_{job['task']}",
                "--all_split", f"--num_folds={job['folds']}"]
    return args, log


def snapshot(job, state):
    """Copy the newest epoch checkpoint of a finished pretraining aside."""
    runs = sorted((REPO / "experiments").glob(f"*/{job['name']}"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        return False, f"no run dir for {job['name']}"
    ckpts = sorted((runs[-1] / "checkpoints").glob("epoch-*.ckpt"))
    if not ckpts:
        return False, f"no epoch ckpt in {runs[-1]}"
    dst = SNAP / job["snap"]
    shutil.copy2(ckpts[-1], dst)
    state.setdefault("snapshots", {})[job["id"]] = str(dst)
    return True, f"{ckpts[-1].name} -> {dst.name}"


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "logs").mkdir(exist_ok=True)
    state_path = BASE / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    done = set(state.get("done", []))
    failed = set(state.get("failed", []))

    jobs = {j["id"]: j for j in build_jobs()}
    order = list(jobs)
    running = {}          # gpu -> (job_id, Popen, start)

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(BASE / "driver.log", "a") as f:
            f.write(line + "\n")

    def save():
        state["done"] = sorted(done)
        state["failed"] = sorted(failed)
        state_path.write_text(json.dumps(state, indent=1))

    log(f"{len(jobs)} jobs total, {len(done)} already done")
    for frac in FRACTIONS:
        log(f"  fraction {frac}: {epochs_for(frac)} epochs "
            f"-> ~{epochs_for(frac) * max(1, int(TOTAL_WINDOWS*frac)//BATCH)} steps")

    while True:
        # reap
        for gpu in list(running):
            jid, proc, started = running[gpu]
            if proc.poll() is None:
                continue
            mins = (time.time() - started) / 60
            job = jobs[jid]
            if proc.returncode == 0:
                if job["kind"] == "pretrain":
                    ok, detail = snapshot(job, state)
                    if not ok:
                        failed.add(jid)
                        log(f"FAIL   {jid} on gpu{gpu} after {mins:.0f}m: {detail}")
                        del running[gpu]
                        save()
                        continue
                    log(f"OK     {jid} on gpu{gpu} in {mins:.0f}m ({detail})")
                else:
                    log(f"OK     {jid} on gpu{gpu} in {mins:.0f}m")
                done.add(jid)
            else:
                failed.add(jid)
                log(f"FAIL   {jid} on gpu{gpu} rc={proc.returncode} after {mins:.0f}m")
            del running[gpu]
            save()

        # dispatch
        for gpu in GPUS:
            if gpu in running:
                continue
            for jid in order:
                if jid in done or jid in failed or jid in {r[0] for r in running.values()}:
                    continue
                job = jobs[jid]
                if any(d in failed for d in job["deps"]):
                    failed.add(jid)
                    log(f"SKIP   {jid}: dependency failed")
                    continue
                if not all(d in done for d in job["deps"]):
                    continue
                args, logfile = command(job, gpu)
                env_prefix = {"XFORMERS_DISABLED": "TRUE",
                              "CUDA_VISIBLE_DEVICES": str(gpu)}
                import os
                env = {**os.environ, **env_prefix}
                with open(logfile, "w") as fh:
                    proc = subprocess.Popen(args, cwd=REPO, stdout=fh,
                                            stderr=subprocess.STDOUT, env=env)
                running[gpu] = (jid, proc, time.time())
                log(f"START  {jid} on gpu{gpu}")
                break

        remaining = [j for j in order if j not in done and j not in failed]
        if not remaining and not running:
            log(f"ALL DONE. {len(done)} succeeded, {len(failed)} failed.")
            save()
            return 0
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
