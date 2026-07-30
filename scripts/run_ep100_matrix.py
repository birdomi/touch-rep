#!/usr/bin/env python
"""Grasp + slip at the epoch-100 snapshot of the three live pretraining runs.

Waits for each run's ``epoch-0100.ckpt`` to appear and finish being written,
copies it into checkpoints/queued_w15_snapshots/, then runs the standard ID
matrix (2 seeds, 4-fold CV) for both tasks at the 15-frame downstream window.

The three encoders differ along two axes against the same recipe:
    fdino_lr4e4  : 42 joints, lr 4e-4   <- the reference arm
    fdino_lr1e4  : 42 joints, lr 1e-4   <- lr axis
    tip_lr4e4    : 10 fingertips, lr 4e-4 <- sensor-extent axis

All three use slip/grasp temporal_w15_cls_d4_fe4. The fingertip checkpoint does
NOT need an in_dim 10 downstream config: under RoPE no parameter shape depends
on in_dim, and the RoPE branch is picked from the runtime token count, so an
in_dim 10 config and the 42 one produce bit-identical outputs on the same
10-sensor BrainCo input.

Every one of these checkpoints is proximity-only (in_chans 1) while the
downstream is BrainCo's 4 channels, so load_encoder leaves 3 of 129 tensors
random: sensor_embed.proj.weight, signal_mean, signal_std. That is the same
condition every prox evaluation today ran under.

``--only label[,label]`` restricts the run to a subset of the three, so the
arms whose checkpoints already exist can be evaluated without waiting on the
slowest run. Each invocation writes its own CSV/MD; merge them when reporting.

Usage:
    python scripts/run_ep100_matrix.py --gpus 0,2 --seeds 0,1 --protocols id
    python scripts/run_ep100_matrix.py --only tip_lr4e4_ep100 --gpus 0,2 --seeds 0,1
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = ROOT / "checkpoints" / "queued_w15_snapshots"

# label -> (source epoch-0100.ckpt, snapshot filename)
RUNS = {
    "fdino_lr4e4_ep100": (
        "experiments/dinov2_temporal_all_pseudo_force_tiny_w15_prox_cls_d4_fe4_fdino_lr4e4"
        "/2026.07.29-16-48/checkpoints/epoch-0100.ckpt",
        "temporal_d4fe4cls_fdino_lr4e4_ep0100.ckpt",
    ),
    "fdino_lr1e4_ep100": (
        "experiments/dinov2_temporal_all_pseudo_force_tiny_w15_prox_cls_d4_fe4_fdino"
        "/2026.07.29-15-20/checkpoints/epoch-0100.ckpt",
        "temporal_d4fe4cls_fdino_lr1e4_ep0100.ckpt",
    ),
    "tip_lr4e4_ep100": (
        "experiments/dinov2_temporal_all_pseudo_force_tiny_tip_w15_prox_cls_d4_fe4_fdino_lr4e4"
        "/2026.07.29-17-47/checkpoints/epoch-0100.ckpt",
        "temporal_d4fe4cls_tip_fdino_lr4e4_ep0100.ckpt",
    ),
}

POLL_SECONDS = 60
# torch.save writes incrementally; only copy once the size stops growing.
STABLE_SECONDS = 30


def pop_only_flag() -> list:
    """Read and remove --only before run_seed_ood_matrix's argparse sees it."""
    if "--only" not in sys.argv:
        return list(RUNS)
    i = sys.argv.index("--only")
    labels = [s.strip() for s in sys.argv[i + 1].split(",") if s.strip()]
    del sys.argv[i:i + 2]
    unknown = [l for l in labels if l not in RUNS]
    if unknown:
        raise SystemExit(f"Unknown --only labels {unknown}; valid: {list(RUNS)}")
    return labels


def wait_and_snapshot(labels) -> dict:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    pending = {k: v for k, v in RUNS.items() if k in labels}
    ready = {}
    while pending:
        for label, (src_rel, dst_name) in list(pending.items()):
            src = ROOT / src_rel
            if not src.exists():
                continue
            size = src.stat().st_size
            time.sleep(STABLE_SECONDS)
            if not src.exists() or src.stat().st_size != size:
                print(f"[wait] {label}: still being written, retrying", flush=True)
                continue
            dst = SNAP_DIR / dst_name
            shutil.copy2(src, dst)
            ready[label] = str(dst.relative_to(ROOT))
            del pending[label]
            print(f"[ready] {label} -> {ready[label]} ({size/1e6:.0f} MB)", flush=True)
        if pending:
            print(f"[wait] still waiting for: {', '.join(sorted(pending))}", flush=True)
            time.sleep(POLL_SECONDS)
    return ready


if __name__ == "__main__":
    labels = pop_only_flag()
    print(f"Waiting for epoch-0100 checkpoints of: {', '.join(labels)} ...", flush=True)
    encoders = wait_and_snapshot(labels)
    print("All checkpoints ready; starting the downstream matrix.\n", flush=True)

    for task in ("grasp_prediction", "slip_detection"):
        base.TASKS[task]["experiment"] = (
            f"brainco/ours_3d/task/{task}/temporal_w15_cls_d4_fe4"
        )
    # Ordered so the reference arm is the first column of the report.
    order = ("fdino_lr4e4_ep100", "fdino_lr1e4_ep100", "tip_lr4e4_ep100")
    base.ENCODERS = {k: encoders[k] for k in order if k in encoders}
    base.main()
