#!/usr/bin/env python
"""Grasp + slip at 100 downstream epochs, seeds 1000-1002, four arms.

The task configs now carry max_epochs 100 and
val_average_epochs [10..100], so EpochAvg spans the whole run rather than its
first half. There is no downstream LR scheduler (scheduler_cfg is null), so the
extra epochs simply continue training at a fixed lr 1e-4.

Arms (fdino_lr1e4 dropped -- it trailed the other two prox arms on both tasks):
    scratch             no pretraining
    tip_lr4e4_ep500     10 fingertips, lr 4e-4
    fdino_lr4e4_ep500   42 joints, lr 4e-4
    jointonly_ep500     no force input at all

Two axes have to be selected because each needs its own task config and fold
count:

    --task grasp   grasp_prediction_0611, 4-fold (590 episodes, random split)
    --task slip    slip_data_v3, 3-fold stratified (pass --num-folds 3)

    --arm force        temporal_w15_cls_d4_fe4[_v3]
    --arm jointonly    temporal_w15_cls_d4_fe4[_v3]_jointonly (input_streams: pos)

The jointonly split is mandatory, not cosmetic: that checkpoint contains
sensor_embed/sensor_block tensors that were never fed and never trained, so the
plain config would load them and route real force through an untrained path.

split_seed stays at 42, so every arm and seed sees the same episode split and
the seeds vary only model init and batch order.

Usage:
    python scripts/run_matrix_ep100.py --task grasp --arm force     --gpus 1,3 --seeds 1000,1001,1002 --num-folds 4
    python scripts/run_matrix_ep100.py --task grasp --arm jointonly --gpus 1,3 --seeds 1000,1001,1002 --num-folds 4
    python scripts/run_matrix_ep100.py --task slip  --arm force     --gpus 1,3 --seeds 1000,1001,1002 --num-folds 3
    python scripts/run_matrix_ep100.py --task slip  --arm jointonly --gpus 1,3 --seeds 1000,1001,1002 --num-folds 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seed_ood_matrix as base  # noqa: E402

SNAP = "checkpoints/queued_w15_snapshots"
FORCE_ARMS = {
    "scratch": "null",
    "tip_lr4e4_ep500": f"{SNAP}/temporal_d4fe4cls_tip_fdino_lr4e4_ep0500.ckpt",
    "fdino_lr4e4_ep500": f"{SNAP}/temporal_d4fe4cls_fdino_lr4e4_ep0500.ckpt",
}
JOINTONLY_ARMS = {
    "jointonly_ep500": f"{SNAP}/temporal_d4fe4cls_jointonly_lr4e4_ep0500.ckpt",
    # The pose-only architecture with no pretraining. Without this cell,
    # jointonly_ep500's margin over `scratch` conflates two things: the value of
    # pretraining on kinematics, and the effect of dropping the sensor stream at
    # all (a pose-only model has 10 patch tokens instead of 52 and no
    # randomly-initialised sensor projection to fight through). Select it with
    # --encoders jointonly_scratch.
    "jointonly_scratch": "null",
}

# (task, arm) -> (TASKS key, experiment config)
CONFIGS = {
    ("grasp", "force"): (
        "grasp_prediction",
        "brainco/ours_3d/task/grasp_prediction/temporal_w15_cls_d4_fe4"),
    ("grasp", "jointonly"): (
        "grasp_prediction",
        "brainco/ours_3d/task/grasp_prediction/temporal_w15_cls_d4_fe4_jointonly"),
    ("slip", "force"): (
        "slip_detection",
        "brainco/ours_3d/task/slip_detection/temporal_w15_cls_d4_fe4_v3"),
    ("slip", "jointonly"): (
        "slip_detection",
        "brainco/ours_3d/task/slip_detection/temporal_w15_cls_d4_fe4_v3_jointonly"),
}


def pop(flag, valid):
    if flag not in sys.argv:
        raise SystemExit(f"{flag} is required (one of {valid})")
    i = sys.argv.index(flag)
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    if v not in valid:
        raise SystemExit(f"Unknown {flag} {v!r}; valid: {valid}")
    return v


if __name__ == "__main__":
    task = pop("--task", ["grasp", "slip"])
    arm = pop("--arm", ["force", "jointonly"])
    key, experiment = CONFIGS[(task, arm)]

    # The default batch 256 leaves grasp with only 4 optimizer steps per epoch
    # (886 train windows), i.e. 400 steps over the whole run. A randomly
    # initialised pose-only model sits on a ~150-step loss plateau before it
    # starts fitting, so it never escapes within that budget. --batch-size
    # trades batch width for step count at the same epoch count.
    # --posnorm standardises the pose stream (see the *_posnorm task config).
    # Only valid for the jointonly arm and only for models trained under the
    # same setting; the existing checkpoints expect raw-scale pose.
    if "--posnorm" in sys.argv:
        sys.argv.remove("--posnorm")
        if task != "grasp":
            raise SystemExit("--posnorm currently only has grasp configs")
        experiment += "_posnorm"
        print(f"posnorm -> {experiment}")

    if "--batch-size" in sys.argv:
        i = sys.argv.index("--batch-size")
        bs = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        base.EXTRA_OVERRIDES = [f"data.train_dataloader.batch_size={bs}"]
        print(f"batch_size override -> {bs}")

    for other in [k for k in base.TASKS if k != key]:
        base.TASKS.pop(other)
    base.TASKS[key]["experiment"] = experiment
    base.ENCODERS = FORCE_ARMS if arm == "force" else JOINTONLY_ARMS
    print(f"task={key}  arm={arm}  experiment={experiment}")
    print(f"encoders={list(base.ENCODERS)}")
    base.main()
