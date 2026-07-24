"""train_task_brainco_angle.py

Downstream grasp prediction using AngleTransformer encoder with either
BraincoAngleGraspDataset (joint_contact + finger_angles) or
BraincoXYZGraspDataset (joint_contact + finger_xyz).

Usage:
    # BrainCo tactile + fingertip XYZ with ours_3d pretraining_all
    XFORMERS_DISABLED=TRUE python train_task_brainco_angle.py \\
        +experiment=brainco/ours_3d/task/grasp_prediction/dinov2_all_rope.yaml

    XFORMERS_DISABLED=TRUE python train_task_brainco_angle.py \\
        +experiment=brainco/ours_vectors/task/grasp_prediction/dinov2_multi_scratch.yaml

    # K-fold
    python train_task_brainco_angle.py \\
        +experiment=brainco/ours_vectors/task/grasp_prediction/dinov2_multi_scratch.yaml \\
        --all_split --num_folds 5
"""

import os
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.utils.data as data
import wandb
from hydra.core.hydra_config import HydraConfig
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict

from tactile_ssl.trainer import Trainer
from tactile_ssl.utils import get_local_rank
from tactile_ssl.utils.logging import get_pylogger, print_config_tree

logger = get_pylogger(__name__)

OmegaConf.register_new_resolver("int_multiply",        lambda a, b: int(a * b))
OmegaConf.register_new_resolver("int_divide",          lambda a, b: a // b)
OmegaConf.register_new_resolver("capitalize",          lambda s: s.title())


# ── wandb ──────────────────────────────────────────────────────────────────────

def init_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        dir=cfg.save_dir,
        id=f"{cfg.id}_{get_local_rank()}",
        group=cfg.group,
        tags=cfg.tags,
        notes=cfg.notes,
    )
    return wandb


# ── dataloader ────────────────────────────────────────────────────────────────

def get_dataloader_brainco_angle_grasp(cfg: DictConfig):
    """K-fold episode split for BrainCo angle/XYZ downstream datasets."""
    import random

    data_cfg   = cfg.data
    fold       = int(cfg.get("fold", 0))
    num_folds  = int(cfg.get("num_folds", 5))
    seed       = int(cfg.get("split_seed", 42))
    val_fold   = fold % num_folds

    dataset = hydra.utils.instantiate(data_cfg.dataset)

    if data_cfg.get("val_dataset") is not None:
        val_dataset = hydra.utils.instantiate(data_cfg.val_dataset)

        train_indices = list(range(len(dataset.windows)))
        val_indices   = list(range(len(val_dataset.windows)))

        # Contact values are already min-max scaled in the dataset. Keep the
        # encoder transform as identity instead of applying train-set z-score.
        num_channels = dataset.windows[train_indices[0]]["joint_contact"].shape[-1]
        signal_mean = torch.zeros(num_channels)
        signal_std = torch.ones(num_channels)

        dataset.computed_signal_mean = signal_mean
        dataset.computed_signal_std  = signal_std
        val_dataset.computed_signal_mean = signal_mean
        val_dataset.computed_signal_std  = signal_std

        CLASS_NAMES = {
            int(label): str(name)
            for label, name in data_cfg.get("class_names", {0: "Fail", 1: "Success"}).items()
        }

        def _ep_names(dset):
            return [Path(ep["path"]).parent.name + "/" + Path(ep["path"]).name for ep in dset.episode_data]

        def _dist(dset, indices):
            return Counter(dset.windows[i]["label"].item() for i in indices)

        lines = [
            "\n=== Explicit Train/Val Dataset Split ===",
            f"  Train: {len(dataset.episode_data)} episodes",
            *[f"    [train] {n}" for n in _ep_names(dataset)],
            f"  Val:   {len(val_dataset.episode_data)} episodes",
            *[f"    [val]   {n}" for n in _ep_names(val_dataset)],
            "Using dataset min-max scaling only (no train mean/std) ...",
            f"[NormStats] signal_mean: {signal_mean.tolist()}",
            f"[NormStats] signal_std:  {signal_std.tolist()}",
            "  (position stream: NOT normalized)",
            "=== Class Distribution ===",
        ]
        for tag, dset, idx in [("Train", dataset, train_indices), ("Val", val_dataset, val_indices)]:
            d = _dist(dset, idx)
            lines.append(f"  {tag}:")
            for cls in sorted(d):
                cnt = d[cls]
                lines.append(f"    [{cls}] {CLASS_NAMES.get(cls, str(cls)):8s}: {cnt:5d}  ({100*cnt/len(idx):.1f}%)")
        lines.append("=" * 26)

        split_log = "\n".join(lines)
        print(split_log)
        log_path = Path(cfg.paths.output_dir) / "split_explicit_val.txt"
        log_path.write_text(split_log)

        g = torch.Generator()
        g.manual_seed(cfg.get("seed", 42))

        train_dset = data.Subset(dataset, train_indices)
        val_dset   = data.Subset(val_dataset, val_indices)

        if data_cfg.get("train_data_budget", 1.0) < 1.0:
            budget = int(len(train_dset) * data_cfg.train_data_budget)
            train_dset, _ = data.random_split(train_dset, [budget, len(train_dset) - budget], generator=g)
        if data_cfg.get("val_data_budget", 1.0) < 1.0:
            budget = int(len(val_dset) * data_cfg.val_data_budget)
            val_dset, _ = data.random_split(val_dset, [budget, len(val_dset) - budget], generator=g)

        print(f"Total windows: {len(train_dset)} train, {len(val_dset)} val")

        train_loader = data.DataLoader(train_dset, generator=g, **dict(data_cfg.train_dataloader))
        val_loader   = data.DataLoader(val_dset,                **dict(data_cfg.val_dataloader))
        return train_loader, val_loader

    # ── Episode → window index map ────────────────────────────────────────────
    ep_window_start: dict = {}
    current = 0
    for ep in dataset.episode_data:
        ep_window_start[ep["path"]] = current
        current += len(ep["window_starts"])

    all_episodes = list(dataset.episode_data)
    rng = random.Random(seed)
    rng.shuffle(all_episodes)
    n_ep = len(all_episodes)

    def _fold_range(k):
        sz = n_ep // num_folds
        s  = k * sz
        e  = s + sz if k < num_folds - 1 else n_ep
        return range(s, e)

    val_range = _fold_range(val_fold)
    train_indices, val_indices = [], []
    train_ep_names, val_ep_names = [], []

    for rank, ep in enumerate(all_episodes):
        n_win = len(ep["window_starts"])
        start = ep_window_start[ep["path"]]
        wins  = list(range(start, start + n_win))
        name  = Path(ep["path"]).parent.name + "/" + Path(ep["path"]).name
        if rank in val_range:
            val_indices.extend(wins);   val_ep_names.append(name)
        else:
            train_indices.extend(wins); train_ep_names.append(name)

    # ── Logging ───────────────────────────────────────────────────────────────
    lines = [
        f"\n=== Episode Split  (val_fold={val_fold}/{num_folds}, "
        f"total={n_ep}, seed={seed}) ===",
        f"  Train: {len(train_ep_names)} episodes",
        *[f"    [train] {n}" for n in train_ep_names],
        f"  Val:   {len(val_ep_names)} episodes",
        *[f"    [val]   {n}" for n in val_ep_names],
    ]

    # Contact values are already min-max scaled in the dataset. Keep the
    # encoder transform as identity instead of applying train-set z-score.
    lines.append("Using dataset min-max scaling only (no train mean/std) ...")
    num_channels = dataset.windows[train_indices[0]]["joint_contact"].shape[-1]
    signal_mean = torch.zeros(num_channels)
    signal_std = torch.ones(num_channels)

    lines.append(f"[NormStats] signal_mean: {signal_mean.tolist()}")
    lines.append(f"[NormStats] signal_std:  {signal_std.tolist()}")
    lines.append("  (position stream: NOT normalized)")

    dataset.computed_signal_mean = signal_mean
    dataset.computed_signal_std  = signal_std

    # ── Class distribution ────────────────────────────────────────────────────
    CLASS_NAMES = {
        int(label): str(name)
        for label, name in data_cfg.get("class_names", {0: "Fail", 1: "Success"}).items()
    }

    def _dist(indices):
        return Counter(dataset.windows[i]["label"].item() for i in indices)

    lines.append("=== Class Distribution ===")
    for tag, idx in [("Train", train_indices), ("Val", val_indices)]:
        d = _dist(idx)
        lines.append(f"  {tag}:")
        for cls in sorted(d):
            cnt = d[cls]
            lines.append(f"    [{cls}] {CLASS_NAMES.get(cls, str(cls)):8s}: {cnt:5d}  ({100*cnt/len(idx):.1f}%)")
    lines.append("=" * 26)

    split_log = "\n".join(lines)
    print(split_log)
    log_path = Path(cfg.paths.output_dir) / f"split_fold{val_fold}.txt"
    log_path.write_text(split_log)

    # ── Subsets + data budget ─────────────────────────────────────────────────
    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42))

    train_dset = data.Subset(dataset, train_indices)
    val_dset   = data.Subset(dataset, val_indices)

    if data_cfg.get("train_data_budget", 1.0) < 1.0:
        budget = int(len(train_dset) * data_cfg.train_data_budget)
        train_dset, _ = data.random_split(train_dset, [budget, len(train_dset) - budget], generator=g)
    if data_cfg.get("val_data_budget", 1.0) < 1.0:
        budget = int(len(val_dset) * data_cfg.val_data_budget)
        val_dset, _ = data.random_split(val_dset, [budget, len(val_dset) - budget], generator=g)

    print(f"Total windows: {len(train_dset)} train, {len(val_dset)} val")

    train_loader = data.DataLoader(train_dset, generator=g, **dict(data_cfg.train_dataloader))
    val_loader   = data.DataLoader(val_dset,                **dict(data_cfg.val_dataloader))
    return train_loader, val_loader


def get_dataloader_brainco_angle_oc(cfg: DictConfig):
    """K-fold episode split for BraincoAngleObjectClassificationDataset."""
    import random

    data_cfg  = cfg.data
    fold      = int(cfg.get("fold", 0))
    num_folds = int(cfg.get("num_folds", 5))
    seed      = int(cfg.get("split_seed", 42))
    val_fold  = fold % num_folds

    dataset = hydra.utils.instantiate(data_cfg.dataset)
    num_classes = len(dataset.classes)

    # ── Patch num_classes into task config ────────────────────────────────────
    with open_dict(cfg):
        cfg.task.model_task.num_classes = num_classes

    # ── Episode → window index map ────────────────────────────────────────────
    ep_window_start: dict = {}
    current = 0
    for ep in dataset.episode_data:
        ep_window_start[ep["path"]] = current
        current += len(ep["window_starts"])

    all_episodes = list(dataset.episode_data)
    rng = random.Random(seed)
    rng.shuffle(all_episodes)
    n_ep = len(all_episodes)

    def _fold_range(k):
        sz = n_ep // num_folds
        s  = k * sz
        e  = s + sz if k < num_folds - 1 else n_ep
        return range(s, e)

    val_range = _fold_range(val_fold)
    train_indices, val_indices = [], []
    train_ep_names, val_ep_names = [], []

    for rank, ep in enumerate(all_episodes):
        n_win = len(ep["window_starts"])
        start = ep_window_start[ep["path"]]
        wins  = list(range(start, start + n_win))
        name  = ep.get("class", "?") + "/" + Path(ep["path"]).name
        if rank in val_range:
            val_indices.extend(wins);   val_ep_names.append(name)
        else:
            train_indices.extend(wins); train_ep_names.append(name)

    # Contact values are already min-max scaled in the dataset.
    num_channels = dataset.windows[train_indices[0]]["joint_contact"].shape[-1]
    signal_mean = torch.zeros(num_channels)
    signal_std = torch.ones(num_channels)

    dataset.computed_signal_mean = signal_mean
    dataset.computed_signal_std  = signal_std

    lines = [
        f"\n=== OC Episode Split  (val_fold={val_fold}/{num_folds}, "
        f"total={n_ep}, seed={seed}, classes={dataset.classes}) ===",
        f"  Train: {len(train_ep_names)} episodes  ({len(train_indices)} windows)",
        f"  Val:   {len(val_ep_names)} episodes  ({len(val_indices)} windows)",
        f"[NormStats] signal_mean: {signal_mean.tolist()}",
        f"[NormStats] signal_std:  {signal_std.tolist()}",
    ]
    split_log = "\n".join(lines)
    print(split_log)
    log_path = Path(cfg.paths.output_dir) / f"split_fold{val_fold}.txt"
    log_path.write_text(split_log)

    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42))
    train_dset = data.Subset(dataset, train_indices)
    val_dset   = data.Subset(dataset, val_indices)

    if data_cfg.get("train_data_budget", 1.0) < 1.0:
        budget = int(len(train_dset) * data_cfg.train_data_budget)
        train_dset, _ = data.random_split(train_dset, [budget, len(train_dset) - budget], generator=g)
    if data_cfg.get("val_data_budget", 1.0) < 1.0:
        budget = int(len(val_dset) * data_cfg.val_data_budget)
        val_dset, _ = data.random_split(val_dset, [budget, len(val_dset) - budget], generator=g)

    print(f"Total windows: {len(train_dset)} train, {len(val_dset)} val")

    train_loader = data.DataLoader(train_dset, generator=g, **dict(data_cfg.train_dataloader))
    val_loader   = data.DataLoader(val_dset,                **dict(data_cfg.val_dataloader))
    return train_loader, val_loader


def get_dataloaders(cfg: DictConfig):
    if cfg.data.sensor in {
        "brainco_angle_grasp_prediction",
        "brainco_xyz_grasp_prediction",
        "brainco_xyz_slip_detection",
    }:
        return get_dataloader_brainco_angle_grasp(cfg), None
    if cfg.data.sensor == "brainco_angle_object_classification":
        return get_dataloader_brainco_angle_oc(cfg), None
    raise NotImplementedError(f"Unknown sensor: {cfg.data.sensor}")


def _balanced_class_weights(train_dataset, num_classes: int) -> torch.Tensor:
    """Compute N/(C*N_c) weights from the training subset only."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for index in range(len(train_dataset)):
        sample = train_dataset[index]
        label = int(torch.as_tensor(sample["label"]).item())
        if not 0 <= label < num_classes:
            raise ValueError(
                f"Training label {label} is outside [0, {num_classes - 1}]"
            )
        counts[label] += 1

    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(
            "Cannot compute balanced class weights because the training split "
            f"contains no samples for classes {missing}"
        )

    weights = counts.sum().float() / (num_classes * counts.float())
    logger.info(
        "Balanced cross-entropy weights from training split: "
        f"counts={counts.tolist()}, weights={weights.tolist()}"
    )
    return weights


# ── resume ─────────────────────────────────────────────────────────────────────

def attempt_resume(cfg: DictConfig):
    if os.path.exists(f"{cfg.paths.output_dir}/config.yaml") and cfg.resume_id:
        job_id = HydraConfig.get().job.id
        for check in ["checkpoints/", "wandb/", "config.yaml"]:
            if not os.path.exists(f"{cfg.paths.output_dir}/{check}"):
                logger.warning(f"Cannot resume: {check} not found")
                return False, cfg
        cfg = OmegaConf.load(f"{cfg.paths.output_dir}/config.yaml")
        OmegaConf.update(cfg, "ckpt_path", f"{cfg.paths.output_dir}/checkpoints/", force_add=True)
        cfg.wandb.id = f"{job_id}_{cfg.experiment_name}"
        logger.info(f"Resuming from {cfg.ckpt_path}")
        return True, cfg
    return False, cfg


# ── train ─────────────────────────────────────────────────────────────────────

def train(cfg: DictConfig):
    resume_state, cfg = attempt_resume(cfg)

    logger.info("Instantiating wandb ...")
    wb = init_wandb(cfg.wandb)
    if not resume_state:
        wb.config.update(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True

    logger.info(f"Instantiating dataset for <{cfg.data.dataset._target_}>")
    (train_loader, val_loader), _ = get_dataloaders(cfg)

    if cfg.task.get("class_weights") == "balanced":
        num_classes = int(cfg.task.model_task.num_classes)
        class_weights = _balanced_class_weights(
            train_loader.dataset, num_classes=num_classes
        )
        with open_dict(cfg.task):
            cfg.task.class_weights = class_weights.tolist()

    logger.info(f"Instantiating model <{cfg.task._target_}>")
    model = hydra.utils.instantiate(cfg.task)

    # ── Inject normalization stats into encoder ───────────────────────────────
    _ds = train_loader.dataset
    while hasattr(_ds, "dataset"):
        _ds = _ds.dataset
    if hasattr(_ds, "computed_signal_mean") and hasattr(model, "model_encoder"):
        enc = model.model_encoder
        if hasattr(enc, "update_stats"):
            enc.update_stats(
                _ds.computed_signal_mean,
                _ds.computed_signal_std,
                # angles are not normalized → no pos stats
            )
            logger.info("Encoder signal normalization stats updated.")

    trainer = Trainer(wandb_logger=wb, **cfg.trainer)
    trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.ckpt_path)

    last = getattr(model, "last_val_metrics", {})
    best = getattr(model, "best_val_metrics", {})

    print(f"\n{'='*40}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*40}")
    print(f"  Last  — Acc: {last.get('accuracy', float('nan')):.4f}  F1: {last.get('f1', float('nan')):.4f}")
    print(f"  Best  — Acc: {best.get('accuracy', float('nan')):.4f}  F1: {best.get('f1', float('nan')):.4f}")
    print(f"{'='*40}\n")

    wb.finish()
    return {"last": last, "best": best}


# ── main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="config", config_name="default_task.yaml")
def main(cfg: DictConfig):
    if cfg.data.get("val_dataset") is not None and cfg.get("all_split", False):
        logger.warning("data.val_dataset is set; ignoring all_split because validation is explicit.")
        train(cfg)
        return

    if cfg.get("all_split", False):
        num_folds = int(cfg.get("num_folds", 5))
        base_id   = cfg.wandb.id
        base_ckpt = cfg.trainer.save_checkpoint_dir
        all_metrics: dict = {}

        for fold in range(num_folds):
            print(f"\n{'='*60}\n  K-FOLD: fold {fold+1}/{num_folds}\n{'='*60}")
            with open_dict(cfg):
                cfg.fold = fold
                cfg.wandb.id                    = f"{base_id}_fold{fold}"
                cfg.trainer.save_checkpoint_dir = f"{base_ckpt}_fold{fold}"
            all_metrics[fold] = train(cfg)

        with open_dict(cfg):
            cfg.wandb.id = base_id

        for tag, label in [("last", "Last Epoch"), ("best", "Best Epoch")]:
            print(f"\n{'='*60}\n  K-FOLD SUMMARY ({label})\n{'='*60}")
            print(f"{'Fold':>6}  {'Accuracy':>10}  {'F1 Score':>10}")
            print("-" * 34)
            accs, f1s = [], []
            for f in range(num_folds):
                m   = all_metrics.get(f, {}).get(tag, {})
                acc = m.get("accuracy", float("nan"))
                f1  = m.get("f1",       float("nan"))
                accs.append(acc); f1s.append(f1)
                print(f"{f:>6}  {acc:>10.4f}  {f1:>10.4f}")
            print("-" * 34)
            print(f"{'Mean':>6}  {np.mean(accs):>10.4f}  {np.mean(f1s):>10.4f}")
            print(f"{'Std':>6}  {np.std(accs):>10.4f}  {np.std(f1s):>10.4f}")
    else:
        train(cfg)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--all_split", action="store_true", default=False)
    parser.add_argument("--num_folds", type=int, default=None)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    if known.all_split:
        sys.argv.append("all_split=true")
    if known.num_folds is not None:
        sys.argv.append(f"num_folds={known.num_folds}")

    torch.set_float32_matmul_precision("medium")
    main()
