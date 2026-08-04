"""
Vision-only grasp detection / slip detection training script.
Uses pre-trained ResNet18 on RGB frames from {episode}/colors/.
No tactile input.

Usage:
  python train_task_brainco_vision.py +experiment=brainco/task/grasp_detection/resnet18
  python train_task_brainco_vision.py +experiment=brainco/task/grasp_detection/resnet18 --all_split --num_folds 5

  # slip detection (episode-level K-fold, class-balanced train sampling)
  python train_task_brainco_vision.py +experiment=brainco/task/slip_detection/resnet18
  python train_task_brainco_vision.py +experiment=brainco/task/slip_detection/resnet18 --all_split --num_folds 4
"""

import os
import re
from pathlib import Path
from collections import Counter
from functools import partial

import hydra
import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig

import wandb
from lightning.fabric import seed_everything

from tactile_ssl.utils import get_local_rank
from tactile_ssl.utils.logging import get_pylogger, print_config_tree
from tactile_ssl.trainer import Trainer
from tactile_ssl.data.brainco_grasp_vision_dataset import BraincoGraspVisionDataset
from tactile_ssl.downstream_task.brainco_grasp_vision_sl import ResNet18GraspModule

# Sensors whose runs are configured through cfg.task and split at the episode
# level exactly like the tactile downstream scripts.
SLIP_VISION_SENSOR = "brainco_slip_detection_vision"
GRASP_W15_VISION_SENSOR = "brainco_grasp_prediction_w15_vision"
EPISODE_FOLD_VISION_SENSORS = {SLIP_VISION_SENSOR, GRASP_W15_VISION_SENSOR}

logger = get_pylogger(__name__)

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b))
OmegaConf.register_new_resolver("int_divide",   lambda a, b: a // b)
OmegaConf.register_new_resolver("capitalize",   lambda s: s.title())


# ── wandb ─────────────────────────────────────────────────────────────────

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


# ── sampling ──────────────────────────────────────────────────────────────

class _EqualClassSampler(data.Sampler[int]):
    """Sample the same number of items from every class each epoch."""

    def __init__(self, labels: torch.Tensor, generator: torch.Generator):
        self.generator = generator
        classes = labels.unique(sorted=True)
        if len(classes) < 2:
            raise ValueError(
                "Balanced sampling requires at least two classes in the train split"
            )
        self.class_indices = [
            torch.nonzero(labels == class_id, as_tuple=False).flatten()
            for class_id in classes
        ]
        self.samples_per_class = len(labels) // len(self.class_indices)
        self.num_samples = self.samples_per_class * len(self.class_indices)

    def __iter__(self):
        sampled = []
        for indices in self.class_indices:
            if self.samples_per_class <= len(indices):
                positions = torch.randperm(
                    len(indices), generator=self.generator
                )[: self.samples_per_class]
            else:
                positions = torch.randint(
                    len(indices),
                    (self.samples_per_class,),
                    generator=self.generator,
                )
            sampled.append(indices[positions])
        epoch_indices = torch.cat(sampled)
        order = torch.randperm(len(epoch_indices), generator=self.generator)
        return iter(epoch_indices[order].tolist())

    def __len__(self):
        return self.num_samples


def _window_labels(dataset: data.Dataset, indices) -> torch.Tensor:
    """Read window labels without decoding any images."""
    base, offsets = dataset, list(indices)
    while isinstance(base, data.Subset):
        offsets = [base.indices[i] for i in offsets]
        base = base.dataset
    return torch.tensor(
        [int(base.windows[i]["label"].item()) for i in offsets], dtype=torch.long
    )


def _balanced_train_loader(
    train_dataset: data.Dataset,
    data_cfg: DictConfig,
    generator: torch.Generator,
) -> data.DataLoader:
    """Build a train loader with optional equal-class sampling."""
    loader_cfg = dict(data_cfg.train_dataloader)
    if not data_cfg.get("balanced_sampling", False):
        return data.DataLoader(train_dataset, generator=generator, **loader_cfg)

    labels = _window_labels(train_dataset, range(len(train_dataset)))
    classes, counts = labels.unique(sorted=True, return_counts=True)
    sampler = _EqualClassSampler(labels, generator)
    loader_cfg.pop("shuffle", None)
    logger.info(
        "Balanced train sampling enabled: "
        f"counts={dict(zip(classes.tolist(), counts.tolist()))}, "
        f"sampled_per_class={sampler.samples_per_class}, "
        f"samples_per_epoch={len(sampler)}"
    )
    return data.DataLoader(train_dataset, sampler=sampler, **loader_cfg)


# ── dataloader ────────────────────────────────────────────────────────────

# K-fold changes only the train/val index split, never the dataset itself, so
# building it once per process avoids re-parsing every episode for each fold.
_DATASET_CACHE: dict = {}


def _instantiate_dataset_cached(dataset_cfg: DictConfig, **overrides):
    key = (OmegaConf.to_yaml(dataset_cfg, resolve=True), tuple(sorted(overrides.items())))
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = hydra.utils.instantiate(dataset_cfg, **overrides)
    else:
        logger.info("Reusing dataset built for an earlier fold.")
    return _DATASET_CACHE[key]


def get_dataloader_episode_fold_vision(cfg: DictConfig):
    """Episode-level K-fold split for RGB window datasets.

    Mirrors train_task_brainco_angle.py so a vision baseline sees the same
    episode partition as the tactile encoders: episodes are shuffled with
    ``split_seed`` and cut into contiguous folds, or — with
    ``fold_by_episode_index`` — stratified so fold k holds out episode k of
    every class (required when a class has only a handful of episodes).
    """
    import random

    data_cfg  = cfg.data
    fold      = int(cfg.get("fold", 0))
    num_folds = int(cfg.get("num_folds", 5))
    seed      = int(cfg.get("split_seed", cfg.get("seed", 42)))
    val_fold  = fold % num_folds
    stratified = bool(cfg.get("fold_by_episode_index", False))

    if data_cfg.get("dataset") is None:
        raise ValueError("data.dataset must be set for episode-fold vision tasks")

    # Augmentation must not leak into validation, so the two splits read from
    # separate instances that share an identical window layout.
    dataset = _instantiate_dataset_cached(data_cfg.dataset, augment=False)
    augment_train = bool(data_cfg.get("augment_train", False))
    train_source = (
        _instantiate_dataset_cached(data_cfg.dataset, augment=True)
        if augment_train else dataset
    )
    if len(train_source.windows) != len(dataset.windows):
        raise RuntimeError(
            "Augmented and evaluation datasets disagree on window count: "
            f"{len(train_source.windows)} vs {len(dataset.windows)}"
        )

    ep_window_start: dict = {}
    current = 0
    for ep in dataset.episode_data:
        ep_window_start[ep["path"]] = current
        current += len(ep["window_starts"])

    all_episodes = list(dataset.episode_data)
    n_ep = len(all_episodes)

    if stratified:
        by_class: dict = {}
        for ep in all_episodes:
            by_class.setdefault(Path(ep["path"]).parent.name, []).append(ep)
        ep_fold = {}
        for eps in by_class.values():
            eps.sort(key=lambda e: e["path"])
            for i, ep in enumerate(eps):
                ep_fold[ep["path"]] = i % num_folds
        is_val = lambda rank, ep: ep_fold[ep["path"]] == val_fold  # noqa: E731
    else:
        rng = random.Random(seed)
        rng.shuffle(all_episodes)

        def _fold_range(k):
            size = n_ep // num_folds
            start = k * size
            end   = start + size if k < num_folds - 1 else n_ep
            return range(start, end)

        val_range = _fold_range(val_fold)
        is_val = lambda rank, ep: rank in val_range  # noqa: E731

    train_indices, val_indices = [], []
    train_ep_names, val_ep_names = [], []

    for rank, ep in enumerate(all_episodes):
        start = ep_window_start[ep["path"]]
        wins  = list(range(start, start + len(ep["window_starts"])))
        name  = Path(ep["path"]).parent.name + "/" + Path(ep["path"]).name
        if is_val(rank, ep):
            val_indices.extend(wins);   val_ep_names.append(name)
        else:
            train_indices.extend(wins); train_ep_names.append(name)

    CLASS_NAMES = {
        int(label): str(name)
        for label, name in data_cfg.get("class_names", {0: "non-slip", 1: "slip"}).items()
    }

    split_kind = "by-episode-index" if stratified else f"shuffled, seed={seed}"
    lines = [
        f"\n=== Vision Episode Split  (val_fold={val_fold}/{num_folds}, "
        f"total={n_ep}, {split_kind}) ===",
        f"  Train: {len(train_ep_names)} episodes  (augment={augment_train})",
        *[f"    [train] {n}" for n in train_ep_names],
        f"  Val:   {len(val_ep_names)} episodes",
        *[f"    [val]   {n}" for n in val_ep_names],
        "=== Class Distribution ===",
    ]
    for tag, idx in [("Train", train_indices), ("Val", val_indices)]:
        dist = Counter(_window_labels(dataset, idx).tolist())
        lines.append(f"  {tag}:")
        for cls in sorted(dist):
            cnt = dist[cls]
            lines.append(
                f"    [{cls}] {CLASS_NAMES.get(cls, str(cls)):8s}: {cnt:5d}  "
                f"({100*cnt/len(idx):.1f}%)"
            )
    lines.append("=" * 26)

    split_log = "\n".join(lines)
    print(split_log)
    (Path(cfg.paths.output_dir) / f"split_fold{val_fold}_vision.txt").write_text(split_log)

    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 42)))

    train_dset = data.Subset(train_source, train_indices)
    val_dset   = data.Subset(dataset, val_indices)

    if data_cfg.get("train_data_budget", 1.0) < 1.0:
        budget = int(len(train_dset) * data_cfg.train_data_budget)
        train_dset, _ = data.random_split(train_dset, [budget, len(train_dset) - budget], generator=g)
    if data_cfg.get("val_data_budget", 1.0) < 1.0:
        budget = int(len(val_dset) * data_cfg.val_data_budget)
        val_dset, _ = data.random_split(val_dset, [budget, len(val_dset) - budget], generator=g)

    print(f"Total windows: {len(train_dset)} train, {len(val_dset)} val")

    train_loader = _balanced_train_loader(train_dset, data_cfg, g)
    val_loader   = data.DataLoader(val_dset, **dict(data_cfg.val_dataloader))
    return train_loader, val_loader


def get_dataloaders(cfg: DictConfig):
    if cfg.data.sensor in EPISODE_FOLD_VISION_SENSORS:
        return get_dataloader_episode_fold_vision(cfg)
    return get_dataloader_vision(cfg)


def get_dataloader_vision(cfg: DictConfig):
    """K-fold contiguous episode split for the vision dataset."""
    data_cfg = cfg.data
    fold      = int(cfg.get("fold", 0))
    num_folds = int(cfg.get("num_folds", 5))

    def _make_dataset(dataset_cfg=None):
        if dataset_cfg is not None:
            return hydra.utils.instantiate(dataset_cfg)
        return BraincoGraspVisionDataset(
            config=data_cfg,
            data_path=data_cfg.data_path,
            num_frames_per_sample=int(data_cfg.get("num_frames_per_sample", 4)),
            img_size=int(data_cfg.get("img_size", 224)),
            augment=bool(data_cfg.get("augment", False)),
        )

    dataset = _make_dataset(data_cfg.get("dataset"))

    if data_cfg.get("val_dataset") is not None:
        val_dataset = _make_dataset(data_cfg.val_dataset)

        train_indices = list(range(len(dataset.windows)))
        val_indices   = list(range(len(val_dataset.windows)))

        CLASS_NAMES = {0: "Fail", 1: "Success"}

        def _ep_names(dset):
            return [Path(ep["path"]).parent.name + "/" + Path(ep["path"]).name for ep in dset.episode_data]

        def _class_dist(dset, indices):
            return Counter(dset.windows[i]["label"].item() for i in indices)

        lines = [
            "\n=== Vision Explicit Train/Val Dataset Split ===",
            f"  Train: {len(dataset.episode_data)} episodes",
            *[f"    [train] {name}" for name in _ep_names(dataset)],
            f"  Val:   {len(val_dataset.episode_data)} episodes",
            *[f"    [val]   {name}" for name in _ep_names(val_dataset)],
            "=== Class Distribution ===",
        ]
        for tag, dset, indices in [("Train", dataset, train_indices), ("Val", val_dataset, val_indices)]:
            dist = _class_dist(dset, indices)
            lines.append(f"  {tag}:")
            for cls in sorted(dist):
                cnt = dist[cls]
                lines.append(f"    [{cls}] {CLASS_NAMES.get(cls, cls):10s}: {cnt:5d}  ({100*cnt/len(indices):.1f}%)")
        lines.append("=" * 26)

        split_log = "\n".join(lines)
        print(split_log)
        split_log_path = Path(cfg.paths.output_dir) / "split_explicit_val_vision.txt"
        split_log_path.write_text(split_log)

        train_dset = data.Subset(dataset, train_indices)
        val_dset   = data.Subset(val_dataset, val_indices)

        g = torch.Generator()
        g.manual_seed(cfg.seed) if cfg.get("seed") else g.seed()

        if data_cfg.get("train_data_budget", 1.0) < 1.0:
            n = int(len(train_dset) * data_cfg.train_data_budget)
            train_dset, _ = data.random_split(train_dset, [n, len(train_dset) - n], generator=g)

        if data_cfg.get("val_data_budget", 1.0) < 1.0:
            n = int(len(val_dset) * data_cfg.val_data_budget)
            val_dset, _ = data.random_split(val_dset, [n, len(val_dset) - n], generator=g)

        print(f"Total windows: {len(train_dset)} train, {len(val_dset)} val")

        train_loader = data.DataLoader(train_dset, **dict(data_cfg.train_dataloader))
        val_loader   = data.DataLoader(val_dset,   **dict(data_cfg.val_dataloader))
        return train_loader, val_loader

    # Sort episodes by episode number
    def _ep_sort_key(ep):
        m = re.search(r'(\d+)', Path(ep["path"]).name)
        return int(m.group(1)) if m else 0

    sorted_episodes = sorted(dataset.episode_data, key=_ep_sort_key)
    num_episodes    = len(sorted_episodes)

    # Contiguous block split
    fold_size = num_episodes // num_folds
    val_start = fold * fold_size
    val_end   = val_start + fold_size if fold < num_folds - 1 else num_episodes
    val_ep_set = set(range(val_start, val_end))

    # Map episode path → window start index
    ep_window_start = {}
    current_idx = 0
    for ep in dataset.episode_data:
        ep_window_start[ep["path"]] = current_idx
        current_idx += len(ep["window_starts"])

    train_indices, val_indices   = [], []
    train_ep_names, val_ep_names = [], []

    for rank, ep in enumerate(sorted_episodes):
        start      = ep_window_start[ep["path"]]
        win_range  = list(range(start, start + len(ep["window_starts"])))
        ep_name    = Path(ep["path"]).parent.name + "/" + Path(ep["path"]).name
        if rank in val_ep_set:
            val_indices.extend(win_range)
            val_ep_names.append(ep_name)
        else:
            train_indices.extend(win_range)
            train_ep_names.append(ep_name)

    CLASS_NAMES = {0: "Fail", 1: "Success"}

    print(f"\n=== Vision K-Fold Split (fold={fold}/{num_folds}, total={num_episodes} episodes) ===")
    print(f"  Train: {len(train_ep_names)} episodes")
    for name in train_ep_names:
        print(f"    [train] {name}")
    print(f"  Val: {len(val_ep_names)} episodes")
    for name in val_ep_names:
        print(f"    [val]   {name}")

    def _class_dist(indices):
        return Counter(dataset.windows[i]["label"].item() for i in indices)

    def _print_dist(name, dist, total):
        print(f"  {name}:")
        for cls in sorted(dist):
            cnt = dist[cls]
            print(f"    [{cls}] {CLASS_NAMES.get(cls, cls):10s}: {cnt:5d}  ({100*cnt/total:.1f}%)")

    print("=== Class Distribution ===")
    _print_dist("Train", _class_dist(train_indices), len(train_indices))
    _print_dist("Val  ", _class_dist(val_indices),   len(val_indices))
    print("=" * 26 + "\n")

    train_dset = data.Subset(dataset, train_indices)
    val_dset   = data.Subset(dataset, val_indices)

    g = torch.Generator()
    g.manual_seed(cfg.seed) if cfg.get("seed") else g.seed()

    if hasattr(data_cfg, "train_data_budget") and data_cfg.train_data_budget < 1.0:
        n = int(len(train_dset) * data_cfg.train_data_budget)
        train_dset, _ = data.random_split(train_dset, [n, len(train_dset) - n], generator=g)

    if hasattr(data_cfg, "val_data_budget") and data_cfg.val_data_budget < 1.0:
        n = int(len(val_dset) * data_cfg.val_data_budget)
        val_dset, _ = data.random_split(val_dset, [n, len(val_dset) - n], generator=g)

    print(f"Total windows: {len(train_dset)} train, {len(val_dset)} val")

    train_loader = data.DataLoader(train_dset, **dict(data_cfg.train_dataloader))
    val_loader   = data.DataLoader(val_dset,   **dict(data_cfg.val_dataloader))
    return train_loader, val_loader


# ── model ─────────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig):
    """Window runs are configured through cfg.task; legacy grasp uses top-level keys."""
    if cfg.data.sensor in EPISODE_FOLD_VISION_SENSORS:
        return hydra.utils.instantiate(cfg.task)

    optim_cfg = partial(
        torch.optim.AdamW,
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler_cfg = partial(
        torch.optim.lr_scheduler.CosineAnnealingLR,
        T_max=cfg.trainer.max_epochs,
    ) if cfg.get("use_scheduler", True) else None

    return ResNet18GraspModule(
        pretrained=cfg.get("pretrained", True),
        freeze_backbone=cfg.get("freeze_backbone", False),
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        num_classes=int(cfg.get("num_classes", 2)),
    )


# ── train ─────────────────────────────────────────────────────────────────

def train(cfg: DictConfig) -> dict:
    if os.path.exists(f"{cfg.paths.output_dir}/config.yaml") and cfg.resume_id:
        logger.info("Resume not supported for vision script — starting fresh.")

    logger.info("Instantiating wandb ...")
    wb = init_wandb(cfg.wandb)
    wb.config.update(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    logger.info("Building vision dataloader ...")
    train_loader, val_loader = get_dataloaders(cfg)

    logger.info("Building ResNet18 model ...")
    model = build_model(cfg)

    trainer = Trainer(wandb_logger=wb, **cfg.trainer)
    trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.ckpt_path)
    wb.finish()

    last = getattr(model, "last_val_metrics", {})
    best = getattr(model, "best_val_metrics", {})
    epoch_avg = getattr(model, "epoch_avg_val_metrics", {})

    def _line(tag, metrics):
        # Slip modules report balanced accuracy; grasp modules report raw accuracy.
        acc = metrics.get("balanced_accuracy", metrics.get("accuracy", float("nan")))
        return (
            f"  {tag} — Acc: {acc:.4f}"
            f"  F1: {metrics.get('f1', float('nan')):.4f}"
        )

    print(f"\n{'='*40}")
    print("  VISION TRAINING COMPLETE")
    print(f"{'='*40}")
    print(_line("Last    ", last))
    print(_line("Best    ", best))
    if epoch_avg:
        print(_line("EpochAvg", epoch_avg))
        print(f"  EpochAvg epochs: {epoch_avg.get('epochs', [])}")
    print(f"{'='*40}\n")

    return last


# ── main ──────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="config", config_name="default_task.yaml")
def main(cfg: DictConfig):
    if cfg.data.get("val_dataset") is not None and cfg.get("all_split", False):
        logger.warning("data.val_dataset is set; ignoring all_split because validation is explicit.")
        train(cfg)
        return

    if cfg.get("all_split", False):
        num_folds     = int(cfg.get("num_folds", 5))
        base_wandb_id = cfg.wandb.id
        base_ckpt_dir = cfg.trainer.save_checkpoint_dir
        all_metrics   = {}

        for fold in range(num_folds):
            print(f"\n{'='*60}")
            print(f"  K-FOLD: Training fold {fold + 1}/{num_folds}")
            print(f"{'='*60}")
            with open_dict(cfg):
                cfg.fold                          = fold
                cfg.wandb.id                      = f"{base_wandb_id}_fold{fold}"
                cfg.trainer.save_checkpoint_dir   = f"{base_ckpt_dir}_fold{fold}"

            all_metrics[fold] = train(cfg)

        with open_dict(cfg):
            cfg.wandb.id = base_wandb_id

        print(f"\n{'='*60}")
        print("  K-FOLD CROSS-VALIDATION SUMMARY (Last Epoch) — Vision")
        print(f"{'='*60}")
        print(f"{'Fold':>6}  {'Accuracy':>10}  {'F1 Score':>10}")
        print(f"{'-'*34}")
        accuracies, f1s = [], []
        for fold in range(num_folds):
            m   = all_metrics.get(fold, {})
            acc = m.get("balanced_accuracy", m.get("accuracy", float("nan")))
            f1  = m.get("f1",       float("nan"))
            accuracies.append(acc)
            f1s.append(f1)
            print(f"{fold:>6}  {acc:>10.4f}  {f1:>10.4f}")
        print(f"{'-'*34}")
        print(f"{'Mean':>6}  {np.mean(accuracies):>10.4f}  {np.mean(f1s):>10.4f}")
        print(f"{'Std':>6}  {np.std(accuracies):>10.4f}  {np.std(f1s):>10.4f}")
        print(f"{'='*60}\n")
    else:
        train(cfg)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--all_split",   action="store_true", default=False)
    parser.add_argument("--num_folds",   type=int, default=None)
    parser.add_argument("--pretrained",  action="store_true", default=True)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    if known.all_split:
        sys.argv.append("all_split=true")
    if known.num_folds is not None:
        sys.argv.append(f"num_folds={known.num_folds}")
    if not known.pretrained:
        sys.argv.append("pretrained=false")
    if known.freeze_backbone:
        sys.argv.append("freeze_backbone=true")

    torch.set_float32_matmul_precision("medium")
    main()
