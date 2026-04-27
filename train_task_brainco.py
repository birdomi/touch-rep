# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
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
from tactile_ssl.data.d360.utils import get_weights, get_experiment_name, get_modality_tag, get_modality_used_tag
from tactile_ssl.trainer import Trainer
from tactile_ssl.utils.combined_dataset import CombinedDataset

logger = get_pylogger(__name__)
OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b))
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b)
OmegaConf.register_new_resolver("d360_expt_name", get_experiment_name)
OmegaConf.register_new_resolver("d360_modal_tag", get_modality_tag)
OmegaConf.register_new_resolver("d360_modal_used_tag", get_modality_used_tag)
OmegaConf.register_new_resolver("capitalize", lambda s: s.title())


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


def get_dataloader_brainco_grasp(cfg: DictConfig):
    """Create dataloaders for BrainCo grasp detection/prediction.

    Episodes are sorted by name and split into ``num_folds`` folds.
    Fold ``fold`` (0-indexed) is used as the validation set (~20% for num_folds=5),
    the remaining folds are used for training (~80%).

    Config keys (top-level):
        fold      (int, default 0): which fold to use as validation
        num_folds (int, default 5): total number of folds (5 → 80/20 split)
    """
    import random
    from pathlib import Path
    from collections import Counter

    data_cfg = cfg.data
    fold = int(cfg.get("fold", 0))
    num_folds = int(cfg.get("num_folds", 5))
    shuffle_seed = int(cfg.get("split_seed", 42))

    # train-val split: val=fold k, train=remaining
    val_fold = fold % num_folds

    # Instantiate the full dataset (all episodes)
    dataset = hydra.utils.instantiate(data_cfg.dataset)

    # Build window start index map (keyed by episode path)
    ep_window_start = {}
    current_idx = 0
    for ep_data in dataset.episode_data:
        ep_window_start[ep_data["path"]] = current_idx
        current_idx += len(ep_data["window_starts"])

    # Shuffle all episodes with fixed seed, then split into num_folds blocks
    all_episodes = list(dataset.episode_data)
    rng = random.Random(shuffle_seed)
    rng.shuffle(all_episodes)
    num_episodes = len(all_episodes)

    def _fold_range(k):
        fold_size = num_episodes // num_folds
        start = k * fold_size
        end = start + fold_size if k < num_folds - 1 else num_episodes
        return range(start, end)

    val_range = _fold_range(val_fold)

    train_indices, val_indices = [], []
    train_ep_names, val_ep_names = [], []

    for rank, ep_data in enumerate(all_episodes):
        num_windows = len(ep_data["window_starts"])
        start = ep_window_start[ep_data["path"]]
        window_range = list(range(start, start + num_windows))
        ep_name = Path(ep_data["path"]).parent.name + "/" + Path(ep_data["path"]).name
        if rank in val_range:
            val_indices.extend(window_range)
            val_ep_names.append(ep_name)
        else:
            train_indices.extend(window_range)
            train_ep_names.append(ep_name)

    lines = []
    lines.append(f"\n=== Episode Random Split (val_fold={val_fold}/{num_folds}, total={num_episodes} episodes, seed={shuffle_seed}) ===")
    lines.append(f"  Train: {len(train_ep_names)} episodes")
    for name in train_ep_names:
        lines.append(f"    [train] {name}")
    lines.append(f"  Val: {len(val_ep_names)} episodes")
    for name in val_ep_names:
        lines.append(f"    [val]   {name}")

    # ── Compute normalization stats from train windows ────────────────────
    lines.append("Computing normalization stats from training data ...")
    all_sensor = torch.stack([dataset.windows[i]["sensor"] for i in train_indices])
    all_poses  = torch.stack([dataset.windows[i]["sensor_poses"] for i in train_indices])

    is_cat_encoder = "brainco_cat" in str(cfg.task.model_encoder._target_)
    signal_source = all_sensor
    pose_source = all_poses

    NUM_CHANS = signal_source.shape[-1]
    sensor_mean_1d = torch.zeros(NUM_CHANS)
    sensor_std_1d  = torch.ones(NUM_CHANS)
    for c in range(NUM_CHANS):
        valid = signal_source[..., c][signal_source[..., c] >= 0].float()
        if valid.numel() > 0:
            sensor_mean_1d[c] = valid.mean()
            sensor_std_1d[c]  = valid.std().clamp(min=1e-6)

    if is_cat_encoder:
        signal_mean = sensor_mean_1d.clone()
        signal_std = sensor_std_1d.clone()
    else:
        NUM_SENSOR_TYPES = 1 # MultiSensorTransformer expects (NUM_SENSORS, in_chans)
        signal_mean = sensor_mean_1d.unsqueeze(0).expand(NUM_SENSOR_TYPES, -1).clone()
        signal_std  = sensor_std_1d.unsqueeze(0).expand(NUM_SENSOR_TYPES, -1).clone()

    pose_flat = pose_source.reshape(-1, 3).float()
    pos_mean  = pose_flat.mean(dim=0)
    pos_std   = pose_flat.std(dim=0).clamp(min=1e-6)

    # pos_mean  = torch.tensor([0.,0.,0.])  # Initialize with zeros
    # pos_std   = torch.tensor([0.015, 0.015, 0.015])  # Initialize with small values

    lines.append(f"[NormStats] signal_mean (per ch): {sensor_mean_1d.tolist()}")
    lines.append(f"[NormStats] signal_std  (per ch): {sensor_std_1d.tolist()}")
    lines.append(f"[NormStats] pos_mean:             {pos_mean.tolist()}")
    lines.append(f"[NormStats] pos_std:              {pos_std.tolist()}")

    dataset.computed_signal_mean = signal_mean
    dataset.computed_signal_std  = signal_std
    dataset.computed_pos_mean    = pos_mean
    dataset.computed_pos_std     = pos_std
    # ─────────────────────────────────────────────────────────────────────

    train_dset = data.Subset(dataset, train_indices)
    val_dset   = data.Subset(dataset, val_indices)

    # Class distribution — object_classification 데이터셋은 object_classes 속성으로 클래스명 제공
    if hasattr(dataset, "object_classes"):
        CLASS_NAMES = {i: name for i, name in enumerate(dataset.object_classes)}
    else:
        CLASS_NAMES = {0: "Fail", 1: "Success"}

    def _class_dist(indices):
        return Counter(dataset.windows[i]["label"].item() for i in indices)

    def _append_dist(lines, name, dist, total):
        lines.append(f"  {name}:")
        for cls in sorted(dist):
            cnt = dist[cls]
            lines.append(f"    [{cls}] {CLASS_NAMES.get(cls, cls):10s}: {cnt:5d}  ({100*cnt/total:.1f}%)")

    lines.append("=== Class Distribution ===")
    _append_dist(lines, "Train", _class_dist(train_indices), len(train_indices))
    _append_dist(lines, "Val  ", _class_dist(val_indices),   len(val_indices))
    lines.append("=" * 26 + "\n")

    # Print to console and save to txt file
    split_log = "\n".join(lines)
    print(split_log)

    split_log_path = Path(cfg.paths.output_dir) / f"split_fold{val_fold}.txt"
    split_log_path.write_text(split_log)
    print(f"[Split log saved to {split_log_path}]")

    # Initialize generator for budget splits
    g = torch.Generator()
    if cfg.get("seed"):
        g.manual_seed(cfg.seed)
    else:
        g.seed()

    # Apply data budget
    if hasattr(data_cfg, "train_data_budget") and data_cfg.train_data_budget < 1.0:
        budget_size = int(len(train_dset) * data_cfg.train_data_budget)
        train_dset, _ = data.random_split(train_dset, [budget_size, len(train_dset) - budget_size], generator=g)

    if hasattr(data_cfg, "val_data_budget") and data_cfg.val_data_budget < 1.0:
        budget_size = int(len(val_dset) * data_cfg.val_data_budget)
        val_dset, _ = data.random_split(val_dset, [budget_size, len(val_dset) - budget_size], generator=g)

    print(f"Total windows: {len(train_dset)} train, {len(val_dset)} val")

    train_loader_args = dict(cfg.data.train_dataloader)
    val_loader_args   = dict(cfg.data.val_dataloader)

    train_dataloader = data.DataLoader(train_dset, generator=g, **train_loader_args)
    val_dataloader   = data.DataLoader(val_dset,   **val_loader_args)
    return train_dataloader, val_dataloader


def get_dataloaders(cfg: DictConfig):
    data_cfg = cfg.data

    if "d360_contact" in data_cfg.sensor:
        train_dataloader, val_dataloader = get_dataloaders_d360_contact_based(cfg)
    elif "d360_classification" in data_cfg.sensor:
        train_dataloader, val_dataloader = get_dataloaders_d360_classification_based(cfg)
    elif "d360" in data_cfg.sensor:
        train_dataloader, val_dataloader = get_dataloaders_d360_based(cfg)
    elif data_cfg.sensor == "xela":
        train_dataloader, val_dataloader = get_dataloader_xela(cfg)
    elif data_cfg.sensor in ("brainco_grasp", "brainco_grasp_prediction", "brainco_grasp_multimodal",
                              "brainco_object_classification"):
        train_dataloader, val_dataloader = get_dataloader_brainco_grasp(cfg)
        return train_dataloader, val_dataloader, None
    else:
        raise NotImplementedError(f"Sensor type '{data_cfg.sensor}' not implemented yet.")
    return train_dataloader, val_dataloader, None


def attempt_resume(cfg: DictConfig):
    ckpt_path = None
    if os.path.exists(f"{cfg.paths.output_dir}/config.yaml") and cfg.resume_id:
        job_id = HydraConfig.get().job.id
        logger.info(f"Attempting to resume experiment with {cfg.resume_id}")
        if not os.path.exists(f"{cfg.paths.output_dir}/checkpoints/"):
            logger.warning(f"Unable to resume: No checkpoints found for experiment with id {job_id}")
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/wandb/"):
            logger.warning(f"Unable to resume: No wandb logs found for experiment with id {job_id}")
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/config.yaml"):
            logger.warning("Could not find a config.yaml file in the resume directory. Using the current config.")
            return False, cfg

        cfg = OmegaConf.load(f"{cfg.paths.output_dir}/config.yaml")

        ckpt_path = f"{cfg.paths.output_dir}/checkpoints/"
        OmegaConf.update(cfg, "ckpt_path", ckpt_path, force_add=True)
        experiment_name = cfg.experiment_name
        cfg.wandb.id = f"{job_id}_{experiment_name}"
        logger.info(
            f"Resuming experiment {job_id} with wandb_id: {cfg.wandb.id} from latest checkpoint at {cfg.ckpt_path}"
        )
        return True, cfg
    return False, cfg


def train(cfg: DictConfig):
    resume_state, cfg = attempt_resume(cfg)

    logger.info("Instantiating wandb ...")
    wandb = init_wandb(cfg.wandb)
    if not resume_state:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    _GLOBAL_SEED = cfg.seed
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info(f"Instantiating dataset & dataloaders for <{cfg.data.dataset._target_}>")
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(cfg)

    logger.info(f"Instantiating model <{cfg.task._target_}>")
    model = hydra.utils.instantiate(cfg.task)

    # Pass normalization stats to encoder if computed
    _ds = train_dataloader.dataset
    while hasattr(_ds, 'dataset'):
        _ds = _ds.dataset
    if hasattr(_ds, 'computed_signal_mean') and hasattr(model, 'model_encoder') and hasattr(model.model_encoder, 'update_stats'):
        model.model_encoder.update_stats(
            _ds.computed_signal_mean,
            _ds.computed_signal_std,
            _ds.computed_pos_mean,
            _ds.computed_pos_std,
        )
        logger.info("Encoder normalization stats updated from training data.")

    trainer = Trainer(wandb_logger=wandb, **cfg.trainer)

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)

    last_metrics = getattr(model, "last_val_metrics", {})
    best_metrics = getattr(model, "best_val_metrics", {})

    print(f"\n{'='*40}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*40}")
    print(f"  Last Epoch  — Accuracy: {last_metrics.get('accuracy', float('nan')):.4f}  F1: {last_metrics.get('f1', float('nan')):.4f}")
    print(f"  Best Epoch  — Accuracy: {best_metrics.get('accuracy', float('nan')):.4f}  F1: {best_metrics.get('f1', float('nan')):.4f}")
    print(f"{'='*40}\n")

    wandb.finish()

    return {
        "last": last_metrics,
        "best": best_metrics,
    }


# @hydra.main(version_base="1.3", config_path="config")
@hydra.main(version_base="1.3", config_path="config", config_name="default_task.yaml")
def main(cfg: DictConfig):
    """
    Main function to train the model
    """
    if cfg.get("all_split", False):
        num_folds = int(cfg.get("num_folds", 4))
        base_wandb_id = cfg.wandb.id
        base_checkpoint_dir = cfg.trainer.save_checkpoint_dir
        all_metrics = {}

        for fold in range(num_folds):
            print(f"\n{'='*60}")
            print(f"  K-FOLD: Training fold {fold + 1}/{num_folds}")
            print(f"{'='*60}")
            with open_dict(cfg):
                cfg.fold = fold
                cfg.wandb.id = f"{base_wandb_id}_fold{fold}"
                cfg.trainer.save_checkpoint_dir = f"{base_checkpoint_dir}_fold{fold}"

            metrics = train(cfg)
            all_metrics[fold] = metrics

        # Restore original wandb id (cosmetic)
        with open_dict(cfg):
            cfg.wandb.id = base_wandb_id

        # Print summary
        for tag, label in [("last", "Last Epoch"), ("best", "Best Epoch")]:
            print(f"\n{'='*60}")
            print(f"  K-FOLD CROSS-VALIDATION SUMMARY ({label})")
            print(f"{'='*60}")
            print(f"{'Fold':>6}  {'Accuracy':>10}  {'F1 Score':>10}")
            print(f"{'-'*34}")
            accuracies, f1s = [], []
            for fold in range(num_folds):
                m = all_metrics.get(fold, {}).get(tag, {})
                acc = m.get("accuracy", float("nan"))
                f1  = m.get("f1",       float("nan"))
                accuracies.append(acc)
                f1s.append(f1)
                print(f"{fold:>6}  {acc:>10.4f}  {f1:>10.4f}")
            print(f"{'-'*34}")
            print(f"{'Mean':>6}  {np.mean(accuracies):>10.4f}  {np.mean(f1s):>10.4f}")
            print(f"{'Std':>6}  {np.std(accuracies):>10.4f}  {np.std(f1s):>10.4f}")
            print(f"{'='*60}")

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
