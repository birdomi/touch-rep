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
    import re
    from pathlib import Path
    from collections import Counter

    data_cfg = cfg.data
    fold = int(cfg.get("fold", 0))
    num_folds = int(cfg.get("num_folds", 5))

    # Instantiate the full dataset (all episodes)
    dataset = hydra.utils.instantiate(data_cfg.dataset)

    # Sort episodes by episode number extracted from path name
    def _ep_sort_key(ep_data):
        match = re.search(r'(\d+)', Path(ep_data["path"]).name)
        return int(match.group(1)) if match else 0

    sorted_episodes = sorted(dataset.episode_data, key=_ep_sort_key)
    num_episodes = len(sorted_episodes)

    # Contiguous block split: divide sorted episodes into num_folds equal blocks.
    # fold k uses block k as val, the rest as train.
    fold_size = num_episodes // num_folds
    val_start = fold * fold_size
    val_end   = val_start + fold_size if fold < num_folds - 1 else num_episodes
    val_ep_set = set(range(val_start, val_end))

    # Build window index lists using sorted episode order
    # We need to map sorted episodes back to indices in dataset.windows.
    # dataset.windows is built in the same order as dataset.episode_data,
    # so we first build a path→start_idx map.
    ep_window_start = {}
    current_idx = 0
    for ep_data in dataset.episode_data:
        ep_window_start[ep_data["path"]] = current_idx
        current_idx += len(ep_data["window_starts"])

    train_indices, val_indices = [], []
    train_ep_names, val_ep_names = [], []

    for rank, ep_data in enumerate(sorted_episodes):
        num_windows = len(ep_data["window_starts"])
        start = ep_window_start[ep_data["path"]]
        window_range = list(range(start, start + num_windows))
        ep_name = Path(ep_data["path"]).parent.name + "/" + Path(ep_data["path"]).name
        if rank in val_ep_set:
            val_indices.extend(window_range)
            val_ep_names.append(ep_name)
        else:
            train_indices.extend(window_range)
            train_ep_names.append(ep_name)

    print(f"\n=== Episode K-Fold Split (fold={fold}/{num_folds}, total={num_episodes} episodes) ===")
    print(f"  Train: {len(train_ep_names)} episodes")
    for name in train_ep_names:
        print(f"    [train] {name}")
    print(f"  Val: {len(val_ep_names)} episodes")
    for name in val_ep_names:
        print(f"    [val]   {name}")

    train_dset = data.Subset(dataset, train_indices)
    val_dset   = data.Subset(dataset, val_indices)

    # Class distribution
    CLASS_NAMES = {0: "Fail", 1: "Success"}

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

    train_dataloader = data.DataLoader(train_dset, **train_loader_args)
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
    elif data_cfg.sensor in ("brainco_grasp", "brainco_grasp_prediction", "brainco_grasp_multimodal"):
        train_dataloader, val_dataloader = get_dataloader_brainco_grasp(cfg)
    else:
        raise NotImplementedError(f"Sensor type '{data_cfg.sensor}' not implemented yet.")
    return train_dataloader, val_dataloader


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
    torch.backends.cudnn.benchmark = True

    logger.info(f"Instantiating dataset & dataloaders for <{cfg.data.dataset._target_}>")
    train_dataloader, val_dataloader = get_dataloaders(cfg)

    logger.info(f"Instantiating model <{cfg.task._target_}>")
    model = hydra.utils.instantiate(cfg.task)

    trainer = Trainer(wandb_logger=wandb, **cfg.trainer)

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)

    wandb.finish()

    return getattr(model, "last_val_metrics", {})


# @hydra.main(version_base="1.3", config_path="config")
@hydra.main(version_base="1.3", config_path="config", config_name="default_task.yaml")
def main(cfg: DictConfig):
    """
    Main function to train the model
    """
    if cfg.get("all_split", False):
        num_folds = int(cfg.get("num_folds", 5))
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
        print(f"\n{'='*60}")
        print("  K-FOLD CROSS-VALIDATION SUMMARY (Last Epoch)")
        print(f"{'='*60}")
        print(f"{'Fold':>6}  {'Accuracy':>10}  {'F1 Score':>10}")
        print(f"{'-'*34}")
        accuracies, f1s = [], []
        for fold in range(num_folds):
            m = all_metrics.get(fold, {})
            acc = m.get("accuracy", float("nan"))
            f1 = m.get("f1", float("nan"))
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
