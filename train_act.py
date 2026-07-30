"""Train an ACT (Action Chunking Transformer) policy on BrainCo episodes.

    # single fold
    python train_act.py +experiment=brainco/act/act_base.yaml

    # all folds
    python train_act.py +experiment=brainco/act/act_base.yaml --all_split --num_folds 5

    # quick smoke test (4 episodes, 2 epochs)
    python train_act.py +experiment=brainco/act/act_tiny_smoke.yaml

Episodes are split at the episode level, never at the frame level: a frame from
a validation episode must never appear in training, otherwise the policy can
memorize the trajectory it is being scored on.
"""

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.utils.data as data
from hydra.core.hydra_config import HydraConfig
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb
from tactile_ssl.trainer import Trainer
from tactile_ssl.utils import get_local_rank
from tactile_ssl.utils.logging import get_pylogger, print_config_tree

logger = get_pylogger(__name__)
OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b))
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b)
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


def get_dataloaders(cfg: DictConfig):
    """Build train/val dataloaders with an episode-level K-fold split.

    Episodes are shuffled with a fixed seed and cut into ``num_folds`` blocks;
    block ``fold`` becomes validation. Normalization stats (qpos / action /
    tactile mean & std) are computed from the training episodes only and
    installed on the shared dataset object, so validation is normalized with
    training statistics.
    """
    data_cfg = cfg.data
    num_folds = int(cfg.get("num_folds", 5))
    fold = int(cfg.get("fold", 0)) % num_folds
    shuffle_seed = int(cfg.get("split_seed", 42))

    dataset = hydra.utils.instantiate(data_cfg.dataset)

    episode_order = list(range(len(dataset.episode_data)))
    random.Random(shuffle_seed).shuffle(episode_order)
    num_episodes = len(episode_order)
    if num_episodes < num_folds:
        raise ValueError(
            f"num_folds={num_folds} but only {num_episodes} episodes were loaded"
        )

    fold_size = num_episodes // num_folds
    start = fold * fold_size
    end = start + fold_size if fold < num_folds - 1 else num_episodes
    val_ranks = set(range(start, end))

    train_episodes, val_episodes = [], []
    train_indices, val_indices = [], []
    for rank, episode_index in enumerate(episode_order):
        entry = dataset.episode_data[episode_index]
        if rank in val_ranks:
            val_episodes.append(episode_index)
            val_indices.extend(entry["sample_indices"])
        else:
            train_episodes.append(episode_index)
            train_indices.extend(entry["sample_indices"])

    def _episode_name(episode_index: int) -> str:
        path = Path(dataset.episode_data[episode_index]["path"])
        return f"{path.parent.name}/{path.name}"

    lines = [
        f"\n=== Episode Split (val_fold={fold}/{num_folds}, "
        f"{num_episodes} episodes, seed={shuffle_seed}) ===",
        f"  Train: {len(train_episodes)} episodes / {len(train_indices)} samples",
    ]
    lines += [f"    [train] {_episode_name(i)}" for i in train_episodes]
    lines.append(f"  Val: {len(val_episodes)} episodes / {len(val_indices)} samples")
    lines += [f"    [val]   {_episode_name(i)}" for i in val_episodes]

    # train/val are Subsets of one dataset object, so the validation samples
    # must be flagged explicitly or they would get the training augmentation.
    dataset.set_eval_indices(val_indices)

    # ── normalization stats from the training episodes only ────────────────
    norm_stats = dataset.compute_norm_stats(train_episodes)
    dataset.set_norm_stats(norm_stats)
    lines.append("=== Normalization stats (train episodes) ===")
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std",
                "tactile_mean", "tactile_std"):
        values = ", ".join(f"{v:.4f}" for v in norm_stats[key].tolist())
        lines.append(f"  {key:13s}: [{values}]")
    lines.append("=" * 26 + "\n")

    split_log = "\n".join(lines)
    print(split_log)
    split_log_path = Path(cfg.paths.output_dir) / f"split_fold{fold}.txt"
    split_log_path.write_text(split_log)
    torch.save(norm_stats, Path(cfg.paths.output_dir) / f"norm_stats_fold{fold}.pt")

    train_dset = data.Subset(dataset, train_indices)
    val_dset = data.Subset(dataset, val_indices)

    generator = torch.Generator()
    if cfg.get("seed"):
        generator.manual_seed(int(cfg.seed))
    else:
        generator.seed()

    def _apply_budget(subset, budget: float):
        if budget >= 1.0:
            return subset
        keep = int(len(subset) * budget)
        subset, _ = data.random_split(
            subset, [keep, len(subset) - keep], generator=generator
        )
        return subset

    train_dset = _apply_budget(train_dset, float(data_cfg.get("train_data_budget", 1.0)))
    val_dset = _apply_budget(val_dset, float(data_cfg.get("val_data_budget", 1.0)))

    print(f"Total samples: {len(train_dset)} train, {len(val_dset)} val")

    train_dataloader = data.DataLoader(
        train_dset, generator=generator, **dict(data_cfg.train_dataloader)
    )
    val_dataloader = data.DataLoader(val_dset, **dict(data_cfg.val_dataloader))
    return train_dataloader, val_dataloader, dataset, norm_stats


def attempt_resume(cfg: DictConfig):
    if os.path.exists(f"{cfg.paths.output_dir}/config.yaml") and cfg.resume_id:
        job_id = HydraConfig.get().job.id
        logger.info(f"Attempting to resume experiment with {cfg.resume_id}")
        if not os.path.exists(f"{cfg.paths.output_dir}/checkpoints/"):
            logger.warning(f"Unable to resume: no checkpoints for job {job_id}")
            return False, cfg
        cfg = OmegaConf.load(f"{cfg.paths.output_dir}/config.yaml")
        OmegaConf.update(cfg, "ckpt_path", f"{cfg.paths.output_dir}/checkpoints/", force_add=True)
        cfg.wandb.id = f"{job_id}_{cfg.experiment_name}"
        logger.info(f"Resuming from {cfg.ckpt_path}")
        return True, cfg
    return False, cfg


def train(cfg: DictConfig):
    resume_state, cfg = attempt_resume(cfg)

    logger.info("Instantiating wandb ...")
    wandb_logger = init_wandb(cfg.wandb)
    if not resume_state:
        wandb_logger.config.update(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    logger.info(f"Instantiating dataset & dataloaders for <{cfg.data.dataset._target_}>")
    train_dataloader, val_dataloader, dataset, norm_stats = get_dataloaders(cfg)

    # Model dimensions come from the data, not from hand-maintained config values.
    with open_dict(cfg):
        cfg.task.model.state_dim = dataset.state_dim
        cfg.task.model.action_dim = dataset.action_dim
        cfg.task.model.chunk_size = dataset.chunk_size
        cfg.task.model.num_cameras = len(dataset.camera_names)
        cfg.task.model.use_tactile = dataset.use_tactile

    logger.info(f"Instantiating model <{cfg.task._target_}>")
    model = hydra.utils.instantiate(cfg.task)
    # Carry the stats inside the checkpoint so inference can (de)normalize.
    model.model.set_norm_stats(norm_stats)

    os.makedirs(cfg.trainer.save_checkpoint_dir, exist_ok=True)
    trainer = Trainer(wandb_logger=wandb_logger, **cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)

    last_metrics = getattr(model, "last_val_metrics", {})
    best_metrics = getattr(model, "best_val_metrics", {})

    print(f"\n{'='*56}")
    print("  TRAINING COMPLETE")
    print(f"{'='*56}")
    for tag, metrics in (("Last", last_metrics), ("Best", best_metrics)):
        print(f"  {tag} — l1: {metrics.get('l1', float('nan')):.5f}  "
              f"l1_infer: {metrics.get('l1_infer', float('nan')):.5f}  "
              f"hold-state baseline: {metrics.get('l1_hold', float('nan')):.5f}")
    print(f"{'='*56}\n")

    wandb_logger.finish()
    return {"last": last_metrics, "best": best_metrics}


@hydra.main(version_base="1.3", config_path="config", config_name="default_act.yaml")
def main(cfg: DictConfig):
    if not cfg.get("all_split", False):
        train(cfg)
        return

    num_folds = int(cfg.get("num_folds", 5))
    base_wandb_id = cfg.wandb.id
    base_checkpoint_dir = cfg.trainer.save_checkpoint_dir
    all_metrics = {}

    for fold in range(num_folds):
        print(f"\n{'='*60}\n  K-FOLD: training fold {fold + 1}/{num_folds}\n{'='*60}")
        with open_dict(cfg):
            cfg.fold = fold
            cfg.wandb.id = f"{base_wandb_id}_fold{fold}"
            cfg.trainer.save_checkpoint_dir = f"{base_checkpoint_dir}_fold{fold}"
        all_metrics[fold] = train(cfg)

    for tag, label in [("last", "Last Epoch"), ("best", "Best Epoch")]:
        print(f"\n{'='*60}")
        print(f"  K-FOLD SUMMARY ({label})")
        print(f"{'='*60}")
        print(f"{'Fold':>6}  {'L1':>10}  {'L1 (infer)':>12}  {'baseline':>10}")
        print("-" * 44)
        l1s, infer_l1s, holds = [], [], []
        for fold in range(num_folds):
            metrics = all_metrics.get(fold, {}).get(tag, {})
            l1 = metrics.get("l1", float("nan"))
            infer = metrics.get("l1_infer", float("nan"))
            hold = metrics.get("l1_hold", float("nan"))
            l1s.append(l1)
            infer_l1s.append(infer)
            holds.append(hold)
            print(f"{fold:>6}  {l1:>10.5f}  {infer:>12.5f}  {hold:>10.5f}")
        print("-" * 44)
        print(f"{'Mean':>6}  {np.mean(l1s):>10.5f}  {np.mean(infer_l1s):>12.5f}  {np.mean(holds):>10.5f}")
        print(f"{'Std':>6}  {np.std(l1s):>10.5f}  {np.std(infer_l1s):>12.5f}  {np.std(holds):>10.5f}")
        print("=" * 60)


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
