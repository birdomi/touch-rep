# train_task_brainco_all.py
#
# all_split 전용 최적화 버전:
#   - 데이터셋을 1회만 로드 후 메모리에 유지
#   - 모든 fold의 split index와 normalization stats를 미리 계산
#   - 각 fold는 Subset 생성 + 텐서 연산만 수행

import os
import random
from pathlib import Path
from collections import Counter

import hydra
import numpy as np
import torch
import torch.utils.data as data
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb
from lightning.fabric import seed_everything

from tactile_ssl.utils import get_local_rank
from tactile_ssl.utils.logging import get_pylogger, print_config_tree
from tactile_ssl.data.d360.utils import get_weights, get_experiment_name, get_modality_tag, get_modality_used_tag
from tactile_ssl.trainer import Trainer

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


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: 데이터셋 1회 로드
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(cfg: DictConfig):
    """데이터셋을 한 번만 인스턴스화하여 반환."""
    logger.info(f"Loading dataset <{cfg.data.dataset._target_}> (once for all folds) ...")
    dataset = hydra.utils.instantiate(cfg.data.dataset)
    logger.info(f"  Total windows: {len(dataset.windows)}, episodes: {len(dataset.episode_data)}")
    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: 전체 fold split을 미리 계산
# ──────────────────────────────────────────────────────────────────────────────

def precompute_all_splits(dataset, num_folds: int, shuffle_seed: int):
    """
    모든 fold의 (train_indices, val_indices, train_ep_names, val_ep_names)를
    한 번에 계산하여 리스트로 반환.

    Returns:
        list of dicts, length = num_folds, each:
            {
              "fold": int,
              "train_indices": List[int],
              "val_indices":   List[int],
              "train_ep_names": List[str],
              "val_ep_names":   List[str],
            }
    """
    # window start index map (episode path → flat window index)
    ep_window_start = {}
    current_idx = 0
    for ep_data in dataset.episode_data:
        ep_window_start[ep_data["path"]] = current_idx
        current_idx += len(ep_data["window_starts"])

    all_episodes = list(dataset.episode_data)
    rng = random.Random(shuffle_seed)
    rng.shuffle(all_episodes)
    num_episodes = len(all_episodes)

    def _fold_range(k):
        fold_size = num_episodes // num_folds
        start = k * fold_size
        end = start + fold_size if k < num_folds - 1 else num_episodes
        return range(start, end)

    results = []
    for fold in range(num_folds):
        val_range = _fold_range(fold)
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

        results.append({
            "fold": fold,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_ep_names": train_ep_names,
            "val_ep_names": val_ep_names,
        })

    logger.info(f"Pre-computed {num_folds} fold splits (seed={shuffle_seed}).")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: fold별 normalization stats 계산 (디스크 I/O 없음, 텐서 연산만)
# ──────────────────────────────────────────────────────────────────────────────

def compute_norm_stats(dataset, train_indices: list, is_cat_encoder: bool):
    """
    이미 메모리에 올라온 dataset.windows에서 train_indices 기준으로
    signal mean/std, pose mean/std를 계산.
    """
    all_sensor = torch.stack([dataset.windows[i]["sensor"] for i in train_indices])
    all_poses  = torch.stack([dataset.windows[i]["sensor_poses"] for i in train_indices])

    NUM_CHANS = all_sensor.shape[-1]
    sensor_mean_1d = torch.zeros(NUM_CHANS)
    sensor_std_1d  = torch.ones(NUM_CHANS)
    for c in range(NUM_CHANS):
        valid = all_sensor[..., c][all_sensor[..., c] >= 0].float()
        if valid.numel() > 0:
            sensor_mean_1d[c] = valid.mean()
            sensor_std_1d[c]  = valid.std().clamp(min=1e-6)

    if is_cat_encoder:
        signal_mean = sensor_mean_1d.clone()
        signal_std  = sensor_std_1d.clone()
    else:
        NUM_SENSOR_TYPES = 1
        signal_mean = sensor_mean_1d.unsqueeze(0).expand(NUM_SENSOR_TYPES, -1).clone()
        signal_std  = sensor_std_1d.unsqueeze(0).expand(NUM_SENSOR_TYPES, -1).clone()

    pose_flat = all_poses.reshape(-1, 3).float()
    pos_mean  = pose_flat.mean(dim=0)
    pos_std   = pose_flat.std(dim=0).clamp(min=1e-6)

    return {
        "signal_mean": signal_mean,
        "signal_std":  signal_std,
        "pos_mean":    pos_mean,
        "pos_std":     pos_std,
        "sensor_mean_1d": sensor_mean_1d,
        "sensor_std_1d":  sensor_std_1d,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: fold별 DataLoader 생성
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloaders_for_fold(cfg: DictConfig, dataset, split: dict, norm_stats: dict):
    """
    미리 계산된 split index로 Subset을 만들고 DataLoader를 반환.
    normalization stats를 dataset에 주입.
    """
    fold = split["fold"]
    train_indices = split["train_indices"]
    val_indices   = split["val_indices"]
    val_fold      = fold

    # norm stats 주입
    dataset.computed_signal_mean = norm_stats["signal_mean"]
    dataset.computed_signal_std  = norm_stats["signal_std"]
    dataset.computed_pos_mean    = norm_stats["pos_mean"]
    dataset.computed_pos_std     = norm_stats["pos_std"]

    # 로그 출력
    lines = []
    num_episodes = len(dataset.episode_data)
    num_folds = cfg.get("num_folds", 4)
    shuffle_seed = int(cfg.get("split_seed", 42))
    lines.append(
        f"\n=== Episode Split (val_fold={val_fold}/{num_folds}, total={num_episodes} episodes, seed={shuffle_seed}) ==="
    )
    lines.append(f"  Train: {len(split['train_ep_names'])} episodes")
    for name in split["train_ep_names"]:
        lines.append(f"    [train] {name}")
    lines.append(f"  Val: {len(split['val_ep_names'])} episodes")
    for name in split["val_ep_names"]:
        lines.append(f"    [val]   {name}")
    lines.append(f"[NormStats] signal_mean (per ch): {norm_stats['sensor_mean_1d'].tolist()}")
    lines.append(f"[NormStats] signal_std  (per ch): {norm_stats['sensor_std_1d'].tolist()}")
    lines.append(f"[NormStats] pos_mean:             {norm_stats['pos_mean'].tolist()}")
    lines.append(f"[NormStats] pos_std:              {norm_stats['pos_std'].tolist()}")

    train_dset = data.Subset(dataset, train_indices)
    val_dset   = data.Subset(dataset, val_indices)

    # Class distribution
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

    split_log = "\n".join(lines)
    print(split_log)

    split_log_path = Path(cfg.paths.output_dir) / f"split_fold{val_fold}.txt"
    split_log_path.write_text(split_log)
    print(f"[Split log saved to {split_log_path}]")

    # Data budget
    g = torch.Generator()
    if cfg.get("seed"):
        g.manual_seed(cfg.seed)
    else:
        g.seed()

    data_cfg = cfg.data
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


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: 단일 fold 학습
# ──────────────────────────────────────────────────────────────────────────────

def train_fold(cfg: DictConfig, train_dataloader, val_dataloader, norm_stats: dict):
    """이미 만들어진 dataloaders로 단일 fold 학습."""
    logger.info("Instantiating wandb ...")
    wb = init_wandb(cfg.wandb)
    wb.config.update(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info(f"Instantiating model <{cfg.task._target_}>")
    model = hydra.utils.instantiate(cfg.task)

    # encoder에 normalization stats 주입
    if hasattr(model, "model_encoder") and hasattr(model.model_encoder, "update_stats"):
        model.model_encoder.update_stats(
            norm_stats["signal_mean"],
            norm_stats["signal_std"],
            norm_stats["pos_mean"],
            norm_stats["pos_std"],
        )
        logger.info("Encoder normalization stats updated from training data.")

    trainer = Trainer(wandb_logger=wb, **cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)

    last_metrics = getattr(model, "last_val_metrics", {})
    best_metrics = getattr(model, "best_val_metrics", {})

    print(f"\n{'='*40}")
    print(f"  TRAINING COMPLETE (fold {cfg.get('fold', '?')})")
    print(f"{'='*40}")
    print(f"  Last Epoch  — Accuracy: {last_metrics.get('accuracy', float('nan')):.4f}  F1: {last_metrics.get('f1', float('nan')):.4f}")
    print(f"  Best Epoch  — Accuracy: {best_metrics.get('accuracy', float('nan')):.4f}  F1: {best_metrics.get('f1', float('nan')):.4f}")
    print(f"{'='*40}\n")

    wb.finish()

    return {"last": last_metrics, "best": best_metrics}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="config", config_name="default_task.yaml")
def main(cfg: DictConfig):
    num_folds    = int(cfg.get("num_folds", 4))
    shuffle_seed = int(cfg.get("split_seed", 42))

    base_wandb_id       = cfg.wandb.id
    base_checkpoint_dir = cfg.trainer.save_checkpoint_dir

    # ── 데이터셋 1회 로드 ──────────────────────────────────────────────────────
    dataset = load_dataset(cfg)

    is_cat_encoder = "brainco_cat" in str(cfg.task.model_encoder._target_)

    # ── 모든 fold의 split + normalization stats를 미리 계산 ───────────────────
    all_splits = precompute_all_splits(dataset, num_folds, shuffle_seed)

    print(f"\nPre-computing normalization stats for {num_folds} folds ...")
    all_norm_stats = []
    for split in all_splits:
        ns = compute_norm_stats(dataset, split["train_indices"], is_cat_encoder)
        all_norm_stats.append(ns)
        print(
            f"  fold {split['fold']}: "
            f"signal_mean={[f'{v:.3f}' for v in ns['sensor_mean_1d'].tolist()]}  "
            f"signal_std={[f'{v:.3f}' for v in ns['sensor_std_1d'].tolist()]}"
        )

    # ── fold별 학습 ───────────────────────────────────────────────────────────
    all_metrics = {}
    for split, norm_stats in zip(all_splits, all_norm_stats):
        fold = split["fold"]
        print(f"\n{'='*60}")
        print(f"  K-FOLD: Training fold {fold + 1}/{num_folds}")
        print(f"{'='*60}")

        with open_dict(cfg):
            cfg.fold = fold
            cfg.wandb.id = f"{base_wandb_id}_fold{fold}"
            cfg.trainer.save_checkpoint_dir = f"{base_checkpoint_dir}_fold{fold}"

        train_loader, val_loader = make_dataloaders_for_fold(cfg, dataset, split, norm_stats)
        metrics = train_fold(cfg, train_loader, val_loader, norm_stats)
        all_metrics[fold] = metrics

    # restore
    with open_dict(cfg):
        cfg.wandb.id = base_wandb_id

    # ── K-Fold 요약 출력 ──────────────────────────────────────────────────────
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


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_folds", type=int, default=None)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    if known.num_folds is not None:
        sys.argv.append(f"num_folds={known.num_folds}")

    torch.set_float32_matmul_precision("medium")
    main()
