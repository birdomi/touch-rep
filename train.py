import os
from typing import Optional

import hydra
import numpy as np
import torch
import torch.utils.data as data
import wandb
from hydra.core.hydra_config import HydraConfig
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict

from tactile_ssl.trainer import Trainer
from tactile_ssl.utils import get_local_rank, get_node_id
from tactile_ssl.utils.logging import get_pylogger, print_config_tree

logger = get_pylogger(__name__)

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b))
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b)


def init_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        dir=cfg.save_dir,
        id=f"{cfg.id}_{get_node_id()}_{get_local_rank()}",
        group=cfg.group,
        tags=cfg.tags,
        notes=cfg.notes,
    )
    return wandb


def _maybe_update_encoder_stats(algorithm, sensor_means, sensor_stds, pos_mean=None, pos_std=None):
    modules = []
    if hasattr(algorithm, "student_encoder_dict") and "backbone" in algorithm.student_encoder_dict:
        modules.append(("student_backbone", algorithm.student_encoder_dict["backbone"]))
    if hasattr(algorithm, "teacher_encoder_dict") and "backbone" in algorithm.teacher_encoder_dict:
        modules.append(("teacher_backbone", algorithm.teacher_encoder_dict["backbone"]))
    if hasattr(algorithm, "encoder"):
        modules.append(("encoder", algorithm.encoder))

    seen = set()
    for name, module in modules:
        if id(module) in seen or not hasattr(module, "update_stats"):
            continue
        seen.add(id(module))

        kwargs = {}
        if sensor_means is not None and sensor_stds is not None:
            kwargs["signal_mean"] = torch.tensor(sensor_means, dtype=torch.float32)
            kwargs["signal_std"] = torch.tensor(sensor_stds, dtype=torch.float32)
        if pos_mean is not None and pos_std is not None:
            kwargs["pos_mean"] = torch.tensor(pos_mean, dtype=torch.float32)
            kwargs["pos_std"] = torch.tensor(pos_std, dtype=torch.float32)
        if kwargs:
            module.update_stats(**kwargs)
            logger.info(f"Called update_stats() on {name}")


def _build_angle_vector_dataset(cfg: DictConfig, data_root: str):
    from tactile_ssl.data.angle_tactile import AngleTactileDataset

    data_cfg = cfg.data
    kwargs = dict(
        data_root=data_root,
        window_size=int(data_cfg.get("window_size", 1)),
        window_stride=int(data_cfg.get("window_stride", 1)),
        train_val_split=float(data_cfg.get("train_val_split", 0.9)),
    )
    train_ds = AngleTactileDataset(split="train", **kwargs)
    val_ds = AngleTactileDataset(split="val", **kwargs)
    return train_ds, val_ds, kwargs


def _compute_contact_stats(sequences):
    contact_chunks = []
    for seq in sequences:
        contact_chunks.append(seq.lh_contact.reshape(-1))
        contact_chunks.append(seq.rh_contact.reshape(-1))
    if not contact_chunks:
        return 0.0, 1.0

    contact_arr = np.concatenate(contact_chunks)
    contact_mean = float(np.mean(contact_arr))
    contact_std = float(np.std(contact_arr))
    return contact_mean, contact_std if contact_std > 1e-6 else 1.0


def _make_angle_vector_dataloaders(cfg: DictConfig, data_root: str, name: str):
    train_ds, val_ds, kwargs = _build_angle_vector_dataset(cfg, data_root)

    logger.info(f"Computing {name} contact normalization stats...")
    contact_mean, contact_std = _compute_contact_stats(train_ds._sequences)
    logger.info(f"  contact mean={contact_mean:.6f} std={contact_std:.6f}")
    logger.info("  angle normalization skipped")

    sensor_mean = [contact_mean]
    sensor_std = [contact_std]
    with open_dict(cfg):
        cfg.data.normalization.mean = sensor_mean
        cfg.data.normalization.std = sensor_std

    for ds in (train_ds, val_ds):
        ds.sensor_mean = sensor_mean
        ds.sensor_std = sensor_std

    logger.info(
        f"{name} vector: {len(train_ds)} train windows, "
        f"{len(val_ds)} val windows (window_size={kwargs['window_size']})"
    )
    return train_ds, val_ds


def get_dataloaders_hot3d_vector_based(cfg: DictConfig):
    data_root = str(cfg.data.get("data_root", "pretraining_dataset/vector_dataset/HOT3D/hot3d"))
    return _make_angle_vector_dataloaders(cfg, data_root, "HOT3D")


def get_dataloaders_arctic_vector_based(cfg: DictConfig):
    data_root = str(cfg.data.get("data_root", "pretraining_dataset/vector_dataset/ARCTIC"))
    return _make_angle_vector_dataloaders(cfg, data_root, "ARCTIC")


def get_dataloaders_taco_vector_based(cfg: DictConfig):
    data_root = str(cfg.data.get("data_root", "pretraining_dataset/vector_dataset/TACO"))
    return _make_angle_vector_dataloaders(cfg, data_root, "TACO")


def get_dataloaders_oakinkv2_vector_based(cfg: DictConfig):
    data_root = str(cfg.data.get("data_root", "pretraining_dataset/vector_dataset/OAKINKV2"))
    return _make_angle_vector_dataloaders(cfg, data_root, "OAKINKV2")


def get_dataloaders_all_vector_based(cfg: DictConfig):
    data_cfg = cfg.data
    paths = [
        ("HOT3D", str(data_cfg.get("hot3d_data_root", "pretraining_dataset/vector_dataset/HOT3D/hot3d"))),
        ("ARCTIC", str(data_cfg.get("arctic_data_root", "pretraining_dataset/vector_dataset/ARCTIC"))),
        ("TACO", str(data_cfg.get("taco_data_root", "pretraining_dataset/vector_dataset/TACO"))),
        ("OAKINKV2", str(data_cfg.get("oakinkv2_data_root", "pretraining_dataset/vector_dataset/OAKINKV2"))),
    ]

    train_dsets, val_dsets, all_train_seqs = [], [], []
    for name, path in paths:
        logger.info(f"Loading {name} from {path}")
        train_ds, val_ds, _ = _build_angle_vector_dataset(cfg, path)
        train_dsets.append(train_ds)
        val_dsets.append(val_ds)
        all_train_seqs.extend(train_ds._sequences)
        logger.info(f"  {name}: train={len(train_ds)} val={len(val_ds)}")

    logger.info("Computing combined contact normalization stats...")
    contact_mean, contact_std = _compute_contact_stats(all_train_seqs)
    sensor_mean = [contact_mean]
    sensor_std = [contact_std]

    with open_dict(cfg):
        cfg.data.normalization.mean = sensor_mean
        cfg.data.normalization.std = sensor_std

    for ds in train_dsets + val_dsets:
        ds.sensor_mean = sensor_mean
        ds.sensor_std = sensor_std

    train_dset = data.ConcatDataset(train_dsets)
    val_dset = data.ConcatDataset(val_dsets)
    train_dset.sensor_mean = sensor_mean
    train_dset.sensor_std = sensor_std
    val_dset.sensor_mean = sensor_mean
    val_dset.sensor_std = sensor_std

    logger.info(f"ALL vector: {len(train_dset)} train windows, {len(val_dset)} val windows")
    return train_dset, val_dset


def _compute_multichannel_contact_stats(
    sequences,
    num_channels: int,
    include_force_delta: bool = False,
    fingertip_only: bool = False,
    channel_indices=None,
):
    num_kept = len(channel_indices) if channel_indices else num_channels
    output_channels = num_kept * (2 if include_force_delta else 1)
    channel_sum = np.zeros(output_channels, dtype=np.float64)
    channel_sq_sum = np.zeros(output_channels, dtype=np.float64)
    count = 0
    for sequence in sequences:
        contact = sequence.joint_contact.numpy()
        if include_force_delta:
            delta = np.zeros_like(contact)
            delta[1:] = contact[1:] - contact[:-1]
            contact = np.concatenate((contact, delta), axis=-1)
        if fingertip_only:
            from tactile_ssl.data.pseudo_force_tactile import FINGERTIP_COLUMNS

            contact = contact[:, FINGERTIP_COLUMNS]
        if channel_indices:
            idx = list(channel_indices)
            if include_force_delta:
                idx += [c + num_channels for c in channel_indices]
            contact = contact[..., idx]

        half = contact.shape[1] // 2
        for hand_contact in (contact[:, :half], contact[:, half:]):
            values = hand_contact.reshape(-1, output_channels).astype(
                np.float64, copy=False
            )
            channel_sum += values.sum(axis=0)
            channel_sq_sum += np.square(values).sum(axis=0)
            count += values.shape[0]

    if count == 0:
        return [0.0] * output_channels, [1.0] * output_channels

    mean = channel_sum / count
    variance = np.maximum(channel_sq_sum / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std <= 1e-6] = 1.0
    return mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()


def get_dataloaders_all_pseudo_force_based(cfg: DictConfig):
    from tactile_ssl.data.pseudo_force_tactile import (
        FORCE_CHANNELS,
        CompactPseudoForceDataset,
    )

    data_cfg = cfg.data
    paths = [
        ("HOT3D", str(data_cfg.hot3d_data_root)),
        ("ARCTIC", str(data_cfg.arctic_data_root)),
        ("TACO", str(data_cfg.taco_data_root)),
        ("OAKINKV2", str(data_cfg.oakinkv2_data_root)),
    ]
    dataset_kwargs = {
        "window_size": int(data_cfg.get("window_size", 1)),
        "window_stride": int(data_cfg.get("window_stride", 1)),
        "train_val_split": float(data_cfg.get("train_val_split", 0.9)),
        "sensor_output_key": str(cfg.algorithm.get("sensor_input_key", "joint_contact")),
        "include_force_delta": bool(data_cfg.get("include_force_delta", False)),
        "fingertip_only": bool(data_cfg.get("fingertip_only", False)),
        "sensor_channels": list(data_cfg.get("sensor_channels") or []) or None,
    }

    train_dsets, val_dsets, all_train_sequences = [], [], []
    for name, path in paths:
        logger.info(f"Loading {name} compact pseudo-force data from {path}")
        train_ds = CompactPseudoForceDataset(data_root=path, split="train", **dataset_kwargs)
        val_ds = CompactPseudoForceDataset(data_root=path, split="val", **dataset_kwargs)
        train_dsets.append(train_ds)
        val_dsets.append(val_ds)
        all_train_sequences.extend(train_ds._sequences)
        logger.info(f"  {name}: train={len(train_ds)} val={len(val_ds)}")

    sensor_channels = dataset_kwargs["sensor_channels"]
    channel_indices = (
        [FORCE_CHANNELS.index(c) for c in sensor_channels]
        if sensor_channels else None
    )
    sensor_mean, sensor_std = _compute_multichannel_contact_stats(
        all_train_sequences,
        len(FORCE_CHANNELS),
        include_force_delta=dataset_kwargs["include_force_delta"],
        fingertip_only=dataset_kwargs["fingertip_only"],
        channel_indices=channel_indices,
    )
    channel_order = tuple(sensor_channels) if sensor_channels else FORCE_CHANNELS
    if dataset_kwargs["include_force_delta"]:
        channel_order = channel_order + tuple(f"delta_{name}" for name in channel_order)
    logger.info(
        f"Pseudo-force channel order={channel_order}, "
        f"mean={sensor_mean}, std={sensor_std}"
    )
    with open_dict(cfg):
        cfg.data.normalization.mean = sensor_mean
        cfg.data.normalization.std = sensor_std

    for dataset in train_dsets + val_dsets:
        dataset.sensor_mean = sensor_mean
        dataset.sensor_std = sensor_std

    train_dset = data.ConcatDataset(train_dsets)
    val_dset = data.ConcatDataset(val_dsets)
    train_dset.sensor_mean = sensor_mean
    train_dset.sensor_std = sensor_std
    val_dset.sensor_mean = sensor_mean
    val_dset.sensor_std = sensor_std

    logger.info(
        f"ALL pseudo-force: {len(train_dset)} train windows, "
        f"{len(val_dset)} val windows"
    )
    return train_dset, val_dset


def get_dataloaders_compact_pseudo_force_based(cfg: DictConfig):
    from tactile_ssl.data.pseudo_force_tactile import (
        FORCE_CHANNELS,
        CompactPseudoForceDataset,
    )

    data_cfg = cfg.data
    dataset_kwargs = {
        "data_root": str(data_cfg.data_root),
        "window_size": int(data_cfg.get("window_size", 1)),
        "window_stride": int(data_cfg.get("window_stride", 1)),
        "train_val_split": float(data_cfg.get("train_val_split", 0.9)),
        "sensor_output_key": str(cfg.algorithm.get("sensor_input_key", "joint_contact")),
        "include_force_delta": bool(data_cfg.get("include_force_delta", False)),
        "fingertip_only": bool(data_cfg.get("fingertip_only", False)),
        "sensor_channels": list(data_cfg.get("sensor_channels") or []) or None,
    }
    train_dset = CompactPseudoForceDataset(split="train", **dataset_kwargs)
    val_dset = CompactPseudoForceDataset(split="val", **dataset_kwargs)

    _sensor_channels = dataset_kwargs["sensor_channels"]
    sensor_mean, sensor_std = _compute_multichannel_contact_stats(
        train_dset._sequences,
        len(FORCE_CHANNELS),
        include_force_delta=dataset_kwargs["include_force_delta"],
        fingertip_only=dataset_kwargs["fingertip_only"],
        channel_indices=(
            [FORCE_CHANNELS.index(c) for c in _sensor_channels]
            if _sensor_channels else None
        ),
    )
    with open_dict(cfg):
        cfg.data.normalization.mean = sensor_mean
        cfg.data.normalization.std = sensor_std

    for dataset in (train_dset, val_dset):
        dataset.sensor_mean = sensor_mean
        dataset.sensor_std = sensor_std

    channel_order = FORCE_CHANNELS
    if dataset_kwargs["include_force_delta"]:
        channel_order = FORCE_CHANNELS + tuple(f"delta_{name}" for name in FORCE_CHANNELS)
    logger.info(
        f"Compact pseudo-force: train={len(train_dset)} val={len(val_dset)}, "
        f"channel order={channel_order}, mean={sensor_mean}, std={sensor_std}"
    )
    return train_dset, val_dset


def get_dataloaders_brainco_based(cfg: DictConfig):
    data_cfg = cfg.data

    def get_brainco_dataset(dataset_cfg: DictConfig, obj_name: str, episode_id: str, obj_class: int):
        data_path = dataset_cfg.get("data_path", "dataset/brainco/pretraining")
        full_path = f"{data_path}/{obj_name}/{episode_id}"
        if not os.path.exists(full_path):
            logger.warning(f"BrainCo dataset path not found: {full_path}")
            return None

        return hydra.utils.instantiate(
            dataset_cfg,
            data_path=full_path,
            object_class=obj_class,
        )

    train_datasets, val_datasets = [], []
    object_classes, object_class_sizes = [], []

    for dataset_l in data_cfg.get("dataset_list", []):
        if dataset_l.type != "teleop":
            continue

        train_episodes = dataset_l.get("train_dataset_ids", [])
        val_episodes = dataset_l.get("val_dataset_ids", [])
        for obj in dataset_l.get("sequence_list", []):
            object_classes.append(obj)
            object_class_sizes.append(0)
            obj_class_idx = len(object_classes) - 1

            for ep in train_episodes:
                ds = get_brainco_dataset(dataset_l.dataset, obj, f"episode_{int(ep):04d}", obj_class_idx)
                if ds is not None:
                    object_class_sizes[-1] += len(ds)
                    train_datasets.append(ds)

            for ep in val_episodes:
                ds = get_brainco_dataset(dataset_l.dataset, obj, f"episode_{int(ep):04d}", obj_class_idx)
                if ds is not None:
                    val_datasets.append(ds)

    if train_datasets:
        train_dset = data.ConcatDataset(train_datasets)
        val_dset = data.ConcatDataset(val_datasets) if val_datasets else None

        if np.sum(object_class_sizes) > 0:
            ratios = np.array(object_class_sizes) / np.sum(object_class_sizes)
            weights = 1.0 / (ratios + 1e-8)
            weights = weights / np.sum(weights)
            with open_dict(cfg):
                cfg.data.object_classes = object_classes
                cfg.data.object_class_weights = weights.tolist()
    else:
        dataset = hydra.utils.instantiate(data_cfg.dataset)
        train_dset_size = int(len(dataset) * data_cfg.get("train_val_split", 0.9))
        train_dset, val_dset = data.random_split(
            dataset,
            [train_dset_size, len(dataset) - train_dset_size],
        )

    # BrainCo tactile values are already scaled by fixed per-channel maxima in
    # the dataset. Keep encoder normalization as identity instead of applying
    # an additional train-set mean/std z-score.
    mean = [0.0] * 4
    std = [1.0] * 4
    if cfg.data.get("normalization"):
        with open_dict(cfg):
            cfg.data.normalization.mean = mean
            cfg.data.normalization.std = std

    train_dset.sensor_mean = mean
    train_dset.sensor_std = std
    if val_dset is not None:
        val_dset.sensor_mean = mean
        val_dset.sensor_std = std

    return train_dset, val_dset


def get_dataloaders_brainco_angle_based(cfg: DictConfig):
    data_cfg = cfg.data
    train_datasets, val_datasets = [], []

    for dataset_l in data_cfg.dataset_list:
        if dataset_l.type != "teleop":
            continue

        train_ids = dataset_l.get("train_dataset_ids", [])
        val_ids = dataset_l.get("val_dataset_ids", [])
        for obj_class_idx, obj in enumerate(dataset_l.get("sequence_list", [])):
            for ep in train_ids:
                full_path = f"{dataset_l.dataset.data_path}/{obj}/episode_{int(ep):04d}"
                if not os.path.exists(full_path):
                    logger.warning(f"BrainCo angle path not found: {full_path}")
                    continue
                train_datasets.append(
                    hydra.utils.instantiate(dataset_l.dataset, data_path=full_path, object_class=obj_class_idx)
                )

            for ep in val_ids:
                full_path = f"{dataset_l.dataset.data_path}/{obj}/episode_{int(ep):04d}"
                if not os.path.exists(full_path):
                    continue
                val_datasets.append(
                    hydra.utils.instantiate(dataset_l.dataset, data_path=full_path, object_class=obj_class_idx)
                )

    if not train_datasets:
        raise ValueError("No training datasets loaded for brainco_angle_vector sensor.")

    train_dset = data.ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dset = data.ConcatDataset(val_datasets) if val_datasets else None

    # BrainCo angle tactile values are already scaled by fixed per-channel
    # maxima in the dataset. Do not apply a second mean/std normalization.
    sensor_mean = [0.0] * 4
    sensor_std = [1.0] * 4

    with open_dict(cfg):
        cfg.data.normalization.mean = sensor_mean
        cfg.data.normalization.std = sensor_std

    train_dset.sensor_mean = sensor_mean
    train_dset.sensor_std = sensor_std
    if val_dset is not None:
        val_dset.sensor_mean = sensor_mean
        val_dset.sensor_std = sensor_std

    logger.info(
        f"BrainCo angle: {len(train_dset)} train windows, "
        f"{len(val_dset) if val_dset else 0} val windows"
    )
    return train_dset, val_dset


def get_dataloaders(cfg: DictConfig):
    train_sampler = None
    if cfg.data.sensor == "brainco":
        train_dset, val_dset = get_dataloaders_brainco_based(cfg)
    elif cfg.data.sensor == "brainco_angle_vector":
        train_dset, val_dset = get_dataloaders_brainco_angle_based(cfg)
    elif cfg.data.sensor == "hot3d_vector":
        train_dset, val_dset = get_dataloaders_hot3d_vector_based(cfg)
    elif cfg.data.sensor == "arctic_vector":
        train_dset, val_dset = get_dataloaders_arctic_vector_based(cfg)
    elif cfg.data.sensor == "taco_vector":
        train_dset, val_dset = get_dataloaders_taco_vector_based(cfg)
    elif cfg.data.sensor == "oakinkv2_vector":
        train_dset, val_dset = get_dataloaders_oakinkv2_vector_based(cfg)
    elif cfg.data.sensor == "all_vector":
        train_dset, val_dset = get_dataloaders_all_vector_based(cfg)
    elif cfg.data.sensor == "all_pseudo_force":
        train_dset, val_dset = get_dataloaders_all_pseudo_force_based(cfg)
    elif cfg.data.sensor == "compact_pseudo_force":
        train_dset, val_dset = get_dataloaders_compact_pseudo_force_based(cfg)
    else:
        raise NotImplementedError(f"Sensor type '{cfg.data.sensor}' is not part of the BrainCo-only setup.")

    sensor_means = train_dset.sensor_mean
    sensor_stds = train_dset.sensor_std

    loader_args = dict(cfg.data.train_dataloader)
    if train_sampler is not None:
        logger.info("Using WeightedRandomSampler for training dataloader.")
        loader_args["shuffle"] = False
        loader_args["sampler"] = train_sampler

    train_dataloader = data.DataLoader(train_dset, **loader_args)
    val_dataloader = data.DataLoader(val_dset, **dict(cfg.data.val_dataloader))
    return train_dataloader, val_dataloader, sensor_means, sensor_stds


def attempt_resume(cfg: DictConfig):
    if os.environ.get("SLURM_RESTART_COUNT") is not None:
        cfg.resume_id = os.environ.get("SLURM_JOB_ID")

    if os.path.exists(f"{cfg.paths.output_dir}/config.yaml") and cfg.resume_id:
        job_id = HydraConfig.get().job.id
        logger.info(f"Attempting to resume experiment with {cfg.resume_id}")
        for required_path in ("checkpoints/", "wandb/", "config.yaml"):
            if not os.path.exists(f"{cfg.paths.output_dir}/{required_path}"):
                logger.warning(f"Unable to resume: missing {required_path}")
                return False, cfg

        cfg = OmegaConf.load(f"{cfg.paths.output_dir}/config.yaml")
        OmegaConf.update(cfg, "ckpt_path", f"{cfg.paths.output_dir}/checkpoints/", force_add=True)
        cfg.wandb.id = f"{job_id}_{cfg.experiment_name}"
        logger.info(f"Resuming from latest checkpoint at {cfg.ckpt_path}")
        return True, cfg

    return False, cfg


def train(cfg: DictConfig):
    resume_state, cfg = attempt_resume(cfg)
    logger.info(f"Resume state: {resume_state}, {cfg.ckpt_path}")

    logger.info("Instantiating wandb ...")
    wb = init_wandb(cfg.wandb)
    if not resume_state:
        wb.config.update(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    logger.info(f"Instantiating dataset & dataloaders for <{cfg.data.sensor}>")
    train_dataloader, val_dataloader, sensor_means, sensor_stds = get_dataloaders(cfg)

    if sensor_means is not None and sensor_stds is not None:
        with open_dict(cfg):
            cfg.algorithm.encoder.normalization = {"mean": sensor_means, "std": sensor_stds}

    train_dset = train_dataloader.dataset
    pos_mean = getattr(train_dset, "pos_mean", None)
    pos_std = getattr(train_dset, "pos_std", None)
    if pos_mean is not None and pos_std is not None:
        with open_dict(cfg):
            cfg.algorithm.encoder.pos_normalization = {"mean": pos_mean, "std": pos_std}

    trainer = Trainer(wandb_logger=wb, **cfg.trainer)

    logger.info(f"Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm = hydra.utils.instantiate(cfg.algorithm)
    _maybe_update_encoder_stats(algorithm, sensor_means, sensor_stds, pos_mean, pos_std)

    trainer.fit(algorithm, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)
    wb.finish()


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(cfg: DictConfig):
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    logger.info(f"# of nodes: {num_nodes}")
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
