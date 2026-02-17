# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from typing import List, Optional
import os

import hydra
import numpy as np
import torch
import torch.utils.data as data
from hydra.core.hydra_config import HydraConfig
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict
from copy import deepcopy

import wandb

from tactile_ssl.trainer import Trainer  # noqa: E402
from tactile_ssl.utils import get_local_rank, get_node_id
from tactile_ssl.utils.logging import get_pylogger, print_config_tree  # noqa: E402
from tactile_ssl.data.d360.utils import get_weights, get_experiment_name, get_modality_tag
from tactile_ssl.utils.combined_dataset import CombinedDataset

logger = get_pylogger(__name__)

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b))
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b)
OmegaConf.register_new_resolver("d360_expt_name", get_experiment_name)
OmegaConf.register_new_resolver("d360_modal_tag", get_modality_tag)


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


def get_dataloaders_magnetic_based(cfg: DictConfig):
    data_cfg = cfg.data

    if data_cfg.sensor == "xela":
        def get_xela_dataset(dataset_cfg: DictConfig, dataset_name: str, d_id: int, object_class: Optional[int] = None):
            data_path = f"{dataset_cfg.data_path}"
            data_files = os.listdir(data_path)
            dataset_name_exists = True in [f in f"{dataset_name}" for f in data_files]
            if not dataset_name_exists:
                print(f"Dataset {dataset_name} not found")
                return None
            dataset = hydra.utils.instantiate(
                dataset_cfg,
                data_path=f"{data_path}/{dataset_name}/{d_id}",
                object_class=object_class,
            )
            return dataset

        train_datasets, val_datasets = [], []
        dataset_list: List = data_cfg.dataset_list
        object_classes = []
        object_class_sizes = []
        for dataset_l in dataset_list:
            if dataset_l.type == "teleop":
                train_dataset_ids, val_dataset_ids = (
                    dataset_l.train_dataset_ids,
                    dataset_l.val_dataset_ids,
                )
                for obj in dataset_l.sequence_list:
                    object_classes.append(obj)
                    object_class_sizes.append(0)
                    for d_id in train_dataset_ids:
                        dataset = get_xela_dataset(
                            dataset_l.dataset, dataset_name=obj, d_id=d_id, object_class=len(object_classes) - 1
                        )
                        if dataset is not None:
                            object_class_sizes[-1] += len(dataset)
                        train_datasets.append(dataset)
                    for d_id in val_dataset_ids:
                        val_datasets.append(
                            get_xela_dataset(
                                dataset_l.dataset, dataset_name=obj, d_id=d_id, object_class=len(object_classes) - 1
                            )
                        )
            elif dataset_l.type == "joystick_control":
                with open_dict(dataset_l.dataset.config):
                    dataset_l.dataset.config.object_label = len(object_classes)
                train_dset, val_dset = hydra.utils.instantiate(dataset_l.dataset)
                object_classes.append("joystick")
                object_class_sizes.append(len(train_dset))
                train_datasets.append(train_dset)
                val_datasets.append(val_dset)

        print(f"Object class sizes: {object_class_sizes}")
        object_class_ratios = object_class_sizes / np.sum(object_class_sizes)
        object_class_weights = 1 / object_class_ratios
        object_class_weights = object_class_weights / np.sum(object_class_weights)
        print(f"Object class weights: {object_class_weights}")

        from tactile_ssl.data.xela.utils import compute_xela_normalization

        xela_mean, xela_std = compute_xela_normalization(train_datasets)
        logger.info(f"Compute Xela normalization: mean={xela_mean}, std={xela_std}")
        
        with open_dict(cfg):
            cfg.data.normalization.mean = xela_mean.tolist()
            cfg.data.normalization.std = xela_std.tolist()
            cfg.data.object_classes = object_classes
            cfg.data.object_class_weights = object_class_weights.tolist()

        for dataset in train_datasets + val_datasets:
            dataset.update_normalization(xela_mean, xela_std)
        train_dset = data.ConcatDataset(train_datasets)
        val_dset = data.ConcatDataset(val_datasets)

    elif data_cfg.sensor == "tdex":
        dataset = hydra.utils.instantiate(data_cfg.dataset)
        train_dset_size = int(len(dataset) * cfg.data.train_val_split)

        train_dset, val_dset = data.random_split(dataset, [train_dset_size, len(dataset) - train_dset_size])
    else:
        raise NotImplementedError

    return train_dset, val_dset


def get_dataloaders_d360_based(cfg: DictConfig):
    data_cfg = cfg.data
    train_main_datasets, val_main_datasets = [], []
    train_supp_datasets, val_supp_datasets = [], []

    dataset_list = data_cfg.dataset_list
    online_probe_dataset_list = (
        data_cfg.online_probe_dataset_list if data_cfg.online_probe_dataset_list is not None else []
    )
    object_labels = []
    action_labels = []
    sizes = []
    for sequence in dataset_list:
        full_dataset = hydra.utils.instantiate(data_cfg.sensor.d360.dataset, sequences=[sequence])
        online_probe_used = sequence in online_probe_dataset_list
        if online_probe_used:
            action_labels.append(full_dataset.action_label[0])
            object_labels.append(full_dataset.object_label[0])
            sizes.append(full_dataset.sequence_sizes[0])

        train_dataset, val_dataset = None, None
        if data_cfg.split.type == "device":
            train_dataset = full_dataset
            val_dataset = deepcopy(train_dataset)
            num_devices = len(train_dataset.imu_acc[0])
            devices = list(range(num_devices))
            device = np.random.randint(len(devices))
            train_dataset.remove(0, [device])
            val_dataset.remove(0, devices[:device] + devices[device + 1 :])
        elif data_cfg.split.type == "random":
            ratio = data_cfg.split.ratio
            train_size = int(ratio * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        elif data_cfg.split.type == "sequence":
            if sequence not in data_cfg.split.test_dataset_list:
                train_dataset = full_dataset
            else:
                val_dataset = full_dataset
        else:
            raise NotImplementedError

        if train_dataset is not None:
            train_main_datasets.append(train_dataset)
            if online_probe_used:
                train_supp_datasets.append(train_dataset)

        if val_dataset is not None:
            val_main_datasets.append(val_dataset)
            if online_probe_used:
                val_supp_datasets.append(val_dataset)

    object_weights = get_weights(object_labels, sizes)
    action_weights = get_weights(action_labels, sizes)

    if not object_weights:
        object_weights["default"] = 1

    if not action_weights:
        action_weights["default"] = 1

    with open_dict(cfg):
        for _, modality in cfg.data.sensor.d360.dataset.config.normalization.items():
            if "normalize" in modality:
                normalize = hydra.utils.instantiate(modality["normalize"])
                modal_avg, modal_std = normalize(train_main_datasets + val_main_datasets)
                modality["avg"] = modal_avg.tolist()
                modality["std"] = modal_std.tolist()
        cfg.data.sensor.d360.dataset.config.data.label.object.cls = list(object_weights.keys())
        cfg.data.sensor.d360.dataset.config.data.label.object.weight = list(object_weights.values())
        cfg.data.sensor.d360.dataset.config.data.label.action.cls = list(action_weights.keys())
        cfg.data.sensor.d360.dataset.config.data.label.action.weight = list(action_weights.values())

    if data_cfg.split.type == "device" or data_cfg.split.type == "sequence":
        for dataset in train_main_datasets + val_main_datasets:
            dataset.update_label_cls("object", object_weights)
            dataset.update_label_cls("action", action_weights)
            dataset.update_normalization(cfg.data.sensor.d360.dataset.config.normalization)
    elif data_cfg.split.type == "random":
        for dataset in train_main_datasets + val_main_datasets:
            dataset.dataset.update_label_cls("object", object_weights)
            dataset.dataset.update_label_cls("action", action_weights)
            dataset.dataset.update_normalization(cfg.data.sensor.d360.dataset.config.normalization)

    train_main_dset = data.ConcatDataset(train_main_datasets)
    val_main_dset = data.ConcatDataset(val_main_datasets)

    train_supp_dset = data.ConcatDataset(train_supp_datasets) if train_supp_datasets else None
    train_dset = CombinedDataset(main_dataset=train_main_dset, supp_dataset=train_supp_dset)

    val_supp_dset = data.ConcatDataset(val_supp_datasets) if val_supp_datasets else None
    val_dset = CombinedDataset(main_dataset=val_main_dset, supp_dataset=val_supp_dset)

    return train_dset, val_dset


def get_dataloaders_gelsight_based(cfg: DictConfig):
    data_cfg = cfg.data
    dataset_list = data_cfg.dataset_list
    
    # Extract sequences from all dataset items
    sequences = []
    dataset_config = None
    
    for item in dataset_list:
        if item.type == 'teleop':
            sequences.extend(item.sequence_list)
            # Use the dataset config from the first matching item
            if dataset_config is None:
                dataset_config = item.dataset
                data_path = item.dataset.data_path
                
    if not sequences:
        raise ValueError("No sequences found in dataset_list")

    print(f"Loading {len(sequences)} sequences from {data_path}")

    # Instantiate dataset with all sequences
    ds = hydra.utils.instantiate(
        dataset_config,
        sequences=sequences,
        data_path=data_path
    )

    # Calculate class weights for classification probe
    object_classes = sequences
    object_class_sizes = np.zeros(len(sequences))
    
    # helper to count per class
    # ds.images is a list of (path, class_id)
    for _, class_id in ds.images:
        object_class_sizes[class_id] += 1
        
    print(f"Object class sizes: {object_class_sizes}")
    
    # Avoid division by zero if a class has 0 images (though unlikely if folders exist)
    # Add small epsilon or handle gracefully
    if np.any(object_class_sizes == 0):
        print("Warning: Some classes have 0 images.")
        object_class_sizes = np.maximum(object_class_sizes, 1)

    object_class_ratios = object_class_sizes / np.sum(object_class_sizes)
    object_class_weights = 1 / object_class_ratios
    object_class_weights = object_class_weights / np.sum(object_class_weights)
    # print(f"Object class weights: {object_class_weights}")

    with open_dict(cfg):
        cfg.data.object_classes = object_classes
        cfg.data.object_class_weights = object_class_weights.tolist()

    # Split
    # Default to 90/10 split if not specified
    split_ratio = 0.9
    train_size = int(split_ratio * len(ds))
    val_size = len(ds) - train_size
    train_dset, val_dset = data.random_split(ds, [train_size, val_size])
    
    # Set default normalization if not present
    if cfg.data.normalization.mean is None:
        cfg.data.normalization.mean = [0.485, 0.456, 0.406]
        cfg.data.normalization.std = [0.229, 0.224, 0.225]

    return train_dset, val_dset



def get_dataloaders_actionsense_based(cfg: DictConfig):
    data_cfg = cfg.data
    dataset_list = data_cfg.dataset_list
    
    # Extract sequences from all dataset items
    sequences = []
    dataset_config = None
    data_path = None
    
    for item in dataset_list:
        if item.type == 'human':
            sequences.extend(item.sequence_list)
            # Use the dataset config from the first matching item
            if dataset_config is None:
                dataset_config = item.dataset
                data_path = hydra.utils.to_absolute_path(item.dataset.data_path)
                
    if not sequences:
        raise ValueError("No sequences found in dataset_list")

    print(f"Loading {len(sequences)} sequences from {data_path}")

    # Instantiate dataset with all sequences
    # dataset_config is e.g. tactile_ssl.data.actionsense_tactile.ActionSenseSSLDataset
    # The config inside dataset_config.config is passed to __init__
    ds = hydra.utils.instantiate(
        dataset_config,
        sequences=sequences,
        data_path=data_path
    )

    # Calculate class weights for classification probe (if needed)
    # ActionSenseSSLDataset has object_classes array
    object_classes = sequences
    object_class_sizes = np.zeros(len(sequences))
    
    # ds.object_classes contains class_id for each frame
    unique, counts = np.unique(ds.object_classes, return_counts=True)
    for class_id, count in zip(unique, counts):
        object_class_sizes[int(class_id)] = count
        
    print(f"Object class sizes: {object_class_sizes}")
    
    if np.any(object_class_sizes == 0):
        print("Warning: Some classes have 0 frames.")
        object_class_sizes = np.maximum(object_class_sizes, 1)

    object_class_ratios = object_class_sizes / np.sum(object_class_sizes)
    object_class_weights = 1 / object_class_ratios
    object_class_weights = object_class_weights / np.sum(object_class_weights)

    with open_dict(cfg):
        cfg.data.object_classes = object_classes
        cfg.data.object_class_weights = object_class_weights.tolist()

    # Split
    split_ratio = 0.9
    train_size = int(split_ratio * len(ds))
    val_size = len(ds) - train_size
    train_dset, val_dset = data.random_split(ds, [train_size, val_size])
    
    # Set default normalization if not present (ActionSense might not strictly need it if not using ImageNet pretraining on images)
    # But if we use images later, we might. For now, we can leave it or set placeholder.
    if cfg.data.normalization.mean is None:
        print("Calculating normalization statistics from data (excluding 0)...")
        data_arr = ds.sensor_data
        
        # Determine channel dim
        # Heuristic: if last dim is small (<=25) or matches in_chans (if we knew it), assume it's C.
        # Otherwise assume (N, C, H, W) -> dim 1.
        channel_dim = -1
        if data_arr.shape[-1] <= 25: 
             channel_dim = -1
        else:
             channel_dim = 1
             
        # Move channel to last for easier iteration
        if channel_dim != -1 and channel_dim != data_arr.ndim - 1:
             data_arr_perm = np.moveaxis(data_arr, channel_dim, -1)
        else:
             data_arr_perm = data_arr
             
        C = data_arr_perm.shape[-1]
        mean = []
        std = []
        
        for c in range(C):
            channel_data = data_arr_perm[..., c]
            valid_mask = channel_data != 0
            valid_data = channel_data[valid_mask]
            
            if valid_data.size > 0:
                mean.append(float(np.mean(valid_data)))
                std.append(float(np.std(valid_data)))
            else:
                mean.append(0.0)
                std.append(1.0)
        
        print(f"Computed normalization - Mean: {mean}, Std: {std}")
        
        with open_dict(cfg):
             cfg.data.normalization.mean = mean
             cfg.data.normalization.std = std

    return train_dset, val_dset


def get_dataloaders_cross_sensor_based(cfg: DictConfig):
    data_cfg = cfg.data
    dataset_source_list = data_cfg.dataset_source_list
    
    train_datasets = []
    val_datasets = []
    
    for source_config_path in dataset_source_list:
        # Resolve absolute path relative to project root or use hydra to find it
        # Assuming paths are relative to project root or can be found by hydra.utils.to_absolute_path
        abs_path = hydra.utils.to_absolute_path(source_config_path)
        logger.info(f"Loading dataset config from: {abs_path}")
        
        # Load the specific sensor config
        sensor_cfg = OmegaConf.load(abs_path)
        
        # Create a temporary full config with the loaded sensor config as 'data'
        # We need to preserve 'paths' and other global settings from the main cfg
        temp_cfg = cfg.copy()
        temp_cfg.data = sensor_cfg
        
        # Dispatch to the appropriate loader based on the sensor type in the loaded config
        # We use the main get_dataloaders function but need to be careful not to recurse strictly
        # or just call the specific helper functions based on sensor type.
        # Calling get_dataloaders(temp_cfg) is cleaner but we need to ensure it returns datasets, not loaders
        # Actually get_dataloaders returns loaders. We need datasets.
        # So we should call the specific helper functions.
        
        if "d360" in sensor_cfg.sensor: # d360 config usually has strict structure
             t_dset, v_dset = get_dataloaders_d360_based(temp_cfg)
        elif sensor_cfg.sensor in ["xela", "tdex"]:
             t_dset, v_dset = get_dataloaders_magnetic_based(temp_cfg)
        elif sensor_cfg.sensor == "gelsight":
             t_dset, v_dset = get_dataloaders_gelsight_based(temp_cfg)
        elif sensor_cfg.sensor == "actionsense":
             t_dset, v_dset = get_dataloaders_actionsense_based(temp_cfg)
        else:
             raise NotImplementedError(f"Sensor type {sensor_cfg.sensor} from {source_config_path} is not supported in cross-sensor loading.")
             
        if t_dset is not None:
            train_datasets.append(t_dset)
        if v_dset is not None:
            val_datasets.append(v_dset)

    if not train_datasets:
        raise ValueError("No training datasets loaded from source list.")

    # Combine all datasets
    # Note: This assumes all datasets return compatible items (dictionaries with same keys/shapes or handled by collate_fn)
    # The CrossSensorDinoV2 algorithm likely expects a dictionary with specific keys.
    # We might need a custom ConcatDataset if we need to track which dataset an item came from, 
    # but torch.utils.data.ConcatDataset is usually sufficient if indices aren't needed.
    
    combined_train_dset = data.ConcatDataset(train_datasets)
    combined_val_dset = data.ConcatDataset(val_datasets) if val_datasets else None
    
    return combined_train_dset, combined_val_dset

def get_dataloaders(cfg: DictConfig):
    if "d360" in cfg.data.sensor:
        train_dset, val_dset = get_dataloaders_d360_based(cfg)
    elif cfg.data.sensor in ["xela", "tdex"]:
        train_dset, val_dset = get_dataloaders_magnetic_based(cfg)
    elif cfg.data.sensor == "gelsight":
        train_dset, val_dset = get_dataloaders_gelsight_based(cfg)
    elif cfg.data.sensor == "actionsense":
        train_dset, val_dset = get_dataloaders_actionsense_based(cfg)
    train_dataloader = data.DataLoader(train_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)
    return train_dataloader, val_dataloader

def attempt_resume(cfg: DictConfig):
    ckpt_path = None
    if os.environ.get("SLURM_RESTART_COUNT") is not None:
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        requeue_count = os.environ.get("SLURM_RESTART_COUNT")
        logger.info(f"requeue count for job {slurm_job_id}: {requeue_count}")
        cfg.resume_id = slurm_job_id
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
    logger.info(f"Resume state: {resume_state}, {cfg.ckpt_path}")
    logger.info("Instantiating wandb ...")
    wandb = init_wandb(cfg.wandb)
    if ~resume_state:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    n_sensors = len(cfg.data.sensor) if OmegaConf.is_list(cfg.data.sensor) or OmegaConf.is_dict(cfg.data.sensor) else 1
    if OmegaConf.is_dict(cfg.data.sensor):
        sensors_type = [sensor for sensor in cfg.data.sensor]
    else:
        sensors_type = [cfg.data.sensor[i].type for i in range(n_sensors)] if n_sensors > 1 else cfg.data.sensor
    logger.info(f"Instantiating dataset & dataloaders for <{sensors_type}>")
    train_dataloader, val_dataloader = get_dataloaders(cfg)

    trainer = Trainer(wandb_logger=wandb, **cfg.trainer)

    logger.info(f"Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm = hydra.utils.instantiate(cfg.algorithm)

    trainer.fit(algorithm, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)

    wandb.finish()


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(cfg: DictConfig):
    """
    Main function to train the model
    """

    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    logger.info(f"# of nodes: {num_nodes}")
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
