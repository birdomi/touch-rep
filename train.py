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
from tactile_ssl.data.cross_sensor_dataset import CrossSensorDatasetWrapper
import torch.nn.functional as F

logger = get_pylogger(__name__)


class SensorIdDatasetWrapper(data.Dataset):
    """Wraps a dataset, injects ``sensor_id`` and zero-pads the sensor channel dim.

    Used to unify BrainCo (in_chans=4) and XELA (in_chans=40) samples so they
    can be collated into the same batch by PyTorch's default_collate.

    Args:
        dataset:      Underlying dataset.
        sensor_id:    Integer sensor type (0=BrainCo, 1=XELA).
        max_channels: Target channel size after padding (default 40 = 10*4).
    """

    def __init__(self, dataset: data.Dataset, sensor_id: int, max_channels: int = 40):
        self.dataset = dataset
        self.sensor_id = sensor_id
        self.max_channels = max_channels

    def __len__(self):
        return len(self.dataset)

    # Keys kept in the output; others (e.g. wrist_positions) are dropped so
    # that BrainCo and XELA samples can be collated in the same batch.
    _KEEP_KEYS = {"sensor", "sensor_poses", "object_classification", "sensor_id"}

    def __getitem__(self, idx):
        sample = dict(self.dataset[idx])
        sensor = sample["sensor"]  # (..., C)
        c = sensor.shape[-1]
        if c < self.max_channels:
            pad = self.max_channels - c
            sensor = F.pad(sensor, (0, pad))
        sample["sensor"] = sensor
        sample["sensor_id"] = torch.tensor(self.sensor_id, dtype=torch.long)
        return {k: v for k, v in sample.items() if k in self._KEEP_KEYS}

    def __getattr__(self, name):
        return getattr(self.dataset, name)

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


def _maybe_update_encoder_stats(algorithm, sensor_means, sensor_stds, pos_mean=None, pos_std=None):
    """Call update_stats() on any available backbone/encoder modules."""
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
            print(kwargs)

            module.update_stats(**kwargs)
            logger.info(f"Called update_stats() on {name}")


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

        mean_as_list = xela_mean.tolist()
        std_as_list = xela_std.tolist()
        
        # Attach stats to the returned dataset object for easier retrieval
        train_dset = data.ConcatDataset(train_datasets)
        train_dset.sensor_mean = mean_as_list
        train_dset.sensor_std = std_as_list
        
        val_dset = data.ConcatDataset(val_datasets)
        val_dset.sensor_mean = mean_as_list
        val_dset.sensor_std = std_as_list

    elif data_cfg.sensor == "tdex":
        dataset = hydra.utils.instantiate(data_cfg.dataset)
        train_dset_size = int(len(dataset) * cfg.data.train_val_split)

        train_dset, val_dset = data.random_split(dataset, [train_dset_size, len(dataset) - train_dset_size])
        
        # Default stats for Tdex
        train_dset.sensor_mean = [0.0] * 3 
        train_dset.sensor_std = [1.0] * 3
        if val_dset:
            val_dset.sensor_mean = [0.0] * 3
            val_dset.sensor_std = [1.0] * 3
    else:
        raise NotImplementedError

    return train_dset, val_dset#, mean_as_list, std_as_list 


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
    ds = hydra.utils.instantiate(
        dataset_config,
        sequences=sequences,
        data_path=data_path
    )

    # Calculate class weights
    object_classes = sequences
    object_class_sizes = np.zeros(len(sequences))
    
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
    
    # Set normalization
    if cfg.data.normalization.mean is None:
        print("Calculating normalization statistics from data (excluding 0)...")
        data_arr = ds.sensor_data
        
        # Determine channel dim
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
        
        # Calculate unified stats across all channels
        all_valid_data = []
        for c in range(C):
            channel_data = data_arr_perm[..., c]
            valid_mask = channel_data != 0
            valid_data = channel_data[valid_mask]
            if valid_data.size > 0:
                all_valid_data.append(valid_data)
        
        if len(all_valid_data) > 0:
            concatenated_data = np.concatenate(all_valid_data)
            unified_mean = float(np.mean(concatenated_data))
            unified_std = float(np.std(concatenated_data))
        else:
            unified_mean = 0.0
            unified_std = 1.0
            
        mean = [unified_mean] * C
        std = [unified_std] * C
                
        # Update config and datasets
        with open_dict(cfg):
            cfg.data.normalization.mean = mean
            cfg.data.normalization.std = std
            
        # For random_split subsets:
        if hasattr(train_dset, "dataset"):
             train_dset.dataset.update_normalization(mean, std)
             if val_dset:
                  val_dset.dataset.update_normalization(mean, std)
        elif hasattr(train_dset, "update_normalization"):
             train_dset.update_normalization(mean, std)
             if val_dset:
                  val_dset.update_normalization(mean, std)
        
        # Attach to dataset object for cross-sensor retrieval
        train_dset.sensor_mean = mean
        train_dset.sensor_std = std
        if val_dset:
            val_dset.sensor_mean = mean
            val_dset.sensor_std = std
            
        logger.info(f"Computed and applied UNIFIED ActionSense stats: Mean={unified_mean}, Std={unified_std}")
        
    else:
        # If config already has it, still attach for cross-sensor consistency
        m = cfg.data.normalization.mean
        s = cfg.data.normalization.std
        train_dset.sensor_mean = m
        train_dset.sensor_std = s
        if val_dset:
            val_dset.sensor_mean = m
            val_dset.sensor_std = s

    return train_dset, val_dset

def get_dataloaders_cross_sensor_based(cfg: DictConfig, is_eval=False):
    if not is_eval:
        data_cfg = cfg.data
    else:
        data_cfg = cfg
    dataset_source_list = data_cfg.dataset_source_list
    
    train_datasets = []
    val_datasets = []
    
    sensor_means = []
    sensor_stds = []
    
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

        # Default values for mean/std
        mean = [0.0] * data_cfg.max_channels
        std = [1.0] * data_cfg.max_channels

        if t_dset is not None:
             # Retrieve pre-computed stats from the loader
             if hasattr(t_dset, "sensor_mean") and hasattr(t_dset, "sensor_std"):
                 mean = t_dset.sensor_mean
                 std = t_dset.sensor_std
                 logger.info(f"Retrieved stats from dataset object for {source_config_path}: Mean={mean}, Std={std}")
             else:
                 # Fallback if not attached (e.g. d360 or gelsight if not updated)
                 # Check config first
                 if hasattr(temp_cfg.data, "normalization") and temp_cfg.data.normalization.mean is not None:
                     m = temp_cfg.data.normalization.mean
                     s = temp_cfg.data.normalization.std
                     if isinstance(m, (list, tuple)):
                         mean = list(m)
                     if isinstance(s, (list, tuple)):
                          std = list(s)
                     logger.info(f"Using config stats for {source_config_path}")
                 else:
                     logger.warning(f"No stats found for {source_config_path}. Using identity.")
             
             sensor_means.append(mean)
             sensor_stds.append(std)

        else:
             # Fallback if no train data? Should not happen
             logger.warning(f"No train data for {source_config_path}. Appending default identity.")
             sensor_means.append(mean)
             sensor_stds.append(std)

        # Wrap with CrossSensorDatasetWrapper
        # Use sensor index in source list as ID (0, 1, ...)
        sensor_id = dataset_source_list.index(source_config_path)
        max_taxels=data_cfg.max_taxels
        max_channels=data_cfg.max_channels

        if t_dset is not None:
            t_dset_wrapped = CrossSensorDatasetWrapper(t_dset, sensor_id=sensor_id, max_taxels=max_taxels, max_channels=max_channels)
            train_datasets.append(t_dset_wrapped)
        if v_dset is not None:
            v_dset_wrapped = CrossSensorDatasetWrapper(v_dset, sensor_id=sensor_id, max_taxels=max_taxels, max_channels=max_channels)
            val_datasets.append(v_dset_wrapped)

    if not train_datasets:
        raise ValueError("No training datasets loaded from source list.")
        
    combined_train_dset = data.ConcatDataset(train_datasets)
    combined_val_dset = data.ConcatDataset(val_datasets) if val_datasets else None
    
    train_sampler = None
    if "sampling_ratios" in data_cfg and data_cfg.sampling_ratios is not None:
        sampling_ratios = data_cfg.sampling_ratios
        if len(sampling_ratios) != len(train_datasets):
            print(f"Warning: sampling_ratios length {len(sampling_ratios)} does not match number of datasets {len(train_datasets)}. Using default sampling.")
        else:
            print(f"Using sampling ratios: {sampling_ratios}")
            weights = []
            for idx, dset in enumerate(train_datasets):
                d_len = len(dset)
                if d_len > 0:
                    # Weight per sample = target_ratio / dataset_size
                    # We normalize so sum(ratios) = 1 (or close to it)
                    w = sampling_ratios[idx] / d_len
                    weights.extend([w] * d_len)
                else:
                    print(f"Warning: Dataset {idx} has 0 length.")
            
            if weights:
                weights = torch.tensor(weights, dtype=torch.double)
                # Replacement=True by default for WeightedRandomSampler
                # num_samples can be sum of lengths or custom
                train_sampler = data.WeightedRandomSampler(weights, len(weights))

    return combined_train_dset, combined_val_dset, train_sampler, sensor_means, sensor_stds

def get_dataloaders_gigahands_based(cfg: DictConfig):
    """Load GigaHands, OakInkV2, or combined dataset for DINOv2 SSL pretraining."""
    from tactile_ssl.data.oakinkv2_tactile import OakInkV2TactileDataset

    data_cfg = cfg.data
    sensor   = data_cfg.get("sensor", "gigahands")
    window_size   = int(data_cfg.get("window_size", 3))
    window_stride = int(data_cfg.get("window_stride", 1))
    train_val_split = float(data_cfg.get("train_val_split", 0.9))
    scenes = list(data_cfg.scenes) if data_cfg.get("scenes") else None

    if sensor == "gigahands_oakinkv2":
        # ── Combined: GigaHands + OakInkV2 ──────────────────────────────────
        gh_kwargs = dict(
            data_root=data_cfg.gigahands_data_root,
            window_size=window_size, window_stride=window_stride,
            train_val_split=train_val_split, scenes=scenes,
        )
        oi_kwargs = dict(
            data_root=data_cfg.oakinkv2_data_root,
            window_size=window_size, window_stride=window_stride,
            train_val_split=train_val_split, scenes=scenes,
        )
        gh_train = GigaHandsTactileDataset(split="train", **gh_kwargs)
        gh_val   = GigaHandsTactileDataset(split="val",   **gh_kwargs)
        oi_train = OakInkV2TactileDataset(split="train",  **oi_kwargs)
        oi_val   = OakInkV2TactileDataset(split="val",    **oi_kwargs)

        # Compute normalization from both train sets combined
        all_contact = (
            [seq.joint_data[..., 3].reshape(-1) for seq in gh_train._sequences] +
            [seq.joint_data[..., 3].reshape(-1) for seq in oi_train._sequences]
        )
        train_datasets = [gh_train, oi_train]
        val_datasets   = [gh_val,   oi_val]
    elif sensor == "oakinkv2_arctic":
        # ── Combined: OakInkV2 + Arctic ─────────────────────────────────────
        oi_kwargs = dict(
            data_root=data_cfg.oakinkv2_data_root,
            window_size=window_size, window_stride=window_stride,
            train_val_split=train_val_split, scenes=scenes,
        )
        ar_kwargs = dict(
            data_root=data_cfg.arctic_data_root,
            window_size=window_size, window_stride=window_stride,
            train_val_split=train_val_split, scenes=scenes,
        )
        oi_train = OakInkV2TactileDataset(split="train", **oi_kwargs)
        oi_val   = OakInkV2TactileDataset(split="val",   **oi_kwargs)
        ar_train = OakInkV2TactileDataset(split="train", **ar_kwargs)
        ar_val   = OakInkV2TactileDataset(split="val",   **ar_kwargs)

        # Compute normalization from both train sets combined
        all_contact = (
            [seq.joint_data[..., 3].reshape(-1) for seq in oi_train._sequences] +
            [seq.joint_data[..., 3].reshape(-1) for seq in ar_train._sequences]
        )
        train_datasets = [oi_train, ar_train]
        val_datasets   = [oi_val,   ar_val]
    else:
        # ── Single dataset ──────────────────────────────────────────────────
        dataset_cls = OakInkV2TactileDataset if sensor in ("oakinkv2", "taco") else GigaHandsTactileDataset
        common_kwargs = dict(
            data_root=data_cfg.data_root,
            window_size=window_size, window_stride=window_stride,
            train_val_split=train_val_split, scenes=scenes,
        )
        train_ds = dataset_cls(split="train", **common_kwargs)
        val_ds   = dataset_cls(split="val",   **common_kwargs)
        all_contact = [seq.joint_data[..., 3].reshape(-1) for seq in train_ds._sequences]
        train_datasets = [train_ds]
        val_datasets   = [val_ds]

    # ── Compute c-channel normalization stats ────────────────────────────────
    logger.info(f"Computing {sensor} contact normalization stats…")
    if all_contact:
        arr      = np.concatenate(all_contact)
        mean_val = float(np.mean(arr))
        std_val  = float(np.std(arr))
        std_val  = std_val if std_val > 1e-6 else 1.0
    else:
        mean_val, std_val = 0.0, 1.0
    logger.info(f"  contact  mean={mean_val:.6f}  std={std_val:.6f}")

    with open_dict(cfg):
        cfg.data.normalization.mean = [mean_val]
        cfg.data.normalization.std  = [std_val]

    for ds in train_datasets + val_datasets:
        ds.update_normalization(mean_val, std_val)

    train_dset = data.ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dset   = data.ConcatDataset(val_datasets)   if len(val_datasets)   > 1 else val_datasets[0]

    train_dset.sensor_mean = [mean_val]
    train_dset.sensor_std  = [std_val]
    val_dset.sensor_mean   = [mean_val]
    val_dset.sensor_std    = [std_val]

    # ── Compute pose normalization stats for OakInkV2 wrist-local poses ─────
    if sensor in ("oakinkv2", "oakinkv2_arctic"):
        logger.info(f"Computing {sensor} pose normalization stats…")
        if sensor == "oakinkv2_arctic":
            pose_sequences = oi_train._sequences + ar_train._sequences
        else:
            pose_sequences = train_ds._sequences
        all_pos = [seq.joint_data[..., :3].reshape(-1, 3) for seq in pose_sequences]
        if all_pos:
            pos_arr = np.concatenate(all_pos, axis=0)
            pos_mean = np.mean(pos_arr, axis=0).tolist()
            pos_std = [float(s) if s > 1e-6 else 1.0 for s in np.std(pos_arr, axis=0)]
        else:
            pos_mean = [0.0, 0.0, 0.0]
            pos_std = [0.01, 0.01, 0.01]
            # pos_std = [1.0, 1.0, 1.0]
        logger.info(f"  pose     mean={pos_mean}  std={pos_std}")
        train_dset.pos_mean = pos_mean
        train_dset.pos_std = pos_std
        val_dset.pos_mean = pos_mean
        val_dset.pos_std = pos_std

    logger.info(
        f"{sensor}: {len(train_dset)} train windows, "
        f"{len(val_dset)} val windows  (window_size={window_size})"
    )
    return train_dset, val_dset


def get_dataloaders_brainco_based(cfg: DictConfig):
    data_cfg = cfg.data
    
    def get_brainco_dataset(dataset_cfg: DictConfig, obj_name: str, episode_id: str, obj_class: int):
        data_path = dataset_cfg.get("data_path", "dataset/brainco/pretraining")
        full_path = f"{data_path}/{obj_name}/{episode_id}"
        if not os.path.exists(full_path):
            print(f"Dataset path {full_path} not found")
            return None
            
        dataset = hydra.utils.instantiate(
            dataset_cfg,
            data_path=full_path,
            object_class=obj_class,
        )
        return dataset

    train_datasets, val_datasets = [], []
    object_classes = []
    object_class_sizes = []
    
    dataset_list = data_cfg.dataset_list
    for dataset_l in dataset_list:
        if dataset_l.type == "teleop":
            train_episodes = dataset_l.get("train_dataset_ids", [])
            val_episodes = dataset_l.get("val_dataset_ids", [])
            for obj in dataset_l.get("sequence_list", []):
                object_classes.append(obj)
                object_class_sizes.append(0)
                obj_class_idx = len(object_classes) - 1
                
                for ep in train_episodes:
                    ep_str = f"episode_{int(ep):04d}"
                    ds = get_brainco_dataset(dataset_l.dataset, obj, ep_str, obj_class_idx)
                    if ds is not None:
                        object_class_sizes[-1] += len(ds)
                        train_datasets.append(ds)
                        
                for ep in val_episodes:
                    ep_str = f"episode_{int(ep):04d}"
                    ds = get_brainco_dataset(dataset_l.dataset, obj, ep_str, obj_class_idx)
                    if ds is not None:
                        val_datasets.append(ds)
    
    if not train_datasets:
        # Fallback to single dataset loading if dataset_list not provided/usable
        dataset = hydra.utils.instantiate(data_cfg.dataset)
        train_dset_size = int(len(dataset) * cfg.data.get("train_val_split", 0.9))
        train_dset, val_dset = data.random_split(dataset, [train_dset_size, len(dataset) - train_dset_size])
    else:
        train_dset = data.ConcatDataset(train_datasets)
        val_dset = data.ConcatDataset(val_datasets) if val_datasets else None

        # Stats
        print(f"Brainco object class sizes: {object_class_sizes}")
        if np.sum(object_class_sizes) > 0:
            object_class_ratios = np.array(object_class_sizes) / np.sum(object_class_sizes)
            object_class_weights = 1 / (object_class_ratios + 1e-8)
            object_class_weights = object_class_weights / np.sum(object_class_weights)
            
            with open_dict(cfg):
                cfg.data.object_classes = object_classes
                cfg.data.object_class_weights = object_class_weights.tolist()

    # Pre-calculated or dummy stats for brainco
    mean = [0.0] * 4
    std = [1.0] * 4

    # Calculate BrainCo dataset statistics from actual sensor tensors.
    # This supports both:
    #   - BraincoSSLDataset / tactile_array      -> sensor[..., 4]
    #   - BraincoGraspDetectionWorldDataset      -> sensor[..., 7]
    print("Calculating Brainco dataset statistics...")
    sensor_windows = []
    src_datasets = train_dset.datasets if hasattr(train_dset, "datasets") else [train_dset]
    for ds in src_datasets:
        inner = ds.dataset if hasattr(ds, "dataset") else ds
        if hasattr(inner, "windows") and len(inner.windows) > 0:
            sensor_windows.extend([w["sensor"].cpu().numpy() for w in inner.windows])
        elif hasattr(inner, "tactile_array"):
            sensor_windows.append(inner.tactile_array.astype(np.float32))

    if sensor_windows:
        all_data_np = np.concatenate(sensor_windows, axis=0)  # (Total_T, num_sensors, channels)
        num_chans = all_data_np.shape[-1]
        calc_mean = []
        calc_std = []
        for c in range(num_chans):
            ch = all_data_np[..., c].astype(np.float32)
            # BrainCo tactile channel-2 uses -1 as invalid sentinel.
            valid = ch[ch >= 0] if c == 2 else ch.reshape(-1)
            if valid.size == 0:
                calc_mean.append(0.0)
                calc_std.append(1.0)
            else:
                calc_mean.append(float(valid.mean()))
                calc_std.append(float(max(valid.std(), 1e-6)))

        mean = calc_mean
        std = calc_std
        print(f"Calculated mean: {mean}")
        print(f"Calculated std: {std}")
    else:
        print("Could not calculate stats, using defaults")
    
    if cfg.data.get("normalization") and cfg.data.normalization.get("mean") is None:
        with open_dict(cfg):
            cfg.data.normalization.mean = mean
            cfg.data.normalization.std = std
    elif cfg.data.get("normalization"):
        mean = cfg.data.normalization.mean
        std = cfg.data.normalization.std

    train_dset.sensor_mean = mean
    train_dset.sensor_std = std
    if val_dset:
        val_dset.sensor_mean = mean
        val_dset.sensor_std = std

    # Compute pose normalization stats from fingertip world/local positions when available
    # all_pos = []
    # src_datasets = train_dset.datasets if hasattr(train_dset, "datasets") else [train_dset]
    # for ds in src_datasets:
    #     inner = ds.dataset if hasattr(ds, "dataset") else ds
    #     if hasattr(inner, "windows") and len(inner.windows) > 0 and "sensor_poses" in inner.windows[0]:
    #         all_pos.extend([w["sensor_poses"].cpu().numpy().reshape(-1, 3) for w in inner.windows])
    #     elif hasattr(inner, "fingertip_rel"):
    #         all_pos.append(inner.fingertip_rel.reshape(-1, 3).astype(np.float32))
    # if all_pos:
    #     pos_arr = np.concatenate(all_pos, axis=0)
    #     pos_mean = np.mean(pos_arr, axis=0).tolist()
    #     pos_std  = [float(s) if s > 1e-6 else 1.0 for s in np.std(pos_arr, axis=0)]
    # else:
    #     pos_mean = [0.0, 0.0, 0.0]
    #     pos_std  = [1.0, 1.0, 1.0]
    # logger.info(f"BrainCo pose mean: {pos_mean}  std: {pos_std}")
    # train_dset.pos_mean = pos_mean
    # train_dset.pos_std  = pos_std
    # if val_dset:
    #     val_dset.pos_mean = pos_mean
    #     val_dset.pos_std  = pos_std

    return train_dset, val_dset


def get_dataloaders_brainco_xela_based(cfg: DictConfig):
    """Load BrainCo and XELA-grouped datasets for multi-sensor joint training.

    BrainCo:    sensor=(1,10,4)  wrapped → sensor=(1,10,40) + sensor_id=0
    XELA:       sensor=(1,10,40)          + sensor_id=1 (from dataset directly)

    Per-sensor 4-channel stats are computed and stored as [[b_mean_4ch], [x_mean_4ch]]
    so that MultiSensorBraincoTransformer can normalise each sensor type independently.

    dataset_list entries must have a 'sensor_type' field:
      sensor_type: brainco      → BraincoSSLDataset  (episode-based loading)
      sensor_type: xela_grouped → XelaSSLDatasetGrouped (sequence/id loading)
    """
    XELA_MAX_CHANNELS = 4    # BrainCo/XELA 모두 in_chans=4로 통일

    data_cfg = cfg.data
    # Keep raw (unwrapped) train datasets for stats computation
    brainco_train_raw: list = []
    xela_train_raw:   list = []
    train_datasets, val_datasets = [], []
    train_dataset_types: list = []   # 'brainco' or 'xela' per dataset in train_datasets
    object_classes: list = []
    object_class_sizes: list = []

    for dataset_l in data_cfg.dataset_list:
        sensor_type = dataset_l.get("sensor_type", "brainco")
        train_ids = dataset_l.get("train_dataset_ids", [])
        val_ids   = dataset_l.get("val_dataset_ids",   [])

        if sensor_type == "brainco":
            # Episode-based loading: data_path/<obj>/<episode_XXXX>/
            for obj in dataset_l.get("sequence_list", []):
                object_classes.append(f"brainco_{obj}")
                object_class_sizes.append(0)
                obj_class_idx = len(object_classes) - 1

                for ep in train_ids:
                    ep_str = f"episode_{int(ep):04d}"
                    full_path = f"{dataset_l.dataset.get('data_path', 'dataset/brainco/pretraining')}/{obj}/{ep_str}"
                    if not os.path.exists(full_path):
                        logger.warning(f"BrainCo path not found: {full_path}")
                        continue
                    ds = hydra.utils.instantiate(
                        dataset_l.dataset, data_path=full_path, object_class=obj_class_idx
                    )
                    object_class_sizes[-1] += len(ds)
                    brainco_train_raw.append(ds)
                    # Wrap: adds sensor_id=0, pads channels 4→40
                    train_datasets.append(
                        SensorIdDatasetWrapper(ds, sensor_id=0, max_channels=XELA_MAX_CHANNELS)
                    )
                    train_dataset_types.append("brainco")

                for ep in val_ids:
                    ep_str = f"episode_{int(ep):04d}"
                    full_path = f"{dataset_l.dataset.get('data_path', 'dataset/brainco/pretraining')}/{obj}/{ep_str}"
                    if not os.path.exists(full_path):
                        continue
                    ds = hydra.utils.instantiate(
                        dataset_l.dataset, data_path=full_path, object_class=obj_class_idx
                    )
                    val_datasets.append(
                        SensorIdDatasetWrapper(ds, sensor_id=0, max_channels=XELA_MAX_CHANNELS)
                    )

        elif sensor_type == "xela_grouped":
            # Sequence/id-based loading: data_path/<sequence>/<id>/
            for seq in dataset_l.get("sequence_list", []):
                object_classes.append(f"xela_{seq}")
                object_class_sizes.append(0)
                obj_class_idx = len(object_classes) - 1

                base_path = dataset_l.dataset.get("data_path", "")

                for d_id in train_ids:
                    full_path = f"{base_path}/{seq}/{d_id}"
                    if not os.path.exists(full_path):
                        logger.warning(f"XELA path not found: {full_path}")
                        continue
                    ds = hydra.utils.instantiate(
                        dataset_l.dataset, data_path=full_path, object_class=obj_class_idx
                    )
                    object_class_sizes[-1] += len(ds)
                    xela_train_raw.append(ds)
                    train_datasets.append(ds)  # already outputs sensor_id=1 and 40 channels
                    train_dataset_types.append("xela")

                for d_id in val_ids:
                    full_path = f"{base_path}/{seq}/{d_id}"
                    if not os.path.exists(full_path):
                        continue
                    ds = hydra.utils.instantiate(
                        dataset_l.dataset, data_path=full_path, object_class=obj_class_idx
                    )
                    val_datasets.append(ds)

        else:
            raise NotImplementedError(f"Unknown sensor_type '{sensor_type}' in brainco_xela dataset_list")

    if not train_datasets:
        raise ValueError("No training datasets loaded for brainco_xela sensor.")

    train_dset = data.ConcatDataset(train_datasets)
    val_dset   = data.ConcatDataset(val_datasets) if val_datasets else None

    # ── Compute per-sensor 4-channel stats ───────────────────────────────────
    # Result shape: [[b0,b1,b2,b3], [x0,x1,x2,x3]]  → (2, 4) in torch
    logger.info("Computing per-sensor normalization stats for brainco_xela…")

    def _nanstats(arrays_4ch):
        """Compute mean/std over a list of (N, 4) arrays, ignoring zeros."""
        if not arrays_4ch:
            return [0.0]*4, [1.0]*4
        combined = np.concatenate(arrays_4ch, axis=0)
        combined = np.where(combined == 0, np.nan, combined)
        m  = np.nanmean(combined, axis=0).tolist()
        s_ = np.nanstd(combined,  axis=0)
        s  = [float(v) if v > 1e-6 else 1.0 for v in s_]
        return m, s

    # BrainCo stats
    brainco_flat = []
    for ds in brainco_train_raw:
        if hasattr(ds, "tactile_array"):
            brainco_flat.append(ds.tactile_array.reshape(-1, 4).astype(np.float32))
    brainco_mean, brainco_std = _nanstats(brainco_flat)
    logger.info(f"  BrainCo mean: {brainco_mean}  std: {brainco_std}")

    # XELA stats (4-channel: xyz + L2 norm, matching _group_sensor_data output)
    xela_flat = []
    for ds in xela_train_raw:
        if hasattr(ds, "xela_array"):
            raw = ds.xela_array[..., 1:].reshape(-1, 3).astype(np.float32)  # (N*368, 3)
            l2  = np.linalg.norm(raw, axis=-1, keepdims=True)               # (N*368, 1)
            xela_flat.append(np.concatenate([raw, l2], axis=-1))            # (N*368, 4)
    xela_mean, xela_std = _nanstats(xela_flat)
    logger.info(f"  XELA mean: {xela_mean}  std: {xela_std}")

    # Per-sensor stats as list-of-lists for Hydra / torch.tensor() compatibility
    per_sensor_mean = [brainco_mean, xela_mean]   # [[4 floats], [4 floats]]
    per_sensor_std  = [brainco_std,  xela_std]

    with open_dict(cfg):
        cfg.data.normalization.mean = per_sensor_mean
        cfg.data.normalization.std  = per_sensor_std
        if object_class_sizes and np.sum(object_class_sizes) > 0:
            ratios  = np.array(object_class_sizes) / np.sum(object_class_sizes)
            weights = 1.0 / (ratios + 1e-8)
            weights = (weights / np.sum(weights)).tolist()
            cfg.data.object_classes       = object_classes
            cfg.data.object_class_weights = weights

    train_dset.sensor_mean = per_sensor_mean
    train_dset.sensor_std  = per_sensor_std
    if val_dset is not None:
        val_dset.sensor_mean = per_sensor_mean
        val_dset.sensor_std  = per_sensor_std

    # ── Compute pose normalization stats (3D xyz, wrist-relative) ────────────
    all_pos = []
    for ds in brainco_train_raw:
        if hasattr(ds, "fingertip_rel"):
            all_pos.append(ds.fingertip_rel.reshape(-1, 3).astype(np.float32))
    for ds in xela_train_raw:
        if hasattr(ds, "grouped_poses"):
            all_pos.append(ds.grouped_poses[..., :3].reshape(-1, 3).astype(np.float32))
    if all_pos:
        pos_arr  = np.concatenate(all_pos, axis=0)
        pos_mean = np.mean(pos_arr, axis=0).tolist()
        pos_std  = [float(s) if s > 1e-6 else 1.0 for s in np.std(pos_arr, axis=0)]
    else:
        pos_mean = [0.0, 0.0, 0.0]
        pos_std  = [1.0, 1.0, 1.0]
    logger.info(f"Pose norm  mean: {pos_mean}  std: {pos_std}")
    train_dset.pos_mean = pos_mean
    train_dset.pos_std  = pos_std
    if val_dset is not None:
        val_dset.pos_mean = pos_mean
        val_dset.pos_std  = pos_std

    # ── 50:50 WeightedRandomSampler ──────────────────────────────────────────
    brainco_total = sum(len(ds) for ds in brainco_train_raw)
    xela_total    = sum(len(ds) for ds in xela_train_raw)
    train_sampler = None
    if brainco_total > 0 and xela_total > 0:
        brainco_w = 0.5 / brainco_total
        xela_w    = 0.5 / xela_total
        sample_weights = []
        for ds, stype in zip(train_datasets, train_dataset_types):
            w = brainco_w if stype == "brainco" else xela_w
            sample_weights.extend([w] * len(ds))
        sample_weights = torch.FloatTensor(sample_weights)
        train_sampler = data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dset),
            replacement=True,
        )
        logger.info(
            f"WeightedRandomSampler: BrainCo {brainco_total} samples, "
            f"XELA {xela_total} samples → 50:50 ratio"
        )

    return train_dset, val_dset, train_sampler


def get_dataloaders(cfg: DictConfig):
    train_sampler = None
    if "d360" in cfg.data.sensor:
        train_dset, val_dset = get_dataloaders_d360_based(cfg)
    elif cfg.data.sensor in ["xela", "tdex"]:
        train_dset, val_dset= get_dataloaders_magnetic_based(cfg)
        sensor_means, sensor_stds = train_dset.sensor_mean, train_dset.sensor_std
    elif cfg.data.sensor == "gelsight":
        train_dset, val_dset = get_dataloaders_gelsight_based(cfg)
    elif cfg.data.sensor == "actionsense":
        train_dset, val_dset = get_dataloaders_actionsense_based(cfg)
    elif cfg.data.sensor in ["gigahands", "oakinkv2", "gigahands_oakinkv2", "oakinkv2_arctic", "taco"]:
        train_dset, val_dset = get_dataloaders_gigahands_based(cfg)
        sensor_means, sensor_stds = train_dset.sensor_mean, train_dset.sensor_std
    elif cfg.data.sensor == "brainco":
        train_dset, val_dset = get_dataloaders_brainco_based(cfg)
        sensor_means, sensor_stds = train_dset.sensor_mean, train_dset.sensor_std
    elif cfg.data.sensor == "brainco_xela":
        train_dset, val_dset, train_sampler = get_dataloaders_brainco_xela_based(cfg)
        sensor_means, sensor_stds = train_dset.sensor_mean, train_dset.sensor_std
    elif cfg.data.sensor == "cross_sensor":
        train_dset, val_dset, train_sampler, sensor_means, sensor_stds = get_dataloaders_cross_sensor_based(cfg)
    else:
        sensor_means, sensor_stds = None, None
    
    loader_args = dict(cfg.data.train_dataloader)
    if train_sampler is not None:
        print("Using WeightedRandomSampler for training dataloader.")
        loader_args['shuffle'] = False
        loader_args['sampler'] = train_sampler
    
    train_dataloader = data.DataLoader(train_dset, **loader_args)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)
    return train_dataloader, val_dataloader, sensor_means, sensor_stds

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
    train_dataloader, val_dataloader, sensor_means, sensor_stds = get_dataloaders(cfg)

    # Inject sensor normalization stats into algorithm config if available
    if sensor_means is not None and sensor_stds is not None:
        with open_dict(cfg):
            # Also inject into encoder config so it's passed during instantiation
            cfg.algorithm.encoder.normalization = {"mean": sensor_means, "std": sensor_stds}

    # Inject pose normalization stats if available
    train_dset = train_dataloader.dataset
    pos_mean = getattr(train_dset, "pos_mean", None)
    pos_std  = getattr(train_dset, "pos_std",  None)
    is_brainco_cat = "brainco_cat" in str(cfg.algorithm.encoder._target_)
    if (not is_brainco_cat) and pos_mean is not None and pos_std is not None:
        with open_dict(cfg):
            cfg.algorithm.encoder.pos_normalization = {"mean": pos_mean, "std": pos_std}

    trainer = Trainer(wandb_logger=wandb, **cfg.trainer)

    logger.info(f"Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm = hydra.utils.instantiate(cfg.algorithm)

    if is_brainco_cat:
        _maybe_update_encoder_stats(algorithm, sensor_means, sensor_stds)
    else:
        _maybe_update_encoder_stats(algorithm, sensor_means, sensor_stds, pos_mean, pos_std)

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
