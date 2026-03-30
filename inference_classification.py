import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import logging
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
import torch.utils.data as data

from tactile_ssl.utils.logging import get_pylogger
# Ensure resolvers are registered by importing train module
from tactile_ssl.build_encoder import build_encoder
from tactile_ssl.data.xela_tactile import XelaSSLDataset
import yaml
import hydra

from tactile_ssl.utils.logging import get_pylogger, print_config_tree  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = get_pylogger(__name__)

config = "config/encoder/xela_sparshskin.yaml"
data_path = "config/data/xela.yaml"
# ckpt_path = "experiments/dinov2_xela_tiny/2026.01.28-17-59/checkpoints/epoch-0050.ckpt"
ckpt_path = "experiments/dinov2_xela_no_cls_tiny/2026.02.19-12-31/checkpoints/epoch-0470.ckpt"


with open(data_path, "r") as f:
    data_cfg = yaml.safe_load(f)
data_cfg["paths"] = {'data_root': "./dataset"}
data_cfg["data"] = {'window_time': 0.1, 'window_overlap': 0.0, 'interpolating_freq':100} 

def get_dataloaders_magnetic_based(cfg: DictConfig):
    data_cfg = cfg

    def get_xela_dataset(dataset_cfg: DictConfig, dataset_name: str, d_id: int, object_class):
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
    # print('##', xela_mean, xela_std)
    logger.info(f"Compute Xela normalization: mean={xela_mean}, std={xela_std}")


    for dataset in train_datasets + val_datasets:
        dataset.update_normalization(xela_mean, xela_std)
    train_dset = data.ConcatDataset(train_datasets)
    val_dset = data.ConcatDataset(val_datasets)

    train_dataloader = data.DataLoader(train_dset, **data_cfg.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **data_cfg.val_dataloader)
    return train_dataloader, val_dataloader, (xela_mean, xela_std)

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get Dataloaders
    logger.info("Loading dataloaders...")


    data_cfg_ = OmegaConf.create(data_cfg)
    num_classes = len(data_cfg_.dataset_list[0].sequence_list)
    train_loader, val_loader, (xela_mean, xela_std) = get_dataloaders_magnetic_based(data_cfg_)


    # 2. Build Algorithm/Model
    model = build_encoder(config, device=device, mode="eval")
    model.register_buffer("xela_mean", torch.tensor(xela_mean))
    model.register_buffer("xela_std", torch.tensor(xela_std))
    model.to(device).float()
    model.eval()

    embed_dim = model.embed_dim
    classifier = nn.Linear(embed_dim, num_classes).to(device)

    # 3. Load Checkpoint
    if ckpt_path and ckpt_path != "./":
        logger.info(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Helper to find model state dict in potential Fabric wrappers
        state_dict = checkpoint
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        # print(state_dict.keys())

        # classifier_state_dict = {
        #     'weight': state_dict['online_probes.1.decoder.probe.0.weight'],
        #     'bias': state_dict['online_probes.1.decoder.probe.0.bias']
        # }
        # classifier.load_state_dict(classifier_state_dict, strict=True)

            
        # Clean state dict keys if necessary (e.g. handle 'module.' or '_forward_module.' prefixes)
        # This is a common issue when loading Lightning/Fabric checkpoints manually
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("teacher_encoder.backbone."):
                k_new = k.replace("teacher_encoder.backbone.", "")
                new_state_dict[k_new] = v
        
        state_dict = new_state_dict
        del state_dict['xela_mean']
        del state_dict['xela_std']

        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning("No checkpoint path provided or path is default './'. Using random weights.")
    # Linear Classifier (Head) 정의

    
    # Optimizer & Loss
    optimizer = optim.SGD(classifier.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    ###  5. [NEW] Training Loop (Optional)
    epochs = 10  # 학습 에폭 수 설정
    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # --- Training Phase ---
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for i, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device, dtype=torch.float32) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, list):
                batch = [v.to(device, dtype=torch.float32) if isinstance(v, torch.Tensor) else v for v in batch]
            
            gt = batch['object_classification'].long() # 라벨은 LongTensor여야 함

            # 1. Extract Features (Frozen Encoder)
            with torch.no_grad():
                tactile_rep = model.forward_features(batch['sensor'])
                cls_embedding = tactile_rep["x_norm_regtokens"].squeeze(1)
                # patch_embedding = tactile_rep["x_norm_patchtokens"].mean(1)
                # Concatenate features
                features = cls_embedding

            # 2. Forward Classifier
            outputs = classifier(features)
            loss = criterion(outputs, gt)

            # 3. Backward & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += gt.size(0)
            correct += (predicted == gt).sum().item()

            if i % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        logger.info(f"End of Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # --- Validation Phase ---
    val_loss = 0
    val_total = 0
    val_correct = 0
    classifier.eval()
    with torch.no_grad():
        for batch in val_loader:
            # Move batch
            if isinstance(batch, dict):
                batch = {k: v.to(device, dtype=torch.float32) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            gt = batch['object_classification'].long()

            # Feature Extraction
            tactile_rep = model.forward_features(batch['sensor'])
            cls_embedding = tactile_rep["x_norm_regtokens"].squeeze(1)
            # patch_embedding = tactile_rep["x_norm_patchtokens"].mean(1)
            # features = torch.cat([cls_embedding, patch_embedding], dim=1)
            features = cls_embedding

            # Prediction
            outputs = classifier(features)
            loss = criterion(outputs, gt)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += gt.size(0)
            val_correct += (predicted == gt).sum().item()

    val_acc = 100 * val_correct / val_total
    logger.info(f"Validation Result - Acc: {val_acc:.2f}%, Loss: {val_loss/len(val_loader):.4f}")
    logger.info("-" * 50)

    logger.info("Training finished.")

if __name__ == "__main__":
    main()
