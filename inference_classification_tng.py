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

config = "config/encoder/tng_encoder.yaml"
data_path = "config/data/tng.yaml"
ckpt_path = "experiments/dinov2_tng_tiny/2026.02.05-09-50/checkpoints/last.ckpt"

with open(data_path, "r") as f:
    data_cfg = yaml.safe_load(f)
data_cfg["paths"] = {'data_root': "./dataset"}
# data_cfg["data"] = {'window_time': 0.1, 'window_overlap': 0.0, 'interpolating_freq':100} 

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

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get Dataloaders
    logger.info("Loading dataloaders...")


    data_cfg_ = OmegaConf.create(data_cfg)
    num_classes = len(data_cfg_.dataset_list[0].sequence_list)
    print(num_classes)

    # Wrap data config to match train.py expectation (cfg.data)
    full_cfg = OmegaConf.create({'data': data_cfg_})
    train_dset, val_dset = get_dataloaders_gelsight_based(full_cfg)
    train_loader = data.DataLoader(train_dset, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(val_dset, batch_size=64, shuffle=False)


    # 2. Build Algorithm/Model
    model = build_encoder(config, device=device, mode="eval")
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

        classifier_state_dict = {
            'weight': state_dict['online_probes.1.decoder.probe.0.weight'],
            'bias': state_dict['online_probes.1.decoder.probe.0.bias']
        }
        classifier.load_state_dict(classifier_state_dict, strict=True)

            
        # Clean state dict keys if necessary (e.g. handle 'module.' or '_forward_module.' prefixes)
        # This is a common issue when loading Lightning/Fabric checkpoints manually
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("teacher_encoder.backbone."):
                k_new = k.replace("teacher_encoder.backbone.", "")
                new_state_dict[k_new] = v
        
        state_dict = new_state_dict
        # Remove xela normalization keys if present
        state_dict.pop('xela_mean', None)
        state_dict.pop('xela_std', None)

        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning("No checkpoint path provided or path is default './'. Using random weights.")
    # Linear Classifier (Head) 정의

    
    # Optimizer & Loss
    optimizer = optim.SGD(classifier.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    ###  5. [NEW] Training Loop (Optional)
    epochs = 5  # 학습 에폭 수 설정
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

            if i % 10 == 0:
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
