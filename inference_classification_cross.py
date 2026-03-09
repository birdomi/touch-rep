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
from train import get_dataloaders_cross_sensor_based

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = get_pylogger(__name__)

config = "config/encoder/cross_sensor_encoder.yaml"
data_path = "config/data/cross_sensor_eval.yaml"
ckpt_path = "experiments/dinov2_cross_sensor_tiny/2026.02.24-12-19/checkpoints/epoch-0120.ckpt"

with open(data_path, "r") as f:
    data_cfg = yaml.safe_load(f)
data_cfg["paths"] = {'data_root': "./dataset"}
data_cfg["data"] = {'window_time': 0.1, 'window_overlap': 0.0, 'interpolating_freq':100} 


def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get Dataloaders
    logger.info("Loading dataloaders...")


    data_cfg_ = OmegaConf.create(data_cfg)
    with open(data_cfg_.dataset_source_list[0], "r") as f:
        dataset1_cfg = yaml.safe_load(f)
    dataset1_cfg = OmegaConf.create(dataset1_cfg)
    loader_args = dict(data_cfg_.train_dataloader)
    num_classes = len(dataset1_cfg.dataset_list[0].sequence_list)


    train_dset, val_dset, train_sampler, sensor_means, sensor_stds = get_dataloaders_cross_sensor_based(data_cfg_, is_eval=True)
    if train_sampler is not None:
        print("Using WeightedRandomSampler for training dataloader.")
        loader_args['shuffle'] = False
        loader_args['sampler'] = train_sampler
    
    train_loader = data.DataLoader(train_dset, **loader_args)
    val_loader = data.DataLoader(val_dset, **data_cfg_.val_dataloader)

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
        print(state_dict.keys())
        # print(state_dict['teacher_encoder.backbone.xela_mean'].shape)
        # print(state_dict['student_encoder.backbone.xela_mean'])


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

        model.load_state_dict(state_dict, strict=True)
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
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, list):
                batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
            
            gt = batch['object_classification']

            # 1. Extract Features (Frozen Encoder)
            with torch.no_grad():
                # print(batch['sensor_ids'])
                tactile_rep = model.forward_features(batch['sensor'], batch['sensor_poses'], sensor_ids=batch['sensor_ids'])
                cls_embedding = tactile_rep["x_norm_regtokens"].squeeze(1)
                # patch_embedding = tactile_rep["x_norm_patchtokens"].mean(1)
                # Concatenate features
                features = cls_embedding
                # print(features.shape)

            # 2. Forward Classifier
            outputs = classifier(features)
            # print(outputs.shape)
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
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            gt = batch['object_classification']

            # Feature Extraction
            tactile_rep = model.forward_features(batch['sensor'], batch['sensor_poses'], sensor_ids=batch['sensor_ids'])
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
