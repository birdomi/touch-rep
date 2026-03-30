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
from tactile_ssl.downstream_task.attentive_pooler import AttentiveClassifier

import yaml
import hydra

print("Imports successful up to hydra", flush=True)

from tactile_ssl.utils.logging import get_pylogger, print_config_tree  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = get_pylogger(__name__)

config = "config/encoder/brainco.yaml"
data_path = "config/data/brainco.yaml"
ckpt_path = "./checkpoints/dinov2_brainco_tiny/epoch-0100_.ckpt"

with open(data_path, "r") as f:
    data_cfg = yaml.safe_load(f)
data_cfg["data"] = {'window_time': 0.01, 'window_overlap': 0.0, 'interpolating_freq':100}
data_path = "./dataset/brainco/pretraining/"

def main():
    print("Starting main", flush=True)
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get Dataloaders
    logger.info("Loading dataloaders...")


    data_cfg_ = OmegaConf.create(data_cfg)
    num_classes = len(data_cfg_.dataset_list[0].sequence_list)
    from train import get_dataloaders_brainco_based
    cfg_wrapper = OmegaConf.create({"data": data_cfg_, "paths":{'data_root':"./dataset"}})

    train_dset, val_dset = get_dataloaders_brainco_based(cfg_wrapper)
    
    train_loader = data.DataLoader(train_dset, **data_cfg_.train_dataloader)
    val_loader = data.DataLoader(val_dset, **data_cfg_.val_dataloader)
    
    brainco_mean, brainco_std = train_dset.sensor_mean, train_dset.sensor_std

    # 2. Build Algorithm/Model
    model = build_encoder(config, device=device, mode="eval")
    model.register_buffer("signal_mean", torch.tensor(brainco_mean))
    model.register_buffer("signal_std", torch.tensor(brainco_std))
    model.to(device).float()
    model.eval()

    embed_dim = model.embed_dim
    classifier = AttentiveClassifier(
        embed_dim=embed_dim, 
        num_classes=num_classes, 
        num_heads=embed_dim // 64
    ).to(device)

    # 3. Load Checkpoint
    if ckpt_path and ckpt_path != "./":
        logger.info(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Helper to find model state dict in potential Fabric wrappers
        state_dict = checkpoint
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        print(state_dict.keys())

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
    optimizer = optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ###  5. [NEW] Training Loop (Optional)
    epochs = 20  # 학습 에폭 수 설정
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
            # with torch.no_grad():
            tactile_rep = model.forward_features(batch['sensor'], batch['sensor_poses'])
            # Since we are using an AttentiveClassifier, we feed all patch tokens (sequence) to the pooler
            features = tactile_rep["x_norm_patchtokens"]

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
                tactile_rep = model.forward_features(batch['sensor'], batch['sensor_poses'])
                # Using patch tokens for AttentiveClassifier
                features = tactile_rep["x_norm_patchtokens"]
                # print(features[0])

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
