
import os
import glob
from pathlib import Path
from typing import List, Optional

import torch
import torch.utils.data as data
from omegaconf import DictConfig
from PIL import Image
import torchvision.transforms.functional as F

class GelsightSSLDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        sequences: List[str],
        data_path: str
    ):
        self.config = config
        self.sequences = sequences
        self.data_path = data_path

        self.images = []
        
        for class_id, sequence in enumerate(self.sequences):
            # Construct the path to the gelsight_frame folder
            # Assuming the sequence corresponds to the folder name in data_path
            sequence_path = os.path.join(self.data_path, str(sequence), "gelsight_frame")
            
            if not os.path.exists(sequence_path):
                print(f"Warning: Path {sequence_path} does not exist. Skipping.")
                continue
            
            # Find all images in the folder
            # We can add more extensions if needed
            image_files = sorted(glob.glob(os.path.join(sequence_path, "*.jpg"))) 
            
            if not image_files:
                    print(f"DEBUG: No images found in {sequence_path}")
                    # Try listing dir
                    # print(f"DEBUG: Dir content: {os.listdir(sequence_path)}")

            # Append tuple of (path, class_id)
            for img_path in image_files:
                self.images.append((img_path, class_id))
                
        print(f"Loaded {len(self.images)} images from {len(self.sequences)} sequences.")
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, class_id = self.images[index]
        
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_tensor = F.to_tensor(img)
                img_tensor = F.resize(img_tensor, (64, 64))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img_tensor = torch.zeros(3, 64, 64) # Assuming 64x64 default
            
        return {
            "sensor": img_tensor,
            "object_classification": class_id
        }
