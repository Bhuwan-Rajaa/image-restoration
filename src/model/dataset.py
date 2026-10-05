import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RestorationDataset(Dataset):
    def __init__(self, metadata_path, dataset_dir="d:/MV/restoration_dataset", transform=None, is_train=False):
        """
        metadata_path: Path to the split JSON (e.g., train.json)
        dataset_dir: Base directory of the dataset
        """
        self.dataset_dir = Path(dataset_dir)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.transform = transform
        self.is_train = is_train
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # Note: No normalization applied here so we can output directly in [0, 1] range via Sigmoid
            ])
            
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        entry = self.metadata[idx]
        
        damaged_path = self.dataset_dir / entry['damaged_image']
        clean_path = self.dataset_dir / entry['clean_image']
        
        damaged_img = Image.open(damaged_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")
        
        if self.is_train:
            # Consistent data augmentation for both images
            # E.g. Random Horizontal Flip
            if torch.rand(1) < 0.5:
                damaged_img = damaged_img.transpose(Image.FLIP_LEFT_RIGHT)
                clean_img = clean_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        if self.transform:
            damaged_tensor = self.transform(damaged_img)
            clean_tensor = self.transform(clean_img)
            
        return damaged_tensor, clean_tensor
