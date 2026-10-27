"""Aligned lossless pairs, optional identity examples, and damage masks."""
import json
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class RestorationPairs(Dataset):
    def __init__(self, root, split, training=False, identity_probability=0.0):
        self.root = Path(root)
        self.metadata = json.loads((self.root / 'metadata' / f'{split}.json').read_text(encoding='utf-8'))
        if not self.metadata:
            raise ValueError(f'Empty {split} dataset')
        self.training = training
        self.identity_probability = identity_probability

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        entry = self.metadata[index]
        with Image.open(self.root / entry['clean_image']) as image:
            clean = pil_to_tensor(image.convert('RGB')).float().div_(255)
        with Image.open(self.root / entry['damaged_image']) as image:
            damaged = pil_to_tensor(image.convert('RGB')).float().div_(255)
        with Image.open(self.root / entry['mask']) as image:
            damage = (pil_to_tensor(image.convert('L')) == 0).float()
        if clean.shape != damaged.shape or clean.shape[-2:] != damage.shape[-2:]:
            raise ValueError(f'Misaligned pair: {entry["source_image"]}')
        if self.training:
            if random.random() < 0.5:
                clean, damaged, damage = (torch.flip(x, (-1,)) for x in (clean, damaged, damage))
            if random.random() < self.identity_probability:
                damaged, damage = clean.clone(), torch.zeros_like(damage)
        return damaged, clean, damage
