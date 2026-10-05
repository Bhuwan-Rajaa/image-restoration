import os
import cv2
import glob
import math
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

# Perlin noise implementation adapted from FilmDamageSimulator
def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def random_perlin_with_numpy(num_samples, noise_array):
    noise_array = noise_array - np.min(noise_array)
    sum_noise = float(noise_array.sum())
    if sum_noise == 0:
        p = None
    else:
        p = noise_array.ravel() / sum_noise
    linear_idx = np.random.choice(noise_array.size, p=p, size=num_samples)
    x, y = np.unravel_index(linear_idx, noise_array.shape)
    return x, y

class DamageSimulatorAdapter:
    def __init__(self, assets_dir="d:/MV/damage dataset"):
        self.assets_dir = Path(assets_dir)
        self.artifact_types = ['dirt', 'dots', 'scratches', 'smut', 'spots', 'sprinkles', 'stain']
        self.assets = {}
        for atype in self.artifact_types:
            folder = self.assets_dir / atype
            if folder.exists():
                images = list(folder.glob("*.png"))
                self.assets[atype] = images
            else:
                self.assets[atype] = []

    def get_random_asset(self, artifact_type):
        if not self.assets.get(artifact_type):
            return None
        return random.choice(self.assets[artifact_type])

    def generate_mask(self, target_size=(256, 256), selected_types=None, severity='medium'):
        """
        severity: light, medium, heavy. Controls number of artifacts placed.
        target_size: (height, width)
        """
        # Create blank mask (255 = clean)
        mask = np.full((target_size[0], target_size[1]), 255, dtype=np.uint8)
        
        if not selected_types:
            return mask

        # Severity mappings
        if severity == 'light':
            count_multiplier = 0.5
        elif severity == 'heavy':
            count_multiplier = 2.0
        else:
            count_multiplier = 1.0

        # Generate perlin noise for placement
        noise_res = (2, 2) # ensure power of 2 factor for shape?
        # Target size is 256, 256, divisible by 2.
        perlin_noise = generate_perlin_noise_2d(target_size, noise_res)
        
        for atype in selected_types:
            # Decide how many to place based on type and severity
            # For 256x256, a few artifacts are enough
            if atype in ['dots', 'sprinkles']:
                count = int(random.randint(5, 15) * count_multiplier)
            elif atype == 'scratches':
                count = int(random.randint(1, 3) * count_multiplier)
            else:
                count = int(random.randint(2, 6) * count_multiplier)

            positions_x, positions_y = random_perlin_with_numpy(count, perlin_noise)

            for i in range(count):
                asset_path = self.get_random_asset(atype)
                if not asset_path: continue
                
                asset_img = Image.open(asset_path).convert("RGBA")
                
                # Random rotation
                angle = random.uniform(0, 360)
                asset_img = asset_img.rotate(angle, resample=Image.BICUBIC, expand=True)
                
                # Random scaling
                scale = random.uniform(0.1, 0.5) 
                if atype == 'scratches':
                    scale = random.uniform(0.5, 1.2) # Scratches can be larger
                
                new_size = (int(asset_img.width * scale), int(asset_img.height * scale))
                if new_size[0] <= 0 or new_size[1] <= 0: continue
                asset_img = asset_img.resize(new_size, Image.LANCZOS)
                
                # Get alpha channel as numpy
                asset_np = np.array(asset_img)
                alpha = asset_np[:, :, 3]
                
                # Coordinates (center placement)
                px, py = positions_x[i], positions_y[i]
                h, w = alpha.shape
                
                y_min = max(0, py - h//2)
                y_max = min(target_size[0], py - h//2 + h)
                x_min = max(0, px - w//2)
                x_max = min(target_size[1], px - w//2 + w)
                
                # Crop alpha if it goes out of bounds
                alpha_y_min = y_min - (py - h//2)
                alpha_y_max = alpha_y_min + (y_max - y_min)
                alpha_x_min = x_min - (px - w//2)
                alpha_x_max = alpha_x_min + (x_max - x_min)
                
                alpha_crop = alpha[alpha_y_min:alpha_y_max, alpha_x_min:alpha_x_max]
                
                # Apply to mask (where alpha is high, set mask to 0)
                region = mask[y_min:y_max, x_min:x_max]
                
                # Any pixel in alpha > 127 is considered damaged (0)
                region[alpha_crop > 127] = 0
                mask[y_min:y_max, x_min:x_max] = region

        return mask

    def apply_damage(self, clean_img_np, mask_np, strength=0.6):
        """
        clean_img_np: RGB image numpy array (H, W, 3), range 0-255
        mask_np: Grayscale mask numpy array (H, W), 255=clean, 0=damaged
        strength: 0.0 to 1.0. 
        """
        clean_img_np = clean_img_np.astype(np.float32)
        mask_norm = mask_np.astype(np.float32) / 255.0
        
        # Add damage as bright marks (simulate film scratches scattering light)
        mask_norm = np.expand_dims(mask_norm, axis=-1)
        damaged = clean_img_np + strength * 255.0 * (1.0 - mask_norm)
        
        damaged = np.clip(damaged, 0, 255).astype(np.uint8)
        return damaged
