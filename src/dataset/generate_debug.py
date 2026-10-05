import os
import random
from pathlib import Path
from PIL import Image
import numpy as np

from damage_adapter import DamageSimulatorAdapter

def main():
    debug_dir = Path("d:/MV/debug_samples")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    clean_splits_dir = Path("d:/MV/data/clean_splits/train/buildings")
    clean_images = list(clean_splits_dir.glob("*.jpg"))
    if not clean_images:
        print("No clean images found in train/buildings.")
        return
        
    source_img_path = random.choice(clean_images)
    print(f"Using source image: {source_img_path.name}")
    
    # Load and preprocess (resize to 256x256)
    clean_img = Image.open(source_img_path).convert("RGB")
    clean_img = clean_img.resize((256, 256), Image.LANCZOS)
    clean_np = np.array(clean_img)
    
    adapter = DamageSimulatorAdapter("d:/MV/damage dataset")
    
    artifact_types = ['dirt', 'dots', 'scratches', 'smut', 'spots', 'sprinkles', 'stain']
    
    for atype in artifact_types:
        print(f"Generating {atype}...")
        mask_np = adapter.generate_mask(target_size=(256, 256), selected_types=[atype], severity='medium')
        damaged_np = adapter.apply_damage(clean_np, mask_np, strength=0.6)
        
        # Create CLEAN | MASK | DAMAGED visualization
        mask_rgb = np.stack((mask_np,)*3, axis=-1)
        combo = np.concatenate((clean_np, mask_rgb, damaged_np), axis=1)
        
        combo_img = Image.fromarray(combo)
        combo_img.save(debug_dir / f"{atype}.png")
        
    # Mixed damage
    print("Generating mixed...")
    mixed_types = random.sample(artifact_types, 3)
    mask_np = adapter.generate_mask(target_size=(256, 256), selected_types=mixed_types, severity='heavy')
    damaged_np = adapter.apply_damage(clean_np, mask_np, strength=0.8)
    
    mask_rgb = np.stack((mask_np,)*3, axis=-1)
    combo = np.concatenate((clean_np, mask_rgb, damaged_np), axis=1)
    
    combo_img = Image.fromarray(combo)
    combo_img.save(debug_dir / "mixed.png")
    
    print("Debug samples generated in d:/MV/debug_samples/")

if __name__ == "__main__":
    main()
