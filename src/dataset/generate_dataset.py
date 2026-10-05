import os
import random
import json
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from damage_adapter import DamageSimulatorAdapter

def setup_directories(base_out, splits, classes):
    base_out = Path(base_out)
    for split in splits:
        for cls in classes:
            (base_out / split / cls / 'clean').mkdir(parents=True, exist_ok=True)
            (base_out / split / cls / 'damaged').mkdir(parents=True, exist_ok=True)
            (base_out / split / cls / 'masks').mkdir(parents=True, exist_ok=True)
    (base_out / 'metadata').mkdir(parents=True, exist_ok=True)

def generate_variant(adapter, clean_img_np, artifact_types):
    num_types = random.randint(1, 3)
    selected_types = random.sample(artifact_types, num_types)
    severity = random.choice(['light', 'medium'])
    
    if severity == 'light':
        strength = random.uniform(0.3, 0.5)
    else:
        strength = random.uniform(0.5, 0.7)
        
    mask_np = adapter.generate_mask(target_size=(256, 256), selected_types=selected_types, severity=severity)
    damaged_np = adapter.apply_damage(clean_img_np, mask_np, strength=strength)
    
    return mask_np, damaged_np, selected_types, severity, strength

def process_image(img_path, out_dir, split, cls, damage_assets_dir, variants_count):
    # This runs in a separate process, so we initialize the adapter here if needed,
    # or just instantiate it per process.
    adapter = DamageSimulatorAdapter(damage_assets_dir)
    artifact_types = adapter.artifact_types
    
    clean_img = Image.open(img_path).convert("RGB")
    clean_img = clean_img.resize((256, 256), Image.LANCZOS)
    clean_np = np.array(clean_img)
    
    clean_save_path = out_dir / split / cls / 'clean' / img_path.name
    clean_img.save(clean_save_path)
    
    local_metadata = []
    
    for v in range(variants_count):
        variant_id = f"v{v+1:02d}"
        variant_stem = f"{img_path.stem}_{variant_id}"
        
        seed = random.randint(0, 999999)
        random.seed(seed)
        np.random.seed(seed)
        
        mask_np, damaged_np, selected_types, severity, strength = generate_variant(adapter, clean_np, artifact_types)
        
        damaged_name = f"{variant_stem}.png"
        mask_name = f"{variant_stem}_mask.png"
        
        damaged_save_path = out_dir / split / cls / 'damaged' / damaged_name
        mask_save_path = out_dir / split / cls / 'masks' / mask_name
        
        Image.fromarray(damaged_np).save(damaged_save_path)
        Image.fromarray(mask_np).save(mask_save_path)
        
        meta_entry = {
            "source_image": f"{cls}/{img_path.name}",
            "clean_image": f"{split}/{cls}/clean/{img_path.name}",
            "damaged_image": f"{split}/{cls}/damaged/{damaged_name}",
            "mask": f"{split}/{cls}/masks/{mask_name}",
            "class": cls,
            "split": split,
            "damage_types": selected_types,
            "severity": severity,
            "strength": round(strength, 3),
            "seed": seed,
            "image_size": [256, 256],
            "variant": variant_id
        }
        local_metadata.append(meta_entry)
        
    return local_metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_splits_dir', type=str, default='d:/MV/data/clean_splits')
    parser.add_argument('--output_dir', type=str, default='d:/MV/restoration_dataset')
    parser.add_argument('--damage_assets_dir', type=str, default='d:/MV/damage dataset')
    parser.add_argument('--variants', type=int, default=2, help='Number of variants per clean image')
    args = parser.parse_args()

    clean_dir = Path(args.clean_splits_dir)
    out_dir = Path(args.output_dir)
    splits = ['train', 'val', 'test']
    classes = ['buildings', 'street']
    
    setup_directories(out_dir, splits, classes)
    
    for split in splits:
        metadata = []
        print(f"Processing split: {split}")
        
        # Gather all image paths for this split
        img_paths = []
        for cls in classes:
            img_paths.extend([(p, cls) for p in (clean_dir / split / cls).glob("*.jpg")])
            
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_image, p, out_dir, split, cls, args.damage_assets_dir, args.variants): (p, cls)
                for (p, cls) in img_paths
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split} Progress"):
                try:
                    result_meta = future.result()
                    metadata.extend(result_meta)
                except Exception as exc:
                    print(f"Image processing generated an exception: {exc}")
                    
        with open(out_dir / 'metadata' / f"{split}.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
    print(f"Dataset generated successfully at {args.output_dir}")

if __name__ == "__main__":
    main()
