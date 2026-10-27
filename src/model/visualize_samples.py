import os
import random
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt

from dataset import RestorationDataset
from restoration_v2 import load_restoration_model

def visualize_and_save(clean_img, damaged_img, mask_img, restored_img, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(clean_img)
    axes[0].set_title("CLEAN")
    axes[0].axis('off')
    
    axes[1].imshow(damaged_img)
    axes[1].set_title("DAMAGED")
    axes[1].axis('off')
    
    axes[2].imshow(mask_img, cmap='gray')
    axes[2].set_title("MASK")
    axes[2].axis('off')
    
    axes[3].imshow(restored_img)
    axes[3].set_title("RESTORED")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=str(Path(__file__).resolve().parents[2] / 'restoration_dataset_v2'))
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).resolve().parents[2] / 'results/qualitative'))
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples to visualize')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_metadata_path = Path(args.dataset_dir) / "metadata" / "test.json"
    test_dataset = RestorationDataset(test_metadata_path, args.dataset_dir, is_train=False)
    
    # Randomly sample diverse metadata entries
    import json
    with open(test_metadata_path, 'r') as f:
        metadata_list = json.load(f)
        
    sampled_indices = random.Random(args.seed).sample(range(len(metadata_list)), min(args.num_samples, len(metadata_list)))
    
    model = load_restoration_model(args.checkpoint, device)
    
    print(f"Generating {len(sampled_indices)} qualitative samples...")
    
    manifest = []
    with torch.inference_mode():
        for i, idx in enumerate(sampled_indices):
            entry = metadata_list[idx]
            
            # Load images
            damaged_path = Path(args.dataset_dir) / entry['damaged_image']
            clean_path = Path(args.dataset_dir) / entry['clean_image']
            mask_path = Path(args.dataset_dir) / entry['mask']
            
            clean_img = Image.open(clean_path).convert("RGB")
            damaged_img = Image.open(damaged_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")
            
            # Prepare tensor for model
            damaged_tensor = test_dataset.transform(damaged_img).unsqueeze(0).to(device)
            restored_tensor = model(damaged_tensor)
            if not torch.isfinite(restored_tensor).all():
                raise ValueError(f'Nonfinite prediction for {entry["damaged_image"]}')
            
            restored_np = restored_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            restored_np = (restored_np * 255).clip(0, 255).astype(np.uint8)
            
            # Create a descriptive filename based on metadata
            damage_str = "_".join(entry['damage_types'])
            cls_str = entry['class']
            severity = entry.get('severity', 'unk')
            filename = f"sample_{i:02d}_{cls_str}_{damage_str}_{severity}.png"
            
            save_path = out_dir / filename
            visualize_and_save(np.array(clean_img), np.array(damaged_img), np.array(mask_img), restored_np, save_path)
            manifest.append({'test_index': idx, 'output': filename, 'metadata': entry})
            
            print(f"Saved: {filename}")
            
    (out_dir / 'manifest.json').write_text(json.dumps({
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'dataset_dir': str(Path(args.dataset_dir).resolve()),
        'seed': args.seed,
        'selection': 'Uniform random sample of test pairs, without replacement; no metric-based selection.',
        'panels': ['clean', 'damaged', 'mask (white=clean, black=damage)', 'restored'],
        'samples': manifest,
    }, indent=2), encoding='utf-8')
    print(f"All qualitative samples saved to {out_dir}")

if __name__ == "__main__":
    main()
