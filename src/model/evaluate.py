import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse
import numpy as np
from pathlib import Path

# Evaluation metrics can be calculated using skimage
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from dataset import RestorationDataset
from unet import ResNetUNet

def mean_absolute_error(img1, img2):
    return np.mean(np.abs(img1 - img2))

def evaluate_metrics(clean_np, damaged_np, restored_np):
    # Calculate Damaged vs Clean
    d_psnr = psnr(clean_np, damaged_np, data_range=1.0)
    d_ssim = ssim(clean_np, damaged_np, data_range=1.0, channel_axis=-1)
    d_mse = mse(clean_np, damaged_np)
    d_mae = mean_absolute_error(clean_np, damaged_np)
    
    # Calculate Restored vs Clean
    r_psnr = psnr(clean_np, restored_np, data_range=1.0)
    r_ssim = ssim(clean_np, restored_np, data_range=1.0, channel_axis=-1)
    r_mse = mse(clean_np, restored_np)
    r_mae = mean_absolute_error(clean_np, restored_np)
    
    return {
        'damaged_vs_clean': {'PSNR': d_psnr, 'SSIM': d_ssim, 'MSE': d_mse, 'MAE': d_mae},
        'restored_vs_clean': {'PSNR': r_psnr, 'SSIM': r_ssim, 'MSE': r_mse, 'MAE': r_mae}
    }

class Evaluator:
    def __init__(self):
        # We will store lists of metrics for overall, per-class, and per-damage-type
        self.results = []
        
    def add_result(self, metadata_entry, metrics):
        self.results.append({
            'class': metadata_entry['class'],
            'damage_types': metadata_entry['damage_types'],
            'metrics': metrics
        })
        
    def aggregate(self, filter_fn=None):
        filtered = self.results if filter_fn is None else [r for r in self.results if filter_fn(r)]
        if not filtered:
            return None
            
        agg = {'damaged_vs_clean': {}, 'restored_vs_clean': {}}
        for key in agg.keys():
            for metric in ['PSNR', 'SSIM', 'MSE', 'MAE']:
                vals = [r['metrics'][key][metric] for r in filtered]
                agg[key][metric] = np.mean(vals)
        return agg

    def print_aggregated(self, title, agg):
        if not agg:
            return
        print(f"\n--- {title} ---")
        print(f"{'Metric':<10} | {'Damaged vs Clean':<20} | {'Restored vs Clean':<20} | {'Improvement':<20}")
        print("-" * 75)
        for metric in ['PSNR', 'SSIM', 'MAE', 'MSE']:
            d_val = agg['damaged_vs_clean'][metric]
            r_val = agg['restored_vs_clean'][metric]
            
            if metric in ['PSNR', 'SSIM']:
                imp = r_val - d_val
                imp_str = f"+{imp:.4f}" if imp > 0 else f"{imp:.4f}"
            else:
                imp = d_val - r_val # lower is better, so positive improvement means reduction in error
                imp_str = f"+{imp:.4f} (reduction)" if imp > 0 else f"{imp:.4f}"
                
            print(f"{metric:<10} | {d_val:<20.4f} | {r_val:<20.4f} | {imp_str:<20}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='d:/MV/restoration_dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_run', action='store_true', help='Run on small subset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_metadata_path = f"{args.dataset_dir}/metadata/test.json"
    test_dataset = RestorationDataset(test_metadata_path, args.dataset_dir, is_train=False)
    
    if args.test_run:
        test_dataset = torch.utils.data.Subset(test_dataset, range(16))
        # Keep original metadata reference for the subset
        metadata_list = [test_dataset.dataset.metadata[i] for i in range(16)]
    else:
        metadata_list = test_dataset.metadata
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model = ResNetUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    evaluator = Evaluator()
    
    print("Evaluating Test Set...")
    with torch.no_grad():
        for i, (damaged, clean) in enumerate(tqdm(test_loader)):
            damaged = damaged.to(device)
            restored = model(damaged)
            
            # Convert to numpy [H, W, C] for skimage
            clean_np = clean.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            damaged_np = damaged.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            restored_np = restored.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            metrics = evaluate_metrics(clean_np, damaged_np, restored_np)
            evaluator.add_result(metadata_list[i], metrics)
            
    # Print Overall
    evaluator.print_aggregated("OVERALL EVALUATION", evaluator.aggregate())
    
    # Class-wise Evaluation
    for cls in ['buildings', 'street']:
        agg = evaluator.aggregate(lambda r: r['class'] == cls)
        evaluator.print_aggregated(f"CLASS: {cls.upper()}", agg)
        
    # Damage-type Evaluation
    artifact_types = ['dirt', 'dots', 'scratches', 'smut', 'spots', 'sprinkles', 'stain']
    for atype in artifact_types:
        # Include if it's the only damage type, or mixed if multiple
        agg = evaluator.aggregate(lambda r: [atype] == r['damage_types'])
        if agg:
            evaluator.print_aggregated(f"DAMAGE: {atype.upper()} (Single)", agg)
            
    # Mixed damage
    agg = evaluator.aggregate(lambda r: len(r['damage_types']) > 1)
    if agg:
        evaluator.print_aggregated("DAMAGE: MIXED", agg)

if __name__ == "__main__":
    main()
