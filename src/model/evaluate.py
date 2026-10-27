import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse
import hashlib
import numpy as np
from pathlib import Path

# Evaluation metrics can be calculated using skimage
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from dataset import RestorationDataset
from restoration_v2 import load_restoration_model

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
    
    metrics = {
        'damaged_vs_clean': {'PSNR': d_psnr, 'SSIM': d_ssim, 'MSE': d_mse, 'MAE': d_mae},
        'restored_vs_clean': {'PSNR': r_psnr, 'SSIM': r_ssim, 'MSE': r_mse, 'MAE': r_mae}
    }
    return {group: {name: float(value) for name, value in values.items()}
            for group, values in metrics.items()}

class Evaluator:
    def __init__(self):
        # We will store lists of metrics for overall, per-class, and per-damage-type
        self.results = []
        
    def add_result(self, metadata_entry, metrics):
        self.results.append({
            'source_image': metadata_entry.get('source_image'),
            'damaged_image': metadata_entry['damaged_image'],
            'class': metadata_entry['class'],
            'damage_types': metadata_entry['damage_types'],
            'severity': metadata_entry.get('severity'),
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
                agg[key][metric] = float(np.mean(vals))
        agg['count'] = len(filtered)
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
    parser.add_argument('--dataset_dir', type=str, default=str(Path(__file__).resolve().parents[2] / 'restoration_dataset_v2'))
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_run', action='store_true', help='Run on small subset')
    parser.add_argument('--output_json', help='Save overall, grouped, and per-image metrics with provenance')
    args = parser.parse_args()

    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_metadata_path = f"{args.dataset_dir}/metadata/test.json"
    test_dataset = RestorationDataset(test_metadata_path, args.dataset_dir, is_train=False)
    
    if args.test_run:
        test_dataset = torch.utils.data.Subset(test_dataset, range(min(16, len(test_dataset))))
        # Keep original metadata reference for the subset
        metadata_list = [test_dataset.dataset.metadata[i] for i in test_dataset.indices]
    else:
        metadata_list = test_dataset.metadata
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model = load_restoration_model(args.checkpoint, device)
    
    evaluator = Evaluator()
    
    print("Evaluating Test Set...")
    with torch.inference_mode():
        for i, (damaged, clean) in enumerate(tqdm(test_loader)):
            damaged = damaged.to(device)
            restored = model(damaged)
            if restored.shape != damaged.shape or not torch.isfinite(restored).all():
                raise ValueError(f'Invalid model output at test index {i}')
            restored = restored.clamp(0, 1)
            
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
    artifact_types = sorted({t for row in metadata_list for t in row['damage_types']})
    for atype in artifact_types:
        # Include if it's the only damage type, or mixed if multiple
        agg = evaluator.aggregate(lambda r: [atype] == r['damage_types'])
        if agg:
            evaluator.print_aggregated(f"DAMAGE: {atype.upper()} (Single)", agg)
            
    # Mixed damage
    agg = evaluator.aggregate(lambda r: len(r['damage_types']) > 1)
    if agg:
        evaluator.print_aggregated("DAMAGE: MIXED", agg)

    if args.output_json:
        ready_path = Path(args.dataset_dir) / 'prepared.json'
        ready = json.loads(ready_path.read_text(encoding='utf-8')) if ready_path.exists() else {}
        with open(args.checkpoint, 'rb') as stream:
            checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        report = {
            'dataset_dir': str(Path(args.dataset_dir).resolve()),
            'dataset_id': ready.get('dataset_id'),
            'split': 'test',
            'subset_only': args.test_run,
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'checkpoint_sha256': checkpoint_sha256,
            'device': str(device),
            'torch_version': str(torch.__version__),
            'metric_definition': 'Per-image RGB metrics averaged equally over pairs; data_range=1; '
                                 'skimage SSIM defaults, channel_axis=-1; output clamped to [0,1]; '
                                 'float32 inference without autocast. Mixed-damage groups overlap '
                                 'with each artifact-present group.',
            'overall': evaluator.aggregate(),
            'by_class': {cls: evaluator.aggregate(lambda r, cls=cls: r['class'] == cls)
                         for cls in sorted({r['class'] for r in evaluator.results})},
            'single_damage': {kind: evaluator.aggregate(lambda r, kind=kind: r['damage_types'] == [kind])
                              for kind in artifact_types},
            'artifact_present': {kind: evaluator.aggregate(lambda r, kind=kind: kind in r['damage_types'])
                                 for kind in artifact_types},
            'mixed_damage': evaluator.aggregate(lambda r: len(r['damage_types']) > 1),
            'by_severity': {severity: evaluator.aggregate(lambda r, severity=severity: r['severity'] == severity)
                            for severity in sorted({r['severity'] for r in evaluator.results if r['severity']})},
            'per_image': evaluator.results,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f'\nSaved metrics for {len(evaluator.results)} pairs to {output_path}')

if __name__ == "__main__":
    main()
