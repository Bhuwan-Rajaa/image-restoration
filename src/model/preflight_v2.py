"""Audit prepared data and run inference-only readiness checks. Never trains."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import load_config, PROJECT_ROOT
from dataset_v2 import RestorationPairs
from losses_v2 import RestorationLoss
from restoration_v2 import RestorationUNet
from train_v2 import check_dataset, seed_worker


def audit_pairs(config):
    ready = check_dataset(config)
    root = Path(config['dataset']['output'])
    size = config['dataset']['image_size']
    report = {}
    for split in ('train', 'val', 'test'):
        rows = json.loads((root / 'metadata' / f'{split}.json').read_text(encoding='utf-8'))
        pair_paths = set()
        counts = Counter()
        min_damage, max_damage = 1.0, 0.0
        for row in tqdm(rows, desc=f'Audit {split}'):
            if row['split'] != split or row['damaged_image'] in pair_paths:
                raise ValueError(f'Duplicate pair or wrong split: {row}')
            pair_paths.add(row['damaged_image'])
            arrays = []
            for key in ('clean_image', 'damaged_image', 'mask'):
                with Image.open(root / row[key]) as image:
                    if image.format != 'PNG' or image.size != (size, size):
                        raise ValueError(f'Wrong format or size: {row[key]}')
                    arrays.append(np.array(image))
            clean, damaged, mask = arrays
            if clean.shape != (size, size, 3) or damaged.shape != clean.shape or mask.shape != (size, size):
                raise ValueError(f'Wrong image channels: {row["source_image"]}')
            if not set(np.unique(mask)) <= {0, 255}:
                raise ValueError('Unexpected mask values')
            if not np.array_equal(clean[mask == 255], damaged[mask == 255]):
                raise ValueError(f'Undamaged pixels differ: {row["damaged_image"]}')
            if not np.any(clean != damaged):
                raise ValueError(f'Empty damage: {row["damaged_image"]}')
            fraction = float(np.mean(mask == 0))
            if abs(fraction - row['damage_fraction']) > 1e-10:
                raise ValueError('Mask coverage differs from metadata')
            min_damage, max_damage = min(min_damage, fraction), max(max_damage, fraction)
            counts.update(row['damage_types'])
        report[split] = {'pairs_checked': len(rows), 'artifact_occurrences': dict(counts),
                         'min_damage_fraction': min_damage, 'max_damage_fraction': max_damage,
                         'outside_mask_difference': 0}
    return ready, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--report', default=str(PROJECT_ROOT / 'runs/preparation_report.json'))
    args = parser.parse_args()
    config = load_config(args.config)
    ready, data_report = audit_pairs(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA not available; the planned GPU training is not ready')
    torch.set_num_threads(2)
    settings = config['training']
    model = RestorationUNet(pretrained=True).to(device).eval()
    criterion = RestorationLoss(settings['ssim_weight'], perceptual=True).to(device)
    data = RestorationPairs(config['dataset']['output'], 'val')
    loader = DataLoader(data, batch_size=settings['batch_size'], num_workers=settings['workers'],
                        pin_memory=True, worker_init_fn=seed_worker)
    iterator = iter(loader)
    damaged, clean, mask = next(iterator)
    del iterator  # Shut down the preview loader workers immediately.
    damaged, clean = damaged.to(device), clean.to(device)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast('cuda', enabled=settings['amp']):
        output = model(damaged)
        loss, components = criterion(output, clean, settings['perceptual_weight'])
    if output.shape != clean.shape or not torch.isfinite(output).all() or not torch.isfinite(loss):
        raise AssertionError('Model/loss readiness check failed')
    identity_error = (output.float() - damaged).abs().max().item()
    if identity_error != 0:
        raise AssertionError('New model should initialize as the identity mapping')
    report = {'training_started': False, 'optimizer_steps': 0, 'dataset_id': ready['dataset_id'],
              'dataset_audit': data_report, 'gpu': torch.cuda.get_device_name(),
              'torch_version': str(torch.__version__), 'batch_size': settings['batch_size'],
              'loader_workers': settings['workers'], 'output_shape': list(output.shape),
              'identity_max_error': identity_error, 'initial_loss': loss.item(),
              'initial_components': {k: v.item() for k, v in components.items()},
              'forward_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
              'memory_note': 'Inference-only measurement; backward/optimizer memory has not been measured.',
              'weights': 'ImageNet ResNet34 and VGG16 loaded successfully; no restoration training performed.'}
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
