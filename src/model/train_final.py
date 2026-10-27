"""Train and export the final RGB restoration model using local prepared data."""
import argparse
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import load_config, PROJECT_ROOT
from dataset_v2 import RestorationPairs
from losses_v2 import MaskedRestorationLoss, structural_similarity
from restoration_v2 import RestorationUNet
from train_v2 import check_dataset, seed_everything, seed_worker, save_checkpoint, write_preview


def cached_weights_only(*args, **kwargs):
    raise RuntimeError('Required pretrained weights are missing from the local cache')


def epoch(model, loader, criterion, device, amp, feature_weight, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    count = updates = skipped = 0
    category_errors = {'hair': [0., 0.], 'hair-short': [0., 0.]}
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc='Training' if training else 'Validation', leave=False):
            damaged, clean, mask, *indices = batch
            damaged, clean, mask = (x.to(device, non_blocking=True) for x in (damaged, clean, mask))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                raw = model(damaged)
                loss, parts = criterion(raw, clean, mask, feature_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; last complete checkpoint is preserved')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not amp and not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradients')
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < before:
                    skipped += 1
                else:
                    updates += 1
            size = clean.size(0)
            count += size
            values = {'loss': loss.detach(), **parts}
            if not training:
                restored = raw.float().clamp(0, 1)
                values['psnr'] = (-10 * (restored-clean).square().mean((1, 2, 3)).clamp_min(1e-12).log10()).mean()
                values['ssim'] = structural_similarity(restored, clean).mean()
                values['clean_mae'] = ((restored-clean).abs()*(1-mask)).sum()/((1-mask).sum()*3).clamp_min(1)
                if indices:
                    for i, index in enumerate(indices[0].tolist()):
                        kinds = loader.dataset.metadata[index]['damage_types']
                        for kind in category_errors:
                            if kinds == [kind]:
                                error = ((restored[i]-clean[i]).abs()*mask[i]).sum()
                                pixels = mask[i].sum()*3
                                category_errors[kind][0] += error.item()
                                category_errors[kind][1] += pixels.item()
            for key, value in values.items():
                totals[key] = totals.get(key, 0.) + value.item()*size
    metrics = {key: value/count for key, value in totals.items()}
    if training:
        if updates == 0:
            raise RuntimeError('No optimizer updates succeeded; refusing to waste training epochs')
        metrics.update(optimizer_updates=updates, skipped_updates=skipped)
    else:
        metrics.update({f'{kind}_mae': error/max(pixels, 1) for kind, (error, pixels) in category_errors.items()})
    return metrics


class IndexedPairs(RestorationPairs):
    def __getitem__(self, index):
        return (*super().__getitem__(index), index)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT/path


def finish(model_path, config, run_dir):
    final = config['final_training']
    saved = torch.load(model_path, map_location='cpu', weights_only=True)
    target = resolve(final['export'])
    target.parent.mkdir(parents=True, exist_ok=True)
    export = {key: saved[key] for key in ('architecture', 'model', 'config', 'dataset_id', 'metrics',
                                         'global_epoch', 'final_epoch', 'parent_checkpoint')}
    save_checkpoint(target, export)
    print(f'Final model exported: {target}; selected epoch {saved["global_epoch"]}', flush=True)
    status = dict(status='trained', checkpoint=str(target), global_epoch=saved['global_epoch'],
                  final_epoch=saved['final_epoch'], metrics=saved['metrics'])
    (run_dir/'status.json').write_text(json.dumps(status, indent=2))
    if final['evaluate_after_training']:
        results = PROJECT_ROOT/'results/final_model'
        results.mkdir(parents=True, exist_ok=True)
        commands = [
            ['evaluate.py', '--dataset_dir', config['dataset']['output'], '--checkpoint', str(target),
             '--output_json', str(results/'metrics.json')],
            ['visualize_samples.py', '--dataset_dir', config['dataset']['output'], '--checkpoint', str(target),
             '--output_dir', str(results/'samples'), '--num_samples', '20', '--seed', '42'],
        ]
        os.environ['MPLBACKEND'] = 'Agg'
        for command in commands:
            with (results/f'{Path(command[0]).stem}.log').open('w') as out, (results/f'{Path(command[0]).stem}.stderr.log').open('w') as err:
                subprocess.run([sys.executable, '-u', str(Path(__file__).with_name(command[0])), *command[1:]],
                               cwd=PROJECT_ROOT, stdout=out, stderr=err, check=True)
        status['status'] = 'complete'
        status['test_metrics'] = str(results/'metrics.json')
        (run_dir/'status.json').write_text(json.dumps(status, indent=2))
        print(f'Final evaluation and samples saved: {results}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT_ROOT/'configs/restoration_final.yaml'))
    parser.add_argument('--resume')
    args = parser.parse_args()
    config = load_config(args.config)
    settings, final = config['training'], config['final_training']
    ready = check_dataset(config)
    run_dir = Path(settings['run_dir'])
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Final run already exists; use --resume with its last.pt')
    initial_path = resolve(final['initial_checkpoint'])
    saved = torch.load(args.resume or initial_path, map_location='cpu', weights_only=True)
    if saved['architecture'] != RestorationUNet.architecture or saved['dataset_id'] != ready['dataset_id']:
        raise ValueError('Checkpoint architecture or dataset mismatch')
    if args.resume and saved['config'] != config:
        raise ValueError('Resume must use the same final-training configuration')
    if saved['config']['dataset'] != config['dataset']:
        raise ValueError('The pretrained model must use the same prepared dataset')
    parent_epoch = saved['parent_epoch'] if args.resume else saved['global_epoch']
    if final['additional_epochs'] < 1 or parent_epoch+final['additional_epochs'] > settings['max_total_epochs']:
        raise ValueError('Requested final training exceeds the total epoch budget')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(2)
    torch.hub.download_url_to_file = cached_weights_only
    seed_everything(config['seed'])
    device = torch.device('cuda')
    amp = settings['amp']
    model = RestorationUNet(pretrained=False).to(device)
    model.load_state_dict(saved['model'])
    model.freeze_encoder(False)
    criterion = MaskedRestorationLoss(settings['ssim_weight'], True, final['mask_weight']).to(device)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': settings['encoder_lr']},
        {'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': settings['decoder_lr']},
    ], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=4, min_lr=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    train_data = RestorationPairs(config['dataset']['output'], 'train', True, settings['identity_probability'])
    val_data = IndexedPairs(config['dataset']['output'], 'val')
    weights = [final['hair_sample_weight'] if {'hair','hair-short'} & set(row['damage_types']) else 1.
               for row in train_data.metadata]
    generator = torch.Generator()
    sampler = WeightedRandomSampler(weights, len(train_data), replacement=True, generator=generator)
    options = dict(batch_size=settings['batch_size'], num_workers=settings['workers'], pin_memory=True,
                   worker_init_fn=seed_worker, persistent_workers=False)
    train_loader = DataLoader(train_data, sampler=sampler, **options)
    val_loader = DataLoader(val_data, shuffle=False, **options)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/'previews').mkdir(exist_ok=True)
    (run_dir/'config.json').write_text(json.dumps(config, indent=2))
    start, bad = (saved['final_epoch'], saved['bad_epochs']) if args.resume else (0, 0)
    if args.resume:
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        best = saved['best_score']
    else:
        metrics = epoch(model, val_loader, criterion, device, amp, settings['perceptual_weight'])
        best = metrics['loss']
        state = dict(architecture=model.architecture, model=model.state_dict(), config=config,
                     dataset_id=ready['dataset_id'], global_epoch=parent_epoch, final_epoch=0,
                     parent_checkpoint=str(initial_path), parent_epoch=parent_epoch, metrics=metrics)
        save_checkpoint(run_dir/'best.pt', state)
        (run_dir/'initial_metrics.json').write_text(json.dumps(metrics, indent=2))
        print(f'Initial validation: {json.dumps(metrics)}', flush=True)
    del saved
    print(f'Final training: {start}/{final["additional_epochs"]} additional epochs; '
          f'prior epochs={parent_epoch}; patience={settings["patience"]}; offline=True', flush=True)
    for index in range(start, final['additional_epochs']):
        if bad >= settings['patience']:
            print('Validation has plateaued; finishing with the best saved model.', flush=True)
            break
        started = time.perf_counter()
        seed_everything(config['seed']+parent_epoch+index)
        generator.manual_seed(config['seed']+parent_epoch+index)
        train_metrics = epoch(model, train_loader, criterion, device, amp, settings['perceptual_weight'], optimizer, scaler)
        metrics = epoch(model, val_loader, criterion, device, amp, settings['perceptual_weight'])
        scheduler.step(metrics['loss'])
        improved = metrics['loss'] < best
        significant = metrics['loss'] < best-settings['min_delta']
        bad = 0 if significant else bad+1
        best = min(best, metrics['loss'])
        state = dict(architecture=model.architecture, model=model.state_dict(), config=config,
                     dataset_id=ready['dataset_id'], global_epoch=parent_epoch+index+1, final_epoch=index+1,
                     parent_checkpoint=str(initial_path), parent_epoch=parent_epoch, metrics=metrics,
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     best_score=best, bad_epochs=bad)
        if improved:
            save_checkpoint(run_dir/'best.pt', state)
        save_checkpoint(run_dir/'last.pt', state)
        if index == 0 or improved and (index+1) % settings['preview_every'] == 0:
            write_preview(model, val_data, run_dir/'previews'/f'epoch_{index+1:03d}.png', device, settings['preview_samples'])
        record = dict(epoch=index+1, global_epoch=parent_epoch+index+1,
                      seconds=round(time.perf_counter()-started, 2),
                      encoder_lr=optimizer.param_groups[0]['lr'], decoder_lr=optimizer.param_groups[1]['lr'])
        record.update({f'train_{k}': v for k,v in train_metrics.items()})
        record.update({f'val_{k}': v for k,v in metrics.items()})
        history = run_dir/'history.csv'
        exists = history.exists()
        with history.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)
        status = dict(status='training', epoch=index+1, global_epoch=parent_epoch+index+1,
                      max_additional_epochs=final['additional_epochs'], bad_epochs=bad,
                      best_score=best, metrics=metrics)
        (run_dir/'status.json').write_text(json.dumps(status, indent=2))
        print(f'Epoch {index+1}/{final["additional_epochs"]} (total {parent_epoch+index+1}): '
              f'PSNR={metrics["psnr"]:.3f}, SSIM={metrics["ssim"]:.4f}, '
              f'hair MAE={metrics["hair_mae"]:.4f}, short-hair MAE={metrics["hair-short_mae"]:.4f}, '
              f'score={metrics["loss"]:.6f}, updates={train_metrics["optimizer_updates"]}, '
              f'seconds={record["seconds"]}', flush=True)
    del criterion, optimizer, model
    torch.cuda.empty_cache()
    finish(run_dir/'best.pt', config, run_dir)


if __name__ == '__main__':
    awake = False
    if os.name == 'nt':
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        print(f'Final training PID={os.getpid()}; automatic sleep prevention={awake}', flush=True)
        main()
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
