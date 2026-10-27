"""Explicitly launched two-stage restoration training; --dry-run never trains."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import load_config
from dataset_v2 import RestorationPairs
from losses_v2 import RestorationLoss, structural_similarity
from restoration_v2 import RestorationUNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def check_dataset(config):
    root = Path(config['dataset']['output'])
    ready_path = root / 'prepared.json'
    if not ready_path.exists():
        raise FileNotFoundError('Run src/dataset/prepare_v2.py before training.')
    ready = json.loads(ready_path.read_text(encoding='utf-8'))
    if ready.get('version') != 2 or ready.get('preview_only'):
        raise ValueError('Training requires a complete v2 dataset, not a preview subset.')
    for key in ('image_size', 'variants', 'artifact_types', 'severities'):
        if ready['config']['dataset'][key] != config['dataset'][key]:
            raise ValueError(f'Dataset configuration differs from prepared data: {key}')
    identities = {}
    for split in ('train', 'val', 'test'):
        rows = json.loads((root / 'metadata' / f'{split}.json').read_text(encoding='utf-8'))
        if len(rows) != ready['splits'][split]['pairs']:
            raise ValueError(f'Incomplete {split} metadata')
        identities[split] = {row['source_image'] for row in rows}
        seen = {t for row in rows for t in row['damage_types']}
        if not set(config['dataset']['artifact_types']) <= seen:
            raise ValueError(f'Missing damage categories in {split}')
    if any(identities[a] & identities[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Source image IDs overlap across splits')
    return ready


def make_optimizer(model, settings, refinement):
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    decoder = [p for p in model.parameters() if id(p) not in encoder_ids]
    decoder_lr = settings['refinement_lr'] if refinement else settings['decoder_lr']
    encoder_lr = decoder_lr * 0.1 if refinement else settings['encoder_lr']
    # Include the encoder from the outset; gradients appear after warmup.
    return torch.optim.AdamW([{'params': list(model.encoder.parameters()), 'lr': encoder_lr},
                             {'params': decoder, 'lr': decoder_lr}], weight_decay=settings['weight_decay'])


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def train_epoch(model, loader, criterion, optimizer, scaler, device, amp, feature_weight):
    model.train()
    totals = dict(loss=0.0, l1=0.0, ssim_loss=0.0, vgg=0.0, vgg_weighted=0.0)
    count = 0
    for damaged, clean, _ in tqdm(loader, desc='Training', leave=False):
        damaged, clean = damaged.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            prediction = model(damaged)
            loss, components = criterion(prediction, clean, feature_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss; stopped without saving invalid weights')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        batch = damaged.size(0)
        count += batch
        totals['loss'] += loss.item() * batch
        for key, value in components.items():
            totals[key] += value.item() * batch
    return {key: value / count for key, value in totals.items()}


@torch.inference_mode()
def validate(model, loader, criterion, device, amp, feature_weight):
    model.eval()
    totals = dict(loss=0.0, l1=0.0, ssim_loss=0.0, vgg=0.0, vgg_weighted=0.0,
                  psnr=0.0, ssim=0.0, input_psnr=0.0, input_ssim=0.0)
    count = 0
    region_errors = {'damaged': 0.0, 'undamaged': 0.0}
    region_counts = {'damaged': 0.0, 'undamaged': 0.0}
    for damaged, clean, mask in tqdm(loader, desc='Validation', leave=False):
        damaged, clean, mask = (x.to(device, non_blocking=True) for x in (damaged, clean, mask))
        with torch.autocast(device_type=device.type, enabled=amp):
            raw = model(damaged)
            loss, components = criterion(raw, clean, feature_weight)
        prediction = raw.float().clamp(0, 1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss')
        batch = damaged.size(0)
        count += batch
        totals['loss'] += loss.item() * batch
        for key, value in components.items():
            totals[key] += value.item() * batch
        for prefix, image in (('', prediction), ('input_', damaged)):
            error = (image - clean).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
            totals[prefix + 'psnr'] += (-10 * error.log10()).sum().item()
            totals[prefix + 'ssim'] += structural_similarity(image, clean).sum().item()
        absolute_error = (prediction - clean).abs()
        for name, region in (('damaged', mask), ('undamaged', 1 - mask)):
            region_errors[name] += (absolute_error * region).sum().item()
            region_counts[name] += region.sum().item() * 3
    metrics = {key: value / count for key, value in totals.items()}
    metrics.update({f'{name}_mae': region_errors[name] / max(region_counts[name], 1) for name in region_errors})
    # A consistent score across stages; independent of the VGG ramp and scale.
    metrics['reconstruction_score'] = metrics['l1'] + criterion.ssim_weight * metrics['ssim_loss']
    return metrics


@torch.inference_mode()
def write_preview(model, dataset, path, device, count):
    model.eval()
    rows = []
    # Fixed validation IDs, with hair examples included when available.
    indices = []
    for kind in ('hair', 'hair-short'):
        match = next((i for i, row in enumerate(dataset.metadata) if row['damage_types'] == [kind]), None)
        if match is not None:
            indices.append(match)
    indices += [i for i in range(len(dataset)) if i not in indices]
    for index in indices[:count]:
        damaged, clean, mask, *_ = dataset[index]
        restored = model(damaged.unsqueeze(0).to(device)).float().clamp(0, 1).cpu()[0]
        # Left to right: clean, damaged, mask (white=damage here), restored.
        rows.extend((clean, damaged, mask.expand_as(clean), restored))
    save_image(torch.stack(rows), path, nrow=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--resume', help='Resume a v2 last.pt snapshot, including optimizer and stage progress')
    parser.add_argument('--dry-run', action='store_true', help='Validate metadata and print plan; no model or optimizer is created')
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config['training']
    ready = check_dataset(config)
    if args.dry_run:
        print(json.dumps({'config': config, 'dataset': ready['splits'], 'training_started': False}, indent=2))
        return
    run_dir = Path(settings['run_dir'])
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location='cpu', weights_only=True)
        required = {'optimizer', 'scheduler', 'scaler', 'global_epoch', 'stage_epoch',
                    'bad_epochs', 'best_stage', 'best_overall', 'config', 'dataset_id', 'model', 'stage'}
        if not required <= resume.keys():
            raise ValueError('Resume requires a complete last.pt training snapshot')
        if resume.get('architecture') != RestorationUNet.architecture or resume['dataset_id'] != ready['dataset_id']:
            raise ValueError('Checkpoint architecture or dataset does not match')
        previous = resume['config']
        # Budget increases are allowed within the explicit total cap. Other training
        # changes require a separate run so saved comparisons remain meaningful.
        permitted = {'reconstruction_epochs', 'refinement_epochs', 'max_total_epochs'}
        if previous['seed'] != config['seed'] or previous['dataset'] != config['dataset']:
            raise ValueError('Cannot change seed or dataset when resuming')
        for key, value in settings.items():
            if key not in permitted and previous['training'][key] != value:
                raise ValueError(f'Cannot change {key} when resuming')
        if settings['max_total_epochs'] < resume['global_epoch']:
            raise ValueError('Epoch cap is less than already completed epochs')
        if resume['stage'] == 'refinement' and settings['reconstruction_epochs'] != previous['training']['reconstruction_epochs']:
            raise ValueError('Reconstruction is already finished; only the refinement budget can be extended')
        for key in ('reconstruction_epochs', 'refinement_epochs'):
            if settings[key] < previous['training'][key]:
                raise ValueError('Resume budgets may be increased, not reduced')
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Run directory is not empty; use --resume or choose another run_dir: {run_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = bool(settings['amp'] and device.type == 'cuda')
    if device.type != 'cuda':
        raise RuntimeError('CUDA GPU is required for this training budget; training was not started')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'previews').mkdir(exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    seed_everything(config['seed'])
    model = RestorationUNet(pretrained=resume is None).to(device)
    if resume:
        model.load_state_dict(resume['model'])
    train_data = RestorationPairs(config['dataset']['output'], 'train', training=True,
                                  identity_probability=settings['identity_probability'])
    val_data = RestorationPairs(config['dataset']['output'], 'val')
    generator = torch.Generator()
    loader_options = dict(batch_size=settings['batch_size'], num_workers=settings['workers'],
                          pin_memory=True, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    global_epoch = resume['global_epoch'] if resume else 0
    best_overall = resume['best_overall'] if resume else math.inf
    stages = [('reconstruction', settings['reconstruction_epochs']), ('refinement', settings['refinement_epochs'])]
    for stage, budget in stages:
        refinement = stage == 'refinement'
        if resume and resume['stage'] == 'refinement' and not refinement:
            continue
        if budget == 0:
            continue
        continuing = resume is not None and resume['stage'] == stage
        if refinement and not continuing:
            saved = torch.load(run_dir / 'best_reconstruction.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(saved['model'])
        criterion = RestorationLoss(settings['ssim_weight'], perceptual=refinement).to(device)
        optimizer = make_optimizer(model, settings, refinement)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        start_epoch, bad_epochs, best_stage = 0, 0, math.inf
        if continuing:
            optimizer.load_state_dict(resume['optimizer'])
            scheduler.load_state_dict(resume['scheduler'])
            scaler.load_state_dict(resume['scaler'])
            start_epoch, bad_epochs, best_stage = resume['stage_epoch'], resume['bad_epochs'], resume['best_stage']
        else:
            # Include the starting model as a candidate; fine-tuning need not improve it.
            initial = validate(model, val_loader, criterion, device, amp,
                               settings['perceptual_weight'] if refinement else 0.0)
            best_stage = initial['loss']
            initial_state = dict(architecture=model.architecture, model=model.state_dict(), config=config,
                dataset_id=ready['dataset_id'], stage=stage, stage_epoch=0, global_epoch=global_epoch,
                metrics=initial)
            save_checkpoint(run_dir / f'best_{stage}.pt', initial_state)
            if initial['reconstruction_score'] < best_overall:
                best_overall = initial['reconstruction_score']
                save_checkpoint(run_dir / 'best.pt', initial_state)
        print(f'Stage: {stage}; completed {start_epoch}/{budget} epochs; AMP={amp}', flush=True)
        for stage_epoch in range(start_epoch, budget):
            if global_epoch >= settings['max_total_epochs'] or bad_epochs >= settings['patience']:
                break
            epoch_start = time.perf_counter()
            seed_everything(config['seed'] + global_epoch)
            generator.manual_seed(config['seed'] + global_epoch)
            model.freeze_encoder(not refinement and stage_epoch < settings['warmup_epochs'])
            feature_weight = settings['perceptual_weight'] * min(1, (stage_epoch + 1) / settings['perceptual_ramp_epochs']) if refinement else 0.0
            training = train_epoch(model, train_loader, criterion, optimizer, scaler, device, amp, feature_weight)
            # Always use the full VGG weight for validation, making stage scores comparable.
            validation = validate(model, val_loader, criterion, device, amp,
                                  settings['perceptual_weight'] if refinement else 0.0)
            scheduler.step(validation['loss'])
            improved = validation['loss'] < best_stage - settings['min_delta']
            best_stage = validation['loss'] if improved else best_stage
            bad_epochs = 0 if improved else bad_epochs + 1
            # Give encoder fine-tuning a full patience window after warmup.
            if not refinement and stage_epoch + 1 <= settings['warmup_epochs']:
                bad_epochs = 0
            global_epoch += 1
            overall_improved = validation['reconstruction_score'] < best_overall
            best_overall = min(best_overall, validation['reconstruction_score'])
            state = dict(architecture=model.architecture, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), config=config,
                         dataset_id=ready['dataset_id'], stage=stage, stage_epoch=stage_epoch + 1,
                         global_epoch=global_epoch, bad_epochs=bad_epochs, best_stage=best_stage,
                         best_overall=best_overall, metrics=validation)
            if improved:
                save_checkpoint(run_dir / f'best_{stage}.pt', state)
            if overall_improved:
                save_checkpoint(run_dir / 'best.pt', state)
            # Commit resumable progress after its referenced best checkpoints.
            save_checkpoint(run_dir / 'last.pt', state)
            if settings['preview_samples'] > 0 and (stage_epoch == 0 or improved and (stage_epoch + 1) % settings['preview_every'] == 0):
                write_preview(model, val_data, run_dir / 'previews' / f'{stage}_{stage_epoch + 1:03d}.png',
                              device, settings['preview_samples'])
            record = {'global_epoch': global_epoch, 'stage': stage, 'stage_epoch': stage_epoch + 1,
                      'seconds': round(time.perf_counter() - epoch_start, 2), 'perceptual_weight': feature_weight,
                      'encoder_lr': optimizer.param_groups[0]['lr'], 'decoder_lr': optimizer.param_groups[1]['lr']}
            record.update({f'train_{k}': v for k, v in training.items()})
            record.update({f'val_{k}': v for k, v in validation.items()})
            history = run_dir / 'history.csv'
            new_file = not history.exists()
            with history.open('a', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                if new_file:
                    writer.writeheader()
                writer.writerow(record)
            print(f'Epoch {global_epoch}: {stage} {stage_epoch + 1}/{budget}, '
                  f'PSNR={validation["psnr"]:.3f}, SSIM={validation["ssim"]:.4f}, '
                  f'loss={validation["loss"]:.5f}, seconds={record["seconds"]}', flush=True)
        resume = None
        del criterion, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        if global_epoch >= settings['max_total_epochs']:
            break
    print(f'Finished after {global_epoch} total epochs. Compare best_reconstruction.pt and best_refinement.pt visually; '
          'best.pt is selected by the fixed reconstruction score.')


if __name__ == '__main__':
    main()
