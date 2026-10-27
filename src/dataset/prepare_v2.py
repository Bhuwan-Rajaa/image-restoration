"""Generate lossless, reproducible pairs including hair; never starts training."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import load_config
from damage_adapter import DamageSimulatorAdapter

_ADAPTER = None


def initialize_worker(assets, artifact_types):
    global _ADAPTER
    _ADAPTER = DamageSimulatorAdapter(assets, artifact_types=artifact_types)


def variant_seed(seed, source, variant):
    return int.from_bytes(hashlib.sha256(f'{seed}:{source}:{variant}'.encode()).digest()[:4], 'little')


def process_source(task):
    source, split, cls, index, config = task
    cfg = config['dataset']
    source = Path(source)
    out = Path(cfg['output'])
    size = cfg['image_size']
    source_id = f'{cls}/{source.name}'
    with Image.open(source) as original:
        clean = np.array(original.convert('RGB').resize((size, size), Image.Resampling.LANCZOS))
    clean_rel = f'{split}/{cls}/clean/{source.stem}.png'
    Image.fromarray(clean).save(out / clean_rel)
    rows = []
    types = cfg['artifact_types']
    for variant in range(cfg['variants']):
        seed = variant_seed(config['seed'], source_id, variant)
        random.seed(seed)
        np.random.seed(seed)
        # Balanced single-type coverage plus mixed variants, independent of worker order.
        anchor = types[(index + variant) % len(types)]
        selected = [anchor]
        if variant > 0:
            selected += random.sample([t for t in types if t != anchor], min(random.randint(1, 2), len(types) - 1))
        severity = random.choice(cfg['severities'])
        low, high = {'light': (0.3, 0.5), 'medium': (0.5, 0.7), 'heavy': (0.7, 0.9)}[severity]
        strength = random.uniform(low, high)
        damaged = clean.copy()
        union = np.full((size, size), 255, dtype=np.uint8)
        polarities = {}
        for kind in selected:
            # The supplied transparent assets define shape; do not replace with drawn lines.
            polarity = 'dark' if kind in ('hair', 'hair-short') else 'bright'
            for attempt in range(40):
                mask = _ADAPTER.generate_mask((size, size), [kind], severity)
                candidate = _ADAPTER.apply_damage(damaged, mask, strength, polarity)
                if np.count_nonzero(np.any(candidate != damaged, axis=2)) >= 16:
                    break
            else:
                raise RuntimeError(f'No visible {kind} damage after 40 attempts: {source_id}')
            damaged = candidate
            union = np.minimum(union, mask)
            polarities[kind] = polarity
        if not np.array_equal(clean[union == 255], damaged[union == 255]):
            raise AssertionError(f'Changed undamaged pixels: {source_id}')
        stem = f'{source.stem}_v{variant + 1:02d}'
        damaged_rel = f'{split}/{cls}/damaged/{stem}.png'
        mask_rel = f'{split}/{cls}/masks/{stem}_mask.png'
        Image.fromarray(damaged).save(out / damaged_rel)
        Image.fromarray(union).save(out / mask_rel)
        rows.append(dict(source_image=source_id, clean_image=clean_rel, damaged_image=damaged_rel,
                         mask=mask_rel, split=split, **{'class': cls}, damage_types=selected,
                         severity=severity, strength=strength, polarity=polarities, seed=seed,
                         image_size=[size, size], variant=f'v{variant + 1:02d}',
                         damage_fraction=float(np.mean(union == 0))))
    return rows


def write_previews(out, rows, types):
    preview = out / 'previews'
    preview.mkdir(exist_ok=True)
    for kind in types:
        entry = next((row for row in rows if row['damage_types'] == [kind]), None)
        if entry is None:
            continue
        panels = [Image.open(out / entry[key]).convert('RGB') for key in ('clean_image', 'mask', 'damaged_image')]
        width, height = panels[0].size
        canvas = Image.new('RGB', (width * 3, height + 25), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, (panel, label) in enumerate(zip(panels, ('Clean', 'Mask: black = damage', f'Damaged: {kind}'))):
            canvas.paste(panel, (index * width, 25))
            draw.text((index * width + 5, 5), label, fill='black')
            panel.close()
        canvas.save(preview / f'{kind}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--output', help='Override output, useful for a preparation preview')
    parser.add_argument('--limit', type=int, help='Maximum source images per split for a preparation preview')
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config['dataset']
    if args.output:
        cfg['output'] = str(Path(args.output).resolve())
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    out = Path(cfg['output'])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty; choose a new directory: {out}')
    adapter = DamageSimulatorAdapter(cfg['assets'], artifact_types=cfg['artifact_types'])
    missing = [kind for kind in adapter.artifact_types if not adapter.assets[kind]]
    if missing:
        raise FileNotFoundError(f'Missing PNG assets: {missing}')
    tasks, identities = {}, {}
    fingerprint = hashlib.sha256(json.dumps({'seed': config['seed'], 'dataset': cfg}, sort_keys=True).encode())
    for script in (Path(__file__), Path(__file__).with_name('damage_adapter.py')):
        fingerprint.update(script.read_bytes())
    for split in ('train', 'val', 'test'):
        sources = []
        for cls in ('buildings', 'street'):
            paths = sorted((Path(cfg['clean_splits']) / split / cls).glob('*.jpg'))
            if not paths:
                raise FileNotFoundError(f'No source images for {split}/{cls}')
            sources.extend((p, cls) for p in paths)
        identities[split] = {f'{cls}/{p.name}' for p, cls in sources}
        if args.limit:
            random.Random(config['seed']).shuffle(sources)
            sources = sources[:args.limit]
        tasks[split] = [(str(p), split, cls, i, config) for i, (p, cls) in enumerate(sources)]
        for p, cls in sources:
            fingerprint.update(f'{split}/{cls}/{p.name}'.encode())
            fingerprint.update(hashlib.sha256(p.read_bytes()).digest())
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if identities[a] & identities[b]:
            raise ValueError(f'Overlapping source IDs in {a}/{b}')
    for kind in adapter.artifact_types:
        for asset in adapter.assets[kind]:
            fingerprint.update(f'{kind}/{asset.name}'.encode())
            fingerprint.update(hashlib.sha256(asset.read_bytes()).digest())
    for split in tasks:
        for cls in ('buildings', 'street'):
            for folder in ('clean', 'damaged', 'masks'):
                (out / split / cls / folder).mkdir(parents=True, exist_ok=True)
    (out / 'metadata').mkdir(exist_ok=True)
    summary = {'version': 2, 'dataset_id': fingerprint.hexdigest(), 'config': config,
               'preview_only': args.limit is not None, 'mask_convention': '255=clean, 0=damage',
               'asset_counts': {k: len(v) for k, v in adapter.assets.items()}, 'splits': {}}
    with ProcessPoolExecutor(max_workers=cfg['workers'], initializer=initialize_worker,
                             initargs=(cfg['assets'], cfg['artifact_types'])) as executor:
        for split, jobs in tasks.items():
            rows = []
            for generated in tqdm(executor.map(process_source, jobs, chunksize=8), total=len(jobs), desc=split):
                rows.extend(generated)
            (out / 'metadata' / f'{split}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            summary['splits'][split] = {'sources': len(jobs), 'pairs': len(rows),
                'artifact_occurrences': dict(Counter(t for row in rows for t in row['damage_types'])),
                'single_type_counts': dict(Counter(row['damage_types'][0] for row in rows if len(row['damage_types']) == 1)),
                'severity_counts': dict(Counter(row['severity'] for row in rows))}
            if split == 'val':
                write_previews(out, rows, cfg['artifact_types'])
    # Only a successfully completed generation receives this readiness marker.
    (out / 'prepared.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary['splits'], indent=2))
    print(f'Prepared {out}. Training has NOT been started.')


if __name__ == '__main__':
    main()
