"""Damage a photo using this project's FilmDamageSimulator asset adapter.

Examples:
    python damage_image.py "photo.jpg" --types scratches hair --output-dir "damaged"
    python damage_image.py "photo.jpg" --types dots,stain --severity light --seed 42

Run without --types for an interactive damage menu and output-folder prompt.
"""
import argparse
import json
import math
from pathlib import Path
import random
import re
import secrets
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src/dataset'))
from damage_adapter import DamageSimulatorAdapter

TYPES = ('dirt', 'dots', 'scratches', 'smut', 'spots', 'sprinkles', 'stain', 'hair', 'hair-short')
ALIASES = {'scratch':'scratches', 'spot':'spots', 'stains':'stain', 'sprinkle':'sprinkles',
           'long-hair':'hair', 'short-hair':'hair-short', 'hair_short':'hair-short'}
STRENGTHS = {'light':(.3, .5), 'medium':(.5, .7), 'heavy':(.7, .9)}


def parse_types(values):
    if isinstance(values, str):
        values = [values]
    selected = []
    for value in values:
        for token in re.split(r'[,\s]+', value.strip().lower()):
            if not token:
                continue
            if token.isdecimal() and 1 <= int(token) <= len(TYPES):
                token = TYPES[int(token)-1]
            token = ALIASES.get(token, token)
            if token not in TYPES:
                raise ValueError(f'Unknown damage type: {token}. Choose from: {", ".join(TYPES)}')
            if token not in selected:
                selected.append(token)
    if not selected:
        raise ValueError('Choose at least one damage type.')
    return selected


def unquote(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def damage_photo(image, selected, severity='medium', seed=None, strength=None, assets_dir=None):
    selected = parse_types(selected)
    if severity not in STRENGTHS:
        raise ValueError('Severity must be light, medium, or heavy.')
    if seed is None:
        seed = secrets.randbits(32)
    if not 0 <= seed < 2**32:
        raise ValueError('Seed must be between 0 and 4294967295.')
    if strength is not None and not 0 < strength <= 1:
        raise ValueError('Strength must be greater than 0 and at most 1.')
    adapter = DamageSimulatorAdapter(assets_dir or ROOT/'damage dataset', artifact_types=selected)
    missing = [kind for kind in selected if not adapter.assets[kind]]
    if missing:
        raise FileNotFoundError(f'Missing PNG damage assets: {", ".join(missing)}')
    image = ImageOps.exif_transpose(image)
    grayscale = image.mode in ('1', 'L', 'LA', 'I', 'I;16', 'F')
    if image.mode in ('RGBA', 'LA') or image.mode == 'P' and 'transparency' in image.info:
        rgba = image.convert('RGBA')
        image = Image.alpha_composite(Image.new('RGBA', image.size, 'white'), rgba).convert('RGB')
    else:
        image = image.convert('RGB')
    clean = np.array(image)
    damaged = clean.copy()
    width, height = image.size
    # The adapter's Perlin grid requires even dimensions. A bounded working
    # mask also avoids allocating large noise fields for camera-sized photos.
    # Only the mask is resized; photo pixels and dimensions are preserved.
    scale = min(1., 512/max(width, height))
    work_width = max(2, math.ceil(width*scale/2)*2)
    work_height = max(2, math.ceil(height*scale/2)*2)
    union = np.full((height, width), 255, dtype=np.uint8)
    random_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        strength = random.uniform(*STRENGTHS[severity]) if strength is None else strength
        polarities, counts = {}, {}
        for kind in selected:
            polarity = 'dark' if kind in ('hair', 'hair-short') else 'bright'
            for _ in range(40):
                mask = adapter.generate_mask((work_height, work_width), [kind], severity)
                mask = np.array(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST))
                candidate = adapter.apply_damage(damaged, mask, strength, polarity)
                changed = int(np.count_nonzero(np.any(candidate != damaged, axis=2)))
                if changed:
                    break
            else:
                raise ValueError(f'Could not make visible {kind} damage on this photo. '
                                 'Try a different type or photo; bright damage cannot show on pure white.')
            damaged = candidate
            union = np.minimum(union, mask)
            polarities[kind], counts[kind] = polarity, changed
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
    if not np.array_equal(clean[union == 255], damaged[union == 255]):
        raise AssertionError('Pixels outside the damage mask changed')
    if np.array_equal(clean, damaged):
        raise ValueError('The selected damage types cancelled each other. Try another seed or type combination.')
    output = Image.fromarray(damaged)
    if grayscale:
        output = output.convert('L')
    info = dict(damage_types=selected, severity=severity, strength=strength, seed=seed,
                polarity=polarities, width=width, height=height,
                generation_size=[work_width, work_height],
                mask_convention='255=clean, 0=damage', damage_fraction=float(np.mean(union == 0)),
                changed_pixels_by_application=counts,
                generator='Local FilmDamageSimulator asset adapter; dark hair and bright other damage')
    return output, Image.fromarray(union), info


def save_outputs(source, folder, damaged, mask, info):
    source, folder = Path(source).resolve(), Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    base = f'{source.stem[:70]}_damaged_{"-".join(info["damage_types"])}_{info["severity"]}'
    number = 1
    while True:
        name = base if number == 1 else f'{base}_{number}'
        paths = {'image':folder/f'{name}.png', 'mask':folder/f'{name}_mask.png', 'settings':folder/f'{name}.json'}
        if not any(path.exists() or path == source for path in paths.values()):
            break
        number += 1
    metadata = dict(info, source_image=str(source), damaged_image=str(paths['image']), mask=str(paths['mask']))
    created = []
    try:
        for key, image in [('image', damaged), ('mask', mask)]:
            with paths[key].open('xb') as stream:
                created.append(paths[key])
                image.save(stream, format='PNG')
        with paths['settings'].open('x', encoding='utf-8') as stream:
            created.append(paths['settings'])
            json.dump(metadata, stream, indent=2)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return paths


def print_types():
    print('\nAvailable damage types:')
    for index, kind in enumerate(TYPES, 1):
        label = {'hair':'hair (long hair)', 'hair-short':'hair-short (short hair)'}.get(kind, kind)
        print(f'  {index}. {label}')
    print('Choose names or numbers, separated by commas. Example: scratches,hair or 3,8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', nargs='?', help='Photo to damage')
    parser.add_argument('--types', nargs='+', help='One or more damage names, separated by spaces or commas')
    parser.add_argument('--severity', choices=list(STRENGTHS))
    parser.add_argument('-o', '--output-dir', help='Folder for damaged image, mask, and settings')
    parser.add_argument('--seed', type=int, help='Repeat a specific damage pattern; otherwise random')
    parser.add_argument('--strength', type=float, help='Optional intensity override: greater than 0, up to 1')
    parser.add_argument('--list-types', action='store_true')
    args = parser.parse_args(argv)
    if args.list_types:
        print_types()
        return 0
    try:
        source = Path(unquote(args.image or input('Image path (or drag a photo into this window): '))).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f'Image not found: {source}')
        interactive = args.types is None
        if interactive:
            print_types()
            selected = parse_types(input('Damage types: '))
        else:
            selected = parse_types(args.types)
        severity = args.severity
        if severity is None:
            severity = (input('Severity: light / medium / heavy [medium]: ').strip().lower() or 'medium') if interactive else 'medium'
        folder = args.output_dir
        if folder is None:
            default = ROOT/'damaged_photos'
            folder = (input(f'Save folder [{default}]: ').strip() or str(default)) if interactive else str(default)
        folder = unquote(folder)
        print(f'Applying {", ".join(selected)} ({severity})...', flush=True)
        with Image.open(source) as image:
            damaged, mask, info = damage_photo(image, selected, severity, args.seed, args.strength)
        paths = save_outputs(source, folder, damaged, mask, info)
        print(f'Damaged photo: {paths["image"]}')
        print(f'Damage mask:   {paths["mask"]}')
        print(f'Settings:      {paths["settings"]}')
        print(f'Seed: {info["seed"]} | {info["width"]} x {info["height"]} pixels | Original preserved.')
        return 0
    except (EOFError, KeyboardInterrupt):
        print('\nCancelled.', file=sys.stderr)
        return 1
    except Exception as error:
        print(f'Could not damage the photo: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
