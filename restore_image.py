"""Restore a photograph with the final trained model. No internet is needed.

Usage:
    python restore_image.py "D:\\Photos\\damaged.jpg"
    python restore_image.py "damaged.png" --output "restored.png"

Without arguments, the script asks for the image path. By default, it saves a
PNG beside the input, adding _restored to its name. Existing files are preserved.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src/model'))


def output_path_for(image_path, requested=None):
    image_path = Path(image_path).resolve()
    if requested:
        output = Path(requested).expanduser().resolve()
        if not output.suffix:
            output = output.with_suffix('.png')
        if output.suffix.lower() != '.png':
            raise ValueError('The output must be a .png file.')
        if output == image_path:
            raise ValueError('Choose an output path different from the input.')
        if output.exists():
            raise FileExistsError(f'Output already exists: {output}')
        return output
    output = image_path.with_name(f'{image_path.stem}_restored.png')
    number = 2
    while output.exists() or output == image_path:
        output = image_path.with_name(f'{image_path.stem}_restored_{number}.png')
        number += 1
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', nargs='?', help='Path to the damaged image')
    parser.add_argument('--input', dest='input_path', help='Alternative to the positional image path')
    parser.add_argument('-o', '--output', help='Output PNG; defaults to INPUT_restored.png beside the image')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--checkpoint', default=str(ROOT/'checkpoints/final_restoration.pt'))
    args = parser.parse_args(argv)
    if args.image and args.input_path:
        parser.error('Supply either an image path or --input, not both.')
    image_name = args.image or args.input_path
    if not image_name:
        try:
            image_name = input('Enter the image path (or drag the image into this window): ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nNo image selected.', file=sys.stderr)
            return 1
    image_name = image_name.strip().strip('"')
    if len(image_name) >= 2 and image_name[0] == image_name[-1] == "'":
        image_name = image_name[1:-1]
    try:
        if not image_name:
            raise ValueError('No image selected.')
        image_path = Path(image_name).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f'Image not found: {image_path}')
        output = output_path_for(image_path, args.output)
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Model checkpoint not found: {checkpoint}')

        from PIL import Image
        from inference import RestorationEngine

        print('Loading the final trained model...', flush=True)
        engine = RestorationEngine(checkpoint, None if args.device == 'auto' else args.device)
        print(f'Restoring: {image_path.name} ({engine.device})', flush=True)
        with Image.open(image_path) as image:
            _, restored, info = engine.restore(image)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            restored.save(stream, format='PNG')
        print(f'Saved: {output}', flush=True)
        print(f'{info["width"]} x {info["height"]} pixels | {info["seconds"]:.2f} seconds', flush=True)
        return 0
    except Exception as error:
        print(f'Could not restore the image: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
