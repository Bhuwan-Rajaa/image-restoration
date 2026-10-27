"""Offline RGB restoration with overlap blending for large photographs."""
from pathlib import Path
import threading
import time

from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from restoration_v2 import load_restoration_model


def tile_positions(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    positions = list(range(0, length-tile_size+1, tile_size-overlap))
    if positions[-1] != length-tile_size:
        positions.append(length-tile_size)
    return positions


class RestorationEngine:
    def __init__(self, checkpoint, device=None, model=None):
        torch.set_num_threads(2)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device).eval() if model is not None else load_restoration_model(checkpoint, self.device)
        self.checkpoint = Path(checkpoint)
        self.lock = threading.Lock()

    @torch.inference_mode()
    def _patch(self, image):
        tensor = pil_to_tensor(image).float().div_(255).unsqueeze(0).to(self.device)
        h, w = tensor.shape[-2:]
        tensor = F.pad(tensor, (0, (-w)%32, 0, (-h)%32), mode='replicate')
        output = self.model(tensor)[0, :, :h, :w].float()
        if not torch.isfinite(output).all():
            raise RuntimeError('The model returned an invalid image')
        return output.clamp(0, 1).cpu()

    def restore(self, image):
        image = ImageOps.exif_transpose(image)
        grayscale = image.mode in ('1', 'L', 'LA', 'I', 'I;16', 'F')
        if image.mode in ('RGBA', 'LA') or image.mode == 'P' and 'transparency' in image.info:
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', image.size, 'white')
            image = Image.alpha_composite(background, rgba).convert('RGB')
        else:
            image = image.convert('RGB')
        width, height = image.size
        started = time.perf_counter()
        tiled = width > 512 or height > 512
        with self.lock:
            if not tiled:
                restored = self._patch(image)
            else:
                tile = 256
                ys, xs = tile_positions(height, tile, 64), tile_positions(width, tile, 64)
                restored = torch.zeros(3, height, width)
                weights = torch.zeros(1, height, width)
                for top in ys:
                    for left in xs:
                        right, bottom = min(width, left+tile), min(height, top+tile)
                        patch = self._patch(image.crop((left, top, right, bottom)))
                        h, w = patch.shape[-2:]
                        window = (torch.hann_window(h, periodic=False)[:, None]*
                                  torch.hann_window(w, periodic=False)[None, :]).clamp_min(.01)
                        restored[:, top:bottom, left:right] += patch*window
                        weights[:, top:bottom, left:right] += window
                restored /= weights
        output = to_pil_image(restored.mul(255).round().clamp(0, 255).to(torch.uint8))
        if grayscale:
            output = output.convert('L')
            image = image.convert('L')
        return image, output, dict(width=width, height=height, seconds=round(time.perf_counter()-started, 2), tiled=tiled)
