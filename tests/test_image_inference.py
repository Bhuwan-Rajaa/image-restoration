"""Check native image dimensions, tile blending, and preservation of input files."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src/model'))
from inference import RestorationEngine
from restore_image import output_path_for


class ImageInferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RestorationEngine('unused.pt', device='cpu', model=torch.nn.Identity())

    def test_tiling_preserves_every_pixel_and_dimensions(self):
        values = np.random.default_rng(7).integers(0, 256, (317, 733, 3), dtype=np.uint8)
        _, restored, info = self.engine.restore(Image.fromarray(values))
        self.assertTrue(info['tiled'])
        self.assertEqual(restored.size, (733, 317))
        np.testing.assert_array_equal(np.array(restored), values)

    def test_grayscale_stays_grayscale(self):
        image = Image.fromarray(np.arange(31*17, dtype=np.uint16).reshape(17, 31).astype(np.uint8))
        _, restored, _ = self.engine.restore(image)
        self.assertEqual(restored.mode, 'L')
        self.assertEqual(restored.size, image.size)
        np.testing.assert_array_equal(np.array(restored), np.array(image))

    def test_exif_orientation_is_applied(self):
        image = Image.new('RGB', (21, 35), 'red')
        image.getexif()[274] = 6
        original, restored, _ = self.engine.restore(image)
        self.assertEqual(original.size, (35, 21))
        self.assertEqual(restored.size, (35, 21))

    def test_default_output_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'photo.png'
            source.write_bytes(b'original')
            previous = Path(directory)/'photo_restored.png'
            previous.write_bytes(b'previous')
            self.assertEqual(output_path_for(source).name, 'photo_restored_2.png')
            self.assertEqual(source.read_bytes(), b'original')
            self.assertEqual(previous.read_bytes(), b'previous')
            with self.assertRaises(ValueError):
                output_path_for(source, source)
            with self.assertRaises(FileExistsError):
                output_path_for(source, previous)


if __name__ == '__main__':
    unittest.main()
