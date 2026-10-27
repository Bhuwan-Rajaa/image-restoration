"""Check selectable asset damage, dimensions, reproducibility, and file safety."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from damage_image import TYPES, damage_photo, parse_types, save_outputs


class DamageImageChecks(unittest.TestCase):
    def test_names_numbers_mixtures_and_unknown_types(self):
        self.assertEqual(parse_types(['3,8', 'hair', 'short-hair']), ['scratches','hair','hair-short'])
        with self.assertRaises(ValueError):
            parse_types(['not-real-damage'])
        with self.assertRaises(ValueError):
            parse_types([''])

    def test_all_types_change_only_masked_pixels_at_odd_dimensions(self):
        image = Image.new('RGB', (317, 239), (125, 125, 125))
        original = np.array(image)
        for seed, kind in enumerate(TYPES):
            with self.subTest(kind=kind):
                damaged, mask, info = damage_photo(image, [kind], seed=seed)
                actual, region = np.array(damaged), np.array(mask)
                self.assertEqual(damaged.size, image.size)
                self.assertEqual(mask.size, image.size)
                self.assertTrue(np.any(actual != original))
                np.testing.assert_array_equal(actual[region == 255], original[region == 255])
                if kind in ('hair','hair-short'):
                    self.assertTrue(np.all(actual <= original))
                else:
                    self.assertTrue(np.all(actual >= original))
                self.assertEqual(info['damage_types'], [kind])
        np.testing.assert_array_equal(np.array(image), original)

    def test_mixed_damage_is_seeded_and_does_not_change_global_rng(self):
        image = Image.new('RGB', (733, 317), 'gray')
        random.seed(123)
        np.random.seed(123)
        state, numpy_state = random.getstate(), np.random.get_state()
        first, mask, info = damage_photo(image, ['scratches','hair'], seed=42)
        self.assertEqual(random.getstate(), state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        second, other_mask, other_info = damage_photo(image, ['scratches','hair'], seed=42)
        np.testing.assert_array_equal(np.array(first), np.array(second))
        np.testing.assert_array_equal(np.array(mask), np.array(other_mask))
        self.assertEqual(info, other_info)
        self.assertEqual(first.size, image.size)
        self.assertLessEqual(max(info['generation_size']), 512)

    def test_output_folder_settings_and_original_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'original.png'
            Image.new('L',(201,119),128).save(source)
            before = source.read_bytes()
            with Image.open(source) as image:
                damaged, mask, info = damage_photo(image, ['hair-short'], severity='light', seed=1)
            self.assertEqual(damaged.mode, 'L')
            folder = Path(directory)/'chosen output'
            first = save_outputs(source, folder, damaged, mask, info)
            second = save_outputs(source, folder, damaged, mask, info)
            self.assertNotEqual(first['image'], second['image'])
            self.assertTrue(all(path.is_file() for path in first.values()))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(json.loads(first['settings'].read_text())['damage_types'], ['hair-short'])

    def test_invalid_options_and_missing_assets_stop_generation(self):
        image = Image.new('RGB',(32,32),'gray')
        for kwargs in [{'severity':'extreme'}, {'strength':0}, {'strength':float('nan')}, {'seed':-1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                damage_photo(image, ['hair'], **kwargs)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            damage_photo(image, ['hair'], assets_dir=directory)


if __name__ == '__main__':
    unittest.main()
