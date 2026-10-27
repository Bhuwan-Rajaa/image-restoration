"""Offline regression checks. No training loop, backward pass, or optimizer step."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/model'))
sys.path.insert(0, str(ROOT / 'src/dataset'))
from damage_adapter import DamageSimulatorAdapter
from dataset_v2 import RestorationPairs
from losses_v2 import RestorationLoss, structural_similarity
from pipeline_config import load_config
from restoration_v2 import RestorationUNet, load_restoration_model
from prepare_v2 import variant_seed


class RestorationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_at_odd_resolution_and_frozen_batchnorm(self):
        model = RestorationUNet(pretrained=False)
        model.freeze_encoder(True)
        model.train()
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        norms = [m for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        before = [m.running_mean.clone() for m in norms]
        image = torch.rand(1, 3, 65, 97)
        with torch.no_grad():
            output = model(image)
        torch.testing.assert_close(output, image, rtol=0, atol=0)
        for norm, mean in zip(norms, before):
            self.assertFalse(norm.training)
            torch.testing.assert_close(norm.running_mean, mean, rtol=0, atol=0)
        model.freeze_encoder(False)
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(not m.training for m in norms))

    def test_structural_loss_identity_and_perturbation(self):
        torch.manual_seed(4)
        image = torch.rand(2, 3, 32, 32)
        criterion = RestorationLoss()
        identity_loss, _ = criterion(image, image)
        other_loss, _ = criterion(image * 0.7, image)
        self.assertLess(abs(identity_loss.item()), 1e-6)
        self.assertGreater(other_loss.item(), identity_loss.item())
        self.assertTrue(torch.isfinite(structural_similarity(torch.zeros_like(image), image)).all())

    def test_hair_uses_assets_and_preserves_background(self):
        random.seed(1)
        np.random.seed(1)
        adapter = DamageSimulatorAdapter(ROOT / 'damage dataset')
        for kind in ('hair', 'hair-short'):
            self.assertTrue(adapter.assets[kind])
            mask = adapter.generate_mask((256, 256), [kind], 'medium')
            self.assertTrue((mask == 0).any())
            clean = np.full((256, 256, 3), 150, dtype=np.uint8)
            damaged = adapter.apply_damage(clean, mask, 0.4, 'dark')
            np.testing.assert_array_equal(damaged[mask == 255], clean[mask == 255])
            self.assertTrue((damaged[mask == 0] < clean[mask == 0]).all())

    def test_identity_augmentation_and_pair_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata').mkdir()
            clean = np.full((32, 32, 3), 80, dtype=np.uint8)
            mask = np.full((32, 32), 255, dtype=np.uint8)
            mask[8:16, 8:16] = 0
            damaged = clean.copy()
            damaged[mask == 0] = 0
            for name, array in (('clean', clean), ('damaged', damaged), ('mask', mask)):
                Image.fromarray(array).save(root / f'{name}.png')
            rows = [{'source_image': 'example', 'clean_image': 'clean.png', 'damaged_image': 'damaged.png', 'mask': 'mask.png'}]
            (root / 'metadata/train.json').write_text(json.dumps(rows))
            dataset = RestorationPairs(root, 'train', True, identity_probability=1)
            x, y, m = dataset[0]
            torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertEqual(m.sum().item(), 0)
            dataset = RestorationPairs(root, 'train', True, identity_probability=0)
            x, y, m = dataset[0]
            self.assertEqual(((x - y).abs() * (1 - m)).sum().item(), 0)

    def test_versioned_checkpoint_load_without_pretrained_download(self):
        model = RestorationUNet(pretrained=False).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'architecture': model.architecture, 'model': model.state_dict()}, path)
            loaded = load_restoration_model(path)
            image = torch.rand(1, 3, 32, 32)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(image), model(image), rtol=0, atol=0)

    def test_config_budget_and_stable_distinct_variant_seeds(self):
        config = load_config()
        self.assertIn('hair', config['dataset']['artifact_types'])
        self.assertIn('hair-short', config['dataset']['artifact_types'])
        settings = config['training']
        self.assertLessEqual(settings['reconstruction_epochs'] + settings['refinement_epochs'], settings['max_total_epochs'])
        self.assertEqual(variant_seed(42, 'a', 0), variant_seed(42, 'a', 0))
        self.assertNotEqual(variant_seed(42, 'a', 0), variant_seed(42, 'a', 1))


if __name__ == '__main__':
    unittest.main()
