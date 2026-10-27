"""Checks for region supervision, validation indexing, and final model export."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/model'))
from losses_v2 import MaskedRestorationLoss, RestorationLoss
from train_final import epoch, finish
from restoration_v2 import RestorationUNet, load_restoration_model


class TinyPairs(Dataset):
    metadata = [{'damage_types': ['hair']}, {'damage_types': ['hair-short']}]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        clean = torch.zeros(3, 32, 32)
        mask = torch.zeros(1, 32, 32)
        mask[:, 4:8, 4:8] = 1
        damaged = clean + mask*(.25 if index == 0 else .5)
        return damaged, clean, mask, index


class FinalTrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_small_masks_receive_equal_region_weight(self):
        clean = torch.zeros(2, 3, 32, 32)
        mask = torch.zeros(2, 1, 32, 32)
        mask[0, :, :1, :1] = 1
        mask[1, :, :16, :16] = 1
        prediction = (mask.expand_as(clean)*.5).clone().requires_grad_()
        criterion = MaskedRestorationLoss(perceptual=False, mask_weight=.1)
        loss, parts = criterion(prediction, clean, mask, 0)
        base, _ = RestorationLoss()(prediction, clean, 0)
        self.assertAlmostEqual(parts['mask_l1'].item(), .5, places=6)
        self.assertAlmostEqual((loss-base).item(), .05, places=6)
        (loss-base).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        torch.testing.assert_close(prediction.grad[0].sum(), prediction.grad[1].sum())

    def test_clean_identity_masks_have_no_region_penalty(self):
        clean = torch.rand(2, 3, 32, 32)
        loss, parts = MaskedRestorationLoss(perceptual=False)(clean, clean, torch.zeros(2, 1, 32, 32), 0)
        self.assertAlmostEqual(loss.item(), 0., places=6)
        self.assertEqual(parts['mask_l1'].item(), 0.)

    def test_validation_tracks_each_hair_category(self):
        data = TinyPairs()
        metrics = epoch(torch.nn.Identity(), DataLoader(data, batch_size=2),
                        MaskedRestorationLoss(perceptual=False), torch.device('cpu'), False, 0)
        self.assertAlmostEqual(metrics['hair_mae'], .25, places=6)
        self.assertAlmostEqual(metrics['hair-short_mae'], .5, places=6)
        self.assertEqual(metrics['clean_mae'], 0.)

    def test_export_loads_for_inference_without_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = RestorationUNet(pretrained=False).eval()
            config = {'final_training': {'export': str(root/'final.pt'), 'evaluate_after_training': False}}
            torch.save(dict(architecture=model.architecture, model=model.state_dict(), config=config,
                            dataset_id='test', metrics={'loss': .1}, global_epoch=36,
                            final_epoch=1, parent_checkpoint='parent.pt', optimizer={'not': 'exported'}), root/'best.pt')
            finish(root/'best.pt', config, root)
            saved = torch.load(root/'final.pt', weights_only=True)
            self.assertNotIn('optimizer', saved)
            loaded = load_restoration_model(root/'final.pt')
            image = torch.rand(1, 3, 32, 32)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(image), model(image), rtol=0, atol=0)
            self.assertEqual(json.loads((root/'status.json').read_text())['status'], 'trained')


if __name__ == '__main__':
    unittest.main()
