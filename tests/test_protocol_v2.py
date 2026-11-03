import types
import unittest

import numpy as np
import torch

from data_provider.data_loader import BallisticDataset
from exp.exp_ballistic import MaskedMSELoss, PhysicalCompositeLoss
from utils.provenance import masked_trajectory_metrics


class MaskedObjectiveTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.pred = torch.randn(4, 6, 3, generator=generator)
        self.true = torch.randn(4, 6, 3, generator=generator)
        self.mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0]],
                                 dtype=torch.float32)

    def test_all_ones_mask_matches_unmasked(self):
        loss = PhysicalCompositeLoss(scale=[2.0, 3.0, 4.0])
        ones = torch.ones(4, 6)
        self.assertAlmostEqual(float(loss(self.pred, self.true)), float(loss(self.pred, self.true, ones)), places=5)
        error = (self.pred - self.true) * torch.tensor([2.0, 3.0, 4.0])
        expected = (torch.sqrt(error.square().mean()) + 0.5 * error.norm(dim=-1).mean()
                    + 0.25 * error[:, -1].norm(dim=-1).mean())
        self.assertAlmostEqual(float(loss(self.pred, self.true)), float(expected), places=4)
        self.assertAlmostEqual(float(MaskedMSELoss()(self.pred, self.true, ones)),
                               float(torch.nn.functional.mse_loss(self.pred, self.true)), places=6)

    def test_masked_steps_do_not_matter(self):
        corrupted = self.pred.clone()
        corrupted[self.mask == 0] = 1e6
        for loss in (PhysicalCompositeLoss(scale=[1.0, 1.0, 1.0]), MaskedMSELoss()):
            self.assertAlmostEqual(float(loss(self.pred, self.true, self.mask)),
                                   float(loss(corrupted, self.true, self.mask)), places=4)

    def test_metrics_use_last_valid_step(self):
        mask = self.mask.numpy().astype(bool)
        metrics = masked_trajectory_metrics(self.pred.numpy(), self.true.numpy(), mask)
        distance = np.linalg.norm(self.pred.numpy() - self.true.numpy(), axis=-1)
        self.assertAlmostEqual(metrics['ade_m'], float(distance[mask].mean()), places=5)
        self.assertAlmostEqual(metrics['fde_m'], float(np.mean([distance[0, 5], distance[1, 2],
                                                                 distance[2, 3], distance[3, 0]])), places=5)


class DatasetMaskTest(unittest.TestCase):
    def test_dataset_returns_future_mask(self):
        args = types.SimpleNamespace(seq_len=2, label_len=0, pred_len=4, scale=False)
        data = np.zeros((2, 6, 3), dtype=np.float32)
        dataset = BallisticDataset(args, data, valid_len=np.array([4, 2]))
        self.assertEqual(dataset[0][4].tolist(), [1, 1, 1, 1])
        self.assertEqual(dataset[1][4].tolist(), [1, 1, 0, 0])
        legacy = BallisticDataset(args, data)
        self.assertEqual(legacy[1][4].tolist(), [1, 1, 1, 1])
        self.assertEqual(tuple(legacy[1][5].shape), (1, 3))

    def test_dataset_scales_anchor_like_targets(self):
        from utils.tools import StandardScaler
        args = types.SimpleNamespace(seq_len=2, label_len=0, pred_len=4, scale=True)
        rng = np.random.default_rng(0)
        data = rng.normal(size=(3, 6, 3)).astype(np.float32) * 100
        anchor = data[:, 2:6] + 1.0
        scaler = StandardScaler(mean=data.mean(axis=(0, 1)), std=data.std(axis=(0, 1)))
        dataset = BallisticDataset(args, data, scaler=scaler, anchor=anchor)
        _, seq_y, _, _, _, scaled_anchor = dataset[1]
        np.testing.assert_allclose(
            dataset.inverse_transform(scaled_anchor.numpy()), anchor[1], rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(dataset.inverse_transform(seq_y.numpy()), data[1, 2:6], rtol=1e-5, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
