import unittest
from argparse import Namespace

import torch

from models.iTransformer import Model


class ITransformerModelTests(unittest.TestCase):
    def test_direct_forecast_shape_and_finiteness(self):
        configs = Namespace(
            seq_len=20,
            pred_len=40,
            enc_in=3,
            d_model=32,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
            e_layers=2,
            activation="gelu",
        )
        model = Model(configs).eval()
        x = torch.randn(4, configs.seq_len, configs.enc_in)

        with torch.no_grad():
            prediction = model(x)

        self.assertEqual(tuple(prediction.shape), (4, configs.pred_len, configs.enc_in))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
