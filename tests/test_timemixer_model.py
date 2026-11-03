import os
import sys
import unittest
from types import SimpleNamespace

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model


class TimeMixerModelTests(unittest.TestCase):
    def test_timemixer_forward_shape(self):
        configs = SimpleNamespace(
            task_name='ballistic_forecast',
            model='TimeMixer',
            seq_len=100,
            pred_len=1000,
            enc_in=3,
            c_out=3,
            d_model=32,
            d_ff=64,
            e_layers=2,
            dropout=0.1,
        )
        model = build_model(configs)
        x = torch.randn(4, configs.seq_len, configs.enc_in)

        y = model(x, None, None, None)

        self.assertEqual(tuple(y.shape), (4, configs.pred_len, configs.c_out))


if __name__ == '__main__':
    unittest.main()
