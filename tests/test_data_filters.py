import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_loader import filter_dataset_by_metadata


class DataFilterTests(unittest.TestCase):
    def test_filter_dataset_by_wind_keeps_only_requested_subset(self):
        data_dict = {
            'X': np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3),
            'meta': {
                'weapon': ['a', 'b', 'c', 'd'],
                'wind': ['constant', 'random', 'constant', 'linear'],
            },
        }

        filtered = filter_dataset_by_metadata(data_dict, wind_filter='constant')

        self.assertEqual(filtered['X'].shape[0], 2)
        self.assertEqual(filtered['meta']['wind'], ['constant', 'constant'])
        self.assertEqual(filtered['meta']['weapon'], ['a', 'c'])


if __name__ == '__main__':
    unittest.main()
