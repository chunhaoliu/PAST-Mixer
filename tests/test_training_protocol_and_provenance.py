import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from exp.exp_ballistic import Exp_Ballistic
from utils.provenance import write_evaluation_artifacts


class TrainingProtocolTests(unittest.TestCase):
    def test_train_does_not_request_test_split(self):
        source = Path(Exp_Ballistic.train.__code__.co_filename).read_text(encoding='utf-8')
        tree = ast.parse(source)
        train_node = next(
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'train'
        )
        requested_flags = []
        for node in ast.walk(train_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == '_get_data':
                for keyword in node.keywords:
                    if keyword.arg == 'flag' and isinstance(keyword.value, ast.Constant):
                        requested_flags.append(keyword.value.value)
        self.assertEqual(requested_flags, ['train', 'val'])


class ProvenanceTests(unittest.TestCase):
    @mock.patch('utils.provenance._project_root')
    @mock.patch('utils.provenance._git_info', return_value={'revision': 'test', 'dirty': False, 'tracked_diff_sha256': '0'})
    @mock.patch('utils.provenance.source_tree_fingerprint', return_value={'sha256': '1', 'file_count': 1})
    def test_artifacts_are_complete_and_non_overwriting(self, _, __, project_root_mock):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project_root_mock.return_value = root
            dataset = root / 'dataset'
            dataset.mkdir()
            (dataset / 'protocol_manifest.json').write_text('{}\n', encoding='utf-8')
            checkpoint = root / 'checkpoints' / 'setting'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'checkpoint.pth').write_bytes(b'checkpoint')
            args = SimpleNamespace(
                results=str(root / 'results'), checkpoints=str(root / 'checkpoints'),
                dataset_dir=str(dataset), protocol_id='protocol', model='PASTMixer',
                model_id='model_s2021', seed=2021, gpu=0,
            )
            pred = np.ones((2, 3, 3), dtype=np.float32)
            true = np.zeros_like(pred)
            write_evaluation_artifacts(args, 'setting', pred, true)
            result = root / 'results' / 'setting'
            self.assertEqual(
                {path.name for path in result.iterdir()},
                {'pred.npy', 'true.npy', 'metrics.json', 'run_manifest.json'},
            )
            manifest = json.loads((result / 'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['schema_version'], 2)
            self.assertEqual(manifest['run']['test_evaluation_count'], 1)
            with self.assertRaises(FileExistsError):
                write_evaluation_artifacts(args, 'setting', pred, true)


if __name__ == '__main__':
    unittest.main()
