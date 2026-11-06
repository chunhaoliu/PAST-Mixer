"""Leave-one-projectile-class-out splits of protocol v2, with anchors and physics baselines.

For each class, train/val drop the class and test keeps only that class (both domains).
The Drag-LS beta prior and B4 setting therefore come from the three training classes.
Run with ``python -B tools/make_holdout_data.py``; existing outputs are skipped.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {'a155': '155mm_artillery', 'a122': '122mm_artillery', 'r130': '130mm_rocket', 'r227': '227mm_rocket'}


def subset(payload, keep):
    index = torch.as_tensor(np.flatnonzero(keep))
    out = {key: value[index] for key, value in payload.items() if torch.is_tensor(value)}
    out['meta'] = {key: [values[i] for i in index.tolist()] for key, values in payload['meta'].items()}
    out['protocol'] = dict(payload['protocol'])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'data/protocol_v2')
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'data/protocol_v2_hold')
    parser.add_argument('--out-root', type=Path, default=ROOT.parent / 'outputs/hold_v3')
    parser.add_argument('--test-source', type=Path, help='Take test.pt from here instead (default: --source)')
    args = parser.parse_args()
    splits = {name: torch.load((args.test_source if name == 'test' and args.test_source else args.source) /
                               f'{name}.pt', weights_only=False) for name in ('train', 'val', 'test')}
    tool = ROOT / 'tools/evaluate_protocol_v2_physics_baselines.py'
    for short, cls in CLASSES.items():
        data_dir = args.data_root / short
        data_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in splits.items():
            path = data_dir / f'{name}.pt'
            if path.exists():
                continue
            is_class = np.asarray(payload['meta']['projectile']) == cls
            split = subset(payload, is_class if name == 'test' else ~is_class)
            split['protocol']['holdout_class'] = cls
            torch.save(split, path)
            print(short, name, len(split['valid_len']), flush=True)
        manifest = data_dir / 'protocol_manifest.json'
        if not manifest.exists():
            source = json.loads((args.source / 'protocol_manifest.json').read_text(encoding='utf-8'))
            source['holdout_class'] = cls
            source['holdout_rule'] = 'train/val exclude holdout_class; test is holdout_class only (both domains)'
            if args.test_source:
                source['test_source'] = str(args.test_source)
            manifest.write_text(json.dumps(source, indent=2), encoding='utf-8')
        # The full baseline run also writes the anchors of every split.
        physics = args.out_root / short / 'physics'
        if not (physics / 'report.json').exists():
            subprocess.run([sys.executable, '-B', str(tool), '--data-dir', str(data_dir), '--out', str(physics)], check=True)


if __name__ == '__main__':
    main()
