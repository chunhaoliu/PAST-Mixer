"""Generate non-operational synthetic trajectories for software smoke tests."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def make_split(rng, count, steps, seq_len, dt):
    t = np.arange(steps, dtype=np.float32)[None, :, None] * dt
    p0 = rng.normal(0.0, 0.2, size=(count, 1, 3)).astype(np.float32)
    velocity = rng.uniform(0.2, 1.0, size=(count, 1, 3)).astype(np.float32)
    acceleration = rng.uniform(-0.03, 0.03, size=(count, 1, 3)).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(count, 1, 3)).astype(np.float32)
    clean = p0 + velocity * t + 0.5 * acceleration * (t ** 2)
    clean += 0.015 * np.sin(0.7 * t + phase)
    observed = clean[:, :seq_len].copy()
    observed += rng.normal(0.0, 0.003, size=observed.shape).astype(np.float32)
    return {'X': torch.from_numpy(clean), 'X_observed': torch.from_numpy(observed)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/demo')
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--pred-len', type=int, default=32)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    steps = args.seq_len + args.pred_len
    for name, count in {'train': 32, 'val': 8, 'test': 8}.items():
        torch.save(make_split(rng, count, steps, args.seq_len, args.dt), output / f'{name}.pt')
    manifest = {
        'protocol_id': 'software_demo_only',
        'purpose': 'Non-operational synthetic data for interface and smoke testing; not used in the paper.',
        'default_observation_mode': 'tracked',
        'sample_interval_s': args.dt,
        'observation_steps': args.seq_len,
        'prediction_steps': args.pred_len,
        'seed': args.seed,
    }
    (output / 'protocol_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
