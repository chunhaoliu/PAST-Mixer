"""Generate non-operational synthetic trajectories for software smoke tests.

Anchor files are placeholders: constant-velocity extrapolations of the observed
window, a linear fitted window, and an arbitrary fitted-beta placeholder. They
do not use Drag-LS, projectile parameters, wind, or a radar tracker and cannot
reproduce the manuscript benchmark.
"""
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
    horizon = steps - seq_len
    valid_len = rng.integers(max(1, horizon // 2), horizon + 1, size=count)
    valid_len[0] = max(1, horizon // 2)
    for i, valid in enumerate(valid_len):
        clean[i, seq_len + valid:] = clean[i, seq_len + valid - 1]
    return {'X': torch.from_numpy(clean), 'X_observed': torch.from_numpy(observed),
            'valid_len': torch.from_numpy(valid_len)}


def placeholder_anchors(observed, horizon, dt):
    """Linear fit and continuation in arbitrary demo units, with dummy beta."""
    times = (np.arange(observed.shape[1]) - (observed.shape[1] - 1)) * dt
    basis = np.stack([np.ones_like(times), times], axis=1)
    coef, *_ = np.linalg.lstsq(
        basis, observed.transpose(1, 0, 2).reshape(len(times), -1), rcond=None)
    coef = coef.reshape(2, len(observed), 3)
    fit = coef[0, :, None] + times[None, :, None] * coef[1, :, None]
    future = np.arange(1, horizon + 1) * dt
    anchor = coef[0, :, None] + future[None, :, None] * coef[1, :, None]
    beta = np.linspace(0.0, 1.0, len(observed), dtype=np.float32)
    return anchor.astype(np.float32), fit.astype(np.float32), beta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/demo')
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--pred-len', type=int, default=32)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.seq_len < 2 or args.pred_len < 2 or args.dt <= 0:
        parser.error('seq-len and pred-len must be at least 2; dt must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    steps = args.seq_len + args.pred_len
    for name, count in {'train': 32, 'val': 8, 'test': 8}.items():
        split = make_split(rng, count, steps, args.seq_len, args.dt)
        torch.save(split, output / f'{name}.pt')
        arrays = placeholder_anchors(split['X_observed'].numpy(), args.pred_len, args.dt)
        for key, array in zip(('anchor_drag', 'anchor_fit_window', 'anchor_beta'), arrays):
            np.save(output / f'{key}_{name}.npy', array)
    manifest = {
        'protocol_id': 'software_demo_only',
        'purpose': 'Non-operational synthetic data for interface and smoke testing; not used in the paper.',
        'default_observation_mode': 'tracked',
        'sample_interval_s': args.dt,
        'observation_steps': args.seq_len,
        'prediction_steps': args.pred_len,
        'anchors': 'Placeholder linear fit/constant-velocity forecast and arbitrary beta; not Drag-LS.',
        'seed': args.seed,
    }
    (output / 'protocol_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
