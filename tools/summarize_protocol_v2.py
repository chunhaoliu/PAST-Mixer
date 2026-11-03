"""Recompute protocol v2 seed summaries and paired ADE bootstrap from saved arrays."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.metrics import evaluate, future_mask  # noqa: E402

NAME = re.compile(r'ballistic_forecast_(?P<name>[A-Za-z0-9]+_(?P<recipe>[A-Za-z0-9]+)_(?P<model>[A-Za-z]+)_s(?P<seed>\d{4}))_')
KEYS = ('rmse', 'ade_m', 'fde_m', 'impact_err_median_m', 'impact_err_p90_m', 'tti_abs_err_mean_s')


def seed_stats(group, path):
    blocks = []
    for run in group:
        block = run['result']
        for key in path:
            block = block[key]
        blocks.append(block)
    return {key: (float(np.mean([b[key] for b in blocks])),
                  float(np.std([b[key] for b in blocks], ddof=1)) if len(blocks) > 1 else 0.0)
            for key in KEYS}


def ade_totals(pred, true, valid_len):
    """Per-trajectory distance sums; preserve the evaluator's valid-step weighting."""
    distance = np.linalg.norm(pred - true, axis=-1)
    return np.where(future_mask(valid_len, pred.shape[1]), distance, 0.0).sum(axis=1)


def paired_bootstrap(left, right, valid_len, n, seed=0):
    """Paired trajectory/seed resampling, then seed-mean ADE (left minus right).

    Inputs map seed IDs to trajectory distance sums. A deterministic baseline
    has the single key None and is broadcast across the learned model's seeds.
    Learned methods must have identical seed sets. Percentile CI is 95%.
    """
    if n <= 0:
        raise ValueError('Bootstrap count must be positive')
    if not left or not right:
        raise ValueError('Bootstrap needs both methods')
    if set(left) != {None} and set(right) != {None} and set(left) != set(right):
        raise ValueError('Paired learned methods must have matching seed sets')
    seeds = sorted(right if set(left) == {None} else left)
    valid_len = np.asarray(valid_len)
    if valid_len.ndim != 1 or not len(valid_len) or np.any(valid_len <= 0):
        raise ValueError('Expected positive trajectory lengths')
    delta = np.stack([np.asarray(left[None if None in left else s]) -
                      np.asarray(right[None if None in right else s]) for s in seeds])
    if delta.shape != (len(seeds), len(valid_len)):
        raise ValueError('Bootstrap trajectories do not align')
    mean = float((delta.sum(axis=1) / valid_len.sum()).mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(n)
    for i in range(n):
        indices = rng.integers(len(valid_len), size=len(valid_len))
        seed_indices = rng.integers(len(seeds), size=len(seeds))
        samples[i] = (delta[seed_indices][:, indices].sum(axis=1) / valid_len[indices].sum()).mean()
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {'mean': mean, 'ci_low': float(lo), 'ci_high': float(hi)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--data-dir', default='data/protocol_v2')
    parser.add_argument('--physics', default='outputs/protocol_v2/physics_baselines')
    parser.add_argument('--prefix', default='', help='Comma-separated run-name prefixes, e.g. main_ or main_R5c_PASTMixer,best_')
    parser.add_argument('--bootstrap', type=int, default=0)
    parser.add_argument('--pairs', default='', help='Left,right pairs separated by semicolons; learned names are variant:Model')
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('--bootstrap must be nonnegative')
    out = Path(args.out)
    prefixes = tuple(args.prefix.split(','))
    meta = torch.load(Path(args.data_dir) / 'test.pt', weights_only=False, map_location='cpu')['meta']

    runs, physics, scores = [], {}, {}
    reference_true = reference_len = None

    def load_arrays(pred_path, true_path, length_path):
        nonlocal reference_true, reference_len
        pred = np.load(pred_path).astype(np.float64)
        true = np.load(true_path).astype(np.float64)
        lengths = np.load(length_path)
        if pred.shape != true.shape or pred.ndim != 3 or pred.shape[-1] != 3:
            raise ValueError(f'Expected matching (N,H,3) arrays: {pred_path}')
        if lengths.shape != (len(true),) or np.any(lengths < 1) or np.any(lengths > true.shape[1]):
            raise ValueError(f'Invalid trajectory lengths: {length_path}')
        if reference_true is None:
            reference_true, reference_len = true, lengths
        elif (not np.array_equal(lengths, reference_len) or true.shape != reference_true.shape or
              not np.allclose(true, reference_true, rtol=1e-6, atol=1e-3)):
            raise ValueError(f'Test trajectories/order differ: {true_path}')
        return evaluate(pred, true, lengths, meta), ade_totals(pred, true, lengths)

    for run_dir in sorted((out / 'results').glob('*')):
        match = NAME.search(run_dir.name)
        if not match or not match.group('name').startswith(prefixes) or not (run_dir / 'pred.npy').exists():
            continue
        result, totals = load_arrays(run_dir / 'pred.npy', run_dir / 'true.npy', run_dir / 'valid_len.npy')
        recipe, model, seed = match.group('recipe'), match.group('model'), int(match.group('seed'))
        method = f'{recipe}:{model}'
        if seed in scores.setdefault(method, {}):
            raise ValueError(f'Duplicate variant/model/seed: {method} s{seed}; restrict --prefix')
        scores[method][seed] = totals
        runs.append({'name': match.group('name'), 'recipe': recipe, 'model': model, 'seed': seed, 'result': result})

    physics_dir = Path(args.physics)
    for pred_path in sorted(physics_dir.glob('pred_*.npy')):
        name = pred_path.stem.removeprefix('pred_')
        result, totals = load_arrays(pred_path, physics_dir / 'true_test.npy', physics_dir / 'valid_len_test.npy')
        physics[name] = [{'result': result}]
        scores[name] = {None: totals}

    groups = {}
    for run in runs:
        groups.setdefault(f"{run['recipe']}:{run['model']}", []).append(run)
    methods = list(physics.items()) + sorted(groups.items())
    header = ('| Method | seeds | RMSE (m) | ADE (m) | FDE@impact (m) | Impact med (m) | Impact p90 (m) | TTI err (s) |\n'
              '|---|---|---|---|---|---|---|---|')

    def cell(value, precision=1):
        return f'{value[0]:.{precision}f} ± {value[1]:.{precision}f}'

    sections = [('Overall', ('overall',))]
    for key in ('domain', 'projectile', 'wind', 't_go_s'):
        values = sorted({value for _, group in methods for value in group[0]['result'].get(key, {})})
        if key == 't_go_s':
            values.sort(key=lambda value: int(value.split('-')[0]))
        for value in values:
            title = f'{key}: {value}'
            if key == 't_go_s':
                interval = value.replace('-', ',') + (']' if value == '15-20' else ')')
                title = f'Lead time: [{interval} s'
            sections.append((title, (key, value)))
    lines = ['Seed mean ± sample std (single deterministic run: std 0).', '']
    for title, path in sections:
        lines += [f'## {title}', '', header]
        for name, group in methods:
            stats = seed_stats(group, path)
            cells = [cell(stats[k], 3 if k == 'tti_abs_err_mean_s' else 1) for k in KEYS]
            lines.append(f'| {name} | {len(group)} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    if args.bootstrap:
        lines += ['## Paired bootstrap ADE difference (m)', '',
                  'Left minus right; shared trajectory indices and paired seed resampling, seed 0; 95% percentile CI.', '',
                  '| Left | Right | Mean difference | 95% CI |', '|---|---|---|---|']
        for pair in filter(None, args.pairs.split(';')):
            names = [name.strip() for name in pair.split(',')]
            if len(names) != 2 or any(name not in scores for name in names):
                parser.error(f'Unknown or malformed pair: {pair}')
            left, right = names
            stat = paired_bootstrap(scores[left], scores[right], reference_len, args.bootstrap)
            lines.append(f"| {left} | {right} | {stat['mean']:.3f} | [{stat['ci_low']:.3f}, {stat['ci_high']:.3f}] |")
        lines.append('')

    lines += ['## Per-seed overall RMSE', '']
    for name, group in sorted(groups.items()):
        per_seed = ', '.join(f"{r['seed']}:{r['result']['overall']['rmse']:.1f}"
                             for r in sorted(group, key=lambda r: r['seed']))
        lines.append(f'- {name}: {per_seed}')
    report = '\n'.join(lines)
    # Keep separate stage summaries when multiple stages share the output root.
    stem = '_'.join(['summary'] + [prefix.rstrip('_') for prefix in prefixes if prefix])
    (out / f'{stem}.md').write_text(report + '\n', encoding='utf-8')
    (out / f'{stem}_runs.json').write_text(json.dumps(runs, indent=1), encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
