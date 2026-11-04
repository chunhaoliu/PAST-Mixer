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

from utils.metrics import evaluate, future_mask, impact_metrics  # noqa: E402
from utils.ballistic_physics import DT, predicted_impact  # noqa: E402

NAME = re.compile(r'ballistic_forecast_(?P<name>[A-Za-z0-9]+_(?P<recipe>[A-Za-z0-9]+)_(?P<model>[A-Za-z]+)_s(?P<seed>\d{4}))_')
KEYS = ('rmse', 'ade_m', 'fde_m', 'impact_err_median_m', 'impact_err_p90_m', 'tti_abs_err_mean_s')
HOLD_CLASSES = {'a155': '155mm_artillery', 'a122': '122mm_artillery',
                'r130': '130mm_rocket', 'r227': '227mm_rocket'}
HOLD_METHODS = ('B4_DragLS', 'B5_DragUKF', 'R4:PASTMixer', 'R5c:PASTMixer',
                'R5c:TimeMixer', 'R5c:TSMixer', 'R5c:Transformer')


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


def paired_impact_bootstrap(left, right, n, seed=0):
    """Seed-mean differences of impact-error mean/median/p90, with paired 95% CIs.

    Inputs map matched seed IDs (or None for a deterministic method) to the
    per-trajectory ``impact_err_m`` returned by utils.metrics. Each resample
    draws shared trajectories, then matched seeds, exactly as paired_bootstrap.
    Quantiles are computed for each method separately, before subtraction.
    """
    if n <= 0:
        raise ValueError('Bootstrap count must be positive')
    if not left or not right:
        raise ValueError('Bootstrap needs both methods')
    if set(left) != {None} and set(right) != {None} and set(left) != set(right):
        raise ValueError('Paired learned methods must have matching seed sets')
    seeds = sorted(right if set(left) == {None} else left)
    a = np.stack([np.asarray(left[None if None in left else s]) for s in seeds])
    b = np.stack([np.asarray(right[None if None in right else s]) for s in seeds])
    if a.ndim != 2 or not a.shape[1] or a.shape != b.shape:
        raise ValueError('Bootstrap trajectories do not align')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite impact errors')

    def statistics(values):
        return np.stack([values.mean(axis=1), np.median(values, axis=1),
                         np.percentile(values, 90, axis=1)])

    difference = (statistics(a) - statistics(b)).mean(axis=1)
    rng = np.random.default_rng(seed)
    samples = np.empty((n, 3))
    for i in range(n):
        indices = rng.integers(a.shape[1], size=a.shape[1])
        seed_indices = rng.integers(len(seeds), size=len(seeds))
        samples[i] = (statistics(a[seed_indices][:, indices]) -
                      statistics(b[seed_indices][:, indices])).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return {key: {'difference': float(difference[j]), 'ci_low': float(lo[j]), 'ci_high': float(hi[j])}
            for j, key in enumerate(('mean', 'median', 'p90'))}


def signed_impact_components(pred, true, lengths):
    """Signed versions of make_paper_figures.Results.read's impact projections.

    Use the nearest nonzero valid backward horizontal difference at true impact.
    The cross product order matches that tool before its absolute value.
    Ground crossing and true TTI are exactly those used by utils.metrics.
    """
    rows = np.arange(len(true))
    true_xy = true[rows, lengths - 1, :2]
    velocity = np.zeros_like(true_xy)
    for i, length in enumerate(lengths):
        differences = np.diff(true[i, :length, :2], axis=0)
        nonzero = np.flatnonzero(np.linalg.norm(differences, axis=1) > 1e-12)
        if not len(nonzero):
            raise ValueError('Undefined true horizontal impact direction')
        velocity[i] = differences[nonzero[-1]]
    direction = velocity / np.linalg.norm(velocity, axis=1)[:, None]
    pred_xy, pred_tti = predicted_impact(pred)
    delta = pred_xy - true_xy
    return {'tti': pred_tti - lengths * DT,
            'along': (delta * direction).sum(axis=1),
            'cross': delta[:, 0] * direction[:, 1] - delta[:, 1] * direction[:, 0]}


def load_pool_methods(out, physics, data_dir, prefix, methods, seeds):
    """Read saved arrays once; retain only per-trajectory statistics on CPU."""
    meta = torch.load(data_dir / 'test.pt', weights_only=False, map_location='cpu')['meta']
    reference_true = reference_len = None
    records = {name: {} for name in methods}

    def read(pred_path, true_path, length_path, mechanism):
        nonlocal reference_true, reference_len
        pred = np.load(pred_path).astype(np.float64)
        true = np.load(true_path).astype(np.float64)
        lengths = np.load(length_path)
        if pred.shape != true.shape or pred.ndim != 3 or pred.shape[-1] != 3:
            raise ValueError(f'Expected matching (N,H,3) arrays: {pred_path}')
        if (lengths.shape != (len(true),) or not np.issubdtype(lengths.dtype, np.integer) or
                np.any(lengths < 1) or np.any(lengths > true.shape[1])):
            raise ValueError(f'Invalid trajectory lengths: {length_path}')
        if not np.isfinite(pred).all() or not np.isfinite(true).all():
            raise ValueError(f'Nonfinite trajectories: {pred_path}')
        if len(meta['projectile']) != len(true):
            raise ValueError(f'Metadata count differs: {data_dir}')
        if reference_true is None:
            reference_true, reference_len = true, lengths
        elif (not np.array_equal(lengths, reference_len) or true.shape != reference_true.shape or
              not np.allclose(true, reference_true, rtol=1e-6, atol=1e-3)):
            raise ValueError(f'Test trajectories/order differ: {true_path}')
        impact = impact_metrics(pred, true, lengths)
        record = {'totals': ade_totals(pred, true, lengths), 'impact': impact['impact_err_m'],
                  'tti_abs': impact['tti_abs_err_s'],
                  'overall': evaluate(pred, true, lengths)['overall']}
        if mechanism:
            record.update(signed_impact_components(pred, true, lengths))
        return record

    for name in methods:
        if ':' not in name:
            records[name][None] = read(physics / f'pred_{name}.npy', physics / 'true_test.npy',
                                       physics / 'valid_len_test.npy', name == 'B4_DragLS')
    for run_dir in sorted((out / 'results').glob('*')):
        match = NAME.search(run_dir.name)
        if not match or not match.group('name').startswith(prefix):
            continue
        name = f"{match.group('recipe')}:{match.group('model')}"
        if name not in records:
            continue
        seed = int(match.group('seed'))
        if seed in records[name]:
            raise ValueError(f'Duplicate variant/model/seed: {name} s{seed}')
        records[name][seed] = read(run_dir / 'pred.npy', run_dir / 'true.npy', run_dir / 'valid_len.npy',
                                   name == 'R5c:PASTMixer')
    for name, runs in records.items():
        if set(runs) != ({None} if ':' not in name else set(seeds)):
            raise ValueError(f'Unexpected/missing seeds for {prefix}{name}: {list(runs)}')
    return records, reference_len, meta


def pooled_statistics(class_ade, impact):
    """Per-seed class mean, maximum class ADE, and pooled impact p90.

    class_ade has shape (seeds, classes). Take each method's max separately
    before a paired subtraction; never take the max of class differences.
    """
    return np.stack([class_ade.mean(axis=1), class_ade.max(axis=1),
                     np.percentile(impact, 90, axis=1)], axis=1)


def paired_pool_bootstrap(left, right, lengths, n=2000, seed=0):
    """Stratified paired trajectory draws and one shared matched-seed draw.

    Methods map class -> seed (or None for physics) -> {totals, impact}.
    Recompute each method's worst class and pooled p90 inside every resample.
    """
    if n <= 0:
        raise ValueError('Bootstrap count must be positive')
    if not lengths or set(left) != set(lengths) or set(right) != set(lengths):
        raise ValueError('Bootstrap classes do not align')
    seed_sets = [set(runs) for method in (left, right) for runs in method.values() if set(runs) != {None}]
    if not seed_sets or any(s != seed_sets[0] for s in seed_sets):
        raise ValueError('Paired learned methods must have matching seed sets')
    seeds = sorted(seed_sets[0])
    arrays = []
    for method in (left, right):
        blocks = []
        for cls, valid_len in lengths.items():
            valid_len = np.asarray(valid_len)
            if valid_len.ndim != 1 or not len(valid_len) or np.any(valid_len <= 0):
                raise ValueError('Expected positive trajectory lengths')
            runs = method[cls]
            block = {key: np.stack([runs[None if None in runs else s][key] for s in seeds])
                     for key in ('totals', 'impact')}
            if any(a.shape != (len(seeds), len(valid_len)) or not np.isfinite(a).all()
                   for a in block.values()):
                raise ValueError('Bootstrap trajectories do not align or are nonfinite')
            blocks.append(block)
        arrays.append(blocks)

    def difference(indices, seed_indices):
        statistics = []
        for blocks in arrays:
            ade = np.stack([b['totals'][seed_indices][:, idx].sum(axis=1) / valid_len[idx].sum()
                            for b, valid_len, idx in zip(blocks, lengths.values(), indices)], axis=1)
            impact = np.concatenate([b['impact'][seed_indices][:, idx] for b, idx in zip(blocks, indices)], axis=1)
            statistics.append(pooled_statistics(ade, impact))
        return (statistics[0] - statistics[1]).mean(axis=0)

    point = difference([np.arange(len(v)) for v in lengths.values()], np.arange(len(seeds)))
    rng = np.random.default_rng(seed)
    samples = np.empty((n, 3))
    for i in range(n):
        indices = [rng.integers(len(v), size=len(v)) for v in lengths.values()]
        samples[i] = difference(indices, rng.integers(len(seeds), size=len(seeds)))
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return {key: {'difference': float(point[j]), 'ci_low': float(lo[j]), 'ci_high': float(hi[j])}
            for j, key in enumerate(('class_mean_ade', 'worst_class_ade', 'impact_p90'))}


def summarize_pool(args):
    root = Path(args.pool_holdout).resolve()
    seeds = (2021, 2022, 2023)
    classes, lengths = {}, {}
    for cls, projectile in HOLD_CLASSES.items():
        records, valid_len, meta = load_pool_methods(
            root, root / cls / 'physics', Path(args.data_dir).parent / 'protocol_v2_hold' / cls,
            f'hold{cls}_', HOLD_METHODS, seeds)
        if (len(valid_len) != 450 or set(meta['projectile']) != {projectile} or
                set(meta['domain']) != {'nominal', 'extrapolative'}):
            raise ValueError(f'Expected 450 {projectile} trajectories in both domains')
        classes[cls], lengths[cls] = records, valid_len

    def run(cls, name, seed):
        runs = classes[cls][name]
        return runs[None if None in runs else seed]

    def cell(values, precision=1):
        return (f'{np.mean(values):.{precision}f} ± '
                f'{np.std(values, ddof=1) if len(values) > 1 else 0.:.{precision}f}')

    lines = ['# Pooled leave-one-class-out summary', '',
             'Four unseen classes, 450 trajectories each (both domains), 1800 total. '
             'Seed mean ± sample std; pooled seed s combines seed s of every class (2021–2023). '
             'Deterministic physics is repeated across those seeds (std 0).', '',
             'ADE within each class uses valid-step weighting from utils.metrics.evaluate; '
             'class-mean ADE gives each class weight 1/4. Worst-class ADE is the per-seed maximum '
             'of the four class ADEs, then summarized across seeds. Impact median/p90 and mean absolute '
             'TTI error are computed over all 1800 trajectories within each pooled seed, then summarized. '
             'Distances are in m; time is in s.', '',
             '| Method | a155 ADE | a122 ADE | r130 ADE | r227 ADE | Class-mean ADE | Worst-class ADE | Impact median | Impact p90 | TTI abs err (s) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for name in HOLD_METHODS:
        ade = np.array([[run(cls, name, s)['overall']['ade_m'] for cls in classes] for s in seeds])
        impacts = np.array([np.concatenate([run(cls, name, s)['impact'] for cls in classes]) for s in seeds])
        tti = [np.concatenate([run(cls, name, s)['tti_abs'] for cls in classes]).mean() for s in seeds]
        values = [*ade.T, ade.mean(axis=1), ade.max(axis=1), np.median(impacts, axis=1),
                  np.percentile(impacts, 90, axis=1)]
        lines.append('| ' + name + ' | ' + ' | '.join([cell(v) for v in values] + [cell(tti, 3)]) + ' |')
    n = args.bootstrap or 2000
    lines += ['', '## Paired bootstrap differences (m)', '',
              f'R5c:PASTMixer minus comparator; {n} resamples, RNG seed 0, percentile 95% CI. '
              'Draw shared trajectories independently within each class, then one shared matched-seed '
              'sample across all classes/methods. Recompute each method’s class maximum and pooled '
              'impact p90 before subtraction in every resample. Negative favors PAST-Mixer.', '',
              '| Comparator | Statistic | Difference | 95% CI |', '|---|---|---|---|']
    left = {cls: records['R5c:PASTMixer'] for cls, records in classes.items()}
    for name in HOLD_METHODS:
        if name == 'R5c:PASTMixer':
            continue
        right = {cls: records[name] for cls, records in classes.items()}
        for key, stat in paired_pool_bootstrap(left, right, lengths, n).items():
            lines.append(f"| {name} | {key} | {stat['difference']:.3f} | [{stat['ci_low']:.3f}, {stat['ci_high']:.3f}] |")

    main, _, main_meta = load_pool_methods(root.parent / 'main_v3', Path(args.physics), Path(args.data_dir),
                                          'main_', ('B4_DragLS', 'R5c:PASTMixer'), range(2021, 2026))
    lines += ['', '## Signed impact error mechanism', '',
              'All entries are trajectory means within a class, then seed mean ± sample std '
              '(holdout: 3 seeds; main test trained on all four classes: 5 seeds). Physics std is 0. '
              'Main-test classes are selected by meta[\'projectile\']. Distances are in m.', '',
              'Impact point/TTI use utils.metrics.impact_metrics and its imported predicted_impact '
              'unchanged, including interpolation and no-crossing fallback '
              '(penultimate predicted sample for horizon > 1, as implemented); truth is last valid xy and '
              'valid_len × DT. Signed TTI = predicted minus true (positive: late). '
              'utils.metrics contains no along/cross definition; projections reuse '
              'tools/make_paper_figures.py Results.read before absolute values: u is the normalized '
              'nearest nonzero backward horizontal difference in valid truth positions. '
              'For e = predicted impact xy − true impact xy, along = e·u (positive: overshoot); '
              'cross = e_x u_y − e_y u_x (positive: right of u in xy). '
              'Correction along = (PAST impact − Drag-LS impact)·u, each at its own predicted '
              'impact time (positive: shifts downrange); this is an endpoint displacement, not a '
              'same-time residual or causal attribution.', '',
              '| Test | Class | n | Method | Signed TTI (s) | Signed along (m) | Signed cross (m) | Correction along (m) |',
              '|---|---|---|---|---|---|---|---|']
    for test in ('holdout', 'main'):
        for cls, projectile in HOLD_CLASSES.items():
            records = classes[cls] if test == 'holdout' else main
            index = np.arange(len(lengths[cls])) if test == 'holdout' else np.flatnonzero(np.asarray(main_meta['projectile']) == projectile)
            if not len(index):
                raise ValueError(f'Missing main-test class: {projectile}')
            baseline = records['B4_DragLS'][None]
            for name in ('R5c:PASTMixer', 'B4_DragLS'):
                runs = records[name]
                cells = [cell([r[key][index].mean() for r in runs.values()], 3 if key == 'tti' else 1)
                         for key in ('tti', 'along', 'cross')]
                correction = cell([(r['along'][index] - baseline['along'][index]).mean() for r in runs.values()]) if name == 'R5c:PASTMixer' else '—'
                lines.append(f'| {test} | {cls} | {len(index)} | {name} | ' + ' | '.join(cells + [correction]) + ' |')
    report = '\n'.join(lines) + '\n'
    (Path(args.report) if args.report else root / 'summary_pool.md').write_text(report, encoding='utf-8')
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--data-dir', default='data/protocol_v2')
    parser.add_argument('--physics', default='outputs/protocol_v2/physics_baselines')
    parser.add_argument('--prefix', default='', help='Comma-separated run-name prefixes, e.g. main_ or main_R5c_PASTMixer,best_')
    parser.add_argument('--bootstrap', type=int, default=0)
    parser.add_argument('--pairs', default='', help='Left,right pairs separated by semicolons; learned names are variant:Model')
    parser.add_argument('--group-by', choices=('projectile',), help='Class x nominal/extrapolative/all tables and CIs')
    parser.add_argument('--impact-bootstrap', action='store_true', help='Also bootstrap impact mean/median/p90 for --pairs')
    parser.add_argument('--runner-up', help='Compare this learned method with the lowest-ADE other learned method in each group')
    parser.add_argument('--physics-methods', default='', help='Comma-separated physics methods; default: all')
    parser.add_argument('--report', help='Write only Markdown to this path instead of the default Markdown/JSON outputs')
    parser.add_argument('--pool-holdout', help='Pool four holdouts under this root into summary_pool.md; --data-dir/--physics locate main-test references; --bootstrap defaults to 2000')
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('--bootstrap must be nonnegative')
    if (args.impact_bootstrap or args.runner_up) and not args.bootstrap:
        parser.error('--impact-bootstrap and --runner-up require --bootstrap')
    if args.pool_holdout:
        summarize_pool(args)
        return
    out = Path(args.out)
    prefixes = tuple(args.prefix.split(','))
    meta = torch.load(Path(args.data_dir) / 'test.pt', weights_only=False, map_location='cpu')['meta']
    class_sections = []
    if args.group_by:
        projectile, domain = np.asarray(meta['projectile']), np.asarray(meta['domain'])
        joint = np.array([f'{p} / {d}' for p, d in zip(projectile, domain)])
        meta = dict(meta, projectile_domain=joint)
        for value in sorted(set(projectile.tolist())):
            for subset in ('nominal', 'extrapolative', 'all'):
                index = np.flatnonzero((projectile == value) & ((domain == subset) if subset != 'all' else True))
                if len(index):
                    path = ('projectile', str(value)) if subset == 'all' else ('projectile_domain', f'{value} / {subset}')
                    class_sections.append((f'projectile: {value} / {subset} (n={len(index)})', path, index))

    runs, physics, scores, impacts = [], {}, {}, {}
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
        kwargs = {'groups': ('projectile', 'projectile_domain')} if args.group_by else {}
        result = evaluate(pred, true, lengths, meta, **kwargs)
        impact = impact_metrics(pred, true, lengths)['impact_err_m'] if args.impact_bootstrap else None
        return result, ade_totals(pred, true, lengths), impact

    for run_dir in sorted((out / 'results').glob('*')):
        match = NAME.search(run_dir.name)
        if not match or not match.group('name').startswith(prefixes) or not (run_dir / 'pred.npy').exists():
            continue
        result, totals, impact = load_arrays(run_dir / 'pred.npy', run_dir / 'true.npy', run_dir / 'valid_len.npy')
        recipe, model, seed = match.group('recipe'), match.group('model'), int(match.group('seed'))
        method = f'{recipe}:{model}'
        if seed in scores.setdefault(method, {}):
            raise ValueError(f'Duplicate variant/model/seed: {method} s{seed}; restrict --prefix')
        scores[method][seed] = totals
        if args.impact_bootstrap:
            impacts.setdefault(method, {})[seed] = impact
        runs.append({'name': match.group('name'), 'recipe': recipe, 'model': model, 'seed': seed, 'result': result})

    physics_dir = Path(args.physics)
    for pred_path in sorted(physics_dir.glob('pred_*.npy')):
        name = pred_path.stem.removeprefix('pred_')
        if args.physics_methods and name not in args.physics_methods.split(','):
            continue
        result, totals, impact = load_arrays(pred_path, physics_dir / 'true_test.npy', physics_dir / 'valid_len_test.npy')
        physics[name] = [{'result': result}]
        scores[name] = {None: totals}
        if args.impact_bootstrap:
            impacts[name] = {None: impact}

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
    if args.group_by:
        sections = [(title, path) for title, path, _ in class_sections]
    lines = ['Seed mean ± sample std (single deterministic run: std 0).', '']
    for title, path in sections:
        lines += [f'## {title}', '', header]
        for name, group in methods:
            stats = seed_stats(group, path)
            cells = [cell(stats[k], 3 if k == 'tti_abs_err_mean_s' else 1) for k in KEYS]
            lines.append(f'| {name} | {len(group)} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    if args.bootstrap:
        pairs = []
        for pair in filter(None, args.pairs.split(';')):
            names = [name.strip() for name in pair.split(',')]
            if len(names) != 2 or any(name not in scores for name in names):
                parser.error(f'Unknown or malformed pair: {pair}')
            pairs.append(names)
        if args.runner_up and (args.runner_up not in groups or len(groups) < 2):
            parser.error('--runner-up needs the named learned method and at least one other learned method')
        lines += ['## Paired bootstrap ADE difference (m)', '',
                  'Left minus right; shared trajectory indices and paired seed resampling, seed 0; 95% percentile CI.', '']
        if args.group_by or args.impact_bootstrap or args.runner_up:
            lines += [f'Bootstrap resamples: {args.bootstrap}. Runner-up (when requested) is the lowest seed-mean ADE other learned method within each reported group.', '']
        bootstrap_sections = class_sections if args.group_by else [('Overall', ('overall',), np.arange(len(reference_len)))]
        for title, path, index in bootstrap_sections:
            if args.group_by:
                lines += [f'### {title}', '']
            selected = list(pairs)
            if args.runner_up:
                other = min((name for name in groups if name != args.runner_up),
                            key=lambda name: seed_stats(groups[name], path)['ade_m'][0])
                selected.append((args.runner_up, other))
            lines += ['| Left | Right | Mean difference | 95% CI |', '|---|---|---|---|']
            for left, right in selected:
                a, b = [{s: values[index] for s, values in scores[name].items()} for name in (left, right)]
                stat = paired_bootstrap(a, b, reference_len[index], args.bootstrap)
                lines.append(f"| {left} | {right} | {stat['mean']:.3f} | [{stat['ci_low']:.3f}, {stat['ci_high']:.3f}] |")
            lines.append('')
            if args.impact_bootstrap:
                lines += ['### Paired impact-point error difference (m)', '',
                          'Seed mean of per-seed statistics; quantiles taken separately for each method, then subtracted. Uses utils.metrics.impact_metrics (predicted ground crossing versus true last valid xy).', '',
                          '| Left | Right | Statistic | Difference | 95% CI |', '|---|---|---|---|---|']
                for left, right in selected:
                    a, b = [{s: values[index] for s, values in impacts[name].items()} for name in (left, right)]
                    for key, stat in paired_impact_bootstrap(a, b, args.bootstrap).items():
                        lines.append(f"| {left} | {right} | {key} | {stat['difference']:.3f} | [{stat['ci_low']:.3f}, {stat['ci_high']:.3f}] |")
                lines.append('')

    lines += ['## Per-seed overall RMSE', '']
    for name, group in sorted(groups.items()):
        per_seed = ', '.join(f"{r['seed']}:{r['result']['overall']['rmse']:.1f}"
                             for r in sorted(group, key=lambda r: r['seed']))
        lines.append(f'- {name}: {per_seed}')
    report = '\n'.join(lines)
    # Keep separate stage summaries when multiple stages share the output root.
    stem = '_'.join(['summary'] + [prefix.rstrip('_') for prefix in prefixes if prefix])
    if args.report:
        Path(args.report).write_text(report + '\n', encoding='utf-8')
    else:
        (out / f'{stem}.md').write_text(report + '\n', encoding='utf-8')
        (out / f'{stem}_runs.json').write_text(json.dumps(runs, indent=1), encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
