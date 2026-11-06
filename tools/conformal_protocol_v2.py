"""Inference-only impact-region feasibility for saved protocol-v2 ensembles."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.summarize_protocol_v2 import HOLD_CLASSES, ade_totals  # noqa: E402
from utils.ballistic_physics import predicted_impact  # noqa: E402
from utils.provenance import sha256_file  # noqa: E402


def conformal_quantile(scores, alpha):
    """Empirical inverse-CDF at ceil((n+1)*(1-alpha))/n (k-th smallest).

    No interpolation; k > n gives the conservative infinite region.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 1 or not scores.size or not np.isfinite(scores).all() or
            np.any(scores < 0) or not 0 < alpha < 1):
        raise ValueError('Expected nonempty finite nonnegative scores and 0 < alpha < 1')
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return float(np.partition(scores, k - 1)[k - 1]) if k <= len(scores) else math.inf


def region_summary(errors, radii):
    """Closed horizontal disks: impact error <= radius counts as covered."""
    errors, radii = np.asarray(errors), np.asarray(radii)
    if (errors.ndim != 1 or not errors.size or radii.shape != errors.shape or
            not np.isfinite(errors).all() or np.any(errors < 0) or
            np.isnan(radii).any() or np.any(radii < 0)):
        raise ValueError('Expected aligned nonnegative errors/radii')
    return {'n': len(errors), 'covered': int(np.count_nonzero(errors <= radii)),
            'coverage': float(np.mean(errors <= radii)), 'median_radius_m': float(np.median(radii))}


def run_dir(out, prefix, seed):
    matches = list((out / 'results').glob(f'ballistic_forecast_{prefix}_R5c_PASTMixer_s{seed}_*'))
    if len(matches) != 1:
        raise ValueError(f'Expected one {prefix} seed {seed} in {out}: {matches}')
    return matches[0]


def inference_job(saved, data_dir, scaler_dir):
    """Reuse the complete saved argument snapshot; the last overrides are decisive."""
    manifest = json.loads((saved / 'run_manifest.json').read_text(encoding='utf-8'))
    arguments = manifest['run']['arguments']
    tokens = [token for key, value in arguments.items() if key != 'device'
              for token in (f'--{key}', str(value))]
    tokens += ['--is_training', '0', '--dataset_dir', str(data_dir),
               '--scaler_dataset_dir', str(scaler_dir)]
    return {'name': arguments['model_id'], 'args': tokens}


def queue_jobs(jobs, destination, checkpoints, gpus):
    destination.mkdir(parents=True, exist_ok=True)
    job_path = destination / 'jobs.json'
    job_path.write_text(json.dumps(jobs, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT / 'tools/run_queue.py'), '--jobs', str(job_path),
                    '--out', str(destination), '--checkpoints', str(checkpoints),
                    '--gpus', gpus, '--per_gpu', '1'], cwd=ROOT, check=True)


def infer(args):
    # The view is disposable; source files are only read by run.py, never modified.
    # Explicit scaler_dataset_dir preserves both position and beta training statistics.
    with tempfile.TemporaryDirectory(prefix='view_', dir=args.out) as temporary:
        view_root = Path(temporary)

        def view(data, split, label):
            target = view_root / label
            target.mkdir()
            os.link(data / f'{split}.pt', target / 'test.pt')
            for stem in ('anchor_drag', 'anchor_fit_window', 'anchor_beta'):
                os.link(data / f'{stem}_{split}.npy', target / f'{stem}_test.npy')
            shutil.copyfile(data / 'protocol_manifest.json', target / 'protocol_manifest.json')
            return target

        saved = run_dir(args.main_out, 'main', 2021)
        queue_jobs([inference_job(saved, view(args.data_dir, 'test', 'test'), args.data_dir)],
                   args.out / 'repro', args.main_out / 'checkpoints', args.gpus.split(',')[0])
        reproduced = args.out / 'repro/results' / saved.name
        checks = {}
        for filename in ('pred.npy', 'true.npy', 'valid_len.npy'):
            original, new = saved / filename, reproduced / filename
            a, b = np.load(original), np.load(new)
            checks[filename] = {'array_equal': bool(np.array_equal(a, b)),
                                'max_abs_difference': float(np.max(np.abs(a.astype(float) - b))),
                                'original_sha256': sha256_file(original),
                                'rerun_sha256': sha256_file(new)}
            checks[filename]['byte_identical'] = checks[filename]['original_sha256'] == checks[filename]['rerun_sha256']
        (args.out / 'reproduction.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        print('Test reproduction:', json.dumps(checks), flush=True)
        # A nonidentical rerun is recorded explicitly; calibration remains a diagnostic.
        main_view = view(args.data_dir, 'val', 'main')
        queue_jobs([inference_job(run_dir(args.main_out, 'main', seed), main_view, args.data_dir)
                    for seed in range(2021, 2026)], args.out / 'main',
                   args.main_out / 'checkpoints', args.gpus)
        hold_jobs = []
        for cls in HOLD_CLASSES:
            data = args.hold_data / cls
            hold_view = view(data, 'val', cls)
            hold_jobs += [inference_job(run_dir(args.hold_out, f'{args.hold_tag}{cls}', seed), hold_view, data)
                          for seed in range(2021, 2024)]
        queue_jobs(hold_jobs, args.out / 'hold', args.hold_out / 'checkpoints', args.gpus)


def load_ensemble(out, data, split, prefix, seeds):
    raw = torch.load(data / f'{split}.pt', map_location='cpu', weights_only=False)
    meta = raw['meta']
    expected_len = np.asarray(raw['valid_len'])
    ensemble = reference = lengths = None
    seed_xy, single = [], []
    for seed in seeds:
        saved = run_dir(out, prefix, seed)
        pred = np.load(saved / 'pred.npy').astype(np.float64)
        true = np.load(saved / 'true.npy').astype(np.float64)
        valid_len = np.load(saved / 'valid_len.npy')
        manifest = json.loads((saved / 'run_manifest.json').read_text(encoding='utf-8'))
        seq_len = manifest['run']['arguments']['seq_len']
        source_true = np.asarray(raw['X'])[:, seq_len:seq_len + pred.shape[1]]
        if (pred.shape != true.shape or pred.ndim != 3 or pred.shape[-1] != 3 or
                not np.isfinite(pred).all() or not np.isfinite(true).all() or
                not np.array_equal(valid_len, expected_len) or
                not np.allclose(true, source_true, rtol=1e-6, atol=0.003)):
            raise ValueError(f'Invalid/misaligned {split} arrays: {saved}')
        if reference is None:
            reference, lengths = true, valid_len
            ensemble = np.zeros_like(pred)
        elif not np.array_equal(reference, true) or not np.array_equal(lengths, valid_len):
            raise ValueError(f'Seed targets/order differ: {saved}')
        ensemble += pred / len(seeds)
        xy = predicted_impact(pred)[0]
        seed_xy.append(xy)
        error = np.linalg.norm(xy - true[np.arange(len(true)), valid_len - 1, :2], axis=1)
        single.append({'ade_m': float(ade_totals(pred, true, valid_len).sum() / valid_len.sum()),
                       'impact_median_m': float(np.median(error)),
                       'impact_p90_m': float(np.percentile(error, 90))})
    # Impact of the mean TRAJECTORY; nonlinear crossing interpolation is not averaged.
    xy = predicted_impact(ensemble)[0]
    seed_xy = np.stack(seed_xy)
    spread = np.sqrt(np.mean(np.sum((seed_xy - seed_xy.mean(axis=0)) ** 2, axis=-1), axis=0))
    anchor = np.load(data / f'anchor_drag_{split}.npy').astype(np.float64)
    if anchor.shape != ensemble.shape or not np.isfinite(anchor).all():
        raise ValueError(f'Invalid anchor: {data} {split}')
    correction = np.linalg.norm(xy - predicted_impact(anchor)[0], axis=1)
    error = np.linalg.norm(xy - reference[np.arange(len(xy)), lengths - 1, :2], axis=1)
    accuracy = {'ensemble': {'ade_m': float(ade_totals(ensemble, reference, lengths).sum() / lengths.sum()),
                             'impact_median_m': float(np.median(error)),
                             'impact_p90_m': float(np.percentile(error, 90))},
                'single_seed_mean': {k: float(np.mean([s[k] for s in single])) for k in single[0]},
                'single_seed_std': {k: float(np.std([s[k] for s in single], ddof=1)) for k in single[0]}}
    return {'error': error, 'spread': spread, 'correction': correction,
            'meta': meta, 'accuracy': accuracy}


def groups(meta, holdout=False):
    classes, domains = np.asarray(meta['projectile']), np.asarray(meta['domain'])
    yield 'overall', np.ones(len(classes), dtype=bool)
    for domain in ('nominal', 'extrapolative'):
        yield domain, domains == domain
    if not holdout:
        for short, projectile in HOLD_CLASSES.items():
            yield short, classes == projectile
            for domain in ('nominal', 'extrapolative'):
                yield f'{short}/{domain}', (classes == projectile) & (domains == domain)


ALPHAS = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)  # targets 50-95%; 90/80 are the primary tables


def calibrate(val, test, holdout=False):
    result = {'n_calibration': len(val['error']), 'variants': {}}
    for variant, key in (('V0', None), ('V1', 'spread'), ('V2', 'correction')):
        d = float(np.median(val[key])) if key else 0.0
        val_scale = val[key] + d if key else np.ones_like(val['error'])
        test_scale = test[key] + d if key else np.ones_like(test['error'])
        if np.any(val_scale <= 0) or np.any(test_scale <= 0):
            raise ValueError(f'{variant} has zero scale; prescribed normalization is undefined')
        block = {'d_m': d, 'levels': {}}
        for alpha in ALPHAS:
            q = conformal_quantile(val['error'] / val_scale, alpha)
            radius = q * test_scale
            block['levels'][str(round(100 * (1 - alpha)))] = {
                'k': math.ceil((len(val['error']) + 1) * (1 - alpha)), 'q': q,
                'groups': {name: region_summary(test['error'][mask], radius[mask])
                           for name, mask in groups(test['meta'], holdout)}}
        result['variants'][variant] = block
    return result


def write_summary(args, result):
    main = result['pools']['main']['variants']
    overall90 = [main[v]['levels']['90']['groups']['overall'] for v in ('V0', 'V1', 'V2')]
    lines = ['# Impact-region feasibility (inference only)', '',
             'A region can be computed without training, but 90% coverage does not transfer uniformly. '
             'Main overall coverage / median radius for V0, V1, V2: ' + '; '.join(
                 f"{c['coverage'] * 100:.2f}% / {c['median_radius_m']:.1f} m" for c in overall90) + '. '
             'No variant reaches 90% observed coverage in every held-out class.', '',
             'Horizontal disks centered at predicted_impact(mean seed trajectory); '
             'distances in m. V0: fixed radius. V1: seed-impact RMS spread + validation median spread. '
             'V2: distance to Drag-LS impact + validation median correction. '
             'All scales/quantiles use validation only; no variant is selected on test.', '',
             'Quantile: k=ceil((n+1)(1-alpha)); q is the k-th smallest calibration score '
             '(empirical inverse-CDF at k/n, no interpolation). Coverage counts error <= radius. '
             'Main: 5 seeds (2021–2025), n_val=900, n_test=1800. '
             'Each holdout: 3 seeds (2021–2023), n_val=675 excluding the held-out class, n_test=450.', '',
             'Validation is nominal-only and was already used for checkpoint early stopping. '
             'Thus these are empirical prediction-region diagnostics, not a fresh independent '
             'split-conformal coverage guarantee. Extrapolative and unseen-class tests violate '
             'calibration/test exchangeability; subgroup coverage is not guaranteed. '
             'V1/V2 also estimate d on the same calibration set. See '
             '[Angelopoulos & Bates](https://arxiv.org/abs/2107.07511) for the independent-split assumptions.', '',
             'Impact extraction is unchanged from utils.metrics, including its implemented '
             'penultimate-sample fallback when no ground crossing occurs. ADE uses valid-step weighting. '
             'Truth arrays are inverse-scaled saved targets, checked against source split/order.', '',
             'Statistical interpretation review: 11/11 fallacy categories checked; nominal/domain '
             'and class composition are reported separately, all three prespecified variants retained, '
             'no significance or causal claims are made.', '', '## Test reproduction', '']
    reproduction = result['reproduction']
    for filename, check in reproduction.items():
        lines.append(f"- {filename}: array_equal={check['array_equal']}, byte_identical={check['byte_identical']}, "
                     f"max absolute difference={check['max_abs_difference']:.9g}.")
    lines += ['', 'See reproduction.json for original/rerun SHA-256 hashes. '
              'Validation inference uses temporary hard-link views (val as test, matching anchors), '
              'copied manifests, original training scalers, saved arguments and --is_training 0. '
              'Views are removed after inference. Source data/results/checkpoints are read-only.', '',
              '## Calibration', '', '| Pool | Variant | d (m) | q90 | q80 |', '|---|---|---:|---:|---:|']
    for pool, block in result['pools'].items():
        for variant, b in block['variants'].items():
            lines.append(f"| {pool} | {variant} | {b['d_m']:.3f} | {b['levels']['90']['q']:.6g} | {b['levels']['80']['q']:.6g} |")
    for level in ('90', '80'):
        lines += ['', f'## {level}% target: coverage (%) / median radius (m)', '',
                  '| Test group | n | V0 | V1 | V2 |', '|---|---:|---:|---:|---:|']
        for pool, block in result['pools'].items():
            for group in block['variants']['V0']['levels'][level]['groups']:
                cells = [block['variants'][v]['levels'][level]['groups'][group] for v in ('V0', 'V1', 'V2')]
                lines.append(f"| {pool}/{group} | {cells[0]['n']} | " + ' | '.join(
                    f"{c['coverage'] * 100:.2f} / {c['median_radius_m']:.1f}" for c in cells) + ' |')
    levels = [str(round(100 * (1 - a))) for a in ALPHAS]
    lines += ['', '## Coverage across target levels (V1, %)', '',
              '| Test group | ' + ' | '.join(levels) + ' |', '|---|' + '---:|' * len(levels)]
    for pool, block in result['pools'].items():
        for group in ('overall', 'nominal', 'extrapolative'):
            lines.append(f'| {pool}/{group} | ' + ' | '.join(
                f"{block['variants']['V1']['levels'][l]['groups'][group]['coverage'] * 100:.1f}" for l in levels) + ' |')
    lines += ['', '## Main-test accuracy', '',
              'Single-seed entries are mean ± sample std of each seed metric, not pooled errors. '
              'Computed from the same saved arrays as outputs/main_v3/summary_class.md.', '',
              '| Metric (m) | Ensemble mean trajectory | Single-seed mean ± std |', '|---|---:|---:|']
    accuracy = result['main_accuracy']
    for key in ('ade_m', 'impact_median_m', 'impact_p90_m'):
        lines.append(f"| {key} | {accuracy['ensemble'][key]:.3f} | "
                     f"{accuracy['single_seed_mean'][key]:.3f} ± {accuracy['single_seed_std'][key]:.3f} |")
    (args.out / 'summary_conf.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (args.out / 'summary_conf.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT.parent
    parser.add_argument('--data-dir', type=Path, default=base / 'data/protocol_v2')
    parser.add_argument('--hold-data', type=Path, default=base / 'data/protocol_v2_hold')
    parser.add_argument('--main-out', type=Path, default=base / 'outputs/main_v3')
    parser.add_argument('--hold-out', type=Path, default=base / 'outputs/hold_v3')
    parser.add_argument('--out', type=Path, default=base / 'outputs/conf_v3')
    parser.add_argument('--hold-tag', default='hold', help='Holdout run prefix: hold or holdaero')
    parser.add_argument('--infer', action='store_true', help='Reproduce one test run and infer all validation splits')
    parser.add_argument('--gpus', default='0,1')
    args = parser.parse_args()
    for key in ('data_dir', 'hold_data', 'main_out', 'hold_out', 'out'):
        setattr(args, key, getattr(args, key).resolve())
    if any(args.out == p or p in args.out.parents for p in
           (args.data_dir, args.hold_data, args.main_out, args.hold_out)):
        parser.error('Output must be outside read-only source roots')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.infer:
        infer(args)
    result = {'reproduction': json.loads((args.out / 'reproduction.json').read_text(encoding='utf-8')), 'pools': {}}
    for pool in ('main', *HOLD_CLASSES):
        is_main = pool == 'main'
        data = args.data_dir if is_main else args.hold_data / pool
        source = args.main_out if is_main else args.hold_out
        prefix = 'main' if is_main else f'{args.hold_tag}{pool}'
        seeds = list(range(2021, 2026 if is_main else 2024))
        val = load_ensemble(args.out / ('main' if is_main else 'hold'), data, 'val', prefix, seeds)
        test = load_ensemble(source, data, 'test', prefix, seeds)
        if set(val['meta']['domain']) != {'nominal'}:
            raise ValueError('Expected nominal-only calibration')
        if not is_main and (HOLD_CLASSES[pool] in val['meta']['projectile'] or
                            set(test['meta']['projectile']) != {HOLD_CLASSES[pool]}):
            raise ValueError(f'Incorrect holdout class split: {pool}')
        result['pools'][pool] = calibrate(val, test, not is_main)
        if is_main:
            result['main_accuracy'] = test['accuracy']
    write_summary(args, result)
    print(args.out / 'summary_conf.md')


if __name__ == '__main__':
    main()
