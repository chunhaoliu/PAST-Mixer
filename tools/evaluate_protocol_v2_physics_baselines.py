"""Model-based baselines on protocol v2 and the drag anchors used by residual learning.

All baselines use only what the networks see, except the UKF, which processes
the raw polar measurements of the same warm-up + window span the tracker uses.

  B1 CV        linear LS fit of the tracked window
  B2 CA        quadratic LS fit of the tracked window
  B3 CV+g      linear LS fit with known gravity
  B4 Drag-LS   MAP fit of [p, v, beta] (generic drag model, prior on beta from
               the training split), RK4 extrapolation
  B5 Drag-UKF  UKF on [p, v, beta] with polar measurements, RK4 extrapolation
  B6 Drag-LS-Std as B4 with a standard drag function of Mach (G1 or G7) and an
               ISA atmosphere; own train prior; table and prior strength selected on val

The beta-prior strength (B4) and the UKF acceleration noise (B5) are selected
on the validation split. B4 predictions for every split are saved as
``anchor_drag_<split>.npy`` next to the dataset (meters, (N, H, 3)), along
with ``anchor_fit_window_<split>.npy`` (N, 100, 3) and ``anchor_beta_<split>.npy``
(N,). ``--anchors-only`` fits available splits; ``--report`` optionally reuses a stored prior and B4 setting.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.radar_measurement import ANGLE_STD_DEG, polar_to_cartesian, sensor_for_domain  # noqa: E402
from utils.ballistic_physics import (  # noqa: E402
    DT, drag_predict, drag_scale, drag_window_trajectory, fit_drag_window, integrate, polynomial_predict, ukf_drag_track,
)
from utils.metrics import evaluate  # noqa: E402

INIT_STEPS = 50  # 0.5 s of the 3 s warm-up initializes the UKF
G = 9.81


def load(data_dir, split):
    payload = torch.load(Path(data_dir) / f'{split}.pt', weights_only=False, map_location='cpu')
    return {
        'X': payload['X'].numpy().astype(np.float64),
        'obs': payload['X_observed'].numpy().astype(np.float64),
        'valid_len': payload['valid_len'].numpy(),
        'polar': payload['polar'].numpy().astype(np.float64),
        'detected': payload['detected'].numpy(),
        'meta': payload['meta'],
        'protocol': payload.get('protocol', {}),
    }


def beta_prior_from_train(train, n=1500, seed=0, model='const'):
    """Per-trajectory best constant beta on clean training data (window end + 5 s)."""
    rng = np.random.default_rng(seed)
    index = rng.choice(len(train['X']), size=min(n, len(train['X'])), replace=False)
    clean = train['X'][index]
    window = train['obs'].shape[1]
    state = np.concatenate([clean[:, window - 1], (clean[:, window] - clean[:, window - 2]) / (2 * DT)], axis=1)
    grid = np.geomspace(1e-5, 3e-3, 60)
    horizon = 500
    target = clean[:, window:window + horizon]
    err = np.empty((len(index), len(grid)))
    for j, beta in enumerate(grid):
        pred = integrate(state, np.full(len(index), beta), horizon, model=model)
        err[:, j] = np.linalg.norm(pred - target, axis=-1).mean(-1)
    best = grid[err.argmin(axis=1)] * drag_scale(state, model)
    return {'mean': float(best.mean()), 'std': float(best.std()), 'median': float(np.median(best)),
            'p05': float(np.percentile(best, 5)), 'p95': float(np.percentile(best, 95)), 'n': int(len(index))}


def run_b4(split, prior, noise, horizon, model='const'):
    params = fit_drag_window(split['obs'], (prior['mean'], prior['std'], noise), model=model)
    return drag_predict(params, horizon, model), params[:, 6]


def save_anchors(data_dir, name, split, prior, noise, horizon, model='const'):
    """Save forecast, fitted observation window and beta from the same B4 (or B6) fit.

    For a drag table the saved beta is the window-end drag level beta * f(M_end)."""
    params = fit_drag_window(split['obs'], (prior['mean'], prior['std'], noise), model=model)
    anchor = drag_predict(params, horizon, model)
    beta = params[:, 6] if model == 'const' else params[:, 6] * drag_scale(params, model)
    data_dir = Path(data_dir)
    np.save(data_dir / f'anchor_drag_{name}.npy', anchor.astype(np.float32))
    np.save(data_dir / f'anchor_fit_window_{name}.npy',
            drag_window_trajectory(params, split['obs'].shape[1], model).astype(np.float32))
    np.save(data_dir / f'anchor_beta_{name}.npy', beta.astype(np.float32))
    print('anchors saved', name, anchor.shape, flush=True)


def run_b5(split, prior, q_accel, horizon, q_beta=0.0):
    domains = np.asarray(split['meta']['domain'])
    sites = np.asarray(split['meta']['radar_site_xyz_m'], dtype=np.float64)
    final = np.empty((len(domains), 7))
    for domain in sorted(set(domains.tolist())):
        idx = np.flatnonzero(domains == domain)
        sensor = split.get('protocol', {}).get('sensor', {}).get(domain, sensor_for_domain(domain))
        severity = sensor['severity']
        angle_std = np.deg2rad(ANGLE_STD_DEG * severity)
        polar = split['polar'][idx]
        detected = split['detected'][idx]
        init_steps = min(INIT_STEPS, polar.shape[1])
        # Initialize from a weighted quadratic LS fit of the first init_steps
        # converted measurements (detections only), state at sample init_steps-1.
        cart, cov = polar_to_cartesian(polar[:, :init_steps], sites[idx], severity, angle_std)
        t = (np.arange(init_steps) - (init_steps - 1)) * DT
        basis = np.stack([np.ones_like(t), t, 0.5 * t**2], axis=1)
        init = np.zeros((len(idx), 7))
        p0 = np.zeros((len(idx), 7, 7))
        var = np.einsum('ntii->nti', cov)  # per-axis measurement variance
        for axis in range(3):
            w = detected[:, :init_steps] / var[..., axis]
            normal = np.einsum('nt,ti,tj->nij', w, basis, basis)
            rhs = np.einsum('nt,ti,nt->ni', w, basis, cart[..., axis])
            # Weak prior on acceleration (gravity + typical drag) keeps the fit well posed.
            normal[:, 2, 2] += 1.0 / 20.0**2
            rhs[:, 2] += (-G if axis == 2 else 0.0) / 20.0**2
            coef_cov = np.linalg.inv(normal)
            coef = np.einsum('nij,nj->ni', coef_cov, rhs)
            init[:, axis], init[:, 3 + axis] = coef[:, 0], coef[:, 1]
            p0[:, axis, axis] = coef_cov[:, 0, 0]
            p0[:, 3 + axis, 3 + axis] = coef_cov[:, 1, 1]
            p0[:, axis, 3 + axis] = p0[:, 3 + axis, axis] = coef_cov[:, 0, 1]
        init[:, 6] = prior['mean']
        p0[:, 6, 6] = prior['std'] ** 2
        final[idx] = ukf_drag_track(polar, detected, sites[idx], init, p0, severity, angle_std,
                                    q_accel=q_accel, q_beta=q_beta, start=init_steps)
    return integrate(final[:, :6], final[:, 6], horizon)


def truth(split, horizon):
    window = split['obs'].shape[1]
    return split['X'][:, window:window + horizon]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/protocol_v2')
    parser.add_argument('--out', default='outputs/protocol_v2/physics_baselines')
    parser.add_argument('--horizon', type=int, default=2000)
    parser.add_argument('--prior-dir', help='Training/validation reference directory (default: --data-dir)')
    parser.add_argument('--fast', action='store_true', help='Skip B5 UKF selection and prediction')
    parser.add_argument('--anchors-only', action='store_true', help='Write anchors for available splits only')
    parser.add_argument('--no-anchors', action='store_true', help='Write baseline predictions only, not the anchor files')
    parser.add_argument('--report', help='Reuse an existing beta prior and B4 selection for --anchors-only')
    parser.add_argument('--anchor-model', choices=('b4', 'b6'), default='b4',
                        help='With --anchors-only --report: anchors from B4 or from the selected B6 table')
    args = parser.parse_args()
    prior_dir = args.prior_dir or args.data_dir
    names = [name for name in ('train', 'val', 'test') if (Path(args.data_dir) / f'{name}.pt').exists()]
    if not names:
        parser.error('No dataset splits found')
    t0 = time.time()
    report = {'prior_dir': str(Path(prior_dir).resolve())}
    anchor_model = 'const'
    if args.anchor_model == 'b6' and not (args.anchors_only and args.report):
        parser.error('--anchor-model b6 requires --anchors-only --report')
    if args.anchors_only and args.report:
        stored = json.loads(Path(args.report).read_text(encoding='utf-8'))
        prior = stored['beta_prior']
        best_noise = float(stored['selected']['B4_noise'])
        if args.anchor_model == 'b6':
            anchor_model = stored['selected']['B6_table']
            prior = stored['beta_prior_std'][anchor_model]
            best_noise = float(stored['selected']['B6_noise'])
    else:
        train = load(prior_dir, 'train')
        prior = beta_prior_from_train(train)
        report['beta_prior'] = prior
        print('beta prior', prior, f'{time.time() - t0:.0f}s', flush=True)
        if not args.anchors_only:
            priors_std = {table: beta_prior_from_train(train, model=table) for table in ('g1', 'g7')}
            report['beta_prior_std'] = priors_std
            print('beta prior std', priors_std, f'{time.time() - t0:.0f}s', flush=True)
        del train
        # Select every train-derived setting on the reference validation split.
        val = load(prior_dir, 'val')
        val_true = truth(val, args.horizon)
        selection = {'B4_noise': {}, 'B5_q_accel': {}, 'B6_noise': {}}
        for noise in (0.25, 1.0, 4.0, 16.0, 64.0):
            if noise < 64.0:  # B4 grid unchanged (its anchors feed training); B6 also tries 64
                pred, _ = run_b4(val, prior, noise, args.horizon)
                selection['B4_noise'][noise] = evaluate(pred, val_true, val['valid_len'])['overall']['ade_m']
            if not args.anchors_only:
                for table, prior_std in priors_std.items():
                    pred, _ = run_b4(val, prior_std, noise, args.horizon, table)
                    selection['B6_noise'][(table, noise)] = evaluate(
                        pred, val_true, val['valid_len'])['overall']['ade_m']
        best_noise = min(selection['B4_noise'], key=selection['B4_noise'].get)
        if not args.fast and not args.anchors_only:
            for q_accel in (0.01, 0.1, 1.0):
                for beta_walk in (0.0, 0.05, 0.2):
                    pred = run_b5(val, prior, q_accel, args.horizon, q_beta=(beta_walk * prior['std']) ** 2)
                    selection['B5_q_accel'][(q_accel, beta_walk)] = evaluate(
                        pred, val_true, val['valid_len'])['overall']['ade_m']
            best_q, best_walk = min(selection['B5_q_accel'], key=selection['B5_q_accel'].get)
            report['selected'] = {'B5_q_accel': best_q, 'B5_beta_walk': best_walk}
        report.setdefault('selected', {})['B4_noise'] = best_noise
        if selection['B6_noise']:
            best_table, best_noise_std = min(selection['B6_noise'], key=selection['B6_noise'].get)
            report['selected'].update(B6_table=best_table, B6_noise=best_noise_std)
        report['validation_selection'] = {k: {str(a): b for a, b in v.items()} for k, v in selection.items()}
        print('selection', report['selected'], f'{time.time() - t0:.0f}s', flush=True)
        del val, val_true, pred
    if args.anchors_only:
        for name in names:
            save_anchors(args.data_dir, name, load(args.data_dir, name), prior, best_noise, args.horizon,
                         anchor_model)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    test = load(args.data_dir, 'test')
    test_true = truth(test, args.horizon)
    obs = test['obs']
    preds = {
        'B1_CV': polynomial_predict(obs, args.horizon, 1),
        'B2_CA': polynomial_predict(obs, args.horizon, 2),
        'B3_CVg': polynomial_predict(obs, args.horizon, 1, gravity=True),
    }
    preds['B4_DragLS'], beta_test = run_b4(test, prior, best_noise, args.horizon)
    preds['B6_DragLSStd'], _ = run_b4(test, priors_std[best_table], best_noise_std, args.horizon, best_table)
    if not args.fast:
        preds['B5_DragUKF'] = run_b5(test, prior, best_q, args.horizon, q_beta=(best_walk * prior['std']) ** 2)
    report['test'] = {}
    for name, pred in preds.items():
        report['test'][name] = evaluate(pred, test_true, test['valid_len'], test['meta'])
        np.save(out / f'pred_{name}.npy', pred.astype(np.float32))
        o = report['test'][name]['overall']
        print(f"{name:11s} RMSE {o['rmse']:9.1f}  ADE {o['ade_m']:9.1f}  FDE {o['fde_m']:9.1f}  "
              f"impact med {o['impact_err_median_m']:8.1f}  TTI {o['tti_abs_err_mean_s']:.3f}", flush=True)
    np.save(out / 'true_test.npy', test_true.astype(np.float32))
    np.save(out / 'valid_len_test.npy', test['valid_len'])

    # Drag anchors for physics-anchored residual learning (all splits).
    for name in ([] if args.no_anchors else names):
        save_anchors(args.data_dir, name, test if name == 'test' else load(args.data_dir, name),
                     prior, best_noise, args.horizon)

    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
