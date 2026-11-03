import numpy as np

from utils.ballistic_physics import DT, predicted_impact
from utils.provenance import masked_trajectory_metrics


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8)))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    return mae, mse, rmse, mape


def landing_error(pred, true):
    """Compute landing point error (horizontal distance at ground impact)."""
    true_h = true[:, 2]
    ground_idx = np.where(true_h <= 0)[0]
    land_idx = ground_idx[0] if len(ground_idx) > 0 else len(true_h) - 1
    return np.sqrt(np.sum((pred[land_idx, :2] - true[land_idx, :2]) ** 2))


def trajectory_rmse_per_axis(pred, true):
    """RMSE for each coordinate axis (D, Z, H)."""
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


def reference_axis_average_rmse(pred, true):
    '''Return the arithmetic mean of the three coordinate-wise RMSE values.'''
    error = np.asarray(pred) - np.asarray(true)
    reduce_axes = tuple(range(error.ndim - 1))
    per_axis = np.sqrt(np.mean(error ** 2, axis=reduce_axes))
    return float(np.mean(per_axis)), per_axis


def future_mask(valid_len, horizon):
    return np.arange(horizon)[None, :] < np.asarray(valid_len)[:, None]


def impact_metrics(pred, true, valid_len):
    """Impact-point and time-to-impact errors from the predicted ground crossing."""
    rows = np.arange(len(pred))
    true_xy = true[rows, np.asarray(valid_len) - 1, :2]
    pred_xy, pred_tti = predicted_impact(pred)
    true_tti = np.asarray(valid_len) * DT
    impact_err = np.linalg.norm(pred_xy - true_xy, axis=1)
    return {
        'impact_err_m': impact_err,
        'tti_abs_err_s': np.abs(pred_tti - true_tti),
    }


def evaluate(pred, true, valid_len, meta=None, groups=('domain', 'projectile', 'wind')):
    """Overall and grouped metrics; ``pred``/``true`` are (N,H,3) in meters."""
    mask = future_mask(valid_len, pred.shape[1])
    per_sample = impact_metrics(pred, true, valid_len)

    def block(index):
        out = masked_trajectory_metrics(pred[index], true[index], mask[index])
        out = {k: out[k] for k in ('sample_count', 'rmse', 'ade_m', 'fde_m')}
        out['impact_err_median_m'] = float(np.median(per_sample['impact_err_m'][index]))
        out['impact_err_mean_m'] = float(np.mean(per_sample['impact_err_m'][index]))
        out['impact_err_p90_m'] = float(np.percentile(per_sample['impact_err_m'][index], 90))
        out['tti_abs_err_mean_s'] = float(np.mean(per_sample['tti_abs_err_s'][index]))
        return out

    everything = np.arange(len(pred))
    result = {'overall': block(everything)}
    if meta is not None:
        for key in groups:
            if key not in meta:
                continue
            values = np.asarray(meta[key])
            result[key] = {str(v): block(np.flatnonzero(values == v)) for v in sorted(set(values.tolist()))}
        if 't_go_s' in meta:
            tgo = np.asarray(meta['t_go_s'])
            result['t_go_s'] = {}
            for lo, hi in ((5, 10), (10, 15), (15, 20)):
                upper = (tgo <= hi) if hi == 20 else (tgo < hi)
                index = np.flatnonzero((tgo >= lo) & upper)
                if len(index):
                    result['t_go_s'][f'{lo}-{hi}'] = block(index)
    return result
