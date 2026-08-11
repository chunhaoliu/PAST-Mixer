import numpy as np


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
