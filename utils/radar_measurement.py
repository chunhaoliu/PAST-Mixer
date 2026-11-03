"""Polar measurement conversion and sensor noise parameters (no tracking)."""
import numpy as np


NOMINAL_SENSOR = {'severity': 1.0, 'detection_probability': 0.98}

EXTRAPOLATIVE_SENSOR = {'severity': 1.6, 'detection_probability': 0.90}

ANGLE_STD_DEG = 0.04

def sensor_for_domain(domain):
    if domain == 'nominal':
        return dict(NOMINAL_SENSOR)
    if domain == 'extrapolative':
        return dict(EXTRAPOLATIVE_SENSOR)
    raise ValueError(f'Unsupported domain={domain!r}')

def _per_row(value):
    """Scalar sensor parameter, or one value per trajectory broadcast over (N, T)."""
    value = np.asarray(value, dtype=np.float64)
    return value if value.ndim == 0 else value[:, None]

def polar_to_cartesian(polar, sites, severity, angle_std):
    """Converted measurements and their Jacobian covariance, shapes (N,T,3), (N,T,3,3)."""
    severity, angle_std = _per_row(severity), _per_row(angle_std)
    r, az, el = polar[..., 0], polar[..., 1], polar[..., 2]
    ce, se, ca, sa = np.cos(el), np.sin(el), np.cos(az), np.sin(az)
    cart = np.stack([r * ce * ca, r * ce * sa, r * se], axis=-1) + sites[:, None, :]
    jac = np.zeros(polar.shape[:-1] + (3, 3))
    jac[..., 0, 0], jac[..., 0, 1], jac[..., 0, 2] = ce * ca, -r * ce * sa, -r * se * ca
    jac[..., 1, 0], jac[..., 1, 1], jac[..., 1, 2] = ce * sa, r * ce * ca, -r * se * sa
    jac[..., 2, 0], jac[..., 2, 2] = se, r * ce
    range_std = severity * (3.0 + 2.5e-4 * np.maximum(r, 0.0))
    polar_var = np.stack(
        [range_std**2, np.broadcast_to(angle_std**2, r.shape), np.broadcast_to(angle_std**2, r.shape)],
        axis=-1,
    )
    cov = np.einsum('...ij,...j,...kj->...ik', jac, polar_var, jac)
    return cart, cov
