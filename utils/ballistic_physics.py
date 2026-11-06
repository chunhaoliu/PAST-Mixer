"""Model-based ballistic predictors for protocol v2 (vectorized over trajectories).

The drag model is generic: exponential standard atmosphere, a constant
ballistic coefficient ``beta = Cd * A / m`` estimated from data, no wind and
no Coriolis term. It deliberately does not reuse the simulator's Mach-dependent
drag law, aerodynamic class parameters or wind, which a fielded predictor
would not know.

``model='g1'`` or ``'g7'`` (baseline B6) replaces the constant drag coefficient
by a public standard drag function of Mach number, with ISA density and speed
of sound; beta then scales the table shape (normalized to 1 at Mach 0.5).
"""
from __future__ import annotations

import numpy as np

DT = 0.01
G = 9.81
RHO0 = 1.225
SCALE_HEIGHT = 7400.0
# Standard drag functions (Mach, Cd), as tabulated in py-ballisticcalc / JBM.
G1_MACH = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975,
                    1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.7,
                    1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0])
G1_CD = np.array([0.2629, 0.2487, 0.2344, 0.2214, 0.2104, 0.2032, 0.2020, 0.2034, 0.2165, 0.2313, 0.2546,
                  0.2901, 0.3415, 0.3734, 0.4084, 0.4448, 0.4805, 0.5136, 0.5427, 0.5677, 0.5883, 0.6053,
                  0.6191, 0.6393, 0.6518, 0.6589, 0.6621, 0.6625, 0.6607, 0.6573, 0.6474, 0.6347, 0.6210,
                  0.6072, 0.5934, 0.5685, 0.5481, 0.5325, 0.5211, 0.5133, 0.5040, 0.5006]) / 0.2032
G7_MACH = np.array([0.0, 0.5, 0.6, 0.65, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95,
                    0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6,
                    1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0])
G7_CD = np.array([0.1198, 0.1194, 0.1194, 0.1197, 0.1202, 0.1207, 0.1215, 0.1226, 0.1242, 0.1266, 0.1306,
                  0.1368, 0.1464, 0.1660, 0.2054, 0.2993, 0.3803, 0.4015, 0.4043, 0.4034, 0.4014, 0.3987,
                  0.3955, 0.3884, 0.3810, 0.3732, 0.3657, 0.3580, 0.3440, 0.3315, 0.3209, 0.3117, 0.3042,
                  0.2980, 0.2864, 0.2752, 0.2643, 0.2533, 0.2424, 0.2154, 0.1935]) / 0.1194
DRAG_TABLES = {'g1': (G1_MACH, G1_CD), 'g7': (G7_MACH, G7_CD)}


def isa(height):
    """ISA density (kg/m^3) and speed of sound (m/s): 6.5 K/km lapse to 11 km, isothermal above."""
    h = np.clip(height, 0.0, 20000.0)
    temp = np.maximum(288.15 - 0.0065 * h, 216.65)
    pressure = np.where(h <= 11000.0, 101325.0 * (temp / 288.15) ** 5.25588,
                        22632.06 * np.exp(-(h - 11000.0) / (287.05287 * 216.65 / G)))
    return pressure / (287.05287 * temp), np.sqrt(1.4 * 287.05287 * temp)


def drag_scale(state, model='const'):
    """Table drag factor at the state's Mach number (1 for the constant model).

    For a table model the beta prior acts on beta * drag_scale(window-end state), the
    drag level the window observes, since beta alone shifts with the Mach regime.
    """
    if model not in DRAG_TABLES:
        return np.ones(len(state))
    _, sound = isa(state[:, 2])
    return np.interp(np.linalg.norm(state[:, 3:6], axis=-1) / sound, *DRAG_TABLES[model])


def drag_acceleration(velocity, height, beta, model='const'):
    """(N,3) acceleration of a point mass with ballistic coefficient beta (N,)."""
    speed = np.linalg.norm(velocity, axis=-1)
    if model in DRAG_TABLES:
        rho, sound = isa(height)
        beta = beta * np.interp(speed / sound, *DRAG_TABLES[model])
    else:
        rho = RHO0 * np.exp(-np.maximum(height, 0.0) / SCALE_HEIGHT)
    accel = -0.5 * (rho * beta * speed)[:, None] * velocity
    accel[:, 2] -= G
    return accel


def _derivative(state, beta, model='const'):
    out = np.empty_like(state)
    out[:, :3] = state[:, 3:]
    out[:, 3:] = drag_acceleration(state[:, 3:], state[:, 2], beta, model)
    return out


def rk4_step(state, beta, dt=DT, model='const'):
    k1 = _derivative(state, beta, model)
    k2 = _derivative(state + 0.5 * dt * k1, beta, model)
    k3 = _derivative(state + 0.5 * dt * k2, beta, model)
    k4 = _derivative(state + dt * k3, beta, model)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(state, beta, steps, dt=DT, model='const'):
    """Positions (N, steps, 3) after each of ``steps`` RK4 steps."""
    out = np.empty((len(state), steps, 3))
    for step in range(steps):
        state = rk4_step(state, beta, dt, model)
        out[:, step] = state[:, :3]
    return out


def _times(window):
    return (np.arange(window) - (window - 1)) * DT  # 0 at the last window sample


def polynomial_fit(track, degree, gravity=False):
    """LS polynomial fit of (N,L,3) tracks; returns coefficients at window end."""
    t = _times(track.shape[1])
    target = track.astype(np.float64).copy()
    if gravity:
        target[..., 2] += 0.5 * G * t**2
    basis = np.stack([t**k for k in range(degree + 1)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, target.transpose(1, 0, 2).reshape(len(t), -1), rcond=None)
    return coef.reshape(degree + 1, track.shape[0], 3)


def polynomial_predict(track, horizon, degree, gravity=False):
    coef = polynomial_fit(track, degree, gravity)
    t = np.arange(1, horizon + 1) * DT
    pred = sum(coef[k][:, None, :] * (t**k)[None, :, None] for k in range(degree + 1))
    if gravity:
        pred[..., 2] -= 0.5 * G * t**2
    return pred


def drag_window_trajectory(params, window=100, model='const'):
    """Positions over the fit window, integrating backward from [p_end,v_end,beta]."""
    n = len(params)
    state = params[:, :6].copy()
    beta = params[:, 6]
    # Integrate backward from the window end so that p, v are end-of-window states.
    positions = np.empty((n, window, 3))
    positions[:, -1] = state[:, :3]
    for step in range(window - 2, -1, -1):
        state = rk4_step(state, beta, -DT, model)
        positions[:, step] = state[:, :3]
    return positions


def _window_residual(params, track, beta_prior, model='const'):
    """Residuals of a backward-integrated drag trajectory against the window."""
    residual = (drag_window_trajectory(params, track.shape[1], model) - track).reshape(len(track), -1)
    if beta_prior is not None:
        mean, std, noise = beta_prior
        beta = params[:, 6] * drag_scale(params, model)
        residual = np.concatenate([residual, (noise * (beta - mean) / std)[:, None]], axis=1)
    return residual


def fit_drag_window(track, beta_prior, iterations=8, init=None, model='const'):
    """Levenberg-Marquardt MAP fit of [p_end, v_end, beta] to (N,L,3) tracks.

    ``beta_prior`` = (mean, std, residual_scale): a Gaussian prior on beta
    expressed in the same units as a position residual of ``residual_scale`` m.
    """
    n = track.shape[0]
    if init is None:
        coef = polynomial_fit(track, 2, gravity=False)
        init = np.concatenate([coef[0], coef[1], np.zeros((n, 1))], axis=1)
        init[:, 6] = beta_prior[0] / drag_scale(init, model)
    params = init.astype(np.float64)
    damping = np.full(n, 1e-3)
    steps = np.array([1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3, 1e-6])
    cost = (_window_residual(params, track, beta_prior, model) ** 2).sum(1)
    for _ in range(iterations):
        base = _window_residual(params, track, beta_prior, model)
        jac = np.empty(base.shape + (7,))
        for j in range(7):
            shifted = params.copy()
            shifted[:, j] += steps[j]
            jac[..., j] = (_window_residual(shifted, track, beta_prior, model) - base) / steps[j]
        jtj = np.einsum('nri,nrj->nij', jac, jac)
        jtr = np.einsum('nri,nr->ni', jac, base)
        diag = np.einsum('nii->ni', jtj)
        system = jtj + damping[:, None, None] * np.einsum('ni,ij->nij', diag + 1e-9, np.eye(7))
        delta = -np.linalg.solve(system, jtr[..., None])[..., 0]
        candidate = params + delta
        candidate[:, 6] = np.maximum(candidate[:, 6], 0.0)
        new_cost = (_window_residual(candidate, track, beta_prior, model) ** 2).sum(1)
        better = new_cost < cost
        params[better] = candidate[better]
        cost[better] = new_cost[better]
        damping = np.where(better, damping * 0.3, damping * 10.0)
    return params


def drag_predict(params, horizon, model='const'):
    return integrate(params[:, :6].copy(), params[:, 6], horizon, model=model)


def _polar_of(state, sites):
    rel = state[..., :3] - sites
    horizontal = np.linalg.norm(rel[..., :2], axis=-1)
    return np.stack([np.linalg.norm(rel, axis=-1), np.arctan2(rel[..., 1], rel[..., 0]),
                     np.arctan2(rel[..., 2], horizontal)], axis=-1)


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ukf_drag_track(polar, detected, sites, init_state, init_cov, severity, angle_std,
                   q_accel=4.0, q_beta=1e-12, start=2):
    """Unscented Kalman filter on [p, v, beta] with polar measurements.

    ``init_state`` (N,7) and ``init_cov`` (N,7,7) describe the state at
    sample ``start - 1``; the filter processes samples start..T-1 and returns
    the window-end mean state (N,7).
    """
    n, steps, _ = polar.shape
    dim = 7
    x = init_state.astype(np.float64).copy()
    p = init_cov.astype(np.float64).copy()
    weight = 1.0 / (2 * dim)
    q = np.zeros((dim, dim))
    q[0:3, 0:3] = np.eye(3) * q_accel * DT**3 / 3
    q[0:3, 3:6] = q[3:6, 0:3] = np.eye(3) * q_accel * DT**2 / 2
    q[3:6, 3:6] = np.eye(3) * q_accel * DT
    q[6, 6] = q_beta * DT
    sites_b = sites[:, None, :]
    for step in range(start, steps):
        chol =np.linalg.cholesky(0.5 * (p + np.swapaxes(p, 1, 2)) + 1e-9 * np.eye(dim))
        offsets = np.sqrt(dim) * np.swapaxes(chol, 1, 2)  # rows are columns of chol
        sigma = np.concatenate([x[:, None] + offsets, x[:, None] - offsets], axis=1)  # (N,14,7)
        flat = sigma.reshape(-1, dim)
        moved = np.empty_like(flat)
        moved[:, :6] = rk4_step(flat[:, :6], flat[:, 6])
        moved[:, 6] = flat[:, 6]
        sigma = moved.reshape(n, 2 * dim, dim)
        x = sigma.mean(axis=1)
        dev = sigma - x[:, None]
        p = weight * np.einsum('nki,nkj->nij', dev, dev) + q
        hit = detected[:, step]
        if not np.any(hit):
            continue
        zs = _polar_of(sigma[hit], sites_b[hit])
        z_mean = zs.mean(axis=1)
        z_mean[:, 1:] = np.arctan2(np.sin(zs[..., 1:]).mean(1), np.cos(zs[..., 1:]).mean(1))
        z_dev = zs - z_mean[:, None]
        z_dev[..., 1:] = _wrap(z_dev[..., 1:])
        range_std = severity * (3.0 + 2.5e-4 * z_mean[:, 0])
        r = np.zeros((int(hit.sum()), 3, 3))
        r[:, 0, 0] = range_std**2
        r[:, 1, 1] = r[:, 2, 2] = angle_std**2
        s = weight * np.einsum('nki,nkj->nij', z_dev, z_dev) + r
        cross = weight * np.einsum('nki,nkj->nij', dev[hit], z_dev)
        gain = cross @ np.linalg.inv(s)
        innovation = polar[hit, step] - z_mean
        innovation[:, 1:] = _wrap(innovation[:, 1:])
        x[hit] = x[hit] + np.einsum('nij,nj->ni', gain, innovation)
        p[hit] = p[hit] - gain @ s @ np.swapaxes(gain, 1, 2)
        x[:, 6] = np.maximum(x[:, 6], 0.0)
    return x


def predicted_impact(pred, dt=DT):
    """First downward ground crossing of (N,H,3) predictions.

    Returns impact xy (N,2) and time-to-impact in seconds (N,); trajectories
    that never cross take the last predicted sample.
    """
    z = pred[..., 2]
    below = z <= 0.0
    has = below.any(axis=1)
    first = np.where(has, below.argmax(axis=1), pred.shape[1] - 1)
    prev = np.maximum(first - 1, 0)
    rows = np.arange(len(pred))
    z0, z1 = z[rows, prev], z[rows, first]
    frac = np.where(has & (first > 0) & (z0 != z1), z0 / np.where(z0 != z1, z0 - z1, 1.0), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    xy = pred[rows, prev, :2] + frac[:, None] * (pred[rows, first, :2] - pred[rows, prev, :2])
    step = np.where(first > 0, prev + frac, first).astype(np.float64)
    tti = (step + 1.0) * dt
    return xy, tti
