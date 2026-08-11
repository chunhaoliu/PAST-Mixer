import torch


def _lag_difference(values, lag):
    if values.ndim != 3:
        raise ValueError('values must have shape (B, L, C)')
    if lag < 1:
        raise ValueError('lag must be positive')
    difference = torch.zeros_like(values)
    if values.shape[1] > lag:
        valid = values[:, lag:, :] - values[:, :-lag, :]
        difference[:, lag:, :] = valid
        difference[:, :lag, :] = valid[:, :1, :]
    return difference


def build_kinematic_state(positions, lag=5, dt=0.01):
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError('positions must have shape (B, L, 3)')
    if dt <= 0:
        raise ValueError('dt must be positive')
    velocity = _lag_difference(positions, lag) / (lag * dt)
    acceleration = _lag_difference(velocity, lag) / (lag * dt)
    return torch.cat((positions, velocity, acceleration), dim=-1)


def build_state_features(positions, mode, lag=5, dt=0.01):
    normalized_mode = str(mode).lower()
    if normalized_mode == 'position':
        return positions
    if normalized_mode == 'kinematics':
        return build_kinematic_state(positions, lag=lag, dt=dt)
    raise ValueError('past_state_mode must be position or kinematics')
