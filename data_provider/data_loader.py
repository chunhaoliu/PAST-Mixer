"""
Ballistic trajectory data loader (TSLib-compatible).
Provides sliding windows over full 2000-step trajectories.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.tools import StandardScaler


def _normalize_filter_values(value):
    if value in (None, '', 'all'):
        return None
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(',') if item.strip()]
        return set(parts) if parts else None
    if isinstance(value, (list, tuple, set)):
        return set(value) or None
    return {value}


PER_SAMPLE_ARRAYS = ('X_observed', 'valid_len', 'polar', 'detected', 'anchor', 'anchor_fit_window', 'anchor_beta')


def filter_dataset_by_metadata(data_dict, wind_filter=None, weapon_filter=None):
    meta = data_dict.get('meta')
    if not meta:
        return data_dict

    wind_values = _normalize_filter_values(wind_filter)
    weapon_values = _normalize_filter_values(weapon_filter)
    if wind_values is None and weapon_values is None:
        return data_dict

    winds = np.array(meta['wind'])
    weapons = np.array(meta['weapon'])
    mask = np.ones(len(winds), dtype=bool)
    if wind_values is not None:
        mask &= np.isin(winds, list(wind_values))
    if weapon_values is not None:
        mask &= np.isin(weapons, list(weapon_values))

    filtered = {'X': data_dict['X'][mask]}
    for key in PER_SAMPLE_ARRAYS:
        if key in data_dict:
            filtered[key] = data_dict[key][mask]
    if 'protocol' in data_dict:
        filtered['protocol'] = data_dict['protocol']
    filtered_meta = {}
    for key, values in meta.items():
        arr = np.array(values)
        filtered_meta[key] = arr[mask].tolist()
    filtered['meta'] = filtered_meta
    return filtered


class BallisticDataset(Dataset):
    """
    Dataset for ballistic trajectory prediction.

    Returns:
        seq_x: first seq_len observed points (batch, seq_len, 3)
        seq_y: next label_len+pred_len points for decoder (batch, label_len+pred_len, 3)
        seq_x_mark: time position marks (batch, seq_len, 1)
        seq_y_mark: time position marks (batch, label_len+pred_len, 1)
        future_mask: 1 for valid future steps (batch, pred_len); all ones
            unless the split stores ``valid_len`` (prediction to impact)
        anchor: scaled model-based extrapolation (batch, pred_len, 3) for
            physics-anchored residual learning, else a (batch, 1, 3) zero
            With anchor_context=fit this item is a dict containing ``anchor``
            and seven-channel ``context`` aligned with the observed window.
    """
    def __init__(
        self,
        args,
        data,
        flag='train',
        scaler=None,
        observed=None,
        observation_mode='auto',
        observed_data=None,
        protocol=None,
        scaler_dataset_dir=None,
        valid_len=None,
        anchor=None,
        anchor_fit_window=None,
        anchor_beta=None,
        beta_stats=None,
    ):
        self.args = args
        self.valid_len = (
            np.asarray(valid_len, dtype=np.int64) if valid_len is not None else None
        )
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.flag = flag
        self.scale = args.scale
        if observed is None:
            observed = observed_data
        self.protocol = protocol if isinstance(protocol, dict) else {}
        requested_mode = str(observation_mode or 'auto').lower()
        if requested_mode == 'auto':
            requested_mode = str(
                self.protocol.get('default_observation_mode', '')
            ).lower()
            if not requested_mode:
                requested_mode = 'tracked' if observed is not None else 'clean'
        self.observation_mode = requested_mode
        self.observed = observed
        self.scaler_dataset_dir = (
            str(scaler_dataset_dir) if scaler_dataset_dir is not None else ''
        )

        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler(
                mean=data.mean(axis=(0, 1)),
                std=data.std(axis=(0, 1)) + 1e-8
            )

        self.context = None
        if getattr(args, 'anchor_context', 'none') == 'fit':
            if str(getattr(args, 'output_mode', '')).lower() != 'physics_anchor':
                raise ValueError('anchor_context=fit requires output_mode=physics_anchor')
            if self.observation_mode != 'tracked' or observed is None:
                raise ValueError('anchor_context=fit requires tracked observations')
            if anchor is None or anchor_fit_window is None or anchor_beta is None or beta_stats is None:
                raise ValueError('anchor_context=fit requires anchor, fit window, beta and train beta statistics')
            if np.shape(anchor) != (len(data), self.pred_len, 3) or np.shape(anchor_fit_window) != (len(data), self.seq_len, 3):
                raise ValueError('Anchor/context arrays do not match protocol v2 sample shapes')
            if np.shape(anchor_beta) != (len(data),):
                raise ValueError('anchor_beta must contain one value per trajectory')
            std = self.scaler.std if self.scale else np.ones(3)
            residual = (observed[:, :self.seq_len] - anchor_fit_window) / std
            preview_indices = np.arange(1, self.seq_len + 1) * self.pred_len // self.seq_len - 1
            preview = (anchor[:, preview_indices] - observed[:, self.seq_len - 1:self.seq_len]) / std
            beta_mean, beta_std = beta_stats
            beta = (np.asarray(anchor_beta) - beta_mean) / beta_std
            beta = np.broadcast_to(beta[:, None, None], (len(data), self.seq_len, 1))
            self.context = np.concatenate([residual, preview, beta], axis=-1).astype(np.float32)

        self.anchor = None
        if anchor is not None:
            anchor = np.asarray(anchor, dtype=np.float32)[:, :self.pred_len]
            self.anchor = (
                self.scaler.transform(anchor).astype(np.float32) if self.scale else anchor
            )

        if self.scale:
            self.data = self.scaler.transform(data)
            self.observed_data = (
                self.scaler.transform(observed) if observed is not None else None
            )
        else:
            self.data = data
            self.observed_data = observed

        self.total_len = data.shape[1]
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        target = self.data[index]
        if self.observed_data is not None and self.observation_mode != 'clean':
            observed = self.observed_data[index]
        else:
            observed = target

        s_begin = 0
        s_end = s_begin + self.seq_len
        if observed.shape[0] < s_end:
            raise ValueError(
                f'Observed trajectory is shorter than seq_len={self.seq_len}: '
                f'{observed.shape[0]}'
            )
        seq_x = observed[s_begin:s_end]

        future = target[s_end:s_end + self.pred_len]
        expected_future = self.pred_len
        if future.shape[0] < expected_future:
            padded_future = np.zeros(
                (expected_future, target.shape[-1]),
                dtype=np.float32,
            )
            padded_future[:future.shape[0]] = future
            future = padded_future

        if self.label_len > 0:
            label_begin = max(0, s_end - self.label_len)
            label = target[label_begin:s_end]
            if label.shape[0] < self.label_len:
                padded_label = np.zeros(
                    (self.label_len, target.shape[-1]),
                    dtype=np.float32,
                )
                padded_label[-label.shape[0]:] = label
                label = padded_label
            seq_y = np.concatenate([label, future], axis=0)
        else:
            seq_y = future

        seq_x_mark = (
            np.arange(s_begin, s_end, dtype=np.float32).reshape(-1, 1)
            / self.total_len
        )
        mark_len = self.label_len + self.pred_len
        seq_y_mark = (
            np.arange(s_end - self.label_len, s_end + self.pred_len, dtype=np.float32)
            .reshape(-1, 1)
            / self.total_len
        )
        if seq_y_mark.shape[0] < mark_len:
            pad_len = mark_len - seq_y_mark.shape[0]
            seq_y_mark = np.concatenate(
                [seq_y_mark, np.zeros((pad_len, 1), dtype=np.float32)],
                axis=0,
            )

        future_mask = np.ones(self.pred_len, dtype=np.float32)
        if self.valid_len is not None:
            future_mask[int(self.valid_len[index]):] = 0.0

        anchor_item = (
            torch.from_numpy(self.anchor[index])
            if self.anchor is not None else torch.zeros(1, target.shape[-1])
        )
        if self.context is not None:
            anchor_item = {'anchor': anchor_item, 'context': torch.from_numpy(self.context[index])}

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            torch.from_numpy(seq_x_mark).float(),
            torch.from_numpy(seq_y_mark).float(),
            torch.from_numpy(future_mask),
            anchor_item,
        )

    def __len__(self):
        return self.n_samples

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def _resolve_dataset_dir(args):
    configured = str(getattr(args, 'dataset_dir', '') or '').strip()
    if configured:
        if os.path.isabs(configured):
            return configured
        return os.path.join(args.root_path, configured)
    return os.path.join(args.root_path, 'data', 'protocol_v2')


def _resolve_scaler_dataset_dir(args, data_dir):
    configured = str(getattr(args, 'scaler_dataset_dir', '') or '').strip()
    if not configured:
        return data_dir
    if os.path.isabs(configured):
        return configured
    return os.path.join(args.root_path, configured)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_split(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def data_provider(args, flag):
    """Compatibility import; new code should use data_provider.data_factory."""
    from data_provider.data_factory import data_provider as factory

    return factory(args, flag)
