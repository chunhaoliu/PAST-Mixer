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
    if 'X_observed' in data_dict:
        filtered['X_observed'] = data_dict['X_observed'][mask]
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
    ):
        self.args = args
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

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            torch.from_numpy(seq_x_mark).float(),
            torch.from_numpy(seq_y_mark).float(),
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
    return os.path.join(args.root_path, 'data', 'ren_dataset')


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
