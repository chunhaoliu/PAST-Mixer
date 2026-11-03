"""TSLib-style dataset factory for the ballistic forecasting task."""

import os

import numpy as np
from torch.utils.data import DataLoader

from data_provider.data_loader import (
    BallisticDataset,
    _load_split,
    _resolve_dataset_dir,
    _resolve_scaler_dataset_dir,
    _to_numpy,
    filter_dataset_by_metadata,
)
from utils.tools import StandardScaler


def data_provider(args, flag):
    """Create one split-specific dataset and loader from the active protocol."""
    configured_dir = str(getattr(args, 'dataset_dir', '') or '').strip()
    data_dir = _resolve_dataset_dir(args)

    split_path = os.path.join(data_dir, f'{flag}.pt')
    if not os.path.exists(split_path) and not configured_dir:
        data_dir = os.path.join(args.root_path, 'data', 'raw')
        split_path = os.path.join(data_dir, f'{flag}.pt')
        if not os.path.exists(split_path):
            raise FileNotFoundError(f'Data not found at {split_path}')

    raw = _load_split(split_path)
    scaler_data_dir = _resolve_scaler_dataset_dir(args, data_dir)
    beta_stats = None
    context_mode = str(getattr(args, 'anchor_context', 'none')).lower()
    if context_mode == 'fit' and str(getattr(args, 'output_mode', '')).lower() != 'physics_anchor':
        raise ValueError('anchor_context=fit requires output_mode=physics_anchor')
    if str(getattr(args, 'output_mode', '')).lower() == 'physics_anchor':
        anchor_path = os.path.join(data_dir, f'anchor_drag_{flag}.npy')
        if not os.path.exists(anchor_path):
            raise FileNotFoundError(
                f'{anchor_path} missing; run tools/evaluate_protocol_v2_physics_baselines.py'
            )
        raw['anchor'] = np.load(anchor_path)
        if context_mode == 'fit':
            for key in ('anchor_fit_window', 'anchor_beta'):
                raw[key] = np.load(os.path.join(data_dir, f'{key}_{flag}.npy'))
            # Always use the full training split, regardless of current split/filter.
            train_beta = np.load(os.path.join(scaler_data_dir, 'anchor_beta_train.npy')).astype(np.float64)
            beta_stats = (train_beta.mean(), max(float(train_beta.std()), 1e-8))
    data_dict = filter_dataset_by_metadata(
        raw,
        wind_filter=getattr(args, 'wind_filter', None),
        weapon_filter=getattr(args, 'weapon_filter', None),
    )
    trajectories = _to_numpy(data_dict['X'])
    observed = data_dict.get('X_observed')
    if observed is not None:
        observed = _to_numpy(observed)

    train_reference = _to_numpy(
        _load_split(os.path.join(scaler_data_dir, 'train.pt'))['X']
    )
    scaler = StandardScaler(
        mean=train_reference.mean(axis=(0, 1)),
        std=train_reference.std(axis=(0, 1)) + 1e-8,
    )

    dataset = BallisticDataset(
        args,
        trajectories,
        flag=flag,
        scaler=scaler,
        observed=observed,
        observation_mode=getattr(args, 'observation_mode', 'auto'),
        protocol=data_dict.get('protocol'),
        scaler_dataset_dir=scaler_data_dir,
        valid_len=(
            _to_numpy(data_dict['valid_len']) if 'valid_len' in data_dict else None
        ),
        anchor=data_dict.get('anchor'),
        anchor_fit_window=data_dict.get('anchor_fit_window'),
        anchor_beta=data_dict.get('anchor_beta'),
        beta_stats=beta_stats,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(flag == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader
