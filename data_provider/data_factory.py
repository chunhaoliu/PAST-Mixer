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

    data_dict = filter_dataset_by_metadata(
        _load_split(split_path),
        wind_filter=getattr(args, 'wind_filter', None),
        weapon_filter=getattr(args, 'weapon_filter', None),
    )
    trajectories = _to_numpy(data_dict['X'])
    observed = data_dict.get('X_observed')
    if observed is not None:
        observed = _to_numpy(observed)

    scaler_data_dir = _resolve_scaler_dataset_dir(args, data_dir)
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
