"""Atomic evaluation artifacts and run provenance for formal experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


SOURCE_DIRS = ('configs', 'data_provider', 'exp', 'layers', 'models', 'scripts', 'tools', 'utils')
SOURCE_SUFFIXES = {'.json', '.py', '.ps1'}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _project_root():
    return Path(__file__).resolve().parents[1]


def source_tree_fingerprint(root):
    root = Path(root).resolve()
    digest = hashlib.sha256()
    count = 0
    for dirname in SOURCE_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES):
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode('utf-8'))
            digest.update(b'\0')
            digest.update(bytes.fromhex(sha256_file(path)))
            count += 1
    for name in ('run.py', 'setup.py'):
        path = root / name
        if path.exists():
            digest.update(name.encode('utf-8'))
            digest.update(b'\0')
            digest.update(bytes.fromhex(sha256_file(path)))
            count += 1
    return {'sha256': digest.hexdigest(), 'file_count': count}


def _git_info(root):
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=root, text=True, stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ['git', 'diff', '--binary', 'HEAD'], cwd=root, stderr=subprocess.DEVNULL,
        )
        return {
            'revision': revision,
            'dirty': bool(status.strip()),
            'tracked_diff_sha256': hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        return {'revision': 'unknown', 'dirty': None, 'tracked_diff_sha256': None}


def _atomic_json_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary name short for deeply nested Windows experiment paths.
    handle, temp_name = tempfile.mkstemp(prefix='.tmp-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_numpy_new(path, array):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary name short for deeply nested Windows experiment paths.
    handle, temp_name = tempfile.mkstemp(prefix='.tmp-', suffix='.npy', dir=path.parent)
    os.close(handle)
    try:
        np.save(temp_name, array)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _argument_snapshot(args):
    snapshot = {}
    for key, value in sorted(vars(args).items()):
        if isinstance(value, torch.device):
            value = str(value)
        if isinstance(value, (str, int, float, bool, type(None))):
            snapshot[key] = value
    return snapshot


def _nvidia_driver_version():
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def write_evaluation_artifacts(args, setting, preds, trues, test_evaluation_count=1):
    """Write arrays, metrics, and manifest exactly once for a run."""
    root = _project_root()
    result_dir = Path(args.results) / setting
    checkpoint_path = Path(args.checkpoints) / setting / 'checkpoint.pth'
    pred_path = result_dir / 'pred.npy'
    true_path = result_dir / 'true.npy'
    metrics_path = result_dir / 'metrics.json'
    manifest_path = result_dir / 'run_manifest.json'
    targets = (pred_path, true_path, metrics_path, manifest_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError('Refusing partial or destructive evaluation write: ' + ', '.join(existing))

    preds = np.asarray(preds)
    trues = np.asarray(trues)
    if preds.shape != trues.shape or preds.ndim != 3:
        raise ValueError(f'Expected matching (N,H,C) arrays, got {preds.shape} and {trues.shape}')
    error = preds - trues
    distances = np.linalg.norm(error, axis=-1)
    metrics = {
        'sample_count': int(preds.shape[0]),
        'horizon_steps': int(preds.shape[1]),
        'mae': float(np.mean(np.abs(error))),
        'mse': float(np.mean(error ** 2)),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'ade_m': float(distances.mean()),
        'fde_m': float(distances[:, -1].mean()),
    }

    _atomic_numpy_new(pred_path, preds)
    _atomic_numpy_new(true_path, trues)
    _atomic_json_new(metrics_path, metrics)

    dataset_manifest_path = Path(args.dataset_dir) / 'protocol_manifest.json'
    if not dataset_manifest_path.is_absolute():
        dataset_manifest_path = root / dataset_manifest_path
    argument_snapshot = _argument_snapshot(args)
    argument_json = json.dumps(argument_snapshot, sort_keys=True, separators=(',', ':')).encode('utf-8')
    gpu_index = int(getattr(args, 'gpu', 0))
    cuda_name = torch.cuda.get_device_name(gpu_index) if torch.cuda.is_available() else None
    gpu_memory = (
        int(torch.cuda.get_device_properties(gpu_index).total_memory)
        if torch.cuda.is_available() else None
    )
    manifest = {
        'schema_version': 2,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_id': getattr(args, 'protocol_id', ''),
        'setting': setting,
        'run': {
            'model': args.model,
            'model_id': args.model_id,
            'seed': int(args.seed),
            'test_evaluation_count': int(test_evaluation_count),
            'arguments': argument_snapshot,
            'arguments_sha256': hashlib.sha256(argument_json).hexdigest(),
        },
        'dataset_manifest': {
            'path': str(dataset_manifest_path),
            'sha256': sha256_file(dataset_manifest_path),
        },
        'artifacts': {
            'pred.npy': {'sha256': sha256_file(pred_path), 'bytes': pred_path.stat().st_size},
            'true.npy': {'sha256': sha256_file(true_path), 'bytes': true_path.stat().st_size},
            'metrics.json': {'sha256': sha256_file(metrics_path), 'bytes': metrics_path.stat().st_size},
            'checkpoint.pth': {'path': str(checkpoint_path), 'sha256': sha256_file(checkpoint_path), 'bytes': checkpoint_path.stat().st_size},
        },
        'metrics': metrics,
        'source': {**_git_info(root), 'tree': source_tree_fingerprint(root)},
        'environment': {
            'python': sys.version,
            'platform': platform.platform(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu': cuda_name,
            'gpu_total_memory_bytes': gpu_memory,
            'nvidia_driver': _nvidia_driver_version(),
        },
    }
    source_snapshot = str(getattr(args, 'source_snapshot', '') or '')
    if source_snapshot:
        snapshot_path = Path(source_snapshot)
        if not snapshot_path.is_absolute():
            snapshot_path = root / snapshot_path
        manifest['source']['snapshot'] = {
            'path': str(snapshot_path),
            'sha256': sha256_file(snapshot_path),
            'bytes': snapshot_path.stat().st_size,
        }
    dataset_fingerprint = str(getattr(args, 'dataset_fingerprint', '') or '')
    if dataset_fingerprint:
        fingerprint_path = Path(dataset_fingerprint)
        if not fingerprint_path.is_absolute():
            fingerprint_path = root / fingerprint_path
        manifest['dataset_fingerprint'] = {
            'path': str(fingerprint_path),
            'sha256': sha256_file(fingerprint_path),
            'bytes': fingerprint_path.stat().st_size,
        }
    _atomic_json_new(manifest_path, manifest)
    return manifest
