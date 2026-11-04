"""
Main entry point for ballistic trajectory prediction experiments.
TSLib-compatible run script.

Usage:
    python run.py --is_training 1 --model PASTMixer --seq_len 100 --pred_len 1000
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from setup import CHECKPOINTS, RESULTS, ROOT_PATH


def parse_args():
    parser = argparse.ArgumentParser(description='Ballistic Trajectory Prediction')

    # Basic config
    parser.add_argument('--task_name', type=str, default='ballistic_forecast')
    parser.add_argument('--is_training', type=int, default=1, help='1=train, 0=test only')
    parser.add_argument('--model_id', type=str, default='taes_projectile_mainline')
    parser.add_argument('--model', type=str, default='PASTMixer')
    parser.add_argument('--protocol_id', type=str, default='')

    # Data
    parser.add_argument('--data', type=str, default='ballistic')
    parser.add_argument('--root_path', type=str, default=ROOT_PATH)
    parser.add_argument('--data_path', type=str, default='data/raw/train.pt')
    parser.add_argument('--dataset_dir', type=str, default='data/protocol_v2',
                        help='Protocol-specific dataset directory')
    parser.add_argument('--scaler_dataset_dir', type=str, default='',
                        help='Original training dataset for position and anchor-beta scaling')
    parser.add_argument('--observation_mode', type=str, default='tracked',
                        choices=['clean', 'tracked'],
                        help='Observed prefix source for protocol datasets')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--freq', type=str, default='t')
    parser.add_argument('--wind_filter', type=str, default='', help='Comma-separated wind types to keep')
    parser.add_argument('--weapon_filter', type=str, default='', help='Comma-separated weapon types to keep')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=100, help='Observed points')
    parser.add_argument('--label_len', type=int, default=0, help='Decoder warm-start length (0 for direct forecasting)')
    parser.add_argument('--pred_len', type=int, default=1000, help='Prediction horizon')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Sampling interval in seconds for causal state features')

    # Model
    parser.add_argument('--enc_in', type=int, default=3)
    parser.add_argument('--dec_in', type=int, default=3)
    parser.add_argument('--c_out', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=256, help='Hidden/model dimension')
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3, help='Number of encoder or mixer layers')
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_physics', type=int, default=0, help='Enable physics residual baseline in PASTMixer')
    parser.add_argument('--physics_mode', type=str, default='none', help='Physics baseline mode: none, learned, kinematic')
    parser.add_argument('--physics_gate_mode', type=str, default='scalar',
                        choices=['scalar', 'adaptive', 'conservative_adaptive'],
                        help='PASTMixer physics baseline gate: scalar, adaptive, or conservative_adaptive')
    parser.add_argument('--physics_gate_max', type=float, default=1.0,
                        help='Maximum multiplier applied to the physics baseline gate')
    parser.add_argument('--physics_gate_delta_max', type=float, default=0.2,
                        help='Maximum per-axis adaptive correction for conservative_adaptive physics gate')
    parser.add_argument('--past_temporal_mode', type=str, default='simple',
                        choices=['multiscale', 'simple'],
                        help='PASTMixer temporal mixer for ablations')
    parser.add_argument('--past_norm_mode', type=str, default='pre',
                        choices=['none', 'pre'],
                        help='PASTMixer state normalization before mixing')
    parser.add_argument('--past_output_mode', type=str, default='displacement',
                        choices=['absolute', 'displacement'],
                        help='PASTMixer output parameterization')
    parser.add_argument('--past_state_mode', type=str, default='kinematics',
                        choices=['position', 'kinematics'],
                        help='PASTMixer observed-state representation')
    parser.add_argument('--past_state_lag', type=int, default=5,
                        help='Finite-difference lag for kinematic features')
    parser.add_argument('--physics_fusion_mode', type=str, default='additive')
    parser.add_argument('--freeze_backbone', type=int, default=0)
    parser.add_argument('--pretrained_backbone_checkpoint', type=str, default='')
    parser.add_argument('--activation', type=str, default='gelu')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--train_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--loss_mode', type=str, default='physical_composite',
                        choices=['mse', 'physical_mse', 'physical_composite'])
    parser.add_argument('--ade_loss_weight', type=float, default=0.5)
    parser.add_argument('--fde_loss_weight', type=float, default=0.25)
    parser.add_argument('--select_metric', type=str, default='loss',
                        choices=['loss', 'val_rmse_m'],
                        help='Checkpoint/early-stopping quantity: own validation loss (historical) '
                             'or global validation coordinate RMSE in meters (common to all models)')

    # Controlled-comparison I/O switches (apply to every model; model_default keeps legacy behaviour)
    parser.add_argument('--input_state', type=str, default='model_default',
                        choices=['model_default', 'position', 'kinematics'],
                        help='Observed-state representation for any model')
    parser.add_argument('--output_mode', type=str, default='model_default',
                        choices=['model_default', 'absolute', 'displacement', 'physics_anchor'],
                        help='Output parameterization for any model; physics_anchor adds a residual '
                             'to the drag-LS extrapolation stored as anchor_drag_<split>.npy')
    parser.add_argument('--residual_gate', choices=['none', 'rezero'], default='none',
                        help='Optional zero-initialized per-position residual gate for physics_anchor')
    parser.add_argument('--anchor_context', choices=['none', 'fit'], default='none',
                        help='Optional drag-fit residual, anchor preview and train-standardized beta inputs')

    # GPU
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')

    # Paths
    parser.add_argument('--checkpoints', type=str, default=CHECKPOINTS)
    parser.add_argument('--results', type=str, default=RESULTS)
    parser.add_argument('--source_snapshot', type=str, default='',
                        help='Optional immutable source snapshot created by the suite runner')
    parser.add_argument('--dataset_fingerprint', type=str, default='',
                        help='Optional dataset split fingerprint created by the suite runner')

    # Extra
    parser.add_argument('--des', type=str, default='ballistic trajectory experiment')
    parser.add_argument('--train_subset', type=int, default=0, help='Use N training samples (0=all)')

    return parser.parse_args()


def make_setting(args, itr_idx):
    return '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_el{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.e_layers,
        itr_idx,
    )


def main():
    args = parse_args()
    from models.io_adapter import resolve_io_modes

    resolve_io_modes(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print(f'Using GPU: cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')
        print('Using CPU')

    print('\n' + '=' * 60)
    print('Ballistic Trajectory Prediction')
    print('=' * 60)
    print(f'Model: {args.model}, seq_len={args.seq_len}, pred_len={args.pred_len}, d_model={args.d_model}')
    print(
        f'Layers: {args.e_layers}, epochs={args.train_epochs}, '
        f'use_physics={args.use_physics}, physics_mode={args.physics_mode}, '
        f'physics_gate_mode={args.physics_gate_mode}, physics_gate_max={args.physics_gate_max}, '
        f'physics_gate_delta_max={args.physics_gate_delta_max}'
    )
    print('=' * 60 + '\n')

    from exp.exp_ballistic import Exp_Ballistic

    if args.is_training:
        for ii in range(args.itr):
            setting = make_setting(args, ii)
            exp = Exp_Ballistic(args)
            print(f'>>>>>>> Training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>> Testing:  {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            if args.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        exp = Exp_Ballistic(args)
        setting = make_setting(args, 0)
        print(f'>>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        if args.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
