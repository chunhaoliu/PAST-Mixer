"""Train and evaluate PAST-Mixer on a compatible trajectory dataset."""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup import CHECKPOINTS, RESULTS, ROOT_PATH


def parse_args():
    parser = argparse.ArgumentParser(description='PAST-Mixer trajectory forecasting')
    parser.add_argument('--task_name', default='ballistic_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', default='past_mixer')
    parser.add_argument('--model', default='PASTMixer')
    parser.add_argument('--protocol_id', default='')
    parser.add_argument('--root_path', default=ROOT_PATH)
    parser.add_argument('--dataset_dir', default='data/demo')
    parser.add_argument('--observation_mode', choices=['clean', 'tracked'], default='tracked')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--enc_in', type=int, default=3)
    parser.add_argument('--c_out', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--past_norm_mode', choices=['none', 'pre'], default='pre')
    parser.add_argument('--past_output_mode', choices=['absolute', 'displacement'], default='displacement')
    parser.add_argument('--past_state_mode', choices=['position', 'kinematics'], default='kinematics')
    parser.add_argument('--past_state_lag', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--train_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lradj', choices=['type1', 'type3', 'cosine'], default='type3')
    parser.add_argument('--loss_mode', choices=['mse', 'physical_composite'], default='physical_composite')
    parser.add_argument('--ade_loss_weight', type=float, default=0.5)
    parser.add_argument('--fde_loss_weight', type=float, default=0.25)
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoints', default=CHECKPOINTS)
    parser.add_argument('--results', default=RESULTS)
    parser.add_argument('--train_subset', type=int, default=0)
    parser.add_argument('--source_snapshot', default='')
    parser.add_argument('--dataset_fingerprint', default='')
    return parser.parse_args()


def make_setting(args, iteration):
    return f'{args.task_name}_{args.model_id}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_el{args.e_layers}_{iteration}'


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from exp.exp_ballistic import Exp_Ballistic
    for iteration in range(args.itr):
        setting = make_setting(args, iteration)
        experiment = Exp_Ballistic(args)
        if args.is_training:
            experiment.train(setting)
            experiment.test(setting)
        else:
            experiment.test(setting, test=1)


if __name__ == '__main__':
    main()
