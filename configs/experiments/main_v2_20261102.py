"""Formal protocol v2 job matrices; print JSON with main, abl, tune, best, rand, stress or win."""
import json
import sys
from itertools import product

DATA = 'data/protocol_v2'
COMMON = ['--is_training', '1', '--dataset_dir', DATA, '--observation_mode', 'tracked',
          '--seq_len', '100', '--label_len', '0', '--pred_len', '2000',
          '--d_model', '256', '--d_ff', '512', '--dropout', '0.1',
          '--train_epochs', '80', '--patience', '15', '--batch_size', '32',
          '--learning_rate', '0.001', '--lradj', 'type3', '--use_physics', '0', '--physics_mode', 'none',
          '--past_temporal_mode', 'simple', '--past_norm_mode', 'pre', '--past_state_lag', '5', '--dt', '0.01',
          '--ade_loss_weight', '0.5', '--fde_loss_weight', '0.25', '--num_workers', '0',
          '--input_state', 'kinematics', '--output_mode', 'physics_anchor',
          '--loss_mode', 'physical_composite', '--select_metric', 'val_rmse_m',
          '--residual_gate', 'none', '--anchor_context', 'fit']
# Long-horizon attention and multiscale runs first, then the proposed method.
MODELS = {
    'Transformer': ['--model', 'Transformer', '--e_layers', '3', '--n_heads', '8'],
    'TimeMixer': ['--model', 'TimeMixer', '--e_layers', '3'],
    'LSTM': ['--model', 'LSTM', '--e_layers', '3'],
    'PASTMixer': ['--model', 'PASTMixer', '--e_layers', '3'],
    'TSMixer': ['--model', 'TSMixer', '--e_layers', '3'],
    'iTransformer': ['--model', 'iTransformer', '--e_layers', '3', '--n_heads', '8'],
}
RECIPES = {
    'R5c': ['--anchor_context', 'fit'],
    'R5': ['--anchor_context', 'none'],
    'R4': ['--anchor_context', 'none', '--output_mode', 'displacement'],
    'R3': ['--anchor_context', 'none', '--input_state', 'position', '--output_mode', 'displacement'],
    'R1': ['--anchor_context', 'none', '--input_state', 'position', '--output_mode', 'absolute', '--loss_mode', 'mse'],
}
ABLATIONS = {
    'pos': ['--input_state', 'position'],
    'mse': ['--loss_mode', 'mse'],
    'rmse': ['--ade_loss_weight', '0', '--fde_loss_weight', '0'],
    'rmseade': ['--fde_loss_weight', '0'],
}
SEEDS = range(2021, 2026)
# Validation-best (min val RMSE) tune configuration per baseline; PASTMixer's is the default (stage main).
TUNED = {'Transformer': (128, 3, 5e-4), 'TimeMixer': (256, 4, 1e-3), 'LSTM': (256, 2, 1e-3),
         'TSMixer': (256, 2, 1e-3), 'iTransformer': (128, 4, 1e-3)}

STRESS = {'n05': {'noise_scale': 0.5}, 'n2': {'noise_scale': 2}, 'n3': {'noise_scale': 3},
          'pd90': {'pd': 0.9}, 'pd70': {'pd': 0.7}, 'pd50': {'pd': 0.5},
          'r2': {'revisit': 2}, 'r5': {'revisit': 5}, 'r10': {'revisit': 10}}


def size_args(width, layers, lr):
    return ['--d_model', str(width), '--d_ff', str(2 * width), '--e_layers', str(layers), '--learning_rate', str(lr)]


def fair_jobs(stage):
    """PASTMixer at its default size and the baselines at their validation-best sizes, R5c, 5 seeds."""
    return [{'name': f'{stage}_R5c_{model}_s{seed}',
             'args': COMMON + MODELS[model] + (size_args(*TUNED[model]) if model in TUNED else []) +
                     (['--dataset_dir', f'{DATA}_rand'] if stage == 'rand' else []) + ['--seed', str(seed)]}
            for model in MODELS for seed in SEEDS]


def jobs(stage, condition=None, source='fair'):
    if stage == 'stress':
        if condition not in STRESS:
            raise ValueError(f'Unknown stress condition: {condition}')
        if source == 'fair':
            trained = [job for job in jobs('main') if job['name'].startswith('main_R5c_PASTMixer_')] + jobs('best')
        elif source == 'rand':
            trained = jobs('rand')
        else:
            raise ValueError(f'Unknown stress source: {source}')
        scaler = f'{DATA}_rand' if source == 'rand' else DATA
        return [{'name': job['name'], 'args': job['args'] + [
            '--is_training', '0', '--dataset_dir', f'{DATA}_stress/{condition}', '--scaler_dataset_dir', scaler]}
                for job in trained]
    if stage == 'rand':
        return fair_jobs('rand')
    if stage == 'win':
        return [{'name': f'win{length}_R5c_{model}_s{seed}',
                 'args': COMMON + MODELS[model] + (size_args(*TUNED[model]) if model in TUNED else []) +
                         ['--seq_len', str(length), '--dataset_dir', f'{DATA}_L{length}', '--seed', str(seed)]}
                for length in (50, 200) for model in MODELS for seed in range(2021, 2024)]
    if stage == 'main':
        variants = [(variant, model, overrides) for variant, overrides in RECIPES.items()
                    for model in MODELS if model != 'iTransformer' or variant == 'R5c']
    elif stage == 'abl':
        variants = [(variant, 'PASTMixer', overrides) for variant, overrides in ABLATIONS.items()]
    elif stage == 'tune':
        variants = [(f'd{width}l{layers}lr{lr_name}', model, size_args(width, layers, lr))
                    for model in MODELS
                    for width, layers, (lr_name, lr) in product(
                        (256, 128), (4, 3, 2), (('1e3', 1e-3), ('5e4', 5e-4)))]
    elif stage == 'best':
        variants = [('R5c', model, size_args(*config)) for model, config in TUNED.items()]
    else:
        raise ValueError(f'Unknown stage: {stage}')
    return [{'name': f'{stage}_{variant}_{model}_s{seed}',
             'args': COMMON + MODELS[model] + overrides + ['--seed', str(seed)]}
            for variant, model, overrides in variants
            for seed in ([2021] if stage == 'tune' else SEEDS)]


if __name__ == '__main__':
    json.dump(jobs(*sys.argv[1:]), sys.stdout, indent=1)
