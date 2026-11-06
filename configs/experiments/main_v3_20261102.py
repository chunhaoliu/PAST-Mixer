"""Protocol v2 job matrices with the RMSE+ADE objective (no impact-step FDE term).

The FDE term supervises each output step only through trajectories that impact
there, which makes predictions jitter after the 5 s minimum lead time. Same
stages and CLI as main_v2_20261102.py; ablation 'fde' restores the old term.
"""
import json
import sys

import main_v2_20261102 as v2

v2.COMMON[v2.COMMON.index('--fde_loss_weight') + 1] = '0'
v2.ABLATIONS = {
    'pos': ['--input_state', 'position'],
    'mse': ['--loss_mode', 'mse'],
    'rmse': ['--ade_loss_weight', '0'],
    'fde': ['--fde_loss_weight', '0.25'],
    'pmse': ['--loss_mode', 'physical_mse'],
}
# Validation-best (min val RMSE) tune configuration per baseline under this objective.
v2.TUNED = {'Transformer': (128, 3, 5e-4), 'TimeMixer': (128, 2, 1e-3), 'LSTM': (256, 2, 1e-3),
            'TSMixer': (256, 3, 5e-4), 'iTransformer': (128, 3, 1e-3)}
if 'protocol_hf' in v2.DATA:  # re-tuned on the hi-fi benchmark (outputs/hf_main, stage tune)
    v2.TUNED = {'Transformer': (128, 3, 1e-3), 'TimeMixer': (128, 2, 1e-3), 'LSTM': (256, 2, 1e-3),
                'TSMixer': (128, 4, 1e-3), 'iTransformer': (256, 2, 5e-4)}
# Leave-one-projectile-class-out: data/protocol_v2_hold/<class> (tools/make_holdout_data.py).
HOLD = ('a155', 'a122', 'r130', 'r227')
HOLD_RUNS = [('R5c', 'PASTMixer'), ('R5c', 'TimeMixer'), ('R5c', 'TSMixer'), ('R5c', 'Transformer'), ('R4', 'PASTMixer')]


def jobs(stage, *rest):
    if stage not in ('hold', 'holdaero'):
        return v2.jobs(stage, *rest)
    # holdaero: same runs on data/protocol_v2_holdaero/<class> (aerodynamically randomized train/val, original test).
    return [{'name': f'{stage}{short}_{recipe}_{model}_s{seed}',
             'args': v2.COMMON + v2.MODELS[model] + v2.RECIPES[recipe] +
                     (v2.size_args(*v2.TUNED[model]) if model in v2.TUNED else []) +
                     ['--dataset_dir', f'{v2.DATA}_{stage}/{short}', '--seed', str(seed)]}
            for short in HOLD for recipe, model in HOLD_RUNS for seed in range(2021, 2024)]

if __name__ == '__main__':
    json.dump(jobs(*sys.argv[1:]), sys.stdout, indent=1)
