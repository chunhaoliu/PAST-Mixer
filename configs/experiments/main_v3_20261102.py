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
}
# Validation-best (min val RMSE) tune configuration per baseline under this objective.
v2.TUNED = {'Transformer': (128, 3, 5e-4), 'TimeMixer': (128, 2, 1e-3), 'LSTM': (256, 2, 1e-3),
            'TSMixer': (256, 3, 5e-4), 'iTransformer': (128, 3, 1e-3)}
jobs = v2.jobs

if __name__ == '__main__':
    json.dump(jobs(*sys.argv[1:]), sys.stdout, indent=1)
