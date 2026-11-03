from types import SimpleNamespace

import torch
import torch.nn as nn

from models.PASTMixer import Model, PASTMixerBlock


def config(**overrides):
    values = dict(
        task_name='ballistic_forecast', seq_len=10, pred_len=6, enc_in=3,
        d_model=16, d_ff=32, e_layers=2, dropout=0.0, dt=0.1,
        past_state_mode='kinematics', past_state_lag=2,
        past_norm_mode='pre', past_output_mode='displacement',
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_model_shape_and_gradient_flow():
    model = Model(config())
    x = torch.randn(4, 10, 3, requires_grad=True)
    y = model(x, None, None, None)
    assert y.shape == (4, 6, 3)
    y.square().mean().backward()
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_ablation_switches_change_only_documented_interfaces():
    position_model = Model(config(past_state_mode='position'))
    absolute_model = Model(config(past_output_mode='absolute'))
    no_norm_block = PASTMixerBlock(config(past_norm_mode='none'))
    assert position_model.kinematic_state_encoder[0].in_features == 3
    assert absolute_model(torch.randn(2, 10, 3), None, None, None).shape == (2, 6, 3)
    assert isinstance(no_norm_block.temporal_norm, nn.Identity)
