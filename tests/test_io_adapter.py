import argparse

import pytest
import torch

from models import build_model
from models.io_adapter import IOAdapter, resolve_io_modes
from models.kinematic_features import build_state_features


def _configs(model, input_state='model_default', output_mode='model_default', **extra):
    args = argparse.Namespace(
        task_name='ballistic_forecast', model=model, seq_len=20, pred_len=30, label_len=0,
        enc_in=3, dec_in=3, c_out=3, d_model=16, d_ff=32, n_heads=2, e_layers=1, d_layers=1,
        dropout=0.0, alpha=0.9, activation='gelu', dt=0.01,
        use_physics=0, physics_mode='none', physics_gate_mode='scalar', physics_gate_max=1.0,
        physics_gate_delta_max=0.2, past_temporal_mode='simple', past_norm_mode='pre',
        past_output_mode='displacement', past_state_mode='kinematics', past_state_lag=5,
        physics_fusion_mode='additive', freeze_backbone=0, pretrained_backbone_checkpoint='',
        input_state=input_state, output_mode=output_mode,
    )
    for key, value in extra.items():
        setattr(args, key, value)
    return resolve_io_modes(args)


@pytest.mark.parametrize('model', ['Transformer', 'TSMixer', 'LSTM', 'TimeMixer', 'iTransformer'])
def test_default_flags_build_unwrapped_legacy_model(model):
    net = build_model(_configs(model))
    assert not isinstance(net, IOAdapter)


@pytest.mark.parametrize('model', ['Transformer', 'TSMixer', 'LSTM', 'TimeMixer'])
@pytest.mark.parametrize('input_state', ['position', 'kinematics'])
@pytest.mark.parametrize('output_mode', ['absolute', 'displacement'])
def test_adapter_shapes_for_all_supported_baselines(model, input_state, output_mode):
    configs = _configs(model, input_state, output_mode)
    net = build_model(configs).eval()
    x = torch.randn(4, 20, 3)
    with torch.no_grad():
        y = net(x, None, None, None)
    assert y.shape == (4, 30, 3)
    wrapped = isinstance(net, IOAdapter)
    assert wrapped == (input_state == 'kinematics' or output_mode == 'displacement')


@pytest.mark.parametrize('model', ['PASTMixer', 'Transformer', 'TSMixer'])
def test_physics_anchor_output_is_backbone_plus_anchor(model):
    torch.manual_seed(0)
    net = build_model(_configs(model, 'kinematics', 'physics_anchor')).eval()
    assert isinstance(net, IOAdapter)
    x = torch.randn(2, 20, 3)
    anchor = torch.randn(2, 30, 3)
    with torch.no_grad():
        y = net(x, None, None, None, anchor=anchor)
        residual = net.backbone(
            x if model == 'PASTMixer' else build_state_features(x, mode='kinematics', lag=5, dt=0.01),
            None, None, None)
    torch.testing.assert_close(y, residual + anchor)
    if model == 'PASTMixer':
        assert net.backbone.past_output_mode == 'absolute'
        assert net.backbone.past_state_mode == 'kinematics'
    with pytest.raises(ValueError):
        net(x, None, None, None)


def test_displacement_output_is_backbone_plus_last_position():
    torch.manual_seed(0)
    net = build_model(_configs('TSMixer', 'position', 'displacement')).eval()
    x = torch.randn(2, 20, 3)
    with torch.no_grad():
        raw = net.backbone(x, None, None, None)
        out = net(x, None, None, None)
    torch.testing.assert_close(out, raw + x[:, -1:, :])


def test_kinematic_backbone_receives_nine_channels():
    net = build_model(_configs('Transformer', 'kinematics', 'absolute')).eval()
    assert net.backbone.input_proj.in_features == 9
    assert net.backbone.output_projection.out_features == 3
    x = torch.randn(2, 20, 3)
    seen = {}
    def capture(module, inputs):
        seen['x'] = inputs[0]

    net.backbone.register_forward_pre_hook(capture)
    with torch.no_grad():
        net(x, None, None, None)
    torch.testing.assert_close(seen['x'], build_state_features(x, 'kinematics', lag=5, dt=0.01))


def test_pastmixer_uses_native_switches_not_adapter():
    configs = _configs('PASTMixer', 'position', 'absolute')
    assert configs.past_state_mode == 'position'
    assert configs.past_output_mode == 'absolute'
    net = build_model(configs)
    assert not isinstance(net, IOAdapter)
    assert net.state_dim == 3


def test_pastmixer_model_default_keeps_submitted_configuration():
    configs = _configs('PASTMixer')
    assert configs.past_state_mode == 'kinematics'
    assert configs.past_output_mode == 'displacement'


@pytest.mark.parametrize('input_state', ['position', 'kinematics'])
@pytest.mark.parametrize('output_mode', ['absolute', 'displacement', 'physics_anchor'])
def test_itransformer_adapter_shapes(input_state, output_mode):
    net = build_model(_configs('iTransformer', input_state, output_mode)).eval()
    x = torch.randn(2, 20, 3)
    kwargs = {'anchor': torch.randn(2, 30, 3)} if output_mode == 'physics_anchor' else {}
    y = net(x, None, None, None, **kwargs)
    assert y.shape == (2, 30, 3)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    if isinstance(net, IOAdapter):
        assert net.position_readout.weight.grad is not None


ALL_BACKBONES = ['PASTMixer', 'Transformer', 'TSMixer', 'TimeMixer', 'LSTM', 'iTransformer']


@pytest.mark.parametrize('model', ALL_BACKBONES)
@pytest.mark.parametrize('anchor_context', ['none', 'fit'])
def test_rezero_starts_exactly_at_anchor_and_gate_learns(model, anchor_context):
    torch.manual_seed(42)
    net = build_model(_configs(model, 'kinematics', 'physics_anchor',
                               residual_gate='rezero', anchor_context=anchor_context))
    x = torch.randn(2, 20, 3)
    anchor = torch.randn(2, 30, 3)
    context = torch.randn(2, 20, 7) if anchor_context == 'fit' else None
    y = net(x, anchor=anchor, context=context)
    assert net.gate.shape == (1, 1, 3)
    assert torch.equal(y, anchor)
    y.square().mean().backward()
    assert torch.isfinite(net.gate.grad).all()
    assert torch.all(net.gate.grad != 0)


@pytest.mark.parametrize('model', ALL_BACKBONES)
@pytest.mark.parametrize('input_state', ['position', 'kinematics'])
def test_fit_context_is_in_backbone_state_and_affects_prediction(model, input_state):
    net = build_model(_configs(model, input_state, 'physics_anchor', anchor_context='fit')).eval()
    x = torch.randn(2, 20, 3)
    anchor = torch.randn(2, 30, 3)
    context = torch.randn(2, 20, 7, requires_grad=True)
    seen = {}
    def capture(module, inputs):
        seen['state'] = inputs[0]
    module = net.backbone.state_norm if model == 'PASTMixer' else net.backbone
    handle = module.register_forward_pre_hook(capture)
    y = net(x, anchor=anchor, context=context)
    handle.remove()
    state = build_state_features(x, input_state, lag=5, dt=0.01)
    torch.testing.assert_close(seen['state'], torch.cat([state, context], dim=-1))
    assert y.shape == (2, 30, 3)
    y.square().mean().backward()
    assert torch.isfinite(context.grad).all()
    assert context.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='context'):
        net(x, anchor=anchor)


@pytest.mark.parametrize('model', ALL_BACKBONES)
def test_new_flags_default_preserves_state_dict_and_output(model):
    implicit = _configs(model, 'kinematics', 'physics_anchor')
    del implicit.residual_gate, implicit.anchor_context
    explicit = _configs(model, 'kinematics', 'physics_anchor', residual_gate='none', anchor_context='none')
    torch.manual_seed(1)
    old = build_model(implicit).eval()
    torch.manual_seed(1)
    new = build_model(explicit).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    x, anchor = torch.randn(2, 20, 3), torch.randn(2, 30, 3)
    assert torch.equal(old(x, anchor=anchor), new(x, anchor=anchor))


@pytest.mark.parametrize('option', ['residual_gate', 'anchor_context'])
def test_physics_options_reject_non_anchor_mode(option):
    with pytest.raises(ValueError, match='requires output_mode=physics_anchor'):
        _configs('TSMixer', **{option: 'rezero' if option == 'residual_gate' else 'fit'})
