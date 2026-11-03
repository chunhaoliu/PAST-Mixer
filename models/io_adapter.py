"""Model-agnostic input/output adapter for controlled baseline comparisons.

Wraps any 3-channel forecaster so that it can receive the same finite-difference
kinematic descriptors as PAST-Mixer and/or predict displacements anchored at the
last tracked position. PAST-Mixer keeps its own built-in implementation.

``physics_anchor`` predicts a residual on top of a model-based extrapolation
(the drag-LS baseline), supplied per sample as ``anchor``. It applies to every
backbone, PAST-Mixer included (which then keeps its native kinematic input).
"""
import copy

import torch
import torch.nn as nn

from models.kinematic_features import build_state_features

MODEL_DEFAULT = 'model_default'
INPUT_STATES = (MODEL_DEFAULT, 'position', 'kinematics')
OUTPUT_MODES = (MODEL_DEFAULT, 'absolute', 'displacement', 'physics_anchor')

UNSUPPORTED_FOR_KINEMATICS = set()
UNSUPPORTED_FOR_DISPLACEMENT = set()


def resolve_io_modes(args):
    """Map --input_state/--output_mode onto the arguments a model consumes.

    For PASTMixer the generic flags set its native past_* switches, so its
    implementation stays identical to the submitted version. For other models
    the flags are kept for the adapter. 'model_default' leaves behaviour unchanged.
    """
    input_state = str(getattr(args, 'input_state', MODEL_DEFAULT)).lower()
    output_mode = str(getattr(args, 'output_mode', MODEL_DEFAULT)).lower()
    if input_state not in INPUT_STATES:
        raise ValueError(f'input_state must be one of {INPUT_STATES}')
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f'output_mode must be one of {OUTPUT_MODES}')
    args.input_state = input_state
    args.output_mode = output_mode
    for name, choices in (('residual_gate', ('none', 'rezero')), ('anchor_context', ('none', 'fit'))):
        value = str(getattr(args, name, 'none')).lower()
        if value not in choices:
            raise ValueError(f'{name} must be one of {choices}')
        if value != 'none' and output_mode != 'physics_anchor':
            raise ValueError(f'{name} requires output_mode=physics_anchor')
        setattr(args, name, value)
    if str(args.model).lower() == 'pastmixer':
        if input_state != MODEL_DEFAULT:
            args.past_state_mode = input_state
        if output_mode == 'physics_anchor':
            args.past_output_mode = 'absolute'
        elif output_mode != MODEL_DEFAULT:
            args.past_output_mode = output_mode
    return args


def needs_adapter(configs):
    output_mode = str(getattr(configs, 'output_mode', MODEL_DEFAULT)).lower()
    if output_mode == 'physics_anchor':
        return True
    if str(configs.model).lower() == 'pastmixer':
        return False
    input_state = str(getattr(configs, 'input_state', MODEL_DEFAULT)).lower()
    return input_state == 'kinematics' or output_mode == 'displacement'


class IOAdapter(nn.Module):
    """Kinematic input features and/or anchored displacement output around a backbone."""

    def __init__(self, configs, backbone_cls, model_name):
        super().__init__()
        self.input_state = str(getattr(configs, 'input_state', MODEL_DEFAULT)).lower()
        self.output_mode = str(getattr(configs, 'output_mode', MODEL_DEFAULT)).lower()
        self.native_state = model_name == 'PASTMixer'
        self.context_channels = 7 if getattr(configs, 'anchor_context', 'none') == 'fit' else 0
        if model_name == 'PASTMixer':
            # PAST-Mixer builds its kinematic state internally.
            self.input_state = MODEL_DEFAULT
        self.lag = int(getattr(configs, 'past_state_lag', 5))
        self.dt = float(getattr(configs, 'dt', 0.01))
        self.position_channels = int(configs.enc_in)
        self.gate = (
            nn.Parameter(torch.zeros(1, 1, int(configs.c_out)))
            if self.output_mode == 'physics_anchor' and getattr(configs, 'residual_gate', 'none') == 'rezero'
            else None
        )
        backbone_configs = copy.copy(configs)
        if self.input_state == 'kinematics':
            backbone_configs.enc_in = 3 * self.position_channels
            backbone_configs.dec_in = 3 * self.position_channels
        if self.native_state:
            backbone_configs.context_channels = self.context_channels
        elif self.context_channels:
            backbone_configs.enc_in += self.context_channels
            backbone_configs.dec_in = backbone_configs.enc_in
        backbone_configs.c_out = self.position_channels
        self.backbone = backbone_cls(backbone_configs)
        # iTransformer forecasts one sequence per input variate (including
        # kinematics/context). Mix those forecasts into position coordinates.
        self.position_readout = (
            nn.Linear(backbone_configs.enc_in, self.position_channels)
            if model_name == 'iTransformer' else nn.Identity()
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None,
                anchor=None, context=None):
        features = x_enc
        if self.input_state == 'kinematics':
            features = build_state_features(x_enc, mode='kinematics', lag=self.lag, dt=self.dt)
        if self.context_channels:
            if context is None or context.shape != (*x_enc.shape[:2], self.context_channels):
                raise ValueError('anchor_context=fit requires context of shape (batch, seq_len, 7)')
            if not self.native_state:
                features = torch.cat([features, context], dim=-1)
        if self.native_state and self.context_channels:
            output = self.backbone(features, x_mark_enc, x_dec, x_mark_dec, context=context)
        else:
            output = self.backbone(features, x_mark_enc, x_dec, x_mark_dec)
        output = self.position_readout(output)
        if self.output_mode == 'displacement':
            output = output + x_enc[:, -1:, :]
        elif self.output_mode == 'physics_anchor':
            if anchor is None:
                raise ValueError('physics_anchor output requires a per-sample anchor')
            if self.gate is not None:
                output = self.gate * output
            output = output + anchor
        return output
