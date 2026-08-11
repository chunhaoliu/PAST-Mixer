"""PAST-Mixer mainline used in the accompanying manuscript."""

import torch.nn as nn

from models.kinematic_features import build_state_features


class SimpleTemporalMixing(nn.Module):
    """Mix information along the observed temporal axis."""

    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.temporal(x)


class ChannelMixing(nn.Module):
    """Mix embedded kinematic-state channels at each time step."""

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 2
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.mlp(x)


class PASTMixerBlock(nn.Module):
    """Pre-normalized temporal and channel mixing block."""

    def __init__(self, configs):
        super().__init__()
        self.temporal = SimpleTemporalMixing(configs.d_model, configs.seq_len, configs.dropout)
        self.channel = ChannelMixing(configs.d_model, configs.d_ff, configs.dropout)
        self.use_pre_norm = str(getattr(configs, 'past_norm_mode', 'none')).lower() == 'pre'
        self.temporal_norm = nn.LayerNorm(configs.d_model) if self.use_pre_norm else nn.Identity()
        self.channel_norm = nn.LayerNorm(configs.d_model) if self.use_pre_norm else nn.Identity()

    def forward(self, x):
        temporal_input = self.temporal_norm(x)
        temporal_output = self.temporal(temporal_input.transpose(1, 2)).transpose(1, 2)
        x = x + (temporal_output - temporal_input)
        channel_input = self.channel_norm(x)
        channel_output = self.channel(channel_input)
        return x + (channel_output - channel_input)


class Model(nn.Module):
    """Direct long-horizon predictor from an observed Cartesian prefix."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.n_layers = configs.e_layers
        self.dropout = configs.dropout
        self.dt = float(getattr(configs, 'dt', 0.01))
        self.past_state_mode = str(getattr(configs, 'past_state_mode', 'position')).lower()
        self.past_state_lag = int(getattr(configs, 'past_state_lag', 5))
        self.past_norm_mode = str(getattr(configs, 'past_norm_mode', 'none')).lower()
        self.past_output_mode = str(getattr(configs, 'past_output_mode', 'absolute')).lower()
        if self.past_state_mode not in {'position', 'kinematics'}:
            raise ValueError(f'Unsupported past_state_mode={self.past_state_mode}')
        if self.past_norm_mode not in {'none', 'pre'}:
            raise ValueError(f'Unsupported past_norm_mode={self.past_norm_mode}')
        if self.past_output_mode not in {'absolute', 'displacement'}:
            raise ValueError(f'Unsupported past_output_mode={self.past_output_mode}')

        self.state_dim = 3 if self.past_state_mode == 'position' else 9
        self.use_physics = False
        self.physics_mode = 'none'
        self.physics_branch = None
        self.physics_gate_mode = 'none'
        self.physics_gate_max = 0.0
        self.physics_gate = None
        self.state_norm = nn.Identity()
        self.kinematic_state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
        )
        self.embedding = self.kinematic_state_encoder
        self.blocks = nn.ModuleList(PASTMixerBlock(configs) for _ in range(configs.e_layers))
        self.final_norm = nn.LayerNorm(configs.d_model) if self.past_norm_mode == 'pre' else nn.Identity()
        self.residual_proj = nn.Linear(configs.d_model, configs.enc_in)
        self.projection = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.pred_len),
            nn.Dropout(configs.dropout),
        )

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        state = build_state_features(
            x_enc,
            mode=self.past_state_mode,
            lag=self.past_state_lag,
            dt=self.dt,
        )
        x = self.kinematic_state_encoder(self.state_norm(state))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        residual = self.projection(x.transpose(1, 2)).transpose(1, 2)
        output = self.residual_proj(residual)
        if self.past_output_mode == 'displacement':
            output = output + x_enc[:, -1:, :]
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name not in {'long_term_forecast', 'short_term_forecast', 'ballistic_forecast'}:
            raise ValueError(f'Task {self.task_name} not supported')
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
