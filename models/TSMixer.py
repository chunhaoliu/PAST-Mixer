"""
Standard TSMixer baseline for direct trajectory forecasting.
"""
import torch
import torch.nn as nn


class TemporalMixing(nn.Module):
    def __init__(self, seq_len, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ChannelMixing(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.temporal = TemporalMixing(seq_len, dropout)
        self.channel = ChannelMixing(d_model, d_ff, dropout)

    def forward(self, x):
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = self.channel(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.embedding = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
        )
        self.blocks = nn.ModuleList([
            MixerBlock(configs.seq_len, configs.d_model, configs.d_ff, configs.dropout)
            for _ in range(configs.e_layers)
        ])
        self.temporal_projection = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.pred_len),
            nn.Dropout(configs.dropout),
        )
        self.output_projection = nn.Linear(configs.d_model, configs.c_out)

    def forecast(self, x_enc):
        x = self.embedding(x_enc)
        for block in self.blocks:
            x = block(x)
        x = self.temporal_projection(x.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(x)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name not in ('ballistic_forecast', 'long_term_forecast', 'short_term_forecast'):
            raise ValueError(f'Task {self.task_name} not supported')
        return self.forecast(x_enc)
