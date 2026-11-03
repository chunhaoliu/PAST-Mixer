"""
TimeMixer-style multiscale forecasting baseline for ballistic trajectories.

This implementation follows the core comparison intent of TimeMixer-like
forecasting: mix temporal patterns at several downsampled resolutions and fuse
them for a direct non-autoregressive prediction horizon.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleMixerBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout),
        )
        self.channel = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm_t(x + self.temporal(x.transpose(1, 2)).transpose(1, 2))
        x = self.norm_c(x + self.channel(x))
        return x


class MultiScaleBranch(nn.Module):
    def __init__(self, input_len, pred_len, enc_in, c_out, d_model, d_ff, e_layers, dropout):
        super().__init__()
        self.input_len = input_len
        self.embedding = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            ScaleMixerBlock(input_len, d_model, d_ff, dropout)
            for _ in range(e_layers)
        ])
        self.temporal_projection = nn.Sequential(
            nn.Linear(input_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len),
            nn.Dropout(dropout),
        )
        self.output_projection = nn.Linear(d_model, c_out)

    def forward(self, x):
        z = self.embedding(x)
        for block in self.blocks:
            z = block(z)
        z = self.temporal_projection(z.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(z)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scales = [1, 2, 4]

        self.branches = nn.ModuleList()
        for scale in self.scales:
            input_len = math.ceil(configs.seq_len / scale)
            self.branches.append(MultiScaleBranch(
                input_len=input_len,
                pred_len=configs.pred_len,
                enc_in=configs.enc_in,
                c_out=configs.c_out,
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                e_layers=configs.e_layers,
                dropout=configs.dropout,
            ))
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))

    def _downsample(self, x, scale):
        if scale == 1:
            return x
        # Average-pool along time while padding only at the sequence tail.
        x_t = x.transpose(1, 2)
        pad = (scale - x_t.shape[-1] % scale) % scale
        if pad:
            x_t = F.pad(x_t, (0, pad), mode='replicate')
        pooled = F.avg_pool1d(x_t, kernel_size=scale, stride=scale)
        return pooled.transpose(1, 2)

    def forecast(self, x_enc):
        weights = torch.softmax(self.scale_logits, dim=0)
        outputs = []
        for weight, scale, branch in zip(weights, self.scales, self.branches):
            x_scale = self._downsample(x_enc, scale)
            outputs.append(weight * branch(x_scale))
        return torch.stack(outputs, dim=0).sum(dim=0)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name not in ('ballistic_forecast', 'long_term_forecast', 'short_term_forecast'):
            raise ValueError(f'Task {self.task_name} not supported')
        return self.forecast(x_enc)
