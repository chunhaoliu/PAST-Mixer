"""
Encoder-only Transformer baseline for direct trajectory forecasting.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.input_proj = nn.Linear(configs.enc_in, configs.d_model)
        self.position = nn.Parameter(torch.randn(1, configs.seq_len, configs.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)
        self.temporal_projection = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.pred_len),
            nn.Dropout(configs.dropout),
        )
        self.output_projection = nn.Linear(configs.d_model, configs.c_out)

    def forecast(self, x_enc):
        x = self.input_proj(x_enc) + self.position
        x = self.encoder(x)
        x = self.temporal_projection(x.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(x)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name not in ('ballistic_forecast', 'long_term_forecast', 'short_term_forecast'):
            raise ValueError(f'Task {self.task_name} not supported')
        return self.forecast(x_enc)
