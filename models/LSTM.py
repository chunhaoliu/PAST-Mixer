"""
Direct LSTM baseline for trajectory forecasting.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        self.input_proj = nn.Linear(configs.enc_in, configs.d_model)
        # Stacked single-layer LSTMs with nn.Dropout between them: same computation as
        # nn.LSTM(num_layers, dropout), but avoids cuDNN's RNN dropout state, which crashes
        # the process at exit on Windows (0xC0000409).
        self.encoder = nn.ModuleList(
            nn.LSTM(configs.d_model, configs.d_model, batch_first=True) for _ in range(configs.e_layers)
        )
        self.encoder_dropout = nn.Dropout(configs.dropout)
        self.horizon_embedding = nn.Parameter(torch.randn(configs.pred_len, configs.d_model) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.c_out),
        )

    def forecast(self, x_enc):
        x = self.input_proj(x_enc)
        for index, layer in enumerate(self.encoder):
            x, (hidden, _) = layer(self.encoder_dropout(x) if index else x)
        context = hidden[-1]
        horizon = self.horizon_embedding.unsqueeze(0).expand(x_enc.shape[0], -1, -1)
        context = context.unsqueeze(1).expand(-1, self.pred_len, -1)
        return self.decoder(torch.cat([context, horizon], dim=-1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name not in ('ballistic_forecast', 'long_term_forecast', 'short_term_forecast'):
            raise ValueError(f'Task {self.task_name} not supported')
        return self.forecast(x_enc)
