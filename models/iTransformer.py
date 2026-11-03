"""iTransformer baseline adapted to the ballistic forecasting interface.

The model follows the inverted-token design: each position variable is a token
whose feature vector spans the complete observed window. Self-attention mixes
the variable tokens before a direct projection to the full forecast horizon.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Direct multi-step iTransformer for multivariate position sequences."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.token_embedding = nn.Linear(self.seq_len, configs.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation=getattr(configs, "activation", "gelu"),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=configs.e_layers,
            norm=nn.LayerNorm(configs.d_model),
        )
        self.horizon_projection = nn.Linear(configs.d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec

        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = centered / stdev

        # [batch, time, variables] -> [batch, variable tokens, time features]
        tokens = self.token_embedding(normalized.transpose(1, 2))
        encoded = self.encoder(tokens)
        forecast = self.horizon_projection(encoded).transpose(1, 2)

        return forecast * stdev[:, :1, :] + means[:, :1, :]
