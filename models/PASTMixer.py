"""PAST-Mixer residual backbone for physics-anchored trajectory prediction.

The paper recipe wraps this backbone with models.io_adapter to add a fitted
drag anchor and seven-channel anchor context. Legacy physics and multiscale
branches remain available for ablations; configs/experiments/main_v3_20261102.py
defines the resubmission settings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.kinematic_features import build_state_features


# ============================================================
# Physics Baseline Branch
# ============================================================

class BallisticParameterEstimator(nn.Module):
    """
    Lightweight MLP that estimates ballistic parameters from observed trajectory.
    Outputs: (v0, theta_h, theta_v, C_drag) — initial velocity magnitude,
    horizontal angle, vertical angle, and effective drag coefficient.
    """
    def __init__(self, obs_len, hidden=64):
        super().__init__()
        # Flatten observed trajectory (obs_len, 3) -> feature vector
        self.encoder = nn.Sequential(
            nn.Linear(obs_len * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # Output heads for each parameter
        self.head_v0 = nn.Linear(hidden, 1)       # muzzle velocity (m/s)
        self.head_th = nn.Linear(hidden, 1)        # azimuth angle (rad)
        self.head_tv = nn.Linear(hidden, 1)        # elevation angle (rad)
        self.head_Cd = nn.Linear(hidden, 1)        # drag coefficient

    def forward(self, x_obs):
        """
        x_obs: (B, obs_len, 3) observed trajectory
        Returns: (v0, theta_h, theta_v, C_drag)
        """
        B = x_obs.shape[0]
        flat = x_obs.reshape(B, -1)
        h = self.encoder(flat)

        v0 = torch.sigmoid(self.head_v0(h)) * 500 + 500   # [500, 1000] m/s
        th = torch.tanh(self.head_th(h)) * 0.2             # [-0.2, 0.2] rad
        tv = torch.sigmoid(self.head_tv(h)) * 1.0 + 0.3   # [0.3, 1.3] rad (~17-75 deg)
        Cd = torch.sigmoid(self.head_Cd(h)) * 0.01 + 0.001 # [0.001, 0.011]

        return v0, th, tv, Cd


class PhysicsBaselineBranch(nn.Module):
    """
    Physics baseline: estimates ballistic parameters from observed trajectory,
    then uses analytical ballistic model to generate baseline trajectory.
    This provides a hard physics prior — the network learns only the residual.
    """
    def __init__(self, obs_len, pred_len, dt=0.01, g=9.81):
        super().__init__()
        self.estimator = BallisticParameterEstimator(obs_len)
        self.pred_len = pred_len
        self.dt = dt
        self.g = g

    def forward(self, x_obs):
        """
        x_obs: (B, obs_len, 3) — observed trajectory (D, Z, H)
        Returns: baseline_trajectory (B, pred_len, 3)
        """
        B, obs_len, _ = x_obs.shape
        device = x_obs.device

        v0, th, tv, Cd = self.estimator(x_obs)

        # Initial velocity components
        vD0 = v0 * torch.cos(tv) * torch.cos(th)
        vZ0 = v0 * torch.cos(tv) * torch.sin(th)
        vH0 = v0 * torch.sin(tv)

        # Use last observed point as initial position
        D0 = x_obs[:, -1, 0:1]
        Z0 = x_obs[:, -1, 1:2]
        H0 = x_obs[:, -1, 2:3]

        # Simple ballistic model (parabolic + linear drag)
        t = torch.arange(self.pred_len, device=device, dtype=torch.float32) * self.dt
        t = t.unsqueeze(0).unsqueeze(-1)  # (1, pred_len, 1)

        # Range: nearly linear with slight drag decay
        D = D0.unsqueeze(1) + vD0.unsqueeze(1) * t * (1.0 - Cd.unsqueeze(1) * t * 0.5)
        # Lateral: linear (no significant lateral force)
        Z = Z0.unsqueeze(1) + vZ0.unsqueeze(1) * t
        # Height: parabolic with gravity
        H = H0.unsqueeze(1) + vH0.unsqueeze(1) * t - 0.5 * self.g * t ** 2
        H = torch.clamp(H, min=0.0)

        baseline = torch.cat([D, Z, H], dim=-1)  # (B, pred_len, 3)
        return baseline


class KinematicBaselineBranch(nn.Module):
    """
    Deterministic quadratic baseline fitted to the observed trajectory.
    This provides a stable physics-inspired prior without a separately trained
    parameter estimator: fit x(t) = a t^2 + b t + c on the observed window and
    extrapolate it to the prediction horizon.
    """
    def __init__(self, obs_len, pred_len, dt=0.01):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.dt = dt

        obs_t = torch.arange(obs_len, dtype=torch.float32) * dt
        pred_t = torch.arange(obs_len, obs_len + pred_len, dtype=torch.float32) * dt

        obs_basis = torch.stack([obs_t ** 2, obs_t, torch.ones_like(obs_t)], dim=1)
        pred_basis = torch.stack([pred_t ** 2, pred_t, torch.ones_like(pred_t)], dim=1)

        self.register_buffer('obs_pinv', torch.linalg.pinv(obs_basis).float())
        self.register_buffer('pred_basis', pred_basis.float())

    def forward(self, x_obs):
        coeffs = torch.einsum('kl,blc->bkc', self.obs_pinv, x_obs)
        baseline = torch.einsum('pk,bkc->bpc', self.pred_basis, coeffs)
        baseline[..., 2] = torch.clamp(baseline[..., 2], min=0.0)
        return baseline


class AdaptivePhysicsGate(nn.Module):
    """
    Sample-adaptive gate for the deterministic physics baseline.
    It predicts one gate per output coordinate from the observed trajectory, so
    the model can rely on the kinematic prior more in compatible regimes and
    relax it when the observed dynamics suggest a mismatch.
    """
    def __init__(self, obs_len, enc_in, hidden=64, init_bias=-2.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_len * enc_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, enc_in)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, init_bias)

    def forward(self, x_obs):
        batch = x_obs.shape[0]
        gate = torch.sigmoid(self.head(self.encoder(x_obs.reshape(batch, -1))))
        return gate.unsqueeze(1)


class PhysicsCompatibilityGate(nn.Module):
    '''Legacy scalar compatibility gate with sample-dependent output.'''
    def __init__(self, obs_len, enc_in, hidden=64, init_bias=-2.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_len * enc_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, 1)
        nn.init.constant_(self.head.bias, init_bias)

    def forward(self, x_obs):
        batch = x_obs.shape[0]
        logits = self.head(self.encoder(x_obs.reshape(batch, -1)))
        return torch.sigmoid(logits).unsqueeze(1)


class BoundedPhysicsCompatibilityGate(nn.Module):
    '''Legacy gate constrained to a narrow, audited logit envelope.'''
    def __init__(
        self,
        obs_len,
        enc_in,
        hidden=64,
        lower_logit=-3.8,
        upper_logit=-2.2,
    ):
        super().__init__()
        if lower_logit >= upper_logit:
            raise ValueError('lower_logit must be smaller than upper_logit')
        self.lower_logit = float(lower_logit)
        self.upper_logit = float(upper_logit)
        self.encoder = nn.Sequential(
            nn.Linear(obs_len * enc_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_obs):
        batch = x_obs.shape[0]
        feature = self.encoder(x_obs.reshape(batch, -1))
        fraction = torch.sigmoid(self.head(feature))
        logits = self.lower_logit + (
            self.upper_logit - self.lower_logit
        ) * fraction
        return torch.sigmoid(logits).unsqueeze(1)


class ConservativeAdaptivePhysicsGate(nn.Module):
    """
    Bounded sample-adaptive correction around the stable scalar physics gate.
    The initial output exactly matches the scalar prior, and training can only
    move each coordinate by a small bounded amount unless the global scalar
    prior itself changes.
    """
    def __init__(self, obs_len, enc_in, hidden=64, init_bias=-2.0, delta_max=0.2):
        super().__init__()
        self.enc_in = enc_in
        self.delta_max = float(delta_max)
        self.logit_base = nn.Parameter(torch.tensor(float(init_bias)))
        self.encoder = nn.Sequential(
            nn.Linear(obs_len * enc_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, enc_in)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_obs):
        batch = x_obs.shape[0]
        base = torch.sigmoid(self.logit_base).view(1, 1, 1)
        base = base.expand(batch, 1, self.enc_in)
        delta = self.delta_max * torch.tanh(self.head(self.encoder(x_obs.reshape(batch, -1))))
        gate = torch.clamp(base + delta.unsqueeze(1), min=0.0, max=1.0)
        return gate


# ============================================================
# Multi-Scale Temporal Mixing
# ============================================================

class TrajectoryTemporalMixing(nn.Module):
    """
    Multi-scale temporal mixing via dilated depthwise convolutions.
    Dilations [1,3,7,15] cover 3 to 31 time steps (0.03s to 0.31s at dt=0.01),
    capturing ballistic dynamics from rapid acceleration to gradual deceleration.
    """
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        dilations = [1, 3, 7, 15]
        n_branches = len(dilations)

        # Dilated depthwise conv branches
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, padding=d, dilation=d, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, 3, padding=d, dilation=d, groups=d_model),
            )
            for d in dilations
        ])

        # Per-scale gates
        self.gates = nn.ModuleList([
            nn.Conv1d(d_model, d_model, 3, padding=d, dilation=d, groups=d_model)
            for d in dilations
        ])

        self.dropout = nn.Dropout(dropout)
        self.fuse = nn.Conv1d(n_branches * d_model, d_model, kernel_size=1)
        self.scale_weights = nn.Parameter(torch.zeros(n_branches))

        # Global MLP for long-range dependency
        hidden = max(d_model // 2, 64)
        self.global_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.GELU(),
            nn.Linear(hidden, seq_len), nn.Dropout(dropout))
        self.mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, D, L) where D = d_model, L = seq_len
        outs = []
        for conv, gate in zip(self.convs, self.gates):
            feat = conv(x)
            g = torch.sigmoid(gate(x))
            outs.append(self.dropout(feat * g))

        # Learnable scale fusion
        w = F.softmax(self.scale_weights, dim=0)
        blended = sum(w[i] * outs[i] for i in range(len(outs)))

        # Conv fusion
        fused = self.fuse(torch.cat(outs, dim=1))

        # Global MLP with gate
        alpha = torch.sigmoid(self.mlp_gate)
        global_out = alpha * self.global_mlp(x)

        return blended + fused + global_out + x  # residual


class SimpleTemporalMixing(nn.Module):
    """
    TSMixer-style temporal MLP used for controlled PAST-Mixer ablations.
    The surrounding PASTMixerBlock keeps the same channel mixer and residual
    structure, so this isolates the multi-scale temporal mixer contribution.
    """
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


# ============================================================
# Physics-Partitioned Channel Mixing
# ============================================================

class PhysicsPartitionedChannelMixing(nn.Module):
    """
    Channel mixing with physics-aware grouping on embedded features.
    - Horizontal group: first d_model//3 dims for D-Z features
    - Vertical group: middle d_model//3 dims for H features
    - Cross-group: all d_model dims for D-Z-H coupling
    Uses independent MLPs per group, then gates between per-group and cross-group.
    """
    def __init__(self, enc_in, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.enc_in = enc_in

        # Feature bottleneck: enc_in → d_model → enc_in
        self.channel_mlp = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, enc_in),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, L, enc_in) — raw trajectory features
        return x + self.channel_mlp(x)


# ============================================================
# PAST-Mixer Block
# ============================================================

class ChannelMixing(nn.Module):
    """Channel mixing on embedded features: d_model → d_ff → d_model."""
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
        # x: (B, L, d_model)
        return x + self.mlp(x)


class PASTMixerBlock(nn.Module):
    """PAST-Mixer block: Multi-Scale Temporal → Channel."""
    def __init__(self, configs):
        super().__init__()
        temporal_mode = str(getattr(configs, 'past_temporal_mode', 'multiscale')).lower()
        if temporal_mode == 'simple':
            self.temporal = SimpleTemporalMixing(
                configs.d_model, configs.seq_len, configs.dropout)
        else:
            self.temporal = TrajectoryTemporalMixing(
                configs.d_model, configs.seq_len, configs.dropout)
        self.channel = ChannelMixing(
            configs.d_model, configs.d_ff, configs.dropout)
        self.use_pre_norm = (
            str(getattr(configs, 'past_norm_mode', 'none')).lower() == 'pre'
        )
        self.temporal_norm = (
            nn.LayerNorm(configs.d_model)
            if self.use_pre_norm else nn.Identity()
        )
        self.channel_norm = (
            nn.LayerNorm(configs.d_model)
            if self.use_pre_norm else nn.Identity()
        )

    def forward(self, x):
        # x: (B, L, d_model)
        # Temporal: transpose → (B, d_model, L) → mix → back to (B, L, d_model)
        temporal_input = self.temporal_norm(x)
        temporal_output = self.temporal(
            temporal_input.transpose(1, 2)
        ).transpose(1, 2)
        x = x + (temporal_output - temporal_input)

        channel_input = self.channel_norm(x)
        channel_output = self.channel(channel_input)
        return x + (channel_output - channel_input)


# ============================================================
# PAST-Mixer Model (TSLib-compatible)
# ============================================================

class Model(nn.Module):
    """
    PAST-Mixer: Physics-Aware Spatio-Temporal Mixer.
    Predicts ballistic trajectories by combining an optional quadratic
    kinematic baseline with a learned temporal/channel residual. The selected
    manuscript model uses the simple temporal mixer and a learned scalar prior
    weight; multi-scale and adaptive-gate branches are retained as ablations.

    Architecture:
        Input(obs) → PhysicsBranch → baseline_trajectory
        Input(obs) → Embed → N×PASTMixerBlock → Project → residual
        Output = baseline + residual
    """
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
        self.past_state_mode = str(
            getattr(configs, 'past_state_mode', 'position')
        ).lower()
        self.past_state_lag = int(getattr(configs, 'past_state_lag', 5))
        self.past_norm_mode = str(
            getattr(configs, 'past_norm_mode', 'none')
        ).lower()
        self.past_output_mode = str(
            getattr(configs, 'past_output_mode', 'absolute')
        ).lower()
        self.physics_fusion_mode = str(
            getattr(configs, 'physics_fusion_mode', 'additive')
        ).lower()
        if self.past_state_mode not in {'position', 'kinematics'}:
            raise ValueError(
                f'Unsupported past_state_mode={self.past_state_mode}'
            )
        if self.past_norm_mode not in {'none', 'pre'}:
            raise ValueError(
                f'Unsupported past_norm_mode={self.past_norm_mode}'
            )
        if self.past_output_mode not in {'absolute', 'displacement'}:
            raise ValueError(
                f'Unsupported past_output_mode={self.past_output_mode}'
            )
        self.state_dim = 3 if self.past_state_mode == 'position' else 9
        self.context_channels = int(getattr(configs, 'context_channels', 0))
        self.state_dim += self.context_channels

        # Physics baseline branch (optional)
        self.use_physics = bool(getattr(configs, 'use_physics', False))
        physics_mode = str(getattr(configs, 'physics_mode', 'none')).lower()
        if physics_mode == 'none' and self.use_physics:
            physics_mode = 'learned'
        self.physics_mode = physics_mode

        if physics_mode == 'learned':
            self.physics_branch = PhysicsBaselineBranch(
                obs_len=configs.seq_len,
                pred_len=configs.pred_len,
            )
        elif physics_mode == 'kinematic':
            self.physics_branch = KinematicBaselineBranch(
                obs_len=configs.seq_len,
                pred_len=configs.pred_len,
            )
        else:
            self.physics_branch = None

        if self.physics_branch is not None:
            self.physics_gate_mode = str(getattr(configs, 'physics_gate_mode', 'scalar')).lower()
            self.physics_gate_max = float(getattr(configs, 'physics_gate_max', 1.0))
            physics_gate_delta_max = float(getattr(configs, 'physics_gate_delta_max', 0.2))
            if self.physics_gate_mode == 'adaptive':
                self.physics_gate = AdaptivePhysicsGate(
                    obs_len=configs.seq_len,
                    enc_in=configs.enc_in,
                    hidden=max(32, min(128, configs.d_model // 2)),
                )
            elif self.physics_gate_mode == 'conservative_adaptive':
                self.physics_gate = ConservativeAdaptivePhysicsGate(
                    obs_len=configs.seq_len,
                    enc_in=configs.enc_in,
                    hidden=max(32, min(128, configs.d_model // 2)),
                    delta_max=physics_gate_delta_max,
                )
            else:
                self.physics_gate_mode = 'scalar'
                self.physics_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            self.physics_gate_mode = 'none'
            self.physics_gate_max = 0.0
            self.physics_gate = None

        # Input embedding: raw (D,Z,H) → d_model
        self.state_norm = nn.Identity()
        self.kinematic_state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
        )
        self.embedding = self.kinematic_state_encoder

        # PAST-Mixer blocks
        self.blocks = nn.ModuleList([
            PASTMixerBlock(configs) for _ in range(configs.e_layers)
        ])
        self.final_norm = (
            nn.LayerNorm(configs.d_model)
            if self.past_norm_mode == 'pre' else nn.Identity()
        )

        # Output projection: d_model → enc_in for residual
        self.residual_proj = nn.Linear(configs.d_model, configs.enc_in)

        # Projection: seq_len → pred_len
        self.projection = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.pred_len),
            nn.Dropout(configs.dropout),
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, context=None):
        """
        TSLib forecast interface.

        Args:
            x_enc: (B, seq_len, enc_in) — observed trajectory
            x_mark_enc: unused (for TSLib compatibility)
            x_dec: unused
            x_mark_dec: unused
        Returns:
            pred: (B, pred_len, enc_in) — predicted future trajectory
        """
        B = x_enc.shape[0]

        # Physics baseline
        if self.physics_branch is not None:
            raw_baseline = self.physics_branch(x_enc)  # (B, pred_len, 3)
            if self.physics_gate_mode in {'adaptive', 'conservative_adaptive'}:
                baseline_scale = self.physics_gate(x_enc)  # (B, 1, 3)
            else:
                baseline_scale = torch.sigmoid(self.physics_gate)
            baseline_scale = self.physics_gate_max * baseline_scale
            baseline = baseline_scale * raw_baseline
        else:
            baseline = 0.0

        # Neural residual path uses only the observed prefix.
        state = build_state_features(
            x_enc,
            mode=self.past_state_mode,
            lag=self.past_state_lag,
            dt=self.dt,
        )
        if self.context_channels:
            if context is None or context.shape != (*x_enc.shape[:2], self.context_channels):
                raise ValueError('Missing or incorrectly shaped PASTMixer context')
            state = torch.cat([state, context], dim=-1)
        state = self.state_norm(state)
        x = self.kinematic_state_encoder(state)

        # PAST-Mixer blocks
        for block in self.blocks:
            x = block(x)  # (B, seq_len, d_model)

        # Project to prediction horizon and map to 3 channels
        x = self.final_norm(x)
        residual = self.projection(x.transpose(1, 2)).transpose(1, 2)  # (B, pred_len, d_model)
        residual = self.residual_proj(residual)  # (B, pred_len, 3)

        # Final output = optional physics baseline plus learned residual.
        output = baseline + residual
        if self.physics_branch is None and self.past_output_mode == 'displacement':
            output = output + x_enc[:, -1:, :]

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, context=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast',
                              'ballistic_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, context=context)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError(f'Task {self.task_name} not supported')
