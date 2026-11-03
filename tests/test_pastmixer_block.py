import os
import sys
import types
import unittest

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.PASTMixer import (
    AdaptivePhysicsGate,
    KinematicBaselineBranch,
    Model as PASTMixerModel,
    PASTMixerBlock,
    SimpleTemporalMixing,
)


class EchoTemporal(nn.Module):
    def forward(self, x):
        return x


class PASTMixerBlockResidualTests(unittest.TestCase):
    def test_temporal_stage_does_not_apply_duplicate_residual(self):
        configs = types.SimpleNamespace(
            d_model=8,
            seq_len=5,
            d_ff=16,
            dropout=0.0,
        )
        block = PASTMixerBlock(configs)
        block.temporal = EchoTemporal()
        block.channel = nn.Identity()

        x = torch.randn(2, configs.seq_len, configs.d_model)
        y = block(x)

        self.assertTrue(
            torch.allclose(y, x),
            'Temporal mixing stage should contribute one residual path, not duplicate the input.',
        )


class PASTMixerAblationTests(unittest.TestCase):
    def test_simple_temporal_ablation_keeps_shape(self):
        configs = types.SimpleNamespace(
            d_model=8,
            seq_len=5,
            d_ff=16,
            dropout=0.0,
            past_temporal_mode='simple',
        )
        block = PASTMixerBlock(configs)
        x = torch.randn(2, configs.seq_len, configs.d_model)
        y = block(x)

        self.assertIsInstance(block.temporal, SimpleTemporalMixing)
        self.assertEqual(y.shape, x.shape)

    def test_pre_norm_adds_temporal_channel_and_output_normalization(self):
        configs = types.SimpleNamespace(
            d_model=8,
            seq_len=5,
            d_ff=16,
            dropout=0.0,
            past_temporal_mode='simple',
            past_norm_mode='pre',
        )
        block = PASTMixerBlock(configs)
        self.assertIsInstance(block.temporal_norm, nn.LayerNorm)
        self.assertIsInstance(block.channel_norm, nn.LayerNorm)

        model_configs = types.SimpleNamespace(
            task_name='ballistic_forecast',
            seq_len=5,
            pred_len=4,
            enc_in=3,
            c_out=3,
            d_model=8,
            d_ff=16,
            e_layers=1,
            dropout=0.0,
            past_temporal_mode='simple',
            past_norm_mode='pre',
            past_state_mode='kinematics',
            past_state_lag=2,
            past_output_mode='displacement',
            dt=0.1,
            use_physics=0,
            physics_mode='none',
        )
        model = PASTMixerModel(model_configs)
        self.assertIsInstance(model.final_norm, nn.LayerNorm)


class KinematicBaselineBranchTests(unittest.TestCase):
    def test_quadratic_trajectory_is_extrapolated_exactly(self):
        branch = KinematicBaselineBranch(obs_len=6, pred_len=4, dt=1.0)

        t_obs = torch.arange(6, dtype=torch.float32)
        t_pred = torch.arange(6, 10, dtype=torch.float32)

        def build_traj(t):
            d = 1.0 + 2.0 * t + 0.5 * t ** 2
            z = -3.0 + 0.25 * t + 0.1 * t ** 2
            h = 5.0 + 4.0 * t - 0.2 * t ** 2
            return torch.stack([d, z, h], dim=-1)

        x_obs = build_traj(t_obs).unsqueeze(0)
        expected = build_traj(t_pred).unsqueeze(0)

        pred = branch(x_obs)

        self.assertTrue(
            torch.allclose(pred, expected, atol=1e-4),
            'Kinematic baseline should exactly reproduce constant-acceleration trajectories.',
        )


class AdaptivePhysicsGateTests(unittest.TestCase):
    def test_adaptive_gate_is_sample_and_axis_specific(self):
        gate = AdaptivePhysicsGate(obs_len=6, enc_in=3, hidden=8)
        x_obs = torch.randn(4, 6, 3)

        weights = gate(x_obs)

        self.assertEqual(weights.shape, (4, 1, 3))
        self.assertTrue(torch.all(weights >= 0.0))
        self.assertTrue(torch.all(weights <= 1.0))

    def test_pastmixer_adaptive_gate_keeps_forecast_shape(self):
        configs = types.SimpleNamespace(
            task_name='ballistic_forecast',
            seq_len=6,
            pred_len=4,
            enc_in=3,
            d_model=8,
            e_layers=1,
            dropout=0.0,
            d_ff=16,
            use_physics=1,
            physics_mode='kinematic',
            physics_gate_mode='adaptive',
            physics_gate_max=0.3,
            past_temporal_mode='simple',
        )
        model = PASTMixerModel(configs)
        x_obs = torch.randn(2, configs.seq_len, configs.enc_in)

        pred = model(x_obs, None, None, None)

        self.assertEqual(model.physics_gate_mode, 'adaptive')
        self.assertEqual(model.physics_gate_max, 0.3)
        self.assertEqual(pred.shape, (2, configs.pred_len, configs.enc_in))

    def test_conservative_adaptive_gate_starts_from_scalar_prior(self):
        configs = types.SimpleNamespace(
            task_name='ballistic_forecast',
            seq_len=6,
            pred_len=4,
            enc_in=3,
            d_model=8,
            e_layers=1,
            dropout=0.0,
            d_ff=16,
            use_physics=1,
            physics_mode='kinematic',
            physics_gate_mode='conservative_adaptive',
            physics_gate_max=1.0,
            physics_gate_delta_max=0.2,
            past_temporal_mode='simple',
        )
        model = PASTMixerModel(configs)
        x_obs = torch.randn(2, configs.seq_len, configs.enc_in)

        weights = model.physics_gate(x_obs)
        expected = torch.full_like(weights, torch.sigmoid(torch.tensor(-2.0)))

        self.assertEqual(model.physics_gate_mode, 'conservative_adaptive')
        self.assertEqual(weights.shape, (2, 1, configs.enc_in))
        self.assertTrue(torch.allclose(weights, expected, atol=1e-6))
        self.assertEqual(model(x_obs, None, None, None).shape, (2, configs.pred_len, configs.enc_in))


if __name__ == '__main__':
    unittest.main()
