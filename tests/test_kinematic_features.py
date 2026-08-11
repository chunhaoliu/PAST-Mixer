import unittest

import torch

from models.PASTMixer import Model
from models.kinematic_features import build_kinematic_state


class KinematicFeatureTests(unittest.TestCase):
    def test_state_has_position_velocity_acceleration_channels(self):
        positions = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)
        state = build_kinematic_state(positions, lag=2, dt=0.1)
        self.assertEqual(tuple(state.shape), (1, 10, 9))

    def test_initial_lag_uses_first_valid_difference(self):
        positions = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)
        state = build_kinematic_state(positions, lag=2, dt=1.0)
        expected_velocity = torch.full((1, 3), 3.0)
        self.assertTrue(
            torch.allclose(state[0, :2, 3:6], expected_velocity)
        )
        self.assertTrue(torch.allclose(state[0, :2, 6:9], torch.zeros(2, 3)))

    def test_state_is_causal_at_each_prefix(self):
        base = torch.randn(1, 12, 3)
        changed = base.clone()
        changed[:, 9:, :] += 1000.0
        base_state = build_kinematic_state(base, lag=2, dt=0.1)
        changed_state = build_kinematic_state(changed, lag=2, dt=0.1)
        self.assertTrue(torch.allclose(base_state[:, :9], changed_state[:, :9]))

    def test_past_state_mode_changes_encoder_input_width(self):
        common = dict(
            task_name='ballistic_forecast',
            seq_len=10,
            pred_len=5,
            enc_in=3,
            c_out=3,
            d_model=16,
            d_ff=32,
            e_layers=1,
            dropout=0.0,
            past_temporal_mode='simple',
            past_norm_mode='pre',
            past_output_mode='displacement',
            past_state_lag=2,
            dt=0.1,
            use_physics=0,
            physics_mode='none',
            physics_gate_mode='scalar',
            physics_gate_max=1.0,
            physics_gate_delta_max=0.2,
            physics_fusion_mode='additive',
        )
        kinematic = Model(type('Config', (), {**common, 'past_state_mode': 'kinematics'}))
        position = Model(type('Config', (), {**common, 'past_state_mode': 'position'}))
        self.assertEqual(kinematic.kinematic_state_encoder[0].in_features, 9)
        self.assertEqual(position.kinematic_state_encoder[0].in_features, 3)


if __name__ == '__main__':
    unittest.main()
