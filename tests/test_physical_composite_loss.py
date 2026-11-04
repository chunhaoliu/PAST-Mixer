import numpy as np
import torch

from exp.exp_ballistic import PhysicalCompositeLoss, PhysicalMSELoss


def test_physical_composite_loss_moves_scale_with_the_criterion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = PhysicalCompositeLoss(
        scale=np.array([2.0, 3.0, 4.0], dtype=np.float32),
    ).to(device)
    prediction = torch.ones((2, 5, 3), device=device)
    target = torch.zeros_like(prediction)

    loss = criterion(prediction, target)

    assert criterion.scale.device.type == device.type
    assert torch.isfinite(loss)
    assert loss.device.type == device.type


def test_physical_mse_loss_weights_coordinates_in_meters():
    criterion = PhysicalMSELoss(scale=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    prediction = torch.ones((2, 4, 3))
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    loss = criterion(prediction, torch.zeros_like(prediction), mask)
    assert torch.isclose(loss, torch.tensor((1.0 + 4.0 + 9.0) / 3))
