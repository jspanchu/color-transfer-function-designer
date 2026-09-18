import torch

from color_transfer_function_designer.lib.losses.ssim import SSIMLoss


def test_ssim_loss_identical_inputs_is_zero():
    # SSIM of an image with itself is 1.0, so 1 - SSIM collapses to 0.
    loss_fn = SSIMLoss(max_channel_value=1.0)
    x = torch.rand(2, 5, 16, 16)  # [N, C, H, W], matching the training path
    loss = loss_fn(x, x)
    assert loss.shape == ()  # scalar
    assert loss.item() < 1e-5


def test_ssim_loss_penalizes_dissimilar_inputs():
    # A structurally different output must score a strictly larger loss than a match.
    loss_fn = SSIMLoss(max_channel_value=1.0)
    x = torch.rand(2, 5, 16, 16)
    y = torch.rand(2, 5, 16, 16)
    assert loss_fn(x, x).item() < loss_fn(x, y).item()


def test_ssim_loss_stays_finite_on_flat_and_anticorrelated_inputs():
    # Regression for two NaN sources: flat regions drove sqrt(variance) negative,
    # and a fractional gamma on the (negative) covariance term made structure**gamma
    # NaN. Both must now stay finite.
    flat = torch.zeros(1, 5, 16, 16)  # zero local variance everywhere
    assert torch.isfinite(SSIMLoss(max_channel_value=1.0)(flat, flat))

    x = torch.rand(1, 5, 16, 16)
    anticorrelated = 1.0 - x  # negative local covariance with x
    loss = SSIMLoss(alpha=2.0, beta=2.0, gamma=0.5, max_channel_value=1.0)(
        x, anticorrelated
    )
    assert torch.isfinite(loss)
