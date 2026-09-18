from collections.abc import Callable

import numpy as np
import torch
from vtkmodules.vtkCommonDataModel import vtkImageData

from color_transfer_function_designer.lib.dataset import (
    SharedSlicePlaneDataset,
    compute_gradient_magnitude,
    load_vtk_image_to_tensor,
)
from color_transfer_function_designer.lib.losses.ssim import SSIMLoss
from color_transfer_function_designer.lib.model import (
    TransferFunctionNet,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lut_from_network(
    model: TransferFunctionNet,
    volume: vtkImageData,
    n_points: int,
) -> tuple:
    """
    Samples the provided `model` at `n_points` scalar values uniformly spaced in the scalar range of `volume` and
    returns the corresponding colors and opacities.

    Returns:
        colors:   [(scalar, (r, g, b)), ...]  — format for state.colors
        opacities: [[scalar, alpha], ...]     — format for state.opacities
        gradient_opacities: [[gradient, alpha], ...]     — format for state.gradient_opacities
    Scalars are mapped to actual scalar_range for widget display.
    Gradient magnitudes are normalized to [0, 1] based on the provided gradient_range before being fed to the network.
    """
    scalar_range = volume.scalar_range
    grad_mag_max = compute_gradient_magnitude(
        load_vtk_image_to_tensor(volume), volume.spacing
    ).max()
    gradient_range = (0.0, grad_mag_max)
    device = next(model.parameters()).device
    t = torch.linspace(0.0, 1.0, n_points, device=device)
    grad_t = torch.linspace(0.0, gradient_range[1], n_points, device=device)
    t = torch.stack((t, grad_t), dim=1)  # (N, 2)

    model.eval()
    with torch.no_grad():
        output = model(t).cpu().numpy()  # (N, 4)
        rgba = output[:, :4]  # (N, 4)
        grad_a = output[:, 4]  # (N,) gradient opacity

    scalars = (
        (t[:, 0] * (scalar_range[1] - scalar_range[0]) + scalar_range[0]).cpu().numpy()
    )
    gradients = (
        (t[:, 1] * (gradient_range[1] - gradient_range[0]) + gradient_range[0])
        .cpu()
        .numpy()
    )
    colors = [
        (float(s), (float(r), float(g), float(b)))
        for s, (r, g, b) in zip(scalars, rgba[:, :3], strict=True)
    ]
    opacities = [[float(s), float(a)] for s, a in zip(scalars, rgba[:, 3], strict=True)]
    gradient_opacities = [
        [float(g), float(a)] for g, a in zip(gradients, grad_a, strict=True)
    ]
    return colors, opacities, gradient_opacities


def convert_lut_to_state_format(
    lut_rgb: np.ndarray, lut_scalar_alpha: np.ndarray, lut_gradient_alpha: np.ndarray
) -> tuple[list, list, list]:
    colors = [(float(s), (float(r), float(g), float(b))) for s, r, g, b in lut_rgb]
    opacities = [[float(s), float(a)] for s, a in lut_scalar_alpha]
    gradient_opacities = [[float(g), float(a)] for g, a in lut_gradient_alpha]
    return colors, opacities, gradient_opacities


def transfer_reference_lut(
    model: TransferFunctionNet,
    tgt_volume: vtkImageData,
    ref_volume: vtkImageData,
    lut_rgb: np.ndarray,
    lut_scalar_alpha: np.ndarray,
    lut_gradient_alpha: np.ndarray,
    n_epochs: int,
    n_slices: int,
    batch_size: int,
    ssim_alpha: float = 1.0,
    ssim_beta: float = 1.0,
    ssim_gamma: float = 1.0,
    ssim_gaussian_window_size: tuple[int, int] = (11, 11),
    ssim_gaussian_sigma: tuple[float, float] = (1.5, 1.5),
    blend_factor_l1_vs_ssim: float = 0.2,
    progress_callback: Callable[[int, int, float, float], None] | None = None,
    lr: float = 1e-3,
    margin: float = 0.25,
    crop_size: int | None = None,
):
    """
    Returns a :class:`color_transfer_function_designer.lib.model.TransferFunctionNet` whose weights and biases
    are learned to map `tgt_volume` scalars to color/opacity similar to the manner in which `lut_rgb`, `lut_scalar_alpha`,
    and `lut_gradient_alpha` map `ref_volume` scalars.

    Parameters
    ----------
    tgt_volume : This is another :class:`vtkImageData` containing real valued scalars (ex: MRI intensies)
    ref_volume : This is a :class:`vtkImageData` with a reference mask (ex: discrete numbers - 0, 1, 2, 3, .. etc)
    lut_rgb : (N, 4) [scalar, r, g, b] in scalar range of `ref_volume`.
    lut_scalar_alpha :  (M, 2) [scalar, a] in scalar range of `ref_volume`.
    lut_gradient_alpha :  (M, 2) [gradient, a] in gradient magnitudes range of `ref_volume`.
    n_slices : number of slices to use for initial training
    batch_size : number of slices per batch
    n_epochs : number of training epochs
    lr : learning rate
    margin : margin for safe slice index ranges
    progress_callback : optional callback function for training progress. arguments to callback are epoch_id, batch_id, progress_percent, loss, all indices start at 1, instead of 0.
    """

    assert (
        lut_rgb.min(0)[0] == lut_scalar_alpha.min(0)[0] == ref_volume.scalar_range[0]
    ), "Reference scalar minimum != lut_rgb, or lut_scalar_alpha minimum value"
    assert (
        lut_rgb.max(0)[0] == lut_scalar_alpha.max(0)[0] == ref_volume.scalar_range[1]
    ), "Reference scalar maximum != lut_rgb, or lut_scalar_alpha maximum value"

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
    )
    l1 = torch.nn.L1Loss()
    # Model outputs (sigmoid) and LUT targets are all normalized to [0, 1], so the
    # SSIM dynamic range is 1.0, not the 8-bit-image default of 255.
    one_minus_ssim_score = SSIMLoss(
        alpha=ssim_alpha,
        beta=ssim_beta,
        gamma=ssim_gamma,
        max_channel_value=1.0,
        gaussian_window_size=ssim_gaussian_window_size,
        gaussian_sigma=ssim_gaussian_sigma,
    )
    dataset = SharedSlicePlaneDataset(
        image_known_lut=ref_volume,
        image_unknown_lut=tgt_volume,
        lut_rgb=lut_rgb,
        lut_scalar_alpha=lut_scalar_alpha,
        lut_gradient_alpha=lut_gradient_alpha,
        number_of_slices=n_slices,
        margin=margin,
        crop_size=crop_size,
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    progress = 0.0
    progress_interval = 100.0 / (n_epochs * n_slices / batch_size)
    xi = blend_factor_l1_vs_ssim
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_id, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()
            pred = model(x)  # x: [N, 2, H, W] -> pred: [N, 5, H, W]
            loss = xi * l1(pred, y) + (1 - xi) * one_minus_ssim_score(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if progress_callback is not None:
                progress += progress_interval
                progress_callback(epoch + 1, batch_id + 1, progress, float(loss.item()))
        scheduler.step(epoch_loss / len(data_loader))

    model.eval()
    return model
