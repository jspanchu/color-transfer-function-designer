import numpy as np
import pytest

from color_transfer_function_designer.lib.model import TransferFunctionNet
from color_transfer_function_designer.lib.transfer import (
    convert_lut_to_state_format,
    device,
    lut_from_network,
    transfer_reference_lut,
)


def test_convert_lut_to_state_format():
    lut_rgb = np.array([[0.0, 1.0, 0.0, 0.0], [3.0, 0.0, 0.0, 1.0]])
    lut_scalar_alpha = np.array([[0.0, 0.0], [3.0, 1.0]])
    lut_gradient_alpha = np.array([[0.0, 0.0], [1.0, 0.8]])
    colors, opacities, gradient = convert_lut_to_state_format(
        lut_rgb, lut_scalar_alpha, lut_gradient_alpha
    )
    assert colors == [(0.0, (1.0, 0.0, 0.0)), (3.0, (0.0, 0.0, 1.0))]
    assert opacities == [[0.0, 0.0], [3.0, 1.0]]
    assert gradient == [[0.0, 0.0], [1.0, 0.8]]


def test_lut_from_network(tgt_volume):
    model = TransferFunctionNet()
    colors, opacities, gradient_opacities = lut_from_network(
        model, tgt_volume, n_points=8
    )
    assert len(colors) == len(opacities) == len(gradient_opacities) == 8
    smin, smax = tgt_volume.scalar_range
    assert colors[0][0] == pytest.approx(smin)
    assert colors[-1][0] == pytest.approx(smax)
    # each color is (scalar, (r, g, b)) of floats
    scalar, rgb = colors[0]
    assert isinstance(scalar, float)
    assert all(isinstance(c, float) for c in rgb)


def test_transfer_reference_lut_with_callback(
    ref_volume, tgt_volume, lut_rgb, lut_scalar_alpha, lut_gradient_alpha
):
    calls = []
    model = transfer_reference_lut(
        TransferFunctionNet().to(device),
        tgt_volume,
        ref_volume,
        lut_rgb,
        lut_scalar_alpha,
        lut_gradient_alpha,
        n_epochs=1,
        n_slices=8,
        batch_size=2,
        progress_callback=lambda *a: calls.append(a),
    )
    assert isinstance(model, TransferFunctionNet)
    assert not model.training  # left in eval mode
    assert calls  # callback fired at least once
    epoch, _batch, progress, _loss = calls[0]
    assert epoch == 1
    assert 0.0 <= progress <= 100.0 + 1e-6


def test_transfer_reference_lut_without_callback(
    ref_volume, tgt_volume, lut_rgb, lut_scalar_alpha, lut_gradient_alpha
):
    model = transfer_reference_lut(
        TransferFunctionNet().to(device),
        tgt_volume,
        ref_volume,
        lut_rgb,
        lut_scalar_alpha,
        lut_gradient_alpha,
        n_epochs=1,
        n_slices=4,
        batch_size=2,
        progress_callback=None,
    )
    assert isinstance(model, TransferFunctionNet)


def test_transfer_reference_lut_min_mismatch_raises(
    ref_volume, tgt_volume, lut_scalar_alpha, lut_gradient_alpha
):
    bad_rgb = np.array(
        [[1.0, 1.0, 0.0, 0.0], [3.0, 0.5, 0.5, 0.5]], dtype=np.float32
    )  # min scalar 1.0 != ref min 0.0
    with pytest.raises(AssertionError):
        transfer_reference_lut(
            TransferFunctionNet(),
            tgt_volume,
            ref_volume,
            bad_rgb,
            lut_scalar_alpha,
            lut_gradient_alpha,
            n_epochs=1,
            n_slices=4,
            batch_size=2,
            progress_callback=None,
        )


def test_transfer_reference_lut_max_mismatch_raises(
    ref_volume, tgt_volume, lut_gradient_alpha
):
    lut_rgb = np.array([[0.0, 1.0, 0.0, 0.0], [3.0, 0.5, 0.5, 0.5]], dtype=np.float32)
    bad_scalar_alpha = np.array(
        [[0.0, 0.0], [5.0, 1.0]], dtype=np.float32
    )  # max 5.0 != ref max 3.0
    with pytest.raises(AssertionError):
        transfer_reference_lut(
            TransferFunctionNet(),
            tgt_volume,
            ref_volume,
            lut_rgb,
            bad_scalar_alpha,
            lut_gradient_alpha,
            n_epochs=1,
            n_slices=4,
            batch_size=2,
            progress_callback=None,
        )
