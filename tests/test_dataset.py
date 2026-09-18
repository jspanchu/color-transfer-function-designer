import numpy as np
import pytest
import torch

from color_transfer_function_designer.lib.dataset import (
    SharedSlicePlaneDataset,
    _interp1d,
    apply_lut_torch,
    compute_gradient_magnitude,
    extract_2d_slice,
    load_vtk_image_to_tensor,
    normalize_volume_scalars,
)


def test_load_vtk_image_to_tensor_shape_and_order(make_image):
    # Distinct values let us verify the C-order reshape to (nz, ny, nx).
    arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # (nz, ny, nx)
    image = make_image(arr)
    tensor = load_vtk_image_to_tensor(image)
    assert tensor.shape == (2, 3, 4)
    torch.testing.assert_close(tensor, torch.from_numpy(arr))


def test_normalize_volume_scalars_range():
    out = normalize_volume_scalars(torch.tensor([2.0, 4.0, 6.0]))
    torch.testing.assert_close(out, torch.tensor([0.0, 0.5, 1.0]))


def test_normalize_volume_scalars_constant_volume():
    # Constant input exercises the max(span, 1e-6) guard (no div-by-zero).
    out = normalize_volume_scalars(torch.full((5,), 7.0))
    assert torch.all(out == 0.0)


def test_compute_gradient_magnitude_ramp():
    # Linear ramp along x with unit spacing -> magnitude 1 everywhere.
    ramp = torch.arange(4.0).reshape(1, 1, 4).expand(3, 3, 4).contiguous()
    mag = compute_gradient_magnitude(ramp, spacing=(1.0, 1.0, 1.0))
    torch.testing.assert_close(mag, torch.ones_like(mag))


def test_compute_gradient_magnitude_requires_3d():
    with pytest.raises(AssertionError):
        compute_gradient_magnitude(torch.zeros(4, 4), spacing=(1.0, 1.0))


def test_extract_2d_slice_each_axis():
    vol = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    assert extract_2d_slice(vol, 0, 1).shape == (3, 4)
    assert extract_2d_slice(vol, 1, 2).shape == (2, 4)
    assert extract_2d_slice(vol, 2, 3).shape == (2, 3)
    torch.testing.assert_close(extract_2d_slice(vol, 0, 0), vol[0])


@pytest.mark.parametrize("bad_idx", [4, -5])
def test_extract_2d_slice_out_of_range_raises(bad_idx):
    vol = torch.zeros(2, 3, 4)
    with pytest.raises(IndexError):
        extract_2d_slice(vol, 0, bad_idx)


def test_interp1d_clamps_out_of_range():
    xp = torch.tensor([0.0, 1.0, 2.0])
    fp = torch.tensor([0.0, 10.0, 20.0])
    x = torch.tensor([-1.0, 0.5, 1.5, 3.0])
    out = _interp1d(x, xp, fp)
    # below range clamps to first value, above range clamps to last.
    torch.testing.assert_close(out, torch.tensor([0.0, 5.0, 15.0, 20.0]))


def test_interp1d_duplicate_knots_guard():
    # Duplicate knot positions exercise the (x1 - x0).clamp(min=1e-9) guard.
    xp = torch.tensor([0.0, 1.0, 1.0, 2.0])
    fp = torch.tensor([0.0, 5.0, 5.0, 10.0])
    out = _interp1d(torch.tensor([1.0]), xp, fp)
    assert torch.isfinite(out).all()


def test_apply_lut_torch_shape_and_values():
    scalar_flat = torch.tensor([0.0, 1.5, 3.0])
    gradient_flat = torch.tensor([0.0, 0.5, 1.0])
    lut_rgb = torch.tensor([[0.0, 1.0, 0.0, 0.0], [3.0, 0.0, 0.0, 1.0]])
    lut_scalar_alpha = torch.tensor([[0.0, 0.0], [3.0, 1.0]])
    lut_gradient_alpha = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    out = apply_lut_torch(
        scalar_flat, gradient_flat, lut_rgb, lut_scalar_alpha, lut_gradient_alpha
    )
    assert out.shape == (5, 3)  # (channel, sample): R G B scalar_a grad_a
    # midpoint scalar -> R interpolates halfway between 1.0 and 0.0.
    torch.testing.assert_close(out[0, 1], torch.tensor(0.5))
    torch.testing.assert_close(out[3, :], torch.tensor([0.0, 0.5, 1.0]))


def _dataset(ref, tgt, luts, **kwargs):
    lut_rgb, lut_scalar_alpha, lut_gradient_alpha = luts
    params = {"number_of_slices": 8, "margin": 0.25}
    params.update(kwargs)
    return SharedSlicePlaneDataset(
        image_known_lut=ref,
        image_unknown_lut=tgt,
        lut_rgb=lut_rgb,
        lut_scalar_alpha=lut_scalar_alpha,
        lut_gradient_alpha=lut_gradient_alpha,
        **params,
    )


def test_dataset_len_and_crop_size_default(
    ref_volume, tgt_volume, lut_rgb, lut_scalar_alpha, lut_gradient_alpha
):
    ds = _dataset(
        ref_volume, tgt_volume, (lut_rgb, lut_scalar_alpha, lut_gradient_alpha)
    )
    assert len(ds) == 8
    assert ds._crop_size == min(ref_volume.dimensions)


def test_dataset_explicit_crop_size(
    ref_volume, tgt_volume, lut_rgb, lut_scalar_alpha, lut_gradient_alpha
):
    ds = _dataset(
        ref_volume,
        tgt_volume,
        (lut_rgb, lut_scalar_alpha, lut_gradient_alpha),
        crop_size=4,
    )
    assert ds._crop_size == 4


def test_dataset_getitem_shapes(
    ref_volume, tgt_volume, lut_rgb, lut_scalar_alpha, lut_gradient_alpha
):
    ds = _dataset(
        ref_volume,
        tgt_volume,
        (lut_rgb, lut_scalar_alpha, lut_gradient_alpha),
        crop_size=4,
    )
    model_inputs, model_outputs = ds[0]
    assert model_inputs.shape == (2, 4, 4)  # [scalar, gradient] channels, 4x4 crop
    assert model_outputs.shape == (5, 4, 4)  # [R, G, B, scalar_a, grad_a] channels


@pytest.mark.parametrize("bad_dim", [0, 1, 2])
def test_dataset_dimension_mismatch(
    bad_dim, make_image, lut_rgb, lut_scalar_alpha, lut_gradient_alpha
):
    base = np.zeros((6, 6, 6), dtype=np.float32)
    ref = make_image(base)
    shape = [6, 6, 6]
    # numpy axis order is (nz, ny, nx); vtk dimensions is (nx, ny, nz).
    shape[2 - bad_dim] = 7
    tgt = make_image(np.zeros(shape, dtype=np.float32))
    with pytest.raises(AssertionError):
        _dataset(ref, tgt, (lut_rgb, lut_scalar_alpha, lut_gradient_alpha))
