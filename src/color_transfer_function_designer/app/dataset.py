import numpy as np
import torch
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vtk_image_to_tensor(vtk_image) -> torch.Tensor:
    """Convert vtkImageData to (nz, ny, nx) float32 array."""
    dims = vtk_image.dimensions  # (nx, ny, nz)
    arr = vtk_to_numpy(vtk_image.point_data.scalars).astype(np.float32)
    return torch.from_numpy(arr).reshape(
        dims[2], dims[1], dims[0]
    )  # C-order: x fastest → (nz, ny, nx)


def normalize_volume_scalars(
    scalars: torch.Tensor,
) -> torch.Tensor:
    """Normalize the volume scalars to [0, 1] range."""
    scalar_min = scalars.min()
    scalar_max = scalars.max()
    scalar_span = max(scalar_max - scalar_min, 1e-6)
    return (scalars - scalar_min) / scalar_span


def compute_gradient_magnitude(scalars: torch.Tensor, spacing: tuple) -> torch.Tensor:
    """Compute the gradient magnitudes in the volume."""
    assert scalars.ndim == 3
    dz, dy, dx = torch.gradient(scalars, spacing=list(reversed(spacing)), edge_order=2)
    return torch.sqrt(dx**2 + dy**2 + dz**2)


def extract_2d_slice(volume: torch.Tensor, axis: int, idx: int) -> torch.Tensor:
    """Extract a 2D slice from (nz, ny, nx) volume perpendicular to `axis` at `idx`."""
    if idx < -volume.shape[axis] or idx >= volume.shape[axis]:
        msg = f"slice 'idx' ({idx}) out of range for 'axis' {axis} of size {volume.shape[axis]}"
        raise IndexError(msg)
    index_tensor = torch.tensor([idx]).to(device=volume.device)
    return torch.index_select(volume, axis, index_tensor).squeeze(axis)


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of fp at query points x, given knots (xp, fp)."""
    idx = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, xp.shape[0] - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    t = ((x - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0.0, 1.0)
    return f0 + t * (f1 - f0)


def apply_lut_torch(
    scalar_flat: torch.Tensor,  # (N,) on device
    gradient_flat: torch.Tensor,  # (N,) on device
    lut_rgb: torch.Tensor,  # (K, 4) on device — col 0 xp, cols 1-3 fp
    lut_scalar_alpha: torch.Tensor,  # (M, 2) on device — col 0 xp, col 1 fp
    lut_gradient_alpha: torch.Tensor,  # (L, 2) on device — col 0 xp, col 1 fp
) -> torch.Tensor:  # (N, 5): R G B scalar_alpha grad_alpha
    xp_rgb = lut_rgb[:, 0]
    r = _interp1d(scalar_flat, xp_rgb, lut_rgb[:, 1])
    g = _interp1d(scalar_flat, xp_rgb, lut_rgb[:, 2])
    b = _interp1d(scalar_flat, xp_rgb, lut_rgb[:, 3])
    alpha = _interp1d(scalar_flat, lut_scalar_alpha[:, 0], lut_scalar_alpha[:, 1])
    grad_alpha = _interp1d(
        gradient_flat, lut_gradient_alpha[:, 0], lut_gradient_alpha[:, 1]
    )
    return torch.stack([r, g, b, alpha, grad_alpha], dim=1)


class SharedSlicePlaneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_known_lut: vtkImageData,
        image_unknown_lut: vtkImageData,
        lut_rgb: np.ndarray,
        lut_scalar_alpha: np.ndarray,
        lut_gradient_alpha: np.ndarray,
        number_of_slices: int,
        margin: float,
        crop_size: int | None = None,
    ):
        """
        Parameters
        ----------
        image_known_lut:
            Volume whose scalar-to-RGBA mapping is already known (e.g. a reference mask
            with an established color transfer function).
        image_unknown_lut:
            Volume whose transfer function is being learned (e.g. a co-registered MRI).
            Must have the same spatial dimensions as `image_known_lut`.
        lut_rgb:
            (K, 4) array — column 0 is the scalar knot position, columns 1-3 are R, G, B
            values in the scalar range of `image_known_lut`.
        lut_scalar_alpha:
            (M, 2) array — column 0 is the scalar knot position, column 1 is opacity,
            in the scalar range of `image_known_lut`.
        lut_gradient_alpha:
            (L, 2) array — column 0 is the gradient-magnitude knot position, column 1 is
            opacity, in the gradient-magnitude range of `image_known_lut`.
        number_of_slices:
            Dataset length — how many random slice samples to draw per epoch.
        margin:
            Fraction of each dimension to exclude at both ends when sampling slice indices,
            avoiding uninformative boundary slices (e.g. 0.25 skips the outer 25% on each
            side).
        crop_size:
            Side length of the square patch randomly cropped from each 2-D slice. Must be
            <= min(volume dimensions) so the crop fits in every slice orientation. Defaults
            to min(dimensions), which is the largest square guaranteed to fit in all three
            orientations and preserves the full slice for isotropic volumes.
        """
        assert image_known_lut.dimensions[0] == image_unknown_lut.dimensions[0], (
            "Size of dimension 0 in image_known_lut != image_unknown_lut"
        )
        assert image_known_lut.dimensions[1] == image_unknown_lut.dimensions[1], (
            "Size of dimension 1 in image_known_lut != image_unknown_lut"
        )
        assert image_known_lut.dimensions[2] == image_unknown_lut.dimensions[2], (
            "Size of dimension 2 in image_known_lut != image_unknown_lut"
        )
        # Known lut
        self._scalars_known_lut = load_vtk_image_to_tensor(image_known_lut).to(device)
        self._gradient_mags_known_lut = compute_gradient_magnitude(
            self._scalars_known_lut, image_known_lut.spacing
        )
        # LUT TBD for this volume
        _scalars_unknown_lut = load_vtk_image_to_tensor(image_unknown_lut).to(device)
        self._scalars_unknown_lut = normalize_volume_scalars(_scalars_unknown_lut)
        _gradient_mags_unknown_lut = compute_gradient_magnitude(
            _scalars_unknown_lut, image_unknown_lut.spacing
        )
        self._gradient_mags_unknown_lut = normalize_volume_scalars(
            _gradient_mags_unknown_lut
        )

        self._number_of_slices = number_of_slices
        self._lut_rgb = torch.from_numpy(lut_rgb).to(device)
        self._lut_scalar_alpha = torch.from_numpy(lut_scalar_alpha).to(device)
        self._lut_gradient_alpha = torch.from_numpy(lut_gradient_alpha).to(device)

        # Per-axis safe slice index ranges
        self.dims = self._scalars_known_lut.shape  # (nz, ny, nx)
        self._axis_ranges = np.zeros((len(self.dims), 2), dtype=np.int64)
        for i, d in enumerate(self.dims):
            lo = int(d * margin)
            hi = int(d * (1.0 - margin))
            self._axis_ranges[i][0] = lo
            self._axis_ranges[i][1] = max(lo + 1, hi)

        # Largest square crop guaranteed to fit in any slice orientation
        self._crop_size = crop_size if crop_size is not None else min(self.dims)

    def __len__(self):
        return self._number_of_slices

    def __getitem__(self, _):
        axis = int(torch.randint(0, 3, (1,)).item())
        lo, hi = self._axis_ranges[axis]
        slice_plane_idx = int(torch.randint(lo, hi, (1,)).item())

        known_s = extract_2d_slice(self._scalars_known_lut, axis, slice_plane_idx)
        known_g = extract_2d_slice(self._gradient_mags_known_lut, axis, slice_plane_idx)
        unknown_s = extract_2d_slice(self._scalars_unknown_lut, axis, slice_plane_idx)
        unknown_g = extract_2d_slice(
            self._gradient_mags_unknown_lut, axis, slice_plane_idx
        )
        assert known_s.shape == unknown_s.shape
        assert known_g.shape == unknown_g.shape

        # Random square crop — same origin applied to all four slices so they stay aligned
        H, W = known_s.shape
        cs = self._crop_size
        top = torch.randint(0, H - cs + 1, (1,)).item()
        left = torch.randint(0, W - cs + 1, (1,)).item()
        known_s = known_s[top : top + cs, left : left + cs]
        known_g = known_g[top : top + cs, left : left + cs]
        unknown_s = unknown_s[top : top + cs, left : left + cs]
        unknown_g = unknown_g[top : top + cs, left : left + cs]

        model_inputs = torch.cat(
            (unknown_s.ravel()[:, None], unknown_g.ravel()[:, None]),
            dim=1,
        )
        model_outputs = apply_lut_torch(
            known_s.ravel(),
            known_g.ravel(),
            self._lut_rgb,
            self._lut_scalar_alpha,
            self._lut_gradient_alpha,
        )
        return model_inputs, model_outputs
