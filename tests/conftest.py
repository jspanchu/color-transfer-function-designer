import numpy as np
import pytest
import torch
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData


def make_vtk_image(np_array: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> vtkImageData:
    """Build a vtkImageData from a 3-D numpy array shaped (nz, ny, nx)."""
    nz, ny, nx = np_array.shape
    image = vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(*spacing)
    image.GetPointData().SetScalars(numpy_to_vtk(np_array.ravel(), deep=True))
    return image


@pytest.fixture
def make_image():
    return make_vtk_image


@pytest.fixture
def ref_volume():
    """Discrete-valued reference volume with scalar range [0, 3]."""
    rng = np.random.default_rng(1337)
    arr = rng.integers(0, 4, (10, 10, 10)).astype(np.int32)
    # Force the extremes so scalar_range is exactly [0, 3] regardless of sampling.
    arr.ravel()[0] = 0
    arr.ravel()[-1] = 3
    return make_vtk_image(arr)


@pytest.fixture
def tgt_volume():
    """Continuous-valued target volume."""
    rng = np.random.default_rng(2025)
    arr = rng.random((10, 10, 10), dtype=np.float32)
    return make_vtk_image(arr)


@pytest.fixture
def lut_rgb():
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 0.0],
            [3.0, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def lut_scalar_alpha():
    return np.array([[0.0, 0.0], [3.0, 1.0]], dtype=np.float32)


@pytest.fixture
def lut_gradient_alpha():
    return np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
