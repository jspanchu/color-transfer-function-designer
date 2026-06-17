import numpy as np
import torch
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData

from color_transfer_function_designer.app.model import TransferFunctionNet
from color_transfer_function_designer.app.transfer import (
    transfer_reference_lut,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_simple_transfer():
    # Synthetic data: tiny MRI volume and simple GT TF
    rng = np.random.default_rng(1337)
    tgt_volume_np = rng.random((10, 10, 10), dtype=np.float32)
    ref_volume_np = rng.integers(0, 4, (10, 10, 10), dtype=np.int32)

    # Convert to vtkImageData
    tgt_volume = vtkImageData()
    tgt_volume.SetDimensions(10, 10, 10)
    tgt_volume.GetPointData().SetScalars(numpy_to_vtk(tgt_volume_np.ravel(), deep=True))

    ref_volume = vtkImageData()
    ref_volume.SetDimensions(10, 10, 10)
    ref_volume.GetPointData().SetScalars(numpy_to_vtk(ref_volume_np.ravel(), deep=True))

    lut_rgb = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 0.0],
            [3.0, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    lut_scalar_alpha = np.array([[0.0, 0.0], [3.0, 1.0]], dtype=np.float32)
    lut_gradient_alpha = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    transfer_reference_lut(
        TransferFunctionNet().to(device),
        tgt_volume,
        ref_volume,
        lut_rgb,
        lut_scalar_alpha,
        lut_gradient_alpha,
        n_epochs=1,
        n_slices=8,
        batch_size=2,
        progress_callback=print,
    )
    print("SIMPLE TRANSFER TEST PASSED")
