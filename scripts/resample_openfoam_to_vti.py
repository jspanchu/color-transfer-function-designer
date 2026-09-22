"""Resample an OpenFOAM case onto a uniform image for GPU volume rendering.

By default the velocity gradient (Q-criterion, vorticity) is computed on the
unstructured grid before resampling. On snappyHexMesh cases the thin near-wall
cells and polyhedra produce near-singular gradients there, giving Q outliers
many orders of magnitude above the physical range. Pass ``--derive-on-image``
to compute Q and vorticity magnitude on the resampled image instead, with the
body interior plus a one-voxel shell zeroed so the zero-velocity interior does
not create a fake shear layer.
"""

import argparse
import logging

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        default="data/scivis/Ahmed-body-snappy/openfoam/case.foam",
        help="Path to the .foam file",
    )
    p.add_argument("--time", type=float, default=1000.0, help="Time step to read")
    p.add_argument(
        "--output", default="data/scivis/ahmed_wake.vti", help="Output .vti path"
    )
    p.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=[-1.3, 3.5, -0.8, 0.8, 0.0, 0.8],
        help="Sampling box; default covers the Ahmed body and near wake",
    )
    p.add_argument(
        "--dims",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=[308, 128, 64],
        help="Sampling dimensions",
    )
    p.add_argument(
        "--derive-on-image",
        action="store_true",
        help="Compute Q and vorticity on the resampled image instead of the "
        "unstructured grid, and zero them inside the body plus a one-voxel shell",
    )
    p.add_argument("--velocity-array", default="U", help="Name of the velocity array")
    return p.parse_args()


def read_case(path: str, time: float) -> vtk.vtkUnstructuredGrid:
    r = vtk.vtkOpenFOAMReader()
    r.SetFileName(path)
    r.UpdateInformation()
    r.SetTimeValue(time)
    r.Update()
    ug = r.GetOutput().GetBlock(0)
    logger.info("read %s at t=%g: %d cells", path, time, ug.GetNumberOfCells())
    return ug


def gradient_filter(velocity_array: str) -> vtk.vtkGradientFilter:
    g = vtk.vtkGradientFilter()
    g.SetInputArrayToProcess(0, 0, 0, 0, velocity_array)
    g.ComputeQCriterionOn()
    g.ComputeVorticityOn()
    g.ComputeGradientOff()
    g.SetQCriterionArrayName("Q")
    g.SetVorticityArrayName("Vorticity")
    return g


def resample(source, bounds, dims) -> vtk.vtkImageData:
    rs = vtk.vtkResampleToImage()
    rs.SetInputConnection(source.GetOutputPort())
    rs.UseInputBoundsOff()
    rs.SetSamplingBounds(*bounds)
    rs.SetSamplingDimensions(*dims)
    rs.Update()
    logger.info("resampled to %s", dims)
    return rs.GetOutput()


def derive_on_image(image: vtk.vtkImageData, velocity_array: str) -> vtk.vtkImageData:
    """Compute Q and |vorticity| on the image and mask the body plus a one-voxel shell."""
    g = gradient_filter(velocity_array)
    g.SetInputData(image)
    g.Update()
    out = g.GetOutput()
    pd = out.GetPointData()

    nx, ny, nz = out.GetDimensions()
    invalid = vtk_to_numpy(pd.GetArray("vtkValidPointMask")).reshape(nz, ny, nx) == 0
    shell = invalid.copy()
    for axis in range(3):
        shell |= np.roll(invalid, 1, axis=axis) | np.roll(invalid, -1, axis=axis)
    shell = shell.ravel()
    logger.info(
        "masked %d body voxels (+ shell: %d)", int(invalid.sum()), int(shell.sum())
    )

    q = vtk_to_numpy(pd.GetArray("Q")).astype(np.float32)
    q[shell] = 0.0
    vort = vtk_to_numpy(pd.GetArray("Vorticity")).astype(np.float32)
    vort[shell] = 0.0
    vort_mag = np.linalg.norm(vort, axis=1)

    for name in ("Q", "Vorticity", "vtkGhostType"):
        pd.RemoveArray(name)
    for name, arr in (("Q", q), ("VorticityMag", vort_mag)):
        va = numpy_to_vtk(arr, deep=1)
        va.SetName(name)
        pd.AddArray(va)

    valid_q = q[~shell]
    logger.info(
        "Q on image: p50 %.3g p99 %.3g p99.9 %.3g max %.3g",
        *np.percentile(valid_q, [50, 99, 99.9, 100]),
    )
    return out


def main() -> None:
    args = parse_args()
    vtk.vtkSMPTools.SetBackend("STDThread")

    ug = read_case(args.case, args.time)
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(ug)

    if args.derive_on_image:
        image = resample(c2p, args.bounds, args.dims)
        image = derive_on_image(image, args.velocity_array)
    else:
        g = gradient_filter(args.velocity_array)
        g.SetInputConnection(c2p.GetOutputPort())
        logger.info("computing gradient on unstructured grid")
        image = resample(g, args.bounds, args.dims)

    w = vtk.vtkXMLImageDataWriter()
    w.SetFileName(args.output)
    w.SetInputData(image)
    w.Write()
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
