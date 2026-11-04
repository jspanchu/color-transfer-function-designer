"""Capture snapshots of the transfer-function training as it progresses.

Runs the same training as the app's "Transfer" button (same seed and default
hyper-parameters) headlessly, and at selected training steps renders offscreen:

* the 3D volume render of the target with the current learned TF,
* the X, Y and Z crop views (clipping plane + colored slice, as in the app's
  crop toolbar at 50%),
* a plot of the learned TF (color, scalar opacity, gradient opacity) with the
  ground-truth TF linearly mapped onto the target range for reference.

The reference volume with the ground-truth TF is rendered once in the same
views, and each step is composed into a grid: reference on top, target below.

Usage:
    uv run python scripts/capture_progress.py [--config config.json] [--out DIR]

Outputs (in --out, default ./progress_captures):
    reference_{3d,x,y,z,tf}.png
    step_NNNN_{3d,x,y,z,tf,grid}.png
    progress.gif   (all grids)
    loss.csv       (loss of every training step)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np
import torch
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import VTK_DOUBLE
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction, vtkPlane
from vtkmodules.vtkFiltersCore import vtkArrayCalculator
from vtkmodules.vtkImagingCore import vtkImageMapToColors
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkRenderingCore import (
    vtkDiscretizableColorTransferFunction,
    vtkImageActor,
    vtkRenderer,
    vtkRenderWindow,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

try:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    sys.exit(f"capture_progress.py needs matplotlib and Pillow: {exc}")

from color_transfer_function_designer.app.dataset import (
    compute_gradient_magnitude,
    load_vtk_image_to_tensor,
)
from color_transfer_function_designer.app.model import TransferFunctionNet
from color_transfer_function_designer.app.transfer import (
    device,
    lut_from_network,
    transfer_reference_lut,
)
from color_transfer_function_designer.app.utils import (
    read_paraview_tf_from_json,
    read_slicer_tf_from_ascii,
)

# Same defaults as App._set_default_parameters / App.__init__.
SEED = 2026
DEFAULTS = {
    "n_epochs": 2,
    "n_slices": 1024,
    "batch_size": 16,
    "lr": 5.0e-3,
    "blend_factor_l1_vs_ssim": 0.2,
    "margin": 0.25,
    "n_lut_sampling_points": 64,
}
BACKGROUND = (0.2, 0.2, 0.2)
VIEWS = ("3d", "x", "y", "z")
# Crop views rendered with the camera looking straight at the crop plane, and
# the world axis shown pointing up on screen: superior (+z) for the coronal Y
# view and -y for the axial Z view.
HEAD_ON_UP = {"y": (0.0, 0.0, 1.0), "z": (0.0, -1.0, 0.0)}


# ----------------------------------------------------------------------------
# Loading (mirrors App._load_* in core.py)
# ----------------------------------------------------------------------------


def load_tf(path: pathlib.Path):
    if path.suffix == ".vp":
        colors, sof, gof = read_slicer_tf_from_ascii(path.read_text())
    elif path.suffix == ".json":
        colors, sof, gof = read_paraview_tf_from_json(json.loads(path.read_text()))
    else:
        sys.exit(f"Unsupported TF file format: {path}")
    # VTK sorts nodes by scalar on insertion; the app reads them back sorted.
    colors = colors[np.argsort(colors[:, 0], kind="stable")]
    sof = sof[np.argsort(sof[:, 0], kind="stable")]
    if gof is None:
        gof = np.array([[0.0, 1.0], [1.0, 1.0]])
    gof = gof[np.argsort(gof[:, 0], kind="stable")]
    return colors, sof, gof


def load_reference(path: pathlib.Path) -> vtkImageData:
    reader = vtkNIFTIImageReader(file_name=str(path))
    reader.Update()
    image = vtkImageData()
    image.ShallowCopy(reader.output)
    return image


def load_target(path: pathlib.Path) -> vtkImageData:
    reader = vtkNIFTIImageReader(file_name=str(path))
    reader.Update()
    scaler = vtkArrayCalculator(
        result_array_name="RealScalars", result_array_type=VTK_DOUBLE
    )
    scaler.AddScalarArrayName("NIFTI")
    scaler.function = f"NIFTI * {reader.rescale_slope} + {reader.rescale_intercept}"
    scaler.input_data = reader.output
    scaler.Update()
    image = vtkImageData()
    image.ShallowCopy(scaler.output)
    return image


def grad_mag_max(volume: vtkImageData) -> float:
    return float(
        compute_gradient_magnitude(load_vtk_image_to_tensor(volume), volume.spacing)
        .max()
        .item()
    )


# ----------------------------------------------------------------------------
# Transfer functions
# ----------------------------------------------------------------------------


def build_vtk_tf(colors, sof, gof, discretize=False):
    """colors: (N, 4) [s, r, g, b]; sof/gof: (M, 2) [x, a]."""
    ctf = vtkDiscretizableColorTransferFunction(allow_duplicate_scalars=True)
    if discretize:
        ctf.discretize = True
        ctf.number_of_values = 256
    ctf.AddRGBPoints(
        numpy_to_vtk(np.ascontiguousarray(colors[:, 0]), deep=1),
        numpy_to_vtk(np.ascontiguousarray(colors[:, 1:]), deep=1),
    )
    ctf.Build()
    otf = vtkPiecewiseFunction(allow_duplicate_scalars=True)
    for s, a in sof:
        otf.AddPoint(float(s), float(a))
    gtf = vtkPiecewiseFunction(allow_duplicate_scalars=True)
    for g, a in gof:
        gtf.AddPoint(float(g), float(a))
    return ctf, otf, gtf


def network_tf(model, volume, n_points):
    colors, opacities, gradient_opacities = lut_from_network(model, volume, n_points)
    colors = np.array([[s, *rgb] for s, rgb in colors], dtype=np.float64)
    return colors, np.array(opacities), np.array(gradient_opacities)


def map_gt_to_target(colors, sof, gof, ref, tgt):
    """Linear map of the GT TF onto the target ranges (App's 'linear_map')."""
    r0, r1 = ref.scalar_range
    t0, t1 = tgt.scalar_range

    def remap(x):
        return (x - r0) / (r1 - r0) * (t1 - t0) + t0

    colors, sof, gof = colors.copy(), sof.copy(), gof.copy()
    colors[:, 0] = remap(colors[:, 0])
    sof[:, 0] = remap(sof[:, 0])
    gof[:, 0] *= grad_mag_max(tgt) / grad_mag_max(ref)
    return colors, sof, gof


# ----------------------------------------------------------------------------
# Offscreen rendering (mirrors App._setup_vtk_pipeline and crop.js)
# ----------------------------------------------------------------------------

# crop.js: cropping the "+x" half keeps "-x", i.e. the normal points to -x.
CROP_NORMALS = {
    "+x": (-1, 0, 0),
    "-x": (1, 0, 0),
    "+y": (0, -1, 0),
    "-y": (0, 1, 0),
    "+z": (0, 0, -1),
    "-z": (0, 0, 1),
}


class OffscreenView:
    def __init__(self, volume_data: vtkImageData, camera: dict | None, size: int):
        self.data = volume_data
        self.window = vtkRenderWindow(off_screen_rendering=True, size=(size, size))
        self.renderer = vtkRenderer(background=BACKGROUND)
        self.window.AddRenderer(self.renderer)

        self.mapper = vtkSmartVolumeMapper(input_data=volume_data)
        self.mapper.SetBlendModeToComposite()
        self.property = vtkVolumeProperty()
        self.property.SetScalarOpacityUnitDistance(1.754420659713536)
        self.property.SetScatteringAnisotropy(0)
        self.volume = vtkVolume(mapper=self.mapper, property=self.property)

        self.colors = vtkImageMapToColors(input_data=volume_data)
        self.image_actor = vtkImageActor(force_opaque=True, visibility=False)
        self.image_actor.mapper.input_connection = self.colors.output_port
        self.plane = vtkPlane()

        self.renderer.AddViewProp(self.image_actor)
        self.renderer.AddVolume(self.volume)
        self.renderer.ResetCamera()
        if camera:
            cam = self.renderer.active_camera
            for key, setter in (
                ("position", cam.SetPosition),
                ("focal_point", cam.SetFocalPoint),
                ("view_up", cam.SetViewUp),
            ):
                if key in camera:
                    setter(*camera[key])
            if "view_angle" in camera:
                cam.SetViewAngle(camera["view_angle"])
            if "parallel_scale" in camera:
                cam.SetParallelScale(camera["parallel_scale"])
            cam.SetParallelProjection(bool(camera.get("parallel_projection", False)))

    def set_tf(self, ctf, otf, gtf):
        self.colors.lookup_table = ctf
        self.property.SetColor(ctf)
        self.property.SetScalarOpacity(otf)
        self.property.SetGradientOpacity(gtf)

    def crop_direction(self, axis: int) -> str:
        """Crop away the half facing the camera so the cut face is visible."""
        cam = self.renderer.active_camera
        facing_plus = cam.position[axis] >= cam.focal_point[axis]
        return ("+" if facing_plus else "-") + "xyz"[axis]

    def set_crop(self, direction: str | None, percent: float = 50.0):
        self.mapper.RemoveAllClippingPlanes()
        if direction is None:
            self.image_actor.visibility = False
            return
        axis = "xyz".index(direction[1])
        b = self.data.bounds
        origin = [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]
        origin[axis] = b[2 * axis] + percent / 100.0 * (b[2 * axis + 1] - b[2 * axis])
        self.plane.origin = origin
        self.plane.normal = CROP_NORMALS[direction]
        self.mapper.AddClippingPlane(self.plane)

        e = list(self.data.extent)
        idx = round(e[2 * axis] + percent / 100.0 * (e[2 * axis + 1] - e[2 * axis]))
        e[2 * axis] = e[2 * axis + 1] = idx
        self.image_actor.display_extent = e
        self.image_actor.visibility = True

    def snapshot(self) -> np.ndarray:
        self.renderer.ResetCameraClippingRange()
        self.window.Render()
        w2i = vtkWindowToImageFilter(input=self.window)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()
        out = w2i.output
        w, h, _ = out.dimensions
        rgb = vtk_to_numpy(out.point_data.scalars).reshape(h, w, 3)
        return np.flipud(rgb).copy()

    def look_head_on(self, direction: str):
        """Move the camera onto the cut side of the crop plane, looking along
        its normal, with HEAD_ON_UP as the view-up. Keeps the focal point,
        distance and view angle of the configured camera."""
        cam = self.renderer.active_camera
        axis = "xyz".index(direction[1])
        n = np.zeros(3)
        n[axis] = 1.0 if direction[0] == "+" else -1.0  # from plane to camera
        focal = np.array(cam.focal_point)
        distance = np.linalg.norm(focal - np.array(cam.position))
        cam.SetPosition(*(focal + distance * n))
        cam.SetViewUp(*HEAD_ON_UP[direction[1]])

    def render_all(self, percent: float) -> dict[str, np.ndarray]:
        images = {}
        self.set_crop(None)
        images["3d"] = self.snapshot()
        cam = self.renderer.active_camera
        saved = (cam.position, cam.focal_point, cam.view_up)
        for axis, name in enumerate("xyz"):
            direction = self.crop_direction(axis)
            self.set_crop(direction, percent)
            if name in HEAD_ON_UP:
                self.look_head_on(direction)
            images[name] = self.snapshot()
            cam.SetPosition(*saved[0])
            cam.SetFocalPoint(*saved[1])
            cam.SetViewUp(*saved[2])
        self.set_crop(None)
        return images


# ----------------------------------------------------------------------------
# TF plot
# ----------------------------------------------------------------------------


def plot_tf(ax_s, ax_g, colors, sof, gof, gt=None, title=""):
    """Scalar-opacity curve filled with the color ramp, plus gradient opacity."""
    x0, x1 = float(colors[:, 0].min()), float(colors[:, 0].max())
    xs = np.linspace(x0, x1, 512)
    ramp = np.stack([np.interp(xs, colors[:, 0], colors[:, c]) for c in (1, 2, 3)], 1)
    alpha = np.interp(xs, sof[:, 0], sof[:, 1])

    im = ax_s.imshow(
        ramp[None], extent=(x0, x1, 0, 1), aspect="auto", interpolation="bilinear"
    )
    (fill,) = ax_s.fill(
        np.r_[xs, x1, x0], np.r_[alpha, 0, 0], facecolor="none", edgecolor="none"
    )
    im.set_clip_path(fill)
    ax_s.plot(xs, alpha, color="white", lw=1.2)
    if gt is not None:
        ax_s.plot(gt[1][:, 0], gt[1][:, 1], "--", color="#ffb000", lw=1, label="GT")
        ax_s.legend(loc="upper right", fontsize=7, frameon=False, labelcolor="white")
    ax_s.set_xlim(x0, x1)
    ax_s.set_ylim(0, 1.02)
    ax_s.set_title(title, color="white", fontsize=9)
    ax_s.set_ylabel("opacity", color="white", fontsize=8)

    ax_g.plot(gof[:, 0], gof[:, 1], color="white", lw=1.2)
    if gt is not None:
        ax_g.plot(gt[2][:, 0], gt[2][:, 1], "--", color="#ffb000", lw=1)
    ax_g.set_ylim(0, 1.02)
    ax_g.set_xlabel("scalar  /  gradient magnitude", color="white", fontsize=8)
    ax_g.set_ylabel("grad. opacity", color="white", fontsize=8)
    for ax in (ax_s, ax_g):
        ax.set_facecolor(BACKGROUND)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#888888")


def render_tf_image(colors, sof, gof, gt, title, size) -> np.ndarray:
    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor=BACKGROUND)
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.35)
    plot_tf(fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), colors, sof, gof, gt, title)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.93, bottom=0.13)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------


def save_png(path: pathlib.Path, img: np.ndarray):
    Image.fromarray(img).save(path)


def compose_grid(rows: list[tuple[str, dict]], title: str) -> np.ndarray:
    keys = (*VIEWS, "tf")
    labels = {"3d": "Volume", "x": "Crop X", "y": "Crop Y", "z": "Crop Z", "tf": "TF"}
    h, w = rows[0][1]["3d"].shape[:2]
    dpi = 100
    fig, axes = plt.subplots(
        len(rows),
        len(keys),
        figsize=(len(keys) * w / dpi, len(rows) * h / dpi + 0.6),
        dpi=dpi,
        facecolor="black",
        squeeze=False,
    )
    for r, (row_label, images) in enumerate(rows):
        for c, key in enumerate(keys):
            ax = axes[r][c]
            ax.imshow(images[key])
            ax.set_axis_off()
            if r == 0:
                ax.set_title(labels[key], color="white", fontsize=12)
        axes[r][0].text(
            -0.03, 0.5, row_label, transform=axes[r][0].transAxes, color="white",
            rotation=90, va="center", ha="right", fontsize=12,
        )  # fmt: skip
    fig.suptitle(title, color="white", fontsize=14)
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.005, wspace=0.02, hspace=0.02)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def snapshot_steps(total: int, every: int | None) -> set[int]:
    if every:
        steps = set(range(0, total + 1, every))
    else:
        steps = {0} | {2**k for k in range(int(math.log2(total)) + 1)}
    return steps | {total}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out", default="progress_captures")
    parser.add_argument("--size", type=int, default=600, help="view size in px")
    parser.add_argument("--crop-percent", type=float, default=50.0)
    parser.add_argument(
        "--every", type=int, default=None,
        help="snapshot every N steps (default: log-spaced 0,1,2,4,8,..,last)",
    )  # fmt: skip
    parser.add_argument("--n-epochs", type=int, default=DEFAULTS["n_epochs"])
    parser.add_argument("--n-slices", type=int, default=DEFAULTS["n_slices"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--gif-ms", type=int, default=700, help="GIF frame time")
    args = parser.parse_args()

    config_path = pathlib.Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    base = pathlib.Path.cwd()
    tf_path = (base / cfg["transfer_function_file"]).resolve()
    ref_path = (base / cfg["reference_volume_file"]).resolve()
    tgt_path = (base / cfg["target_volume_file"]).resolve()
    camera = cfg.get("camera")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    model = TransferFunctionNet().to(device=device)

    gt_colors, gt_sof, gt_gof = load_tf(tf_path)
    ref = load_reference(ref_path)
    tgt = load_target(tgt_path)
    print(f"reference {ref.dimensions} range {ref.scalar_range}")
    print(f"target    {tgt.dimensions} range {tgt.scalar_range}")
    gt_on_tgt = map_gt_to_target(gt_colors, gt_sof, gt_gof, ref, tgt)

    # Reference row, rendered once.
    ref_view = OffscreenView(ref, camera, args.size)
    ref_view.set_tf(*build_vtk_tf(gt_colors, gt_sof, gt_gof, discretize=True))
    ref_images = ref_view.render_all(args.crop_percent)
    ref_images["tf"] = render_tf_image(
        gt_colors, gt_sof, gt_gof, None, "Ground truth TF", args.size
    )
    for key, img in ref_images.items():
        save_png(out / f"reference_{key}.png", img)

    tgt_view = OffscreenView(tgt, camera, args.size)
    batches_per_epoch = math.ceil(args.n_slices / args.batch_size)
    total = args.n_epochs * batches_per_epoch
    steps = snapshot_steps(total, args.every)
    print(f"{total} training steps; snapshots at {sorted(steps)}")

    grids: list[pathlib.Path] = []
    losses: list[tuple[int, int, int, float]] = []

    def capture(step: int, epoch: int, loss: float | None):
        colors, sof, gof = network_tf(model, tgt, DEFAULTS["n_lut_sampling_points"])
        tgt_view.set_tf(*build_vtk_tf(colors, sof, gof))
        images = tgt_view.render_all(args.crop_percent)
        images["tf"] = render_tf_image(
            colors, sof, gof, gt_on_tgt, f"Learned TF (step {step})", args.size
        )
        for key, img in images.items():
            save_png(out / f"step_{step:04d}_{key}.png", img)
        loss_txt = "untrained" if loss is None else f"loss {loss:.4f}"
        grid = compose_grid(
            [("Reference (GT)", ref_images), ("Target (learned)", images)],
            f"Step {step}/{total}  ·  epoch {epoch}/{args.n_epochs}  ·  {loss_txt}",
        )
        grid_path = out / f"step_{step:04d}_grid.png"
        save_png(grid_path, grid)
        grids.append(grid_path)
        print(f"  captured step {step} ({loss_txt})")

    def on_progress(epoch_id, batch_id, _progress, loss):
        # Runs on the training thread between optimizer steps, so the snapshot
        # reflects exactly the weights after this step. lut_from_network
        # restores the model's training mode afterwards.
        step = (epoch_id - 1) * batches_per_epoch + batch_id
        losses.append((step, epoch_id, batch_id, loss))
        if step in steps:
            capture(step, epoch_id, loss)

    capture(0, 0, None)
    transfer_reference_lut(
        model,
        tgt_volume=tgt,
        ref_volume=ref,
        lut_rgb=gt_colors,
        lut_scalar_alpha=gt_sof,
        lut_gradient_alpha=gt_gof,
        n_epochs=args.n_epochs,
        n_slices=args.n_slices,
        batch_size=args.batch_size,
        blend_factor_l1_vs_ssim=DEFAULTS["blend_factor_l1_vs_ssim"],
        progress_callback=on_progress,
        lr=args.lr,
        margin=DEFAULTS["margin"],
    )

    with (out / "loss.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "batch", "loss"])
        writer.writerows(losses)

    frames = [Image.open(p).convert("RGB") for p in grids]
    durations = [args.gif_ms] * (len(frames) - 1) + [args.gif_ms * 4]
    frames[0].save(
        out / "progress.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"Wrote {len(grids)} snapshots to {out.resolve()}")


if __name__ == "__main__":
    main()
