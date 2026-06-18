import asyncio
import copy
import ctypes
import json
import logging
import os
import pathlib
import time

import numpy as np
import numpy.typing as npt
import torch
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from trame.app import TrameApp, asynchronous
from trame.decorators import change, controller
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import client, color_opacity_editor, vtklocal, vuetify3
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import VTK_DOUBLE
from vtkmodules.vtkCommonDataModel import (
    vtkImageData,
    vtkPiecewiseFunction,
    vtkPlane,
)
from vtkmodules.vtkFiltersCore import vtkArrayCalculator
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkRenderingCore import (
    vtkDiscretizableColorTransferFunction,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

from color_transfer_function_designer.app import module
from color_transfer_function_designer.app.dataset import (
    compute_gradient_magnitude,
    load_vtk_image_to_tensor,
)
from color_transfer_function_designer.app.file import (
    FileBrowser,
    FileDialog,
)
from color_transfer_function_designer.app.logger import install_handlers
from color_transfer_function_designer.app.model import TransferFunctionNet
from color_transfer_function_designer.app.transfer import (
    convert_lut_to_state_format,
    lut_from_network,
    transfer_reference_lut,
)
from color_transfer_function_designer.app.utils import (
    read_paraview_tf_from_json,
    read_slicer_tf_from_ascii,
    write_slicer_vp,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class App(TrameApp):
    logger = logging.getLogger("color_transfer_function_designer.app.core.TrainingApp")
    install_handlers(logger)

    _SLICE_DIRECTIONS = (
        ("+X", "+x"),
        ("-X", "-x"),
        ("+Y", "+y"),
        ("-Y", "-y"),
        ("+Z", "+z"),
        ("-Z", "-z"),
    )

    def __init__(
        self,
        data_directory,
        config: str | None = None,
        server=None,
    ):
        super().__init__(server, client_type="vue3")
        self.server.enable_module(module)
        self.state.trame__title = "Color Transfer Function Designer"
        self.file_browser = FileBrowser(home=data_directory)

        torch.manual_seed(2026)
        os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkEGLRenderWindow"

        self._set_default_parameters()

        # Model
        self._model = TransferFunctionNet().to(device=device)
        self._model_init_state = copy.deepcopy(self._model.state_dict())

        # Task manager
        self._transfer_executor_future: asyncio.Future | None = None
        self._pending_tasks = set()
        self._last_progress_update_time: float = 0.0

        # Used to record file type when FileDialog is opened.
        self._requested_file_type = ""

        # VTK.wasm views
        self._tgt_html_view: vtklocal.LocalView | None = None
        self._ref_html_view: vtklocal.LocalView | None = None
        (
            self._tgt_wnd,
            self._tgt_renderer,
            self._tgt_volume,
            self._tgt_plane,
        ) = self._setup_vtk_pipeline()
        (
            self._ref_wnd,
            self._ref_renderer,
            self._ref_volume,
            self._ref_plane,
        ) = self._setup_vtk_pipeline()

        # Ground-truth transfer function data (in reference scalar range)
        self._gt_ctf = vtkDiscretizableColorTransferFunction(
            allow_duplicate_scalars=True, discretize=True, number_of_values=256
        )
        self._gt_sof = vtkPiecewiseFunction(allow_duplicate_scalars=True)
        self._gt_gof = vtkPiecewiseFunction(allow_duplicate_scalars=True)

        # Input volumes
        self._tgt_volume_data = vtkImageData()
        self._ref_volume_data = vtkImageData()

        # Editor widget state
        self.state.ref_hist_y_range = []
        self.state.ref_histograms = []
        self.state.ref_colors = []
        self.state.ref_opacities = []
        self.state.ref_scalar_range = []
        self.state.tgt_hist_y_range = []
        self.state.tgt_histograms = []
        self.state.tgt_colors = []
        self.state.tgt_opacities = []
        self.state.tgt_scalar_range = []

        # File upload state
        self.state.transfer_function_file = ""
        self.state.reference_volume_file = ""
        self.state.target_volume_file = ""

        # Progress state
        self.state.progress_percent = 0

        # Transfer function state
        self.state.target_lut_source = "linear_map"

        # Button disabled states
        self.state.allow_transfer = False

        # Dialog visibility
        self.state.show_transfer_dialog = False

        # Slice/crop plane state (per view). Position is a percentage [0, 100]
        # along the active axis so it stays valid when the direction changes.
        for key in ("ref", "tgt"):
            setattr(self.state, f"{key}_crop_enabled", False)
            setattr(self.state, f"{key}_crop_direction", "+x")
            setattr(self.state, f"{key}_crop_position", 50)

        # When linked, the crop plane / camera of one view drives the other.
        # Camera sync runs entirely client-side (see CAMERA_SYNC_JS); these wasm
        # ids let the client resolve each view's renderer/active camera. They are
        # populated when the LocalViews are created in _generate_ui().
        self.state.crop_linked = False
        self.state.camera_linked = False
        # While a transfer runs the tgt view is repeatedly re-`update()`d, which
        # would race a client-side crop apply; lock the tgt crop UI meanwhile.
        self.state.tgt_crop_locked = False
        self._ref_renderer_wasm_id = None
        self._tgt_renderer_wasm_id = None
        self._ref_plane_wasm_id = None
        self._tgt_plane_wasm_id = None

        # Set after a volume (re)loads so the next view `updated` event re-applies
        # that view's clipping plane on the client (loading resets the wasm mapper).
        self._ref_crop_apply_pending = False
        self._tgt_crop_apply_pending = False

        # Set when a transfer finishes so the next tgt view `updated` event
        # unlocks the tgt crop UI and re-applies the (transfer-wiped) crop.
        self._tgt_crop_unlock_pending = False

        self._generate_ui()

        if config is not None:
            self._load_from_config(config)

    def _set_default_parameters(self):
        # default training parameters
        self.state.n_epochs = 2
        self.state.n_slices = 1024
        self.state.batch_size = 16
        self.state.learning_rate = 5.0e-3

        # default volume parameters
        self.state.slice_plane_margin = 0.25

        # default lut parameters
        self.state.n_lut_sampling_points = 64

    @change("n_slices")
    def on_n_slices_change(self, n_slices, **_):
        self.state.batch_size = min(n_slices, self.state.batch_size)

    @change("target_lut_source")
    def on_target_lut_source_change(self, **_):
        if self._tgt_volume_data.number_of_points == 0:
            return
        if self.state.target_lut_source == "nn":
            ctf, otf, gof = self._build_network_transfer_functions(
                self._tgt_volume_data,
                snapshot_lut_in_state=True,
            )
            self._tgt_volume.property.SetColor(ctf)
            self._tgt_volume.property.SetScalarOpacity(otf)
            self._tgt_volume.property.SetGradientOpacity(gof)
            assert self._tgt_html_view is not None
            self._tgt_html_view.update()
        elif self.state.target_lut_source == "linear_map":
            ctf, otf, gof = self._map_ground_truth_lut_to_tgt_volume_scalars(
                self._tgt_volume_data,
                snapshot_lut_in_state=True,
            )
            self._tgt_volume.property.SetColor(ctf)
            self._tgt_volume.property.SetScalarOpacity(otf)
            self._tgt_volume.property.SetGradientOpacity(gof)
            assert self._tgt_html_view is not None
            self._tgt_html_view.update()
        else:
            self.logger.error("Unknown target_lut_source!")

    @change("camera_linked")
    def on_camera_linked_change(self, camera_linked, **_):
        # On linking, snap the target view onto the reference camera. The
        # sync itself runs client-side; see CAMERA_SYNC_JS / triggerCameraSync.
        if camera_linked:
            assert self._ref_html_view is not None
            assert self._tgt_html_view is not None
            self.ctrl.camera_sync_once(
                {
                    "srcRefName": self._ref_html_view.ref_name,
                    "dstRefName": self._tgt_html_view.ref_name,
                }
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ctrl(self):
        return self.server.controller

    @property
    def state(self):
        return self.server.state

    @property
    def ground_truth_colors(self):
        data_size = self._gt_ctf.size * 4
        data_pointer_str = self._gt_ctf.data_pointer
        address_str = data_pointer_str.split("_")[1]
        address = int(address_str, 16)
        buffer = (ctypes.c_double * data_size).from_address(address)
        return np.frombuffer(buffer, dtype=np.float64).reshape(-1, 4)

    @property
    def ground_truth_opacities(self):
        opacities = []
        for i in range(self._gt_sof.size):
            values = [0.0] * 4
            self._gt_sof.GetNodeValue(i, values)
            opacities.extend(values[:2])
        return np.array(opacities, dtype=np.float64).reshape(-1, 2)

    @property
    def ground_truth_gradient_opacities(self):
        opacities = []
        for i in range(self._gt_gof.size):
            values = [0.0] * 4
            self._gt_gof.GetNodeValue(i, values)
            opacities.extend(values[:2])
        return np.array(opacities, dtype=np.float64).reshape(-1, 2)

    # ------------------------------------------------------------------
    # File handlers
    # ------------------------------------------------------------------

    @controller.set("open_file_dialog")
    def open_file_dialog(self, file_type):
        self.logger.debug("open_file_dialog:file-type='%s'", file_type)
        self.state.file_dialog_is_open = not self.state.file_dialog_is_open
        self._requested_file_type = file_type

    @controller.add("on_file_open")
    def on_file_open(self, path):
        self.logger.debug("on_file_open %s", path)
        if self._requested_file_type == "transfer_function":
            self._load_transfer_function(path)
        elif self._requested_file_type == "reference_volume":
            self._load_reference_volume(path)
        elif self._requested_file_type == "target_volume":
            self._load_target_volume(path)

    @controller.add("on_file_save")
    def on_file_save(self, path):
        self.logger.debug("on_file_save %s", path)
        if self._tgt_volume_data.number_of_points == 0:
            self.logger.error("No target volume loaded, cannot export.")
            return
        colors, opacities, gradient_opacities = lut_from_network(
            self._model,
            self._tgt_volume_data,
            n_points=self.state.n_lut_sampling_points,
        )
        lut_rgb = np.array([[s, r, g, b] for s, (r, g, b) in colors])
        lut_scalar_alpha = np.array(opacities)
        lut_gradient_alpha = np.array(gradient_opacities)
        write_slicer_vp(path, lut_rgb, lut_scalar_alpha, lut_gradient_alpha)
        self.logger.info("Exported transfer function to %s", path)

    def open_export_dialog(self):
        self.state.file_dialog_save_mode = True
        self.state.file_dialog_save_filename = ""
        self.state.file_dialog_is_open = True

    # ------------------------------------------------------------------
    # Local-path file loaders (used by on_file_open and _load_from_config)
    # ------------------------------------------------------------------

    def _load_transfer_function(self, path: pathlib.Path) -> None:
        self.logger.debug("Loading TF from %s", str(path))
        self._unload_transfer_function()

        # Load file
        if path.suffix == ".vp":
            with path.open("rb") as f:
                content = f.read()
                (
                    colors,
                    scalar_opacities,
                    gradient_opacities,
                ) = read_slicer_tf_from_ascii(content.decode())
                self._initialize_transfer_functions(
                    colors, scalar_opacities, gradient_opacities
                )
        elif path.suffix == ".json":
            with path.open("rb") as f:
                (
                    colors,
                    scalar_opacities,
                    gradient_opacities,
                ) = read_paraview_tf_from_json(json.load(f))
                self._initialize_transfer_functions(
                    colors, scalar_opacities, gradient_opacities
                )
        else:
            self.logger.error("Unsupported TF file format: %s", path)
            return

        self.state.transfer_function_file = str(path)

        if (
            self._tgt_volume_data.number_of_points > 0
            and self._ref_volume_data.number_of_points > 0
        ):
            self._apply_gt_tf_to_reference_volume()
        self._check_all_loaded()

    def _unload_transfer_function(self):
        # Reset state
        self.state.ref_colors = []
        self.state.ref_opacities = []
        self.state.ref_scalar_range = []
        self.state.transfer_function_file = ""
        self._gt_gof.RemoveAllPoints()
        self._gt_sof.RemoveAllPoints()
        self._gt_ctf.RemoveAllPoints()
        self._check_all_loaded()

    def _load_reference_volume(self, path: pathlib.Path) -> None:
        # Reset without pushing an update: loading issues a single update() below.
        # A second update() here would let the client coalesce them and emit a
        # premature `updated` (before the new mapper is deserialized), firing the
        # crop re-apply against a not-yet-created mapper.
        self._reset_reference_view()
        # Read reference volume
        reader = vtkNIFTIImageReader(file_name=str(path))
        reader.Update()
        self._ref_volume_data.ShallowCopy(reader.output)
        if self._ref_volume_data.number_of_points == 0:
            return
        self.logger.debug(
            "Loaded reference range [%f,%f] from %s",
            self._ref_volume_data.scalar_range[0],
            self._ref_volume_data.scalar_range[1],
            reader.file_name,
        )
        self._ref_volume.mapper.input_data = self._ref_volume_data

        # Directly apply GT TF to reference volume
        self._apply_gt_tf_to_reference_volume()

        # Set state
        self.state.reference_volume_file = str(path)

        # Render
        self._ref_renderer.AddVolume(self._ref_volume)
        self._check_all_loaded()
        self._ref_renderer.ResetCamera()
        # Re-apply the crop plane client-side once the wasm mapper has synced.
        self._ref_crop_apply_pending = True
        assert self._ref_html_view is not None
        self._ref_html_view.update(push_camera=True)

    def _reset_reference_view(self):
        # Drop the client's cached wasm handles before the mapper is pruned from
        # the wasm scene by the next update() (else a watcher-fired crop apply
        # would invoke RemoveAllClippingPlanes on a dead object id).
        self.ctrl.crop_sync_teardown("ref")
        # Reset state
        self.state.reference_volume_file = ""
        self.state.ref_crop_enabled = False
        # Reset renderer
        self._ref_renderer.RemoveAllViewProps()
        self._ref_volume_data.Initialize()

    def _unload_reference_volume(self):
        self._reset_reference_view()
        assert self._ref_html_view is not None
        self._ref_html_view.update()
        self._check_all_loaded()

    def _load_target_volume(self, path: pathlib.Path) -> None:
        # Reset without pushing an update: loading issues a single update() below.
        # A second update() here would let the client coalesce them and emit a
        # premature `updated` (before the new mapper is deserialized), firing the
        # crop re-apply against a not-yet-created mapper.
        self._reset_target_view()
        # Read volume
        reader = vtkNIFTIImageReader(file_name=str(path))
        reader.Update()
        scaler = vtkArrayCalculator(
            result_array_name="RealScalars", result_array_type=VTK_DOUBLE
        )
        scaler.AddScalarArrayName("NIFTI")
        scaler.function = f"NIFTI * {reader.rescale_slope} + {reader.rescale_intercept}"
        scaler.input_data = reader.output
        scaler.Update()
        self._tgt_volume_data.ShallowCopy(scaler.output)
        if self._tgt_volume_data.number_of_points == 0:
            return
        self.logger.debug(
            "Loaded target volume scalar range [%f, %f] from %s",
            self._tgt_volume_data.scalar_range[0],
            self._tgt_volume_data.scalar_range[1],
            reader.file_name,
        )

        self._tgt_volume.mapper.input_data = self._tgt_volume_data
        if self.state.target_lut_source == "nn":
            ctf, otf, gof = self._build_network_transfer_functions(
                self._tgt_volume_data,
                snapshot_lut_in_state=True,
            )
            self._tgt_volume.property.SetColor(ctf)
            self._tgt_volume.property.SetScalarOpacity(otf)
            self._tgt_volume.property.SetGradientOpacity(gof)
        elif self.state.target_lut_source == "linear_map":
            ctf, otf, gof = self._map_ground_truth_lut_to_tgt_volume_scalars(
                self._tgt_volume_data,
                snapshot_lut_in_state=True,
            )
            self._tgt_volume.property.SetColor(ctf)
            self._tgt_volume.property.SetScalarOpacity(otf)
            self._tgt_volume.property.SetGradientOpacity(gof)
        else:
            self.logger.error("Unknown target_lut_source!")

        # Set state
        self.state.target_volume_file = str(path)

        # Render
        self._tgt_renderer.AddVolume(self._tgt_volume)
        self._check_all_loaded()
        self._tgt_renderer.ResetCamera()
        # Re-apply the crop plane client-side once the wasm mapper has synced.
        self._tgt_crop_apply_pending = True
        assert self._tgt_html_view is not None
        self._tgt_html_view.update(push_camera=True)

    def _reset_target_view(self):
        # Drop the client's cached wasm handles before the mapper is pruned from
        # the wasm scene by the next update() (else a watcher-fired crop apply
        # would invoke RemoveAllClippingPlanes on a dead object id).
        self.ctrl.crop_sync_teardown("tgt")
        # Reset state
        self.state.target_volume_file = ""
        self.state.tgt_colors = []
        self.state.tgt_opacities = []
        self.state.tgt_gradient_opacities = []
        self.state.tgt_scalar_range = []
        self.state.tgt_crop_enabled = False
        # Reset renderer
        self._tgt_renderer.RemoveAllViewProps()
        self._tgt_volume_data.Initialize()

    def _unload_target_volume(self):
        self._reset_target_view()
        assert self._tgt_html_view is not None
        self._tgt_html_view.update()
        self._check_all_loaded()

    def _load_from_config(self, config_path: str) -> None:
        self.logger.debug("Load configuration from %s", config_path)
        try:
            with pathlib.Path(config_path).open() as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.error("Could not load config %s: %s", config_path, exc)
            return

        if tf := cfg.get("transfer_function_file"):
            path = pathlib.Path(tf)
            if path.is_file():
                self._load_transfer_function(path)
            else:
                self.logger.error("transfer_function_file not found: %s", tf)

        if ref := cfg.get("reference_volume_file"):
            path = pathlib.Path(ref)
            if path.is_file():
                self._load_reference_volume(path)
            else:
                self.logger.error("reference_volume_file not found: %s", ref)

        if tgt := cfg.get("target_volume_file"):
            path = pathlib.Path(tgt)
            if path.is_file():
                self._load_target_volume(path)
            else:
                self.logger.error("target_volume_file not found: %s", tgt)
        # auto confirm
        # self.on_confirm_transfer()

    # ------------------------------------------------------------------
    # Permission dialog controllers
    # ------------------------------------------------------------------

    @controller.set("on_open_init_dialog")
    def on_open_init_dialog(self):
        self.state.show_init_dialog = True

    @controller.set("on_open_transfer_dialog")
    def on_open_transfer_dialog(self):
        self.state.show_transfer_dialog = True

    @controller.set("on_confirm_transfer")
    def on_confirm_transfer(self):
        self.state.allow_transfer = False
        self.state.show_transfer_dialog = False
        self.state.transfer_complete = False
        self.state.target_lut_source = "nn"
        # Lock the tgt crop UI for the duration of the transfer (unlocked once
        # the final tgt view update lands; see _on_target_view_updated).
        self.state.tgt_crop_locked = True
        self._queue_task(self._execute_transfer())

    # ------------------------------------------------------------------
    # Task manager
    # ------------------------------------------------------------------
    def _queue_task(self, coroutine):
        self._last_progress_update_time = 0.0
        task = asynchronous.create_task(coroutine)
        self._pending_tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task):
        self._pending_tasks.discard(task)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def reset_target_lut(self):
        if self.state.target_lut_source == "nn":
            self._model.load_state_dict(self._model_init_state)
            ctf, otf, gof = self._build_network_transfer_functions(
                self._tgt_volume_data, snapshot_lut_in_state=True
            )
            self.state.allow_transfer = True
            self._tgt_volume.property.SetColor(ctf)
            self._tgt_volume.property.SetScalarOpacity(otf)
            self._tgt_volume.property.SetGradientOpacity(gof)
            assert self._tgt_html_view is not None
            self._tgt_html_view.update()
        elif self.state.target_lut_source == "linear_map":
            ctf, otf, gof = self._map_ground_truth_lut_to_tgt_volume_scalars(
                self._tgt_volume_data, snapshot_lut_in_state=True
            )
            self.state.allow_transfer = True
            self._tgt_volume.property.SetColor(ctf)
            self._tgt_volume.property.SetScalarOpacity(otf)
            self._tgt_volume.property.SetGradientOpacity(gof)
            assert self._tgt_html_view is not None
            self._tgt_html_view.update()
        else:
            self.logger.error("Unknown target_lut_source!")

    async def _execute_transfer(self):
        self.logger.debug("Transferring lookup tables using network...")

        loop = asyncio.get_running_loop()

        def transfer_progress_callback(
            epoch_id: int, batch_id: int, progress_percent: float, loss: float
        ) -> None:
            self.logger.debug(
                "epoch: %d/%d - batch: %d - transfer progress: (%.2f) - loss: %.6f",
                epoch_id,
                self.state.n_epochs,
                batch_id,
                progress_percent,
                loss,
            )
            now = time.monotonic()
            if (
                # first progress event
                self._last_progress_update_time == 0.0
                # minimum 2 seconds between progress emissions
                or now - self._last_progress_update_time >= 2.0
                # last progress event
                or progress_percent == 100.0
            ):
                self._last_progress_update_time = now
                loop.call_soon_threadsafe(
                    self._on_transfer_progress, progress_percent, loss
                )

        self._transfer_executor_future = loop.run_in_executor(
            None,
            lambda: transfer_reference_lut(
                self._model,
                tgt_volume=self._tgt_volume_data,
                ref_volume=self._ref_volume_data,
                lut_rgb=self.ground_truth_colors,
                lut_scalar_alpha=self.ground_truth_opacities,
                lut_gradient_alpha=self.ground_truth_gradient_opacities,
                n_epochs=self.state.n_epochs,
                n_slices=self.state.n_slices,
                batch_size=self.state.batch_size,
                progress_callback=transfer_progress_callback,
                lr=self.state.learning_rate,
                margin=self.state.slice_plane_margin,
            ),
        )

        self._model = await self._transfer_executor_future
        self.logger.debug("Transfer complete!")
        with self.state:
            self.state.progress_percent = 100
            ctf, otf, gof = self._build_network_transfer_functions(
                self._tgt_volume_data, snapshot_lut_in_state=True
            )
            self._tgt_volume.property.SetColor(ctf)
            self._tgt_volume.property.SetScalarOpacity(otf)
            self._tgt_volume.property.SetGradientOpacity(gof)
            assert self._tgt_html_view is not None
            self._tgt_html_view.update()
            self.state.allow_transfer = False
            # The tgt crop UI stays locked until this final update lands on the
            # client, at which point _on_target_view_updated re-enables it.
            self._tgt_crop_unlock_pending = True

        return self._transfer_executor_future

    def _on_transfer_progress(self, progress_percent, loss):
        assert progress_percent >= 0, (
            "progress_percent must be greater than or equal to 0"
        )
        assert progress_percent <= 100, (
            "progress_percent must be less than or equal to 100"
        )
        with self.state:
            self.state.progress_percent = int(progress_percent)
            self.state.training_loss = loss
            ctf, otf, gof = self._build_network_transfer_functions(
                self._tgt_volume_data, snapshot_lut_in_state=True
            )
            self._tgt_volume.property.SetColor(ctf)
            self._tgt_volume.property.SetScalarOpacity(otf)
            self._tgt_volume.property.SetGradientOpacity(gof)
            assert self._tgt_html_view is not None
            self._tgt_html_view.update()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_all_loaded(self) -> None:
        ok = (
            self._ref_volume_data.number_of_points > 0
            and self._tgt_volume_data.number_of_points > 0
            and self._gt_ctf.size > 0
            and self._gt_sof.size > 0
            and self._gt_gof.size > 0
        )
        self.logger.debug("_check_all_loaded:ok= %d", ok)
        self.state.allow_transfer = ok

    def _initialize_transfer_functions(
        self,
        colors: npt.NDArray[np.float64],
        scalar_opacities: npt.NDArray[np.float64],
        gradient_opacities: npt.NDArray[np.float64] | None,
    ):
        self._gt_ctf.AddRGBPoints(
            numpy_to_vtk(colors[:, 0], deep=1),
            numpy_to_vtk(colors[:, 1:], deep=1),
        )
        self._gt_ctf.Build()

        for scalar, alpha in scalar_opacities:
            self._gt_sof.AddPoint(scalar, alpha)

        if gradient_opacities is not None:
            for gradient, alpha in gradient_opacities:
                self._gt_gof.AddPoint(gradient, alpha)

        # Set state
        self.state.ref_opacities = scalar_opacities.tolist()
        self.state.ref_colors = [(v, (r, g, b)) for (v, r, g, b) in colors.tolist()]
        self.state.ref_scalar_range = (
            float(colors[:, 0].min()),
            float(colors[:, 0].max()),
        )

    def _apply_gt_tf_to_reference_volume(self) -> None:
        self._ref_volume.property.SetColor(self._gt_ctf)
        self._ref_volume.property.SetScalarOpacity(self._gt_sof)
        self._ref_volume.property.SetGradientOpacity(self._gt_gof)

    def _map_ground_truth_lut_to_tgt_volume_scalars(
        self, volume: vtkImageData, snapshot_lut_in_state=False
    ) -> tuple:
        tgt_scalar_range = volume.scalar_range
        tgt_scalar_span = tgt_scalar_range[1] - tgt_scalar_range[0]
        ref_scalar_range = self._ref_volume_data.scalar_range
        ref_scalar_span = ref_scalar_range[1] - ref_scalar_range[0]

        ref_grad_mag_max = (
            compute_gradient_magnitude(
                load_vtk_image_to_tensor(self._ref_volume_data),
                self._ref_volume_data.spacing,
            )
            .max()
            .item()
        )
        tgt_grad_mag_max = (
            compute_gradient_magnitude(load_vtk_image_to_tensor(volume), volume.spacing)
            .max()
            .item()
        )
        new_rgb = self.ground_truth_colors.copy()
        new_alpha = self.ground_truth_opacities.copy()
        new_grad_alpha = self.ground_truth_gradient_opacities.copy()
        new_rgb[:, 0] -= ref_scalar_range[0]
        new_rgb[:, 0] /= ref_scalar_span
        new_rgb[:, 0] *= tgt_scalar_span
        new_rgb[:, 0] += tgt_scalar_range[0]
        new_alpha[:, 0] -= ref_scalar_range[0]
        new_alpha[:, 0] /= ref_scalar_span
        new_alpha[:, 0] *= tgt_scalar_span
        new_alpha[:, 0] += tgt_scalar_range[0]
        new_grad_alpha[:, 0] *= tgt_grad_mag_max / ref_grad_mag_max

        if snapshot_lut_in_state:
            (
                self.state.tgt_colors,
                self.state.tgt_opacities,
                self.state.tgt_gradient_opacities,
            ) = convert_lut_to_state_format(new_rgb, new_alpha, new_grad_alpha)
            self.state.tgt_scalar_range = list(volume.scalar_range)
        ctf = vtkDiscretizableColorTransferFunction(allow_duplicate_scalars=True)
        ctf.AddRGBPoints(
            numpy_to_vtk(new_rgb[:, 0], deep=1),
            numpy_to_vtk(new_rgb[:, 1:], deep=1),
        )
        ctf.Build()

        otf = vtkPiecewiseFunction()
        for scalar, alpha in new_alpha:
            otf.AddPoint(float(scalar), float(alpha))

        gof = vtkPiecewiseFunction()
        for gradient, new_alpha in new_grad_alpha:
            gof.AddPoint(float(gradient), float(new_alpha))
        return ctf, otf, gof

    def _build_network_transfer_functions(
        self, volume: vtkImageData, snapshot_lut_in_state=False
    ) -> tuple:
        """Build VTK CTF/OTF from 64 network samples."""
        colors, opacities, gradient_opacities = lut_from_network(
            self._model, volume, n_points=self.state.n_lut_sampling_points
        )
        if snapshot_lut_in_state:
            self.state.tgt_colors = colors
            self.state.tgt_opacities = opacities
            self.state.tgt_gradient_opacities = gradient_opacities
            self.state.tgt_scalar_range = list(volume.scalar_range)

        scalars_np = np.array([s for s, _ in colors], dtype=np.float64)
        rgb_np = np.array([list(rgb) for _, rgb in colors], dtype=np.float64)

        ctf = vtkDiscretizableColorTransferFunction(allow_duplicate_scalars=True)
        ctf.AddRGBPoints(
            numpy_to_vtk(scalars_np.copy(), deep=1),
            numpy_to_vtk(rgb_np.copy(), deep=1),
        )
        ctf.Build()

        otf = vtkPiecewiseFunction()
        for scalar, alpha in opacities:
            otf.AddPoint(float(scalar), float(alpha))

        gof = vtkPiecewiseFunction()
        for gradient, alpha in gradient_opacities:
            gof.AddPoint(float(gradient), float(alpha))

        return ctf, otf, gof

    def _setup_vtk_pipeline(self):
        window = vtkRenderWindow(interactor=vtkRenderWindowInteractor())
        window.interactor.interactor_style.SetCurrentStyleToTrackballCamera()
        renderer = vtkRenderer(background=(0.2, 0.2, 0.2))
        window.AddRenderer(renderer)

        mapper = vtkSmartVolumeMapper()
        mapper.SetBlendModeToComposite()

        volume_actor = vtkVolume()
        volume_actor.mapper = mapper

        volume_property = vtkVolumeProperty()
        volume_actor.SetProperty(volume_property)
        volume_property.ShadeOff()
        volume_property.SetScalarOpacityUnitDistance(1.754420659713536)
        volume_property.SetScatteringAnisotropy(0)

        # Slice/crop plane. Attached to the mapper on demand.
        crop_plane = vtkPlane()

        return window, renderer, volume_actor, crop_plane

    # ------------------------------------------------------------------
    # Linked views (client-side camera + slice/crop plane sync)
    # ------------------------------------------------------------------

    def _apply_crop_client(self, key: str, relink: bool = False) -> None:
        """Forward a view's wasm handles + volume bounds to the client and have
        it (re)apply the clipping plane there. All ongoing crop changes are
        handled in the browser via state watchers. When ``relink`` is set and the
        views are linked, the client re-adopts the reference crop first.
        See color_transfer_function_designer.app.module.serve.crop.js.
        """
        if key == "ref":
            html_view, mapper = self._ref_html_view, self._ref_volume.mapper
            plane_id, volume_data = self._ref_plane_wasm_id, self._ref_volume_data
        else:
            html_view, mapper = self._tgt_html_view, self._tgt_volume.mapper
            plane_id, volume_data = self._tgt_plane_wasm_id, self._tgt_volume_data
        if html_view is None or volume_data is None or plane_id is None:
            return
        self.ctrl.crop_sync_init(
            {
                "key": key,
                "refName": html_view.ref_name,
                "mapperId": html_view.get_wasm_id(mapper),
                "planeId": plane_id,
                "bounds": list(volume_data.bounds),
                "relink": relink,
            }
        )

    def _on_reference_view_updated(self, **_):
        self._init_reference_view_camera_sync()
        if self._ref_crop_apply_pending:
            self._ref_crop_apply_pending = False
            self._apply_crop_client("ref")

    def _on_target_view_updated(self, **_):
        self._init_target_view_camera_sync()
        if self._tgt_crop_apply_pending:
            self._tgt_crop_apply_pending = False
            self._apply_crop_client("tgt")
        if self._tgt_crop_unlock_pending:
            # The final transfer update has landed: re-enable the tgt crop UI and
            # re-apply the crop (transfer updates wiped the client clip). No more
            # tgt updates are in flight, so there is nothing left to race.
            self._tgt_crop_unlock_pending = False
            self.state.tgt_crop_locked = False
            self._apply_crop_client("tgt", relink=self.state.crop_linked)

    def _init_reference_view_camera_sync(self, **_):
        """Hand the renderer wasm ids to the client once the reference view is ready.
        Fires on every LocalView ``updated`` event.
        """
        if self._ref_renderer_wasm_id is None or self._tgt_renderer_wasm_id is None:
            return
        assert self._ref_html_view is not None
        assert self._tgt_html_view is not None
        self.ctrl.camera_sync_init(
            {
                "srcRefName": self._ref_html_view.ref_name,
                "dstRefName": self._tgt_html_view.ref_name,
                "srcRendererId": self._ref_renderer_wasm_id,
            }
        )

    def _init_target_view_camera_sync(self, **_):
        """Hand the renderer wasm ids to the client once the target view is ready.
        Fires on every LocalView ``updated`` event.
        """
        if self._ref_renderer_wasm_id is None or self._tgt_renderer_wasm_id is None:
            return
        assert self._ref_html_view is not None
        assert self._tgt_html_view is not None
        self.ctrl.camera_sync_init(
            {
                "srcRefName": self._tgt_html_view.ref_name,
                "dstRefName": self._ref_html_view.ref_name,
                "srcRendererId": self._tgt_renderer_wasm_id,
            }
        )

    # ------------------------------------------------------------------
    # Editor node callbacks (stubs)
    # ------------------------------------------------------------------

    def on_opacity_node_modified(self, _index, _node):
        self.logger.debug("Opacity node %d modified to %s", _index, str(_node))
        self._gt_sof.SetNodeValue(_index, [*_node, 0.5, 0.0])
        assert self._ref_html_view is not None
        self._ref_html_view.update()
        self.state.allow_transfer = True

    def on_opacity_node_added(self, _index, _node):
        pass

    def on_opacity_node_removed(self, _index):
        pass

    def on_color_node_modified(self, _index, _node):
        self.logger.debug("Color node %d modified to %s", _index, str(_node))
        self._gt_ctf.SetNodeValue(_index, [_node[0], *_node[1], 0.5, 0.0])
        assert self._ref_html_view is not None
        self._ref_html_view.update()
        self.state.allow_transfer = True

    def on_color_node_added(self, _index, _node):
        pass

    def on_color_node_removed(self, _index):
        pass

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def _ref_color_opacity_editor(self):
        return color_opacity_editor.ColorOpacityEditor(
            classes="align-center",
            v_if=("transfer_function_file.length > 0",),
            style="width: 100%; max-height: 150px;",
            v_model_colorNodes="ref_colors",
            v_model_opacityNodes=("ref_opacities",),
            scalar_range=("ref_scalar_range",),
            opacity_node_modified=(self.on_opacity_node_modified, "$event"),
            opacity_node_added=(self.on_opacity_node_added, "$event"),
            opacity_node_removed=(self.on_opacity_node_removed, "[$event]"),
            color_node_modified=(self.on_color_node_modified, "$event"),
            color_node_added=(self.on_color_node_added, "$event"),
            color_node_removed=(self.on_color_node_removed, "[$event]"),
            histograms=("ref_histograms",),
            histograms_range=("ref_hist_y_range",),
            show_histograms=("show_histograms",),
            histograms_color=("histograms_color", [0, 0, 0, 0.25]),
            background_shape=("background_shape",),
            background_opacity=("background_opacity",),
            handle_radius=7,
            line_width=2,
            viewport_padding=("viewport_padding", [8, 8]),
            handle_color=("handle_color", [0.125, 0.125, 0.125, 1]),
            handle_border_color=("handle_border_color", [0.75, 0.75, 0.75, 1]),
        )

    def _tgt_color_opacity_editor(self):
        return color_opacity_editor.ColorOpacityEditor(
            classes="align-center",
            style="width: 100%; max-height: 150px;",
            v_model_colorNodes="tgt_colors",
            v_model_opacityNodes=("tgt_opacities",),
            scalar_range=("tgt_scalar_range",),
            opacity_node_modified=(lambda _, __: None, "$event"),
            opacity_node_added=(lambda _, __: None, "$event"),
            opacity_node_removed=(lambda _: None, "[$event]"),
            color_node_modified=(lambda _, __: None, "$event"),
            color_node_added=(lambda _, __: None, "$event"),
            color_node_removed=(lambda _: None, "[$event]"),
            histograms=("tgt_histograms",),
            histograms_range=("tgt_hist_y_range",),
            show_histograms=("show_histograms",),
            histograms_color=("histograms_color", [0, 0, 0, 0.25]),
            background_shape=("background_shape",),
            background_opacity=("background_opacity",),
            handle_radius=7,
            line_width=2,
            viewport_padding=("viewport_padding", [8, 8]),
            handle_color=("handle_color", [0.125, 0.125, 0.125, 1]),
            handle_border_color=("handle_border_color", [0.75, 0.75, 0.75, 1]),
        )

    def _crop_plane_toolbar(self, key: str, volume_state: str):
        """Toolbar to enable/orient/position the crop plane for a view.

        Shared by both LocalViews; ``key`` is "ref" or "tgt" and ``volume_state``
        is the state variable holding that view's loaded file path.
        """
        enabled = f"{key}_crop_enabled"
        direction = f"{key}_crop_direction"
        position = f"{key}_crop_position"
        # The tgt crop UI is locked while a transfer runs (see tgt_crop_locked).
        lock_expr = "tgt_crop_locked" if key == "tgt" else "false"
        with vuetify3.VToolbar(
            density="compact",
            color="transparent",
            flat=True,
            v_show=(f"{volume_state}.length > 0",),
        ):
            # Link toggles shared by both views: mirror the crop plane / camera.
            with vuetify3.VBtn(
                size="small",
                prepend_icon=(
                    "crop_linked ? 'mdi-link-variant' : 'mdi-link-variant-off'",
                ),
                density="compact",
                variant=("crop_linked ? 'tonal' : 'text'",),
                color=("crop_linked ? 'primary' : ''",),
                click="crop_linked = !crop_linked",
                classes="ml-2",
            ):
                vuetify3.VTooltip(
                    "Link crop plane across both views",
                    activator="parent",
                    location="bottom",
                )
            with vuetify3.VBtn(
                size="small",
                prepend_icon=("camera_linked ? 'mdi-link' : 'mdi-link-off'",),
                density="compact",
                variant=("camera_linked ? 'tonal' : 'text'",),
                color=("camera_linked ? 'primary' : ''",),
                click="camera_linked = !camera_linked",
            ):
                vuetify3.VTooltip(
                    "Link camera across both views",
                    activator="parent",
                    location="bottom",
                )
            vuetify3.VDivider(vertical=True, classes="mx-2")
            vuetify3.VSwitch(
                v_model=(enabled,),
                label="Crop",
                color="primary",
                density="compact",
                hide_details=True,
                inset=True,
                disabled=(lock_expr,),
                classes="flex-grow-0 mx-2",
            )
            with vuetify3.VBtnToggle(
                v_model=(direction,),
                mandatory=True,
                density="compact",
                variant="outlined",
                divided=True,
                disabled=(f"!{enabled} || {lock_expr}",),
                classes="mx-2",
            ):
                for label, value in self._SLICE_DIRECTIONS:
                    vuetify3.VBtn(label, value=value, size="small")
            vuetify3.VSlider(
                v_model=(position,),
                min=0,
                max=100,
                step=1,
                prepend_icon="mdi-arrow-split-vertical",
                thumb_label=True,
                hide_details=True,
                density="compact",
                disabled=(f"!{enabled} || {lock_expr}",),
                classes="mx-2",
            )

    def _generate_ui(self):
        self.logger.debug("Show page 1")
        with SinglePageWithDrawerLayout(self.server) as self.ui:
            # Permission dialog 1 - model transfer
            with vuetify3.VDialog(
                model_value=("show_transfer_dialog",),
                max_width=440,
                persistent=True,
            ):
                with vuetify3.VCard():
                    vuetify3.VCardTitle("Transfer transfer function?")
                    vuetify3.VCardText(
                        f"The network will transfer the provided lookup tables from the reference volume to the scalar volume using {self.state.n_epochs} epochs. Proceed?"
                    )
                    with vuetify3.VCardActions():
                        vuetify3.VSpacer()
                        vuetify3.VBtn(
                            "Cancel",
                            variant="text",
                            click="show_transfer_dialog = false",
                        )
                        vuetify3.VBtn(
                            "Confirm",
                            color="primary",
                            variant="tonal",
                            click=self.on_confirm_transfer,
                        )

            # File dialog
            FileDialog(is_open=False, file_browser=self.file_browser)

            # Client-side camera sync between the two LocalViews.
            # `camera_sync_init` captures the renderer id for a view and observes its camera for modifications
            # `camera_sync_once` snaps the views together when linking.
            self.ctrl.camera_sync_init = client.JSEval(
                exec="utils.colorTransferFunctionDesignerCamera.setup($event.srcRefName, $event.dstRefName, $event.srcRendererId)",
            ).exec
            self.ctrl.camera_sync_once = client.JSEval(
                exec="utils.colorTransferFunctionDesignerCamera.sync($event.srcRefName, $event.dstRefName)",
            ).exec
            self.ctrl.crop_sync_init = client.JSEval(
                exec="utils.colorTransferFunctionDesignerCrop.setup($event)",
            ).exec
            self.ctrl.crop_sync_teardown = client.JSEval(
                exec="utils.colorTransferFunctionDesignerCrop.teardown($event)",
            ).exec

            with self.ui.drawer:
                with vuetify3.VCard():
                    vuetify3.VCardTitle("Training")
                    vuetify3.VNumberInput(
                        v_model=("n_epochs",),
                        label="Epochs",
                        min=(1,),
                        max=(30,),
                        step=(1,),
                        control_variant="split",
                        classes="mx-2",
                    )
                    vuetify3.VNumberInput(
                        v_model=("n_slices",),
                        label="Slices",
                        min=(256,),
                        max=(4096,),
                        step=(256,),
                        control_variant="split",
                        classes="mx-2",
                    )
                    vuetify3.VNumberInput(
                        v_model=("batch_size",),
                        label="Batch Size",
                        control_variant="split",
                        classes="mx-2",
                    )
                    vuetify3.VNumberInput(
                        v_model=("learning_rate",),
                        min=(1.0e-5,),
                        max=(0.1,),
                        step=(1.0e-6,),
                        precision=(7,),
                        label="Learning rate",
                        control_variant="split",
                        classes="mx-2",
                    )
                with vuetify3.VCard():
                    vuetify3.VCardTitle("2D Slice plane")
                    vuetify3.VNumberInput(
                        v_model=("slice_plane_margin",),
                        label="Margin",
                        min=(0.0),
                        max=(1.0,),
                        step=(0.05,),
                        precision=(3,),
                        control_variant="split",
                        classes="mx-2",
                    )
                with vuetify3.VCard():
                    vuetify3.VCardTitle("Lookup table")
                    vuetify3.VNumberInput(
                        v_model=("n_lut_sampling_points",),
                        label="No. of sampling points",
                        min=(8),
                        max=(64,),
                        step=(2,),
                        control_variant="split",
                        classes="mx-2",
                    )
                vuetify3.VSpacer()
                with vuetify3.VCardActions():
                    vuetify3.VBtn(
                        "Reset",
                        color="primary",
                        variant="tonal",
                        click=self._set_default_parameters,
                    )

            with self.ui.content:
                with vuetify3.VContainer(
                    fluid=True,
                    classes="fill-height d-flex flex-column pa-4",
                    style="gap: 16px;",
                ):
                    # Row - Color-opacity editor
                    with vuetify3.VRow(
                        classes="d-flex align-center justify-center flex-grow-0",
                        style="width: 100%",
                    ):
                        with vuetify3.VCol(
                            cols="12",
                            md="6",
                            classes="d-flex flex-column",
                        ):
                            with vuetify3.VCard(
                                variant="outlined",
                                style="min-height: 150px;",
                                classes="d-flex flex-column",
                            ):
                                with vuetify3.VRow(
                                    classes="flex-shrink-0 justify-end ma-0",
                                    v_show=("transfer_function_file.length > 0",),
                                ):
                                    vuetify3.VBtn(
                                        icon="mdi-close",
                                        size="small",
                                        density="compact",
                                        variant="text",
                                        click=self._unload_transfer_function,
                                    )
                                with vuetify3.VRow(
                                    classes="flex-grow-1 align-center justify-center ma-0",
                                    v_show=("transfer_function_file.length == 0",),
                                ):
                                    vuetify3.VBtn(
                                        text="1. Load a transfer function file (.vp, .json)",
                                        click=(
                                            self.ctrl.open_file_dialog,
                                            "['transfer_function']",
                                        ),
                                    )
                                self._ref_color_opacity_editor()
                        with vuetify3.VCol(
                            cols="12",
                            md="6",
                            classes="d-flex flex-column",
                            v_if=("target_volume_file.length > 0",),
                        ):
                            with vuetify3.VCard(
                                variant="outlined",
                                style="min-height: 150px",
                                classes="d-flex flex-column",
                            ):
                                with vuetify3.VRow(
                                    classes="flex-shrink-0 justify-end ma-0",
                                ):
                                    vuetify3.VBtn(
                                        icon="mdi-download",
                                        size="small",
                                        density="compact",
                                        variant="text",
                                        click=self.open_export_dialog,
                                    )
                                    vuetify3.VBtn(
                                        icon="mdi-restart",
                                        size="small",
                                        density="compact",
                                        variant="text",
                                        click=self.reset_target_lut,
                                    )
                                self._tgt_color_opacity_editor()

                    # Row — 3D views
                    with vuetify3.VRow(classes="flex-grow-1", style="width: 100%"):
                        with vuetify3.VCol(
                            cols="12",
                            md="6",
                            classes="d-flex flex-column",
                        ):
                            with vuetify3.VCard(
                                variant="outlined",
                                classes="d-flex flex-grow-1 flex-column",
                            ):
                                with vuetify3.VRow(
                                    classes="flex-shrink-0 justify-end ma-0",
                                    v_show=("reference_volume_file.length > 0",),
                                ):
                                    vuetify3.VBtn(
                                        icon="mdi-close",
                                        size="small",
                                        density="compact",
                                        variant="text",
                                        click=self._unload_reference_volume,
                                    )
                                with vuetify3.VRow(
                                    classes="flex-grow-1 align-center justify-center ma-0",
                                    v_show=("reference_volume_file.length === 0",),
                                ):
                                    vuetify3.VBtn(
                                        text="2. Load a reference volume (*.nii.gz)",
                                        click=(
                                            self.ctrl.open_file_dialog,
                                            "['reference_volume']",
                                        ),
                                    )
                                self._crop_plane_toolbar("ref", "reference_volume_file")
                                self._ref_html_view = vtklocal.LocalView(
                                    self._ref_wnd,
                                    ref="ref_view",
                                    # interactive_ratio=1,
                                    style="width: 100%; min-height: 0;",
                                    v_show=("reference_volume_file.length > 0",),
                                    updated=self._on_reference_view_updated,
                                )
                                self._ref_renderer_wasm_id = (
                                    self._ref_html_view.get_wasm_id(self._ref_renderer)
                                )
                                self._ref_plane_wasm_id = (
                                    self._ref_html_view.register_vtk_object(
                                        self._ref_plane
                                    )
                                )
                        with vuetify3.VCol(
                            cols="12",
                            md="6",
                            classes="d-flex flex-column",
                        ):
                            with vuetify3.VCard(
                                variant="outlined",
                                classes="d-flex flex-grow-1 flex-column",
                            ):
                                with vuetify3.VRow(
                                    classes="flex-shrink-0 justify-end ma-0",
                                    v_show=("target_volume_file.length > 0",),
                                ):
                                    vuetify3.VBtn(
                                        icon="mdi-close",
                                        size="small",
                                        density="compact",
                                        variant="text",
                                        click=self._unload_target_volume,
                                    )
                                with vuetify3.VRow(
                                    classes="flex-grow-1 align-center justify-center ma-0",
                                    v_show=("target_volume_file.length === 0",),
                                ):
                                    vuetify3.VBtn(
                                        text="3. Load the volume with continuous scalar intensities (*.nii.gz)",
                                        click=(
                                            self.ctrl.open_file_dialog,
                                            "['target_volume']",
                                        ),
                                    )
                                self._crop_plane_toolbar("tgt", "target_volume_file")
                                self._tgt_html_view = vtklocal.LocalView(
                                    self._tgt_wnd,
                                    ref="tgt_view",
                                    # interactive_ratio=1,
                                    style="width: 100%; min-height: 0;",
                                    v_show=("target_volume_file.length > 0",),
                                    updated=self._on_target_view_updated,
                                )
                                self._tgt_renderer_wasm_id = (
                                    self._tgt_html_view.get_wasm_id(self._tgt_renderer)
                                )
                                self._tgt_plane_wasm_id = (
                                    self._tgt_html_view.register_vtk_object(
                                        self._tgt_plane
                                    )
                                )

                    # Row — actions
                    with vuetify3.VRow(classes="flex-grow-0", style="width: 100%"):
                        with vuetify3.VCol(cols="12"):
                            with vuetify3.VCardActions():
                                vuetify3.VSpacer()
                                with vuetify3.VCard(
                                    variant="tonal",
                                    classes="d-flex",
                                ):
                                    with vuetify3.VBtnToggle(
                                        v_model=("target_lut_source",),
                                        mandatory=True,
                                        rounded=True,
                                        border=True,
                                    ):
                                        vuetify3.VBtn("Neural network", value="nn")
                                        vuetify3.VBtn("Linear map", value="linear_map")
                                vuetify3.VBtn(
                                    "Transfer",
                                    color="primary",
                                    variant="tonal",
                                    click=self.on_open_transfer_dialog,
                                    disabled=("!allow_transfer",),
                                )
                            vuetify3.VProgressLinear(
                                model_value=("progress_percent",),
                                v_if=("progress_percent > 0 && progress_percent < 100"),
                                color="primary",
                                height=8,
                                rounded=True,
                            )
