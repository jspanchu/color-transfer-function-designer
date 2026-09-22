# CFD Features app: implementation plan

Goal: a mini application (`apps/cfd_features`, console script
`ctfd-cfd-features`) that loads a resampled CFD volume, classifies every voxel
into named flow features with a small network, lets the analyst correct the
network by scribbling on a slice, and renders the features as a composited RGBA
volume. Trained detectors are saved as JSON and can be applied to other volumes
offline.

First dataset: `data/scivis/ahmed_wake.vti` (snappyHexMesh Ahmed body, SST,
corrected inlet). Second dataset for cross-solver testing:
`data/scivis/AhmedBody-BARAMCFD` (realizable k-epsilon). AhmedML runs come
later.

## Decisions already made

- Per-voxel multi-label training. No image-space loss.
- Network outputs feature probabilities, not colors. Colors and opacities are
  user-chosen in the palette and blended after inference.
- Inputs are non-dimensional so detectors transfer between cases.
- Rendering in version one is a precomputed RGBA array through the stock GPU ray
  cast mapper. No custom shader.
- UI is trame with the React client and trame-mui. The medical app stays on Vue.
- Labels come from rule detectors (soft, weight 1, class balanced) overridden by
  scribbles (hard, weight about 10). Body voxels and a one-voxel shell have
  weight 0.

## React client conventions

Verified in `examples/react_smoke.py` (MUI slider driving a vtklocal volume
view). These differ from the Vue patterns in the medical app:

- Server: `get_server(client_type="react")`; layout from `trame.ui.mui`.
- State to prop: `value=react.Bind("opacity")`, not `value=("opacity",)`.
- Prop to state:
  `on_change=react.Callback("opacity = Number($event.target.value)")`, not a Vue
  arrow-function string. `$event` is the DOM event.
- vtklocal view events use the `on_*` names (see
  `trame_vtklocal/widgets/vtklocal.py`).
- The Vue-only pieces (`ui/file.py`, the color-opacity editor) are not used.

Dataset pressure conventions for the adapter in M1:

- snappy Ahmed: kinematic gauge pressure, `rho = 1`, `p_ref = 0`.
- BARAM Ahmed: absolute pressure in Pa, `rho = 1.2`, `p_ref` = freestream value
  (about 1.012e5), has `epsilon` instead of `omega`.
- AhmedML: inlet 1 m/s, geometry parameters in mm; check coordinate units before
  choosing crop bounds; k is resolved from `UPrime2Mean`, not modeled.

## Data contracts

### Feature vector (fixed order, all dimensionless)

| index | name            | definition                                                     |
| ----- | --------------- | -------------------------------------------------------------- |
| 0     | `cp`            | (p - p*ref) / (0.5 * rho \_ U_inf^2). For kinematic p, rho = 1 |
| 1     | `ux`            | U_x / U_inf                                                    |
| 2     | `speed`         | \|U\| / U_inf                                                  |
| 3     | `q`             | Q \* L^2 / U_inf^2                                             |
| 4     | `vort`          | \|omega\| \* L / U_inf                                         |
| 5     | `tke`           | k / U_inf^2                                                    |
| 6     | `log_nut_ratio` | log10(nut / nu + 1)                                            |

Reference values: `U_inf`, `L`, `nu`, `rho`, `p_ref`, plus the flow axis
(default +x). Read from the case when possible, editable in the app.

### Features (fixed order, four to fit one RGBA texture later)

| index | name          | rule (soft threshold, see below) |
| ----- | ------------- | -------------------------------- |
| 0     | vortex core   | `q > 2` and `cp < -0.1`          |
| 1     | shear layer   | `vort > 5` and `q < 0`           |
| 2     | recirculation | `ux < -0.05`                     |
| 3     | wake          | `speed < 0.85` or `tke > 0.003`  |

Soft threshold: `s(x, t) = sigmoid((x - t) / (0.1 * |t|))`. Conjunctions
multiply, disjunctions use `1 - (1 - a)(1 - b)`. Thresholds are constructor
arguments; the values above were chosen from the baseline percentiles and will
be tuned by eye.

### Weights file (JSON)

```json
{
  "version": 1,
  "inputs": ["cp", "ux", "speed", "q", "vort", "tke", "log_nut_ratio"],
  "features": ["vortex_core", "shear_layer", "recirculation", "wake"],
  "input_mean": [...7], "input_std": [...7],
  "layers": [{"w": [[...]], "b": [...]}, ...],
  "reference": {"U_inf": 40, "L": 1.044, "nu": 1.5e-5, "rho": 1.0, "p_ref": 0.0}
}
```

Plain lists so a NumPy forward pass in ParaView needs no torch.

### Scribbles file (JSON)

```json
{"version": 1, "volume": "ahmed_wake.vti", "dims": [308, 128, 64],
 "strokes": [{"feature": 0, "label": 1, "voxels": [flat indices...]}, ...]}
```

### Output volume

Input arrays plus `feat_<name>` (float32 probability per feature),
`features_rgba` (4-component uint8), and the existing `vtkValidPointMask`.

## Loss and training

- Loss: weighted multi-label BCE, `sum(w * bce(p, y)) / sum(w)`, sigmoid
  outputs.
- Weights: 0 inside body and shell; rule labels weight 1 times a per-feature
  balance factor `0.5 / mean(y_f)` clipped to [1, 50]; scribble voxels weight
  `scribble_weight` (default 10) for the scribbled feature only.
- Network: 7 -> 32 -> 32 -> 4, ReLU, sigmoid output. Inputs standardized with
  the stored mean and std. No positional embedding.
- Sampler: each step draws 64k voxels, half from feature positives (equal share
  per feature), half from negatives, plus every scribble voxel.
- Optimizer: Adam, lr 3e-3, 300 steps for a fresh fit, 100 steps for a continue.
- Validation split: hold out the downstream slab `x > 2.5 m` (scaled: x/L >
  2.4). Never random voxel splits; never a y-half split (mirror symmetry).
- Metrics: per-feature IoU at 0.5 against rule labels and against scribbles in
  the held-out slab. Reported in the app after each training run.

## Milestones

### M0. Environment (half a day)

- pyproject: add `trame-mui`, register `ctfd-cfd-features`, add the two new lib
  modules to `[tool.coverage.report] include`.
- Create `apps/cfd_features/{__init__.py, main.py, core.py}` mirroring
  `apps/medical` but with `get_server(client_type="react")`.
- Acceptance: `ctfd-cfd-features` starts and shows an empty MUI page.

### M1. `lib/flow_features.py` (2 days)

Pure NumPy/Torch, 100% covered.

- `ReferenceValues` dataclass and `reference_from_openfoam(case_dir)` helper
  (reads U_inf from `0.orig/U` or `0/U`, nu from transport properties; L and rho
  are arguments).
- `feature_vector(image, ref) -> (N, 7) float32` from a vtkImageData holding
  `U`, `p`, `Q`, `VorticityMag`, `k`, `nut`. Uses `load_vtk_image_to_tensor`
  style helpers from `dataset.py`.
- `body_mask(image, dilate=1) -> (N,) bool` from `vtkValidPointMask`.
- `RuleThresholds` dataclass and `rule_labels(x, thresholds) -> (N, 4)` soft
  labels.
- `assemble_labels(rule, mask, scribbles, scribble_weight) -> (y, w)`.
- `holdout_mask(image, axis, fraction)` for the validation slab.
- Tests: synthetic 8x8x8 volumes with analytic fields. Solid-body rotation gives
  Q > 0 and known vorticity; plane shear gives Q < 0 with the same vorticity;
  reversed flow triggers recirculation; a constant field gives all-zero labels.
  Check the seven inputs against hand-computed values, threshold monotonicity,
  masking, scribble override, and the holdout slab shape.

### M2. `lib/feature_net.py` (2 days)

- `FeatureNet(n_in=7, hidden=32, n_out=4)`.
- `Standardizer` (mean/std, fit on valid voxels).
- `train(net, x, y, w, steps, lr, batch, progress_cb) -> history` with the
  balanced sampler and a progress callback shaped like the medical app's
  `(step, loss)`.
- `predict(net, x, chunk=1<<20) -> (N, 4)` probabilities.
- `blend_rgba(probs, colors, opacities) -> (N, 4) uint8` with the blend rule:
  color = probability-weighted average of feature colors; alpha = 1 - prod(1 -
  p_f \* a_f).
- `save_weights(path, net, standardizer, ref)` / `load_weights(path)` and
  `predict_numpy(weights_dict, x)` (pure NumPy, used by tests to prove the JSON
  round-trips and by the future ParaView filter).
- `metrics(probs, y, w, mask) -> {feature: iou}`.
- Tests: shapes, loss decreases on a separable toy problem, sampler balance,
  torch vs NumPy forward agreement within 1e-6, blend edge cases (all zero, one
  feature, overlapping), JSON round trip.

### M3. Offline scripts (1 day)

- `scripts/train_features.py --volume --ref ... --scribbles --out weights.json`
  prints metrics and writes `*_features.vti` with probabilities and RGBA.
- `scripts/apply_features.py --weights --volume --out` for batch evaluation on
  other volumes.
- Acceptance: run on `ahmed_wake.vti`, load the output in ParaView with Map
  Scalars off, see vortex tubes, shear layers, recirculation, wake in one
  render. Run `apply` on the BARAM volume (after resampling it with the same
  script and a pressure adapter) and eyeball the result. This is the first
  transfer test.

### M4. App: volume view and palette (3 days)

- `core.py`: `App(TrameApp)` with state for file path, reference values, four
  palette entries (visible, color, opacity), training params, progress, metrics.
- Layout (`trame.ui.mui.SinglePageLayout`): toolbar with path field, Load,
  Train, Reset, progress; content as a three-column `mui.Grid`: slice view
  placeholder, `vtklocal.LocalView` of the RGBA volume, palette panel.
- Palette rows: `Switch`, color swatch (`TextField type=color` is enough),
  `Slider` opacity, chip showing "rule" or "trained", stroke count.
- On Load: read vti, compute feature vector and rule labels, predict with rules
  only, blend, upload RGBA to the mapper (4 components, independent components
  off, opacity from the fourth channel).
- On Train: run `train` in a task like `_execute_transfer` in the medical app,
  stream progress, then predict, blend, refresh, show metrics.
- Palette changes re-blend only (no inference) and refresh.
- Acceptance: load, see rule-based render, change a color, train, see the render
  update, metrics appear.

### M5. Slice view and scribbles (3 days)

- Slice view: second `vtklocal.LocalView` with a `vtkImageSlice` of the chosen
  background field plus a second slice of the RGBA feature array on top,
  parallel projection, axis and index controls.
- Scribble overlay: a plain DOM canvas over the slice view, served from
  `ui/module/serve/scribble.js`. Pointer events record pixel strokes; on release
  it calls a trame trigger with the stroke pixels, brush radius, active feature,
  and label. Python converts pixels to voxel indices via the slice camera (or by
  rendering the slice at a known pixel-to-voxel mapping with parallel projection
  and a fixed viewport, which is simpler and what version one does).
- Label store in `App`: list of strokes, undo last, clear feature, save/load
  JSON.
- Train continues from current weights when strokes exist.
- Acceptance: paint a stroke, train, watch the slice overlay and the volume
  change where painted; save scribbles, restart, load, retrain reproduces.

### M6. Validation and export (1 day)

- Holdout slab toggle; metrics table in the palette panel for train and holdout.
- Export button: write `*_features.vti` and `weights.json`.
- Acceptance: the numbers in the app match `scripts/train_features.py`.

### Later, not in this plan

- AhmedML download-and-resample pipeline and the split described in the notes.
- Compare filter (two volumes, category array).
- ParaView Python plugin that loads `weights.json` and runs `predict_numpy`.
- Shader-side blending of four probability channels with palette uniforms.

## Order and estimate

M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6, about two and a half weeks. M3 is the
first point with a shareable result and the first transfer test; do not skip it
to get to the UI sooner.

## Risks

- trame-mui is new (1.0.0). Widget prop names may need reading the generated
  Python. Mitigation: keep the UI to Grid, Box, Switch, Slider, Button,
  TextField, Chip, Typography.
- Pixel-to-voxel mapping for scribbles. Mitigation: parallel projection with a
  fixed viewport and camera per slice so the mapping is a closed-form affine.
- Rule thresholds may be wrong for the BARAM case even after scaling, because
  RANS k differs from resolved k. Mitigation: thresholds are data, not code;
  scribbles override.
- Body shell masking hides near-wall features. Acceptable for version one; the
  boundary layer is not among the four features.
