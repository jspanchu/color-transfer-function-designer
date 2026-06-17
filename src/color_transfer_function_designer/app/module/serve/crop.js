(() => {
  let cropSyncInitialized = false;

  const cropViews = {
    ref: { refName: null, mapperId: null, planeId: null, bounds: null },
    tgt: { refName: null, mapperId: null, planeId: null, bounds: null },
  };

  // Toolbar direction -> clipping-plane normal / axis
  // A volume mapper keeps the half-space the normal points
  // toward, so e.g. cropping the "+x" half (keeping "-x") needs a -x normal.
  const CROP_NORMALS = {
    "+x": [-1, 0, 0],
    "-x": [1, 0, 0],
    "+y": [0, -1, 0],
    "-y": [0, 1, 0],
    "+z": [0, 0, -1],
    "-z": [0, 0, 1],
  };
  const CROP_AXIS = { "+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2 };

  // Sync a view's wasm clipping plane to its current crop toolbar state. The
  // plane is axis-aligned; position is a percentage [0, 100] along the active
  // axis so it stays valid when the direction changes.
  async function applyCrop(key) {
    const cfg = cropViews[key];
    if (!cfg || cfg.mapperId === null || cfg.bounds === null) return;
    const view = window.trame.refs[cfg.refName];
    if (!view) return;
    const mapper = view.getVtkObject(cfg.mapperId);
    const plane = view.getVtkObject(cfg.planeId);
    if (!mapper || !plane) return;

    await mapper.removeAllClippingPlanes();
    if (window.trame.state.get(key + "_crop_enabled")) {
      const direction = window.trame.state.get(key + "_crop_direction");
      const percent = Number(window.trame.state.get(key + "_crop_position"));
      const b = cfg.bounds;
      const axis = CROP_AXIS[direction];
      const origin = [
        0.5 * (b[0] + b[1]),
        0.5 * (b[2] + b[3]),
        0.5 * (b[4] + b[5]),
      ];
      origin[axis] =
        b[2 * axis] + (percent / 100.0) * (b[2 * axis + 1] - b[2 * axis]);
      await plane.setOrigin(origin);
      await plane.setNormal(CROP_NORMALS[direction]);
      await mapper.addClippingPlane(plane);
    }
    // Redraw the view client-side.
    view.render();
  }

  // Copy the crop toolbar state of one view onto the other. trame skips no-op
  // sets, so mirroring back from the destination terminates (its values already
  // equal the source's) -- no feedback loop.
  function mirrorCrop(src, dst) {
    for (const prop of ["enabled", "direction", "position"]) {
      window.trame.state.set(
        dst + "_crop_" + prop,
        window.trame.state.get(src + "_crop_" + prop),
      );
    }
  }

  function onCropChange(key) {
    if (!cropSyncInitialized) return;
    if (window.trame.state.get("crop_linked")) {
      mirrorCrop(key, key === "ref" ? "tgt" : "ref");
    }
    applyCrop(key);
  }

  function onCropLinkedChange() {
    if (!cropSyncInitialized) return;
    // On linking, make the tgt view adopt the ref crop plane.
    if (window.trame.state.get("crop_linked")) mirrorCrop("ref", "tgt");
  }

  window.trame.utils.colorTransferFunctionDesignerCrop = {
    setup: (payload) => {
      const cfg = cropViews[payload.key];
      cfg.refName = payload.refName;
      cfg.mapperId = payload.mapperId;
      cfg.planeId = payload.planeId;
      cfg.bounds = payload.bounds;

      if (!cropSyncInitialized) {
        cropSyncInitialized = true;
        window.trame.state.watch(
          ["ref_crop_enabled", "ref_crop_direction", "ref_crop_position"],
          () => onCropChange("ref"),
        );
        window.trame.state.watch(
          ["tgt_crop_enabled", "tgt_crop_direction", "tgt_crop_position"],
          () => onCropChange("tgt"),
        );
        window.trame.state.watch(["crop_linked"], () => onCropLinkedChange());
      }
      applyCrop(payload.key);
    },
  };
})();
