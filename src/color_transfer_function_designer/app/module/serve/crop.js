(() => {
  let cropSyncInitialized = false;

  const cropViews = {
    ref: {
      bounds: null,
      extent: null,
      imageActorId: null,
      mapperId: null,
      planeId: null,
      refName: null,
    },
    tgt: {
      bounds: null,
      extent: null,
      imageActorId: null,
      mapperId: null,
      planeId: null,
      refName: null,
    },
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
    const imageActor = view.getVtkObject(cfg.imageActorId);
    if (!mapper || !plane) return;

    await mapper.removeAllClippingPlanes();
    if (window.trame.state.get(key + "_crop_enabled")) {
      const direction = window.trame.state.get(key + "_crop_direction");
      const percent = Number(window.trame.state.get(key + "_crop_position"));
      const b = cfg.bounds;
      const axis = CROP_AXIS[direction]; // 0, 1, 2
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

      if (imageActor) {
        const e = cfg.extent; // [x0, x1, y0, y1, z0, z1]
        const lo = e[2 * axis],
          hi = e[2 * axis + 1];
        const idx = Math.round(lo + (percent / 100) * (hi - lo));
        const de = [...e];
        de[2 * axis] = idx;
        de[2 * axis + 1] = idx;
        await imageActor.setDisplayExtent(...de);
        await imageActor.setVisibility(1);
      }
    } else if (imageActor) {
      await imageActor.setVisibility(0);
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
    // While a transfer is running the tgt view is being re-`update()`d on the
    // server, so any tgt crop op would race that deserialization. Block both the
    // mirror into tgt and a direct tgt apply until the transfer unlocks it.
    const tgtLocked = window.trame.state.get("tgt_crop_locked");
    if (window.trame.state.get("crop_linked")) {
      const dst = key === "ref" ? "tgt" : "ref";
      if (!(dst === "tgt" && tgtLocked)) mirrorCrop(key, dst);
    }
    if (!(key === "tgt" && tgtLocked)) applyCrop(key);
  }

  function onCropLinkedChange() {
    if (!cropSyncInitialized) return;
    if (window.trame.state.get("tgt_crop_locked")) return;
    // On linking, make the tgt view adopt the ref crop plane.
    if (window.trame.state.get("crop_linked")) mirrorCrop("ref", "tgt");
  }

  window.trame.utils.colorTransferFunctionDesignerCrop = {
    setup: async (payload) => {
      const cfg = cropViews[payload.key];
      cfg.bounds = payload.bounds;
      cfg.extent = payload.extent;
      cfg.imageActorId = payload.imageActorId;
      cfg.mapperId = payload.mapperId;
      cfg.planeId = payload.planeId;
      cfg.refName = payload.refName;

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
      // `relink` re-adopts the ref crop after a transfer unlocks the tgt view.
      if (payload.relink && window.trame.state.get("crop_linked")) {
        mirrorCrop("ref", payload.key);
      }
      await applyCrop(payload.key);
    },
    // Drop a view's cached wasm handles before its volume is unloaded. The
    // mapper object is pruned from the wasm scene on the next `update()`, so a
    // stale handle would otherwise invoke RemoveAllClippingPlanes on a dead id.
    teardown: (key) => {
      const cfg = cropViews[key];
      if (!cfg) return;
      cfg.bounds = null;
      cfg.extent = null;
      cfg.imageActorId = null;
      cfg.mapperId = null;
      cfg.planeId = null;
    },
  };
})();
