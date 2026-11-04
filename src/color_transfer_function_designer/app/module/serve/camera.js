(() => {
  let suppressCameraSync = false;
  // refName -> { rendererId, cameraId, renderer, camera, cameraObserver }
  const viewObservers = new Map();

  function copyCamera(srcRefName, dstRefName) {
    if (suppressCameraSync) return;
    if (!window.trame.state.get("camera_linked")) return;
    const src = viewObservers.get(srcRefName);
    const dst = viewObservers.get(dstRefName);
    const dstView = window.trame.refs[dstRefName];
    if (!src || !dst || !dstView) return;
    const srcCamera = src.camera;
    const dstCamera = dst.camera;
    suppressCameraSync = true;
    try {
      dstCamera.setPosition(srcCamera.getPosition());
      dstCamera.setFocalPoint(srcCamera.getFocalPoint());
      dstCamera.setViewUp(srcCamera.getViewUp());
      dstCamera.setViewAngle(srcCamera.getViewAngle());
      dstCamera.setParallelScale(srcCamera.getParallelScale());
      dstCamera.setParallelProjection(srcCamera.getParallelProjection());
      dst.renderer.resetCameraClippingRange();
    } finally {
      suppressCameraSync = false;
    }
    // Redraw the destination view client-side.
    dstView.render();
  }

  window.trame.utils.colorTransferFunctionDesignerCamera = {
    setup: (refName, dstRefName, rendererId, cameraId) => {
      const view = window.trame.refs[refName];
      if (!view || !rendererId || !cameraId) return; // retry on next `updated`
      // `setup` fires on every view `updated` event; the renderer and camera
      // are persistent, so build the proxies and observe once. Re-observing
      // each update would stack duplicate ModifiedEvent observers, and
      // rebuilding proxies would re-serialize.
      const existing = viewObservers.get(refName);
      if (
        existing &&
        existing.rendererId === rendererId &&
        existing.cameraId === cameraId
      )
        return;
      if (existing) existing.camera.$unObserve(existing.cameraObserver);
      const renderer = view.getVtkObject(rendererId);
      const camera = view.getVtkObject(cameraId);
      const entry = {
        rendererId,
        cameraId,
        renderer,
        camera,
        cameraObserver: camera.$observe("ModifiedEvent", () =>
          copyCamera(refName, dstRefName),
        ),
      };
      viewObservers.set(refName, entry);
    },
    sync: (srcRefName, dstRefName) => {
      // One-time snap of the target view onto the reference view on link.
      if (!viewObservers.has(srcRefName) || !viewObservers.has(dstRefName))
        return;
      copyCamera(srcRefName, dstRefName);
    },
  };
})();
