(() => {
  let cameraSyncInitialized = false;
  let suppressCameraSync = false;
  const viewObservers = new Map();

  // Copy the active-camera view params from src to dst, refit the dst clipping
  // range (the two volumes have different bounds) and redraw the dst view. The
  // suppress flag swallows the dst ModifiedEvent fired during set()/clipping so
  // the mirror does not echo back into an infinite loop.
  async function copyCamera(srcRefName, dstRefName) {
    if (suppressCameraSync) return;
    if (!window.trame.state.get("camera_linked")) return;
    const srcView = window.trame.refs[srcRefName];
    const dstView = window.trame.refs[dstRefName];
    if (!srcView || !dstView) return;
    const srcRendererId = viewObservers.get(srcRefName).rendererId;
    const dstRendererId = viewObservers.get(dstRefName).rendererId;
    const srcRenderer = srcView.getVtkObject(srcRendererId);
    const dstRenderer = dstView.getVtkObject(dstRendererId);
    const srcCamera = await srcRenderer.getActiveCamera();
    const dstCamera = await dstRenderer.getActiveCamera();
    suppressCameraSync = true;
    try {
      await dstCamera.setPosition(srcCamera.position);
      await dstCamera.setFocalPoint(srcCamera.focalPoint);
      await dstCamera.setViewUp(srcCamera.viewUp);
      await dstCamera.setViewAngle(srcCamera.viewAngle);
      await dstCamera.setParallelScale(srcCamera.parallelScale);
      await dstCamera.setParallelProjection(srcCamera.parallelProjection);
      await dstRenderer.resetCameraClippingRange();
    } finally {
      suppressCameraSync = false;
    }
    // Redraw the destination view client-side.
    window.trame.refs[dstRefName].render();
  }

  window.trame.utils.colorTransferFunctionDesignerCamera = {
    setup: async (refName, dstRefName, rendererId) => {
      const view = window.trame.refs[refName];
      if (!view) return; // a view not mounted yet -> retry on next `updated`
      // `setup` fires on every view `updated` event; the renderer (and its
      // camera) is persistent, so observe it once. Re-observing each update
      // would stack duplicate ModifiedEvent observers.
      const existing = viewObservers.get(refName);
      if (existing && existing.rendererId === rendererId) return;
      let entry = { rendererId: 0, cameraObserver: 0 };
      const renderer = view.getVtkObject(rendererId);
      const camera = await renderer.getActiveCamera();
      entry.rendererId = rendererId;
      entry.cameraObserver = camera.observe("ModifiedEvent", () =>
        copyCamera(refName, dstRefName),
      );
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
