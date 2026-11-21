/** Own the objects and tracked resources created for one scene overlay. */
export function createOverlayScope(scene, resources) {
  const entries = [];
  let disposed = false;
  function dispose() {
    if (disposed) return;
    disposed = true;
    const errors = [];
    for (const { object, geometry, material } of entries) {
      try { scene.remove(object); } catch (error) { errors.push(error); }
      try { resources.release(geometry); } catch (error) { errors.push(error); }
      try { resources.release(material); } catch (error) { errors.push(error); }
    }
    entries.length = 0;
    if (errors.length) throw new AggregateError(errors, "Failed to dispose overlay resources.");
  }
  function add(object, geometry, material) {
    if (disposed) throw new Error("overlay scope is disposed");
    resources.track(geometry);
    try { resources.track(material); } catch (error) {
      try { resources.release(geometry); } catch (cleanupError) { throw new AggregateError([error, cleanupError], "Overlay tracking and cleanup failed."); }
      throw error;
    }
    entries.push({ object, geometry, material });
    scene.add(object);
  }
  return { add, dispose };
}

/** Retain the previous overlay until the new overlay and point update succeed. */
export function replaceOverlay(previous, create, updatePoints, report = () => {}) {
  const next = create();
  try { updatePoints(); } catch (error) {
    try { next(); } catch (cleanupError) { throw new AggregateError([error, cleanupError], "Point update and overlay cleanup failed."); }
    throw error;
  }
  try { previous(); } catch (error) { try { report(error); } catch {} }
  return next;
}
