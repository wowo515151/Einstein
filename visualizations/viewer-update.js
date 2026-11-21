const latestUpdates = new WeakMap();

/** Report only the latest rejected viewer update while the page remains active. */
export function updateViewer(viewer, data, { isClosed, report }) {
  if (!viewer) return Promise.resolve();
  const request = {};
  latestUpdates.set(viewer, request);
  let update;
  try { update = viewer.update(data); } catch (error) { update = Promise.reject(error); }
  return Promise.resolve(update).catch(error => {
    if (!isClosed() && latestUpdates.get(viewer) === request) report(error);
  }).finally(() => {
    if (latestUpdates.get(viewer) === request) latestUpdates.delete(viewer);
  });
}
