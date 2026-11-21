/** Report rejected viewer updates while the page remains active. */
export function updateViewer(viewer, data, { isClosed, report }) {
  if (!viewer) return Promise.resolve();
  return viewer.update(data).catch(error => {
    if (!isClosed()) report(error);
  });
}
