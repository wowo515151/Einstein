/** Remove consecutive points that cannot define a spatial direction. */
export function distinctPathPoints(rows, epsilon = 1e-12) {
  const result = [];
  for (const row of rows) {
    const last = result.at(-1);
    if (!last || Math.hypot(row.x - last.x, row.y - last.y, row.z - last.z) > epsilon) result.push(row);
  }
  return result;
}

/** Match the twin-clock path colors to their selected-time markers. */
export function twinPathColor(label) {
  return label === "stay-home clock" ? 0x55b9ff : 0xff8f9c;
}
