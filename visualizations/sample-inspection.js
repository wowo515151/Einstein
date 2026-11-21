/** Format one displayed sample, using the same normalized coordinates as the scene. */
export function describeSample(rows, number) {
  if (!Number.isInteger(number) || number < 1 || number > rows.length) throw new RangeError(`Choose a sample from 1 to ${rows.length}.`);
  const row = rows[number-1];
  return `Selected sample ${number} of ${rows.length}: ${String(row.label ?? row.id)}. Display coordinates: (${row.x.toFixed(3)}, ${row.y.toFixed(3)}, ${row.z.toFixed(3)}).`;
}

/** Describe a point retained on screen after its replacement data failed to render. */
export function describeDisplayedSample(row) {
  return `Selected displayed point: ${String(row.label ?? row.id)}. Display coordinates: (${row.x.toFixed(3)}, ${row.y.toFixed(3)}, ${row.z.toFixed(3)}). The displayed point does not match the current sample data.`;
}
