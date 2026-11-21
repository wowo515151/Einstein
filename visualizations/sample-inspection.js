import {Vector4} from "../src/algebra.js";

/** Format one displayed sample, using the same normalized coordinates as the scene. */
export function describeSample(rows, number) {
  if (!Number.isInteger(number) || number < 1 || number > rows.length) throw new RangeError(`Choose a sample from 1 to ${rows.length}.`);
  const row = rows[number-1];
  const details=[];
  if(Number.isFinite(row.ct))details.push(`Coordinate time ct: ${row.ct.toFixed(3)}`);
  if(Number.isFinite(row.worldlineX))details.push(`Worldline x: ${row.worldlineX.toFixed(3)}`);
  if(Number.isFinite(row.properTime))details.push(`Clock proper time: ${row.properTime.toFixed(3)}`);
  const vectorText=value=>Array.isArray(value)&&value.length===3&&value.every(Number.isFinite)?`(${value.map(component=>component.toFixed(3)).join(", ")})`:null;
  const position=vectorText(row.physicalPosition),electric=vectorText(row.electricField),magnetic=vectorText(row.magneticField);
  if(position)details.push(`Physical slice position: ${position}`);
  if(electric)details.push(`Electric field: ${electric}`);
  if(magnetic)details.push(`Magnetic field: ${magnetic}`);
  if(Number.isFinite(row.intervalSquared))details.push(`Squared interval from origin: ${row.intervalSquared.toExponential(3)}`);
  const causalNames=new Map([[Vector4.CAUSAL_TIMELIKE,"timelike"],[Vector4.CAUSAL_NULL,"null"],[Vector4.CAUSAL_SPACELIKE,"spacelike"]]);
  if(causalNames.has(row.causalType))details.push(`Vector classification: ${causalNames.get(row.causalType)}`);
  if(Number.isFinite(row.metricNorm))details.push(`Metric tangent norm: ${row.metricNorm.toExponential(3)}`);
  const suffix=details.length?` ${details.join(". ")}.`:"";
  return `Selected sample ${number} of ${rows.length}: ${String(row.label ?? row.id)}. Display coordinates: (${row.x.toFixed(3)}, ${row.y.toFixed(3)}, ${row.z.toFixed(3)}).${suffix}`;
}

/** Describe a point retained on screen after its replacement data failed to render. */
export function describeDisplayedSample(row) {
  return `Selected displayed point: ${String(row.label ?? row.id)}. Display coordinates: (${row.x.toFixed(3)}, ${row.y.toFixed(3)}, ${row.z.toFixed(3)}). The displayed point does not match the current sample data.`;
}
