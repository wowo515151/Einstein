import { Connection4, Curvature4, EMFieldTensor, Event4, FourMomentum, FourVelocity, Geodesic4, LightCone, LorentzTransform, Metric4, SpacetimeInterval, Vector4, integrateRK4 } from "../src/relativity/index.js";

const MAX_SAMPLES = 320;
const PI2 = Math.PI * 2;
let geodesicCache = null;
let particleCache = null;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const finite = (value, name) => {
  if (!Number.isFinite(value)) throw new TypeError(`${name} must be finite`);
  return value;
};
const point = (id, x, y, z, label = id) => ({ id, x, y, z, label });

/** Compose two collinear velocities using the library's four-velocity and Lorentz operations. */
export function composeCollinearVelocities(firstBeta, relativeBeta) {
  finite(firstBeta, "first velocity beta"); finite(relativeBeta, "relative velocity beta");
  if (Math.abs(firstBeta) >= 1 || Math.abs(relativeBeta) >= 1) throw new RangeError("velocity fractions must be strictly subluminal");
  const relative = FourVelocity.fromVelocity3([relativeBeta, 0, 0], 1);
  const inLab = LorentzTransform.boost([-firstBeta, 0, 0]).applyVector(relative.components);
  return new FourVelocity(inLab.components, { c: 1 }).toVelocity3()[0];
}

export const scenes = Object.freeze([
  { id: "minkowski", title: "Minkowski spacetime", category: "Special relativity", summary: "This spacetime diagram compares a stationary observer's vertical worldline with a moving observer's sloped worldline. The colored axes show x and ct; switch frames to see the Lorentz-transformed coordinates while the physical interval stays fixed.", limit: "Exact straight-worldline model. The displayed projection is a 3D view of selected 4D coordinates.", parameter: "Observer speed β = v/c", model: "exact" },
  { id: "light-cone", title: "Light cone as an evolving sphere", category: "Causal structure", summary: "Each point forms a spherical cross-section of light emitted at one event. Move time through the past and future to watch the sphere shrink to the cone's apex and grow again; its radius follows r = c|Δt|.", limit: "Exact flat-spacetime geometry with c normalized to 1 for the display.", parameter: "Cone time", model: "exact" },
  { id: "lorentz", title: "Lorentz contraction and simultaneity", category: "Special relativity", summary: "The colored lines trace both ends of a rod through time, and the highlighted slices mark events judged simultaneous in each frame. Change frames to compare the measured rod length and see why simultaneity tilts between observers.", limit: "Exact one-dimensional Lorentz boost visualized in a normalized coordinate frame.", parameter: "Boost speed β = v/c", model: "exact" },
  { id: "twins", title: "Twin-paradox worldlines", category: "Relativistic motion", summary: "Two paths show a stay-at-home clock and a traveler who turns around and reunites with it. Follow the paths to compare their accumulated proper times; the extra depth separates the lines visually and is not a physical coordinate.", limit: "Exact proper-time result for the declared piecewise-constant-velocity path; acceleration duration is idealized as zero. Display depth separates the paths for visibility and is not a physical coordinate.", parameter: "Travel speed β = v/c", model: "exact" },
  { id: "aberration", title: "Velocity addition and aberration", category: "Relativistic motion", summary: "The star-like dots show incoming light directions across the sky, and the boost changes their apparent directions. The readout composes two collinear velocities with the Lorentz transform and shows the result remains below light speed.", limit: "Exact collinear velocity composition and direction transformation for ideal light rays; star points are sampled over a finite angular grid.", parameter: "Boost speed β = v/c", model: "exact" },
  { id: "fields", title: "Electric and magnetic field mixing", category: "Electromagnetism", summary: "The arrows show electric and magnetic field vectors before and after a boost. Change the observer speed to see how the fields mix while the displayed electromagnetic invariants remain unchanged.", limit: "Uniform-field special-relativity model. Axis scaling is normalized and does not represent a laboratory apparatus.", parameter: "Boost speed β = v/c", model: "exact" },
  { id: "particle", title: "Charged-particle motion", category: "Electromagnetism", summary: "The curved track follows a charged test particle moving through a uniform magnetic field, with a marker showing its current position. Playback advances the numerically integrated orbit; the speed control changes its initial motion.", limit: "Numerically integrated test-charge path. Radiation, back-reaction, and field gradients are omitted.", parameter: "Speed fraction β = v/c", model: "numerical" },
  { id: "geodesic", title: "Gravitational lensing rays", category: "General relativity", summary: "Colored light-ray paths curve around a central mass in a softened weak-field spacetime. Compare their changing directions and the readout's null-geodesic check to see how curvature bends light in this teaching model.", limit: "Softened, static weak-field model for teaching lensing; not a calibrated or exact black-hole solution.", parameter: "Field strength μ/r₀", model: "pedagogical" },
  { id: "curvature", title: "Curved-space embedding grid", category: "General relativity", summary: "The wire grid maps radial and angular positions onto a curved surface; its height visualizes the embedding and its color shows Gaussian curvature. Watch the radial grid spacing and surface color change with the curvature control.", limit: "Pedagogical surface-of-revolution metric. The embedding height is not a fourth spatial dimension or a literal force field.", parameter: "Curvature scale", model: "pedagogical" },
  { id: "field-slices", title: "Animated 4D field explorer", category: "Fields and solutions", summary: "A moving slice through a synthetic plane wave displays electric and magnetic arrows, observer paths, and translucent surfaces where the electric field has selected values. Playback advances the wave through time; the raised surface encodes field strength rather than a spatial height.", limit: "Synthetic plane-wave field for visualization; it is not a measured or independently validated physical solution.", parameter: "Field frequency", model: "synthetic numerical" },
]);

/** Resolve a scene URL fragment, defaulting to the first scene for unknown values. */
export function sceneFromFragment(fragment) {
  return scenes.find(scene => `#${scene.id}` === fragment) ?? scenes[0];
}

export function sceneAccessibleLabel(scene, orbitAvailable, keyboardAvailable = orbitAvailable) {
  const mouse = orbitAvailable ? " Drag to orbit, scroll to zoom." : "";
  const keyboard = keyboardAvailable ? " Use arrow keys to orbit and plus or minus to zoom." : "";
  return `Three dimensional ${scene.title} scene.${mouse}${keyboard} Use the time and model parameter controls below.`;
}

/** Coordinate meanings used by the scene axis legend; display coordinates are normalized unless noted. */
export const sceneAxes = Object.freeze({
  minkowski: Object.freeze({ x: "x position", y: "ct time", z: "projection depth" }),
  "light-cone": Object.freeze({ x: "x position", y: "y position", z: "z position" }),
  lorentz: Object.freeze({ x: "x position", y: "ct time", z: "projection depth" }),
  twins: Object.freeze({ x: "x position", y: "ct time", z: "illustrative path separation" }),
  aberration: Object.freeze({ x: "light direction x", y: "light direction y", z: "light direction z" }),
  fields: Object.freeze({ x: "display/vector x component", y: "display/vector y component", z: "display/vector z component" }),
  particle: Object.freeze({ x: "x position", y: "y position", z: "z position" }),
  geodesic: Object.freeze({ x: "x coordinate", y: "y coordinate", z: "z coordinate" }),
  curvature: Object.freeze({ x: "radial chart x", y: "embedding height", z: "radial chart z" }),
  "field-slices": Object.freeze({ x: "x position", y: "display height (0.45 Eₓ)", z: "wave direction z" }),
});

function spherePoints(time, count) {
  const signedTime = (2 * time - 1) * 2.5;
  const radius = Math.abs(signedTime);
  const n = Math.min(MAX_SAMPLES, Math.max(12, Math.floor(count)));
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const directions = Array.from({ length: n }, (_, i) => {
    const y = 1 - 2 * (i + .5) / n;
    const r = Math.sqrt(1 - y * y);
    const theta = goldenAngle * i;
    return [r * Math.cos(theta), y, r * Math.sin(theta)];
  });
  const events = new LightCone(new Event4([0, 0, 0, 0]), { branch: signedTime < 0 ? "past" : "future" }).sample(directions, radius);
  return events.map(([ct, x, y, z], index) => ({ ...point(`cone-${index}`, x, y, z), ct, causalType: new Vector4([ct, x, y, z]).causalType() }));
}
function gridPoints(scale = 1, count = 18, zMode = () => 0) {
  const points = [];
  for (let i = 0; i < count; i += 1) for (let j = 0; j < count; j += 1) {
    const x = (i / (count - 1) * 2 - 1) * scale;
    const z = (j / (count - 1) * 2 - 1) * scale;
    points.push(point(`grid-${i}-${j}`, x, zMode(x, z), z));
  }
  return points.slice(0, MAX_SAMPLES);
}
/** Induced radial metric and Gaussian curvature for h(r)=A exp(-a r²). */
export function curvatureEmbeddingValues(beta, x, z) {
  finite(beta, "beta"); finite(x, "x"); finite(z, "z");
  if (beta < 0 || beta >= 1) throw new RangeError("beta must be in [0, 1)");
  const amplitude = .55 * beta, decay = .38, radiusSquared = x * x + z * z;
  const radius = Math.sqrt(radiusSquared), profile = Math.exp(-decay * radiusSquared);
  if (profile === 0) return Object.freeze({ radialMetric: 1, gaussianCurvature: 0 });
  const first = -2 * decay * amplitude * radius * profile;
  const second = 2 * decay * amplitude * profile * (2 * decay * radiusSquared - 1);
  const radialMetric = 1 + first * first;
  const gaussianCurvature = radius > 1e-12
    ? first * second / (radius * radialMetric * radialMetric)
    : 4 * decay * decay * amplitude * amplitude;
  return Object.freeze({ radialMetric, gaussianCurvature });
}
/** Compare the induced surface with a direct-product spacetime metric via the library. */
export function curvatureEmbeddingLibraryCheck(beta,x,z) {
  curvatureEmbeddingValues(beta,x,z);
  const metric = ([,u,,v]) => {
    const factor=-.76*.55*beta*Math.exp(-.38*(u*u+v*v)),hx=factor*u,hz=factor*v;
    return [[-1,0,0,0],[0,1+hx*hx,0,hx*hz],[0,0,1,0],[0,hx*hz,0,1+hz*hz]];
  };
  return new Curvature4(metric,{step:1e-4}).ricciScalar([0,x,0,z])/2;
}
export function fieldIsosurfaces(beta, time, { level = .6, extent = 2 } = {}) {
  finite(beta, "beta"); finite(time, "time"); finite(level, "level"); finite(extent, "extent");
  if (beta < 0 || beta >= 1) throw new RangeError("beta must be in [0,1)");
  if (level <= 0 || level >= 1) throw new RangeError("level must be in (0,1)");
  if (extent <= 0) throw new RangeError("extent must be positive");
  const waveNumber = 1 + beta, phase = (time % 1) * PI2, bound = extent * waveNumber, ranges = [];
  if (!Number.isFinite(bound)) throw new RangeError("isosurface extent is too large");
  for (const isoLevel of [-level, level]) {
    const isoAngle = Math.acos(isoLevel);
    for (const sign of [-1, 1]) {
      const offset = phase + sign * isoAngle;
      const first = Math.ceil((-bound - offset) / PI2), last = Math.floor((bound - offset) / PI2);
      if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last)) throw new RangeError("isosurface extent is too large");
      ranges.push({ offset, first, last, isoLevel });
    }
  }
  if (ranges.reduce((sum, range) => sum + Math.max(0, range.last - range.first + 1), 0) > 4096) throw new RangeError("isosurface count exceeds 4096");
  const surfaces = [];
  for (const {offset, first, last, isoLevel} of ranges) for (let cycle = first; cycle <= last; cycle++) {
    const z = (offset + cycle * PI2) / waveNumber;
    if (Math.abs(z) <= extent) surfaces.push({ z, level: isoLevel });
  }
  return surfaces.sort((left, right) => left.z - right.z);
}

/** A normalized vacuum plane wave, with one playback cycle equal to one wave period. */
export function planeWaveState(beta,time,z) {
  finite(beta,"beta");finite(time,"time");finite(z,"z");
  if(beta<0||beta>=1)throw new RangeError("beta must be in [0,1)");
  const waveNumber=1+beta,ct=PI2*time/waveNumber,amplitude=Math.cos(waveNumber*(z-ct));
  const field=EMFieldTensor.fromEB([amplitude,0,0],[0,amplitude,0],{c:1});
  return {ct,electric:field.electricField(),magnetic:field.magneticField()};
}

/** Two inertial worldlines lifted onto field-amplitude heights at each event. */
export function fieldWorldlines(beta, time) {
  finite(beta, "beta"); finite(time, "time");
  if (beta < 0 || beta >= 1 || time < 0 || time > 1) throw new RangeError("beta and time must be in their supported ranges");
  return [-1, 1].map((direction) => {
    const eventAt = t => {
      const ct = PI2 * t / (1 + beta);
      const event = new Event4([ct, direction * ((.2 + .6 * beta) * ct - .4), .45 * direction, direction * .6]);
      return { ct: event.position[0], x: event.position[1], y: .45 * planeWaveState(beta,t,event.position[3]).electric[0], z: event.position[3], physicalY: event.position[2] };
    };
    return { path: Array.from({ length: 25 }, (_, i) => eventAt(i / 24)), current: eventAt(time) };
  });
}
/** Selected events on the worldlines, expressed in the displayed observer frame. */
export function selectedWorldlineEvents(sceneId,beta,time,observerFrame="lab") {
  finite(beta,"beta");finite(time,"time");
  if(beta<0||beta>=1||time<0||time>1)throw new RangeError("beta and time must be in their supported ranges");
  if(!["minkowski","lorentz"].includes(sceneId)||!["lab","moving"].includes(observerFrame))throw new RangeError("unsupported scene or observer frame");
  const sourceCt=4*time-2,transform=observerFrame==="moving"?LorentzTransform.boost([beta,0,0]):null;
  const positions=sceneId==="minkowski"?[0,beta*sourceCt]:[-2,2];
  return positions.map(sourceX=>{
    const [ct,x]=transform?.applyVector([sourceCt,sourceX,0,0]).components??[sourceCt,sourceX,0,0];
    return {x,y:ct,z:0};
  });
}
function twinDisplayDepth(ct) { return ct === 0 || ct === 4 ? 0 : .12 * Math.sin(Math.PI * ct / 4); }
function scenePoints(sceneId, time, beta, observerFrame = "lab") {
  if (sceneId === "light-cone") return spherePoints(time, 220);
  if (sceneId === "lorentz") {
    const transform = observerFrame === "moving" ? LorentzTransform.boost([beta, 0, 0]) : null;
    const displayEvent = (ct, x) => transform?.applyVector(new Vector4([ct, x, 0, 0])).components ?? [ct, x, 0, 0];
    const endpoints = ["left", "right"].flatMap((endpoint) => Array.from({ length: 48 }, (_, i) => {
      const sourceCt = i / 47 * 4 - 2, sourceX = endpoint === "left" ? -2 : 2;
      const [ct, x] = displayEvent(sourceCt, sourceX);
      const interval = new Vector4([ct, x, 0, 0]);
      return { ...point(`${endpoint}-endpoint-${i}`, x, ct, 0, `${endpoint} rod endpoint worldline`), ct, sourceCt, sourceX, intervalSquared: interval.normSquared(), causalType: interval.causalType() };
    }));
    const restSimultaneity = Array.from({ length: 48 }, (_, i) => {
      const sourceX = (i / 47 - .5) * 4, [ct, x] = displayEvent(0, sourceX);
      return { ...point(`rod-rest-simultaneity-${i}`, x, ct, 0, "rod-rest simultaneity slice"), ct };
    });
    const movingSimultaneity = Array.from({ length: 48 }, (_, i) => {
      const sourceX = (i / 47 - .5) * 4, sourceCt = beta * sourceX, [ct, x] = displayEvent(sourceCt, sourceX);
      return { ...point(`moving-frame-simultaneity-${i}`, x, ct, 0, "moving-observer simultaneity slice"), ct };
    });
    return [...endpoints, ...restSimultaneity, ...movingSimultaneity];
  }
  if (sceneId === "minkowski") {
    const transform = observerFrame === "moving" ? LorentzTransform.boost([beta, 0, 0]) : null;
    return ["laboratory observer", "moving observer"].flatMap((label) => Array.from({ length: 100 }, (_, i) => {
      const t = i / 99 * 4 - 2;
      const x = label === "moving observer" ? beta * t : 0;
      const coordinates = transform?.applyVector(new Vector4([t, x, 0, 0])).components ?? [t, x, 0, 0];
      const interval = new Vector4(coordinates);
      const prefix = label === "moving observer" ? "moving" : "lab";
      return { ...point(`${prefix}-worldline-${i}`, coordinates[1], coordinates[0], 0, label), ct: coordinates[0], intervalSquared: interval.normSquared(), causalType: interval.causalType() };
    }));
  }
  if (sceneId === "twins") {
    const count = 141, travelerRate = new SpacetimeInterval([1,beta,0,0], {c:1,atol:0}).properTime();
    return Array.from({ length: count * 2 }, (_, index) => {
      const observer = index < count ? "stay-home" : "traveler";
      const sample = index % count;
      const t = sample / (count - 1) * 4;
      const x = observer === "stay-home" ? 0 : (t <= 2 ? beta * t : beta * (4 - t));
      return {...point(`${observer}-${sample}`, x - 2 * beta, t - 2, ((observer === "stay-home" ? -1 : 1) * twinDisplayDepth(t)) || 0, `${observer} clock`),ct:t,worldlineX:x,properTime:observer === "stay-home" ? t : t * travelerRate};
    });
  }
  if (sceneId === "aberration") {
    const boost = LorentzTransform.boost([beta, 0, 0]);
    const n = 140;
    return Array.from({ length: n }, (_, i) => {
      const u = 1 - 2 * (i + .5) / n;
      const phi = Math.PI * (1 + Math.sqrt(5)) * i;
      const r = Math.sqrt(1 - u * u);
      const x = r * Math.cos(phi), y = u, z = r * Math.sin(phi);
      const transformed = boost.applyVector(new Vector4([1, x, y, z])).components;
      return point(`ray-${i}`, transformed[1] / transformed[0], transformed[2] / transformed[0], transformed[3] / transformed[0]);
    });
  }
  if (sceneId === "fields") {
    const field = transformedFieldState(beta);
    return [...vectorGlyphs("electric", field.electric, [-1.2, 0, 0]), ...vectorGlyphs("magnetic", field.magnetic, [1.2, 0, 0])];
  }
  if (sceneId === "particle") {
    if (!particleCache || particleCache.beta !== beta) {
      const field = EMFieldTensor.fromEB([0, 0, 0], [0, 0, 1], { c: 1 });
      const initialVelocity = FourVelocity.fromVelocity3([beta, 0, 0], 1).components;
      const initialState = [0, 0, 0, 0, ...initialVelocity];
      const rows = integrateRK4((parameter, state) => {
        const tangent = state.slice(4), covariantTangent = [-tangent[0], tangent[1], tangent[2], tangent[3]];
        return [...tangent, ...field.lorentzForce(1, covariantTangent)];
      }, initialState, 0, PI2, 240);
      particleCache = { beta, rows };
    }
    const rows = particleCache.rows;
    const last = Math.max(0, Math.min(rows.length - 1, Math.floor(time * (rows.length - 1))));
    return rows.slice(0, last + 1).map(([parameter, state], i) => {
      const [ct, x, y, z] = state;
      return { ...point(`orbit-${i}`, x, y, z), ct, parameter, fourVelocity: state.slice(4) };
    });
  }
  if (sceneId === "geodesic") {
    // A bounded bundle of null geodesics makes the bending field visible in 3D.
    // This is a softened, static weak-field teaching model, not an exact lens.
    const mu = .008 + .045 * beta;
    const softening = .35;
    const metric = ([, x, y, z]) => {
      // Connection4 supplies Dual4 coordinates for the forward derivative path.
      const phi = typeof x === "number"
        ? -mu / Math.sqrt(x * x + y * y + z * z + softening * softening)
        : x.multiply(x).add(y.multiply(y)).add(z.multiply(z)).add(softening * softening).sqrt().pow(-1).multiply(-mu);
      const temporal = typeof phi === "number" ? -(1 + 2 * phi) : phi.multiply(2).add(1).negate();
      const spatial = typeof phi === "number" ? 1 - 2 * phi : phi.multiply(-2).add(1);
      return [[temporal, 0, 0, 0], [0, spatial, 0, 0], [0, 0, spatial, 0], [0, 0, 0, spatial]];
    };
    if (!geodesicCache || geodesicCache.beta !== beta) {
      const connection = new Connection4(metric, { metricDerivative: "forward" });
      const paths = [];
      for (let ray = 0; ray < 8; ray++) {
        const angle = ray * PI2 / 8;
        const impactParameter = ray % 2 === 0 ? .65 : 1.3;
        const impactDirection = [Math.cos(angle), Math.sin(angle)];
        const initialEvent = [0, -3, impactParameter * impactDirection[0], impactParameter * impactDirection[1]];
        const initialMetric = new Metric4(metric(initialEvent));
        const spatialTangent = [1, 0, 0];
        const spatialNorm = initialMetric.dot([0, ...spatialTangent], [0, ...spatialTangent]);
        const timeTangent = Math.sqrt(spatialNorm / -initialMetric.components[0][0]);
        const samples = new Geodesic4(connection, initialEvent, [timeTangent, ...spatialTangent]).integrate([0, 6], { steps: 38 });
        paths.push(samples.map(({ parameter, event, tangent }, i) => {
          const [ct, x, y, z] = event.position;
          const metricAtEvent = metric(event.position);
          const metricNorm = metricAtEvent.reduce((sum, row, axis) => sum + row[axis] * tangent[axis] ** 2, 0);
          return {
            ...point(`geodesic-ray-${ray}-${i}`, x, y, z),
            label: `light ray ${ray + 1}`,
            ct,
            parameter,
            tangent: tangent.slice(),
            metricNorm,
            impactParameter,
            impactDirection: impactDirection.slice(),
          };
        }));
      }
      geodesicCache = { beta, paths };
    }
    const last = Math.max(0, Math.floor(time * 38));
    return geodesicCache.paths.flatMap((path) => path.slice(0, last + 1).map((row) => ({
      ...row,
      tangent: row.tangent.slice(),
      impactDirection: row.impactDirection.slice(),
    })));
  }
  if (sceneId === "curvature") {
    return gridPoints(2.2, 17, (x, z) => .55 * beta * Math.exp(-.38 * (x * x + z * z))).map((row) => ({
      ...row,
      ...curvatureEmbeddingValues(beta, row.x, row.z),
    }));
  }
  if (sceneId === "field-slices") {
    const fieldsByZ = new Map();
    const fieldAt = z => { if (!fieldsByZ.has(z)) fieldsByZ.set(z, planeWaveState(beta, time, z)); return fieldsByZ.get(z); };
    const slice = (_x, z) => .45 * fieldAt(z).electric[0];
    const rows = gridPoints(2, 16, slice).map(row=>{
      const field=fieldAt(row.z);
      return {...row,ct:field.ct,physicalPosition:[row.x,0,row.z],electricField:field.electric.slice(),magneticField:field.magnetic.slice()};
    });
    const positions = [-2, -2 / 3, 2 / 3, 2];
    for (let i = 0; i < positions.length; i++) for (let j = 0; j < positions.length; j++) {
      const x = positions[i], z = positions[j], y = slice(x, z);
      const field=fieldAt(z);
      rows.push(...vectorGlyphs(`electric-slice-${i}-${j}`, field.electric, [x, y, z]));
      rows.push(...vectorGlyphs(`magnetic-slice-${i}-${j}`, field.magnetic, [x, y, z]));
    }
    return rows;
  }
  return [];
}

/** Unit-mass four-momentum and invariant for a sampled particle state. */
export function particleMomentumState(row) {
  const momentum = new FourMomentum(row.fourVelocity, { c: 1 });
  return { components: momentum.components, invariantMass: momentum.invariantMass() };
}

function vectorGlyphs(id, vector, origin) {
  const magnitude = Math.hypot(...vector);
  const displayScale = magnitude > 0 ? Math.min(1, 1 / magnitude) : 0;
  const tip = origin.map((value, index) => value + vector[index] * displayScale);
  return [
    point(`${id}-tail`, ...origin, `${id} field origin`),
    { ...point(`${id}-tip`, ...tip, `${id} field vector tip`), vector: vector.slice() },
  ];
}

/** Field transformation and invariants use the reusable relativity library. */
export function transformedFieldState(beta) {
  finite(beta, "beta");
  if (beta < 0 || beta >= 1) throw new RangeError("beta must be in [0, 1)");
  const rest = EMFieldTensor.fromEB([0, 0, 1], [0, 0, 0], { c: 1 });
  const transformed = rest.transform(LorentzTransform.boost([beta, 0, 0]));
  return Object.freeze({
    electric: Object.freeze(transformed.electricField()),
    magnetic: Object.freeze(transformed.magneticField()),
    invariants: Object.freeze(transformed.invariants()),
  });
}

/** Proper times for the idealized out-and-back twin paths, with an instant turn at ct=2. */
export function twinClockReadings(beta, time = 1) {
  finite(beta, "beta");
  finite(time, "time");
  if (beta < 0 || beta >= 1) throw new RangeError("beta must be in [0, 1)");
  if (time < 0 || time > 1) throw new RangeError("time must be in [0, 1]");
  const elapsed = 4 * time;
  const leg = new SpacetimeInterval([1, beta, 0, 0], { c: 1, atol: 0 });
  return Object.freeze({
    stayHome: elapsed,
    traveler: elapsed * leg.properTime(),
    reunion: Object.freeze([4, 0, 0, 0]),
  });
}

/** Display positions of the two clocks at the selected trip time. */
export function twinClockPositions(beta, time = 1) {
  const { stayHome: ct } = twinClockReadings(beta, time);
  const travelerX = ct <= 2 ? beta * ct : beta * (4 - ct);
  const depth = twinDisplayDepth(ct);
  return [
    { x: -2 * beta, y: ct - 2, z: -depth || 0 },
    { x: travelerX - 2 * beta, y: ct - 2, z: depth },
  ];
}

/** Deterministic, bounded sample data for the ten educational scene models. */
export function buildSceneData(sceneId, { time = 0, beta = .6, observerFrame = "lab" } = {}) {
  finite(time, "time");
  finite(beta, "beta");
  if (beta < 0 || beta >= 1) throw new RangeError("beta must be in [0, 1)");
  if (!["lab", "moving"].includes(observerFrame)) throw new TypeError("observerFrame must be lab or moving");
  const scene = scenes.find((item) => item.id === sceneId);
  if (!scene) throw new RangeError(`Unknown scene: ${sceneId}`);
  const normalizedTime = clamp(time, 0, 1);
  const rows = scenePoints(sceneId, normalizedTime, beta, observerFrame).slice(0, MAX_SAMPLES);
  return rows.map((row) => Object.freeze({
    ...row,
    x: finite(row.x, "x"), y: finite(row.y, "y"), z: finite(row.z, "z"),
    time: normalizedTime,
    model: scene.model,
  }));
}

export const sampleLimit = MAX_SAMPLES;
