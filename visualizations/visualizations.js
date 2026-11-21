import { createViewerWithRuntime } from "../src/three-viz/core/mount.js";
import { pointCloudPlotter } from "../src/three-viz/plotters/point-cloud-plotter.js";
import { createCarPaintMaterial, retainCarPaintEnvironment } from "../src/three-viz/plotters/shared/materials.js";
import { buildSceneData, composeCollinearVelocities, fieldIsosurfaces, fieldWorldlines, particleMomentumState, sceneAccessibleLabel, sceneFromFragment, scenes, sceneAxes, selectedWorldlineEvents, transformedFieldState, twinClockReadings, twinClockPositions } from "./scene-models.js";
import { createLabeledBox } from "../src/three-viz/objects/labeled-box.js";
import { Vector4 } from "../src/relativity/index.js";
import { createPlaybackController } from "./playback.js";
import { updateViewer } from "./viewer-update.js";
import { distinctPathPoints, twinPathColor } from "./path-points.js";
import { createOverlayScope, replaceOverlay } from "./overlay-scope.js";
import { moveCameraWithKey } from "./camera-keyboard.js";
import { describeSample, describeDisplayedSample } from "./sample-inspection.js";

const byId = (id) => document.getElementById(id);
const sceneList = byId("scene-list");
const host = byId("canvas-host");
const fallback = byId("renderer-fallback");
const status = byId("scene-status");
const timeSlider = byId("time-slider");
const speedSlider = byId("speed-slider");
const observerFrameSelect = byId("observer-frame");
const observerFrameControl = byId("observer-frame-control");
const playButton = byId("play-button");
const prefersReducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

let activeScene = sceneFromFragment(location.hash);
let time = 0;
let beta = Number(speedSlider.value);
let observerFrame = "lab";
let viewer;
let pageClosed = false;
let currentRows = [];
const playback = createPlaybackController({
  reducedMotion: () => prefersReducedMotion.matches,
  available: () => !pageClosed,
  now: () => performance.now(),
  onChange(playing) {
    playButton.setAttribute("aria-pressed", String(playing));
    playButton.textContent = playing ? "Pause animation" : "Play animation";
  },
  onFrame(next) { time = next; timeSlider.value = String(next); updateRows(); },
});

function updateDescription() {
  byId("scene-category").textContent = activeScene.category.toLocaleUpperCase();
  byId("scene-title").textContent = activeScene.title;
  updateSceneLabel();
  byId("scene-summary").textContent = activeScene.summary;
  byId("scene-limit").textContent = `${activeScene.model.toLocaleUpperCase()} MODEL · ${activeScene.limit}`;
  byId("parameter-hint").textContent = `${activeScene.parameter}. Values are normalized for display.`;
  for (const axis of ["x", "y", "z"]) byId(`axis-${axis}-meaning`).textContent = sceneAxes[activeScene.id][axis];
  byId("axis-key").setAttribute("aria-label", `Coordinate axes: ${["x", "y", "z"].map((axis) => `${axis}: ${sceneAxes[activeScene.id][axis]}`).join("; ")}. Values are normalized for display unless the scene states otherwise.`);
  speedSlider.setAttribute("aria-label", activeScene.parameter);
  speedSlider.disabled = activeScene.id === "light-cone";
  observerFrameControl.hidden = activeScene.id !== "minkowski" && activeScene.id !== "lorentz";
  observerFrameSelect.options[0].textContent = activeScene.id === "lorentz" ? "Rod rest frame" : "Laboratory frame";
  observerFrameSelect.options[1].textContent = "Moving observer frame";
  observerFrameSelect.value = observerFrame;
  for (const button of sceneList.querySelectorAll("button")) {
    if (button.dataset.scene === activeScene.id) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  }
}

function updateSceneLabel() {
  const orbitAvailable = Boolean(viewer && globalThis.EINSTEIN_THREE_RUNTIME?.OrbitControls);
  byId("scene").setAttribute("aria-label", sceneAccessibleLabel(activeScene, orbitAvailable, Boolean(viewer)));
}

function currentData() {
  return buildSceneData(activeScene.id, { time, beta, observerFrame });
}

function updateModelReadout(rows) {
  const readout = byId("scene-readout");
  if (activeScene.id === "minkowski") {
    const marker = selectedWorldlineEvents("minkowski", beta, time, observerFrame)[1];
    const observed = new Vector4([marker.y, marker.x, 0, 0]);
    const classification = marker.x === 0 && marker.y === 0 ? "coincident with the origin" : observed.causalName();
    readout.textContent = `${observerFrame === "moving" ? "Moving observer frame" : "Laboratory frame"}: the moving worldline is ${observerFrame === "moving" ? "at rest" : "x = βct"}. Markers show events at laboratory ct = ${(4*time-2).toFixed(2)}. Moving marker: s² = ${observed.normSquared().toFixed(4)}; ${classification}.`;
  } else if (activeScene.id === "light-cone") {
    const samples = rows;
    const first = samples[0];
    const firstType = new Vector4([first.ct, first.x, first.y, first.z]).causalName();
    const branch = first.ct < 0 ? "past" : first.ct > 0 ? "future" : "apex";
    readout.textContent = `Library check: ${samples.length} ${branch} samples; ct = ${first.ct.toFixed(3)}, classified ${firstType}.`;
  } else if (activeScene.id === "lorentz") {
    const selectedLabel = observerFrame === "moving" ? "moving-observer simultaneity slice" : "rod-rest simultaneity slice";
    const selectedSlice = rows.filter((row) => row.label === selectedLabel);
    const properSlice = buildSceneData("lorentz", { beta, observerFrame: "lab" }).filter((row) => row.label === "rod-rest simultaneity slice");
    const measuredLength = Math.abs(selectedSlice.at(-1).x - selectedSlice[0].x);
    const properLength = Math.abs(properSlice.at(-1).x - properSlice[0].x);
    readout.textContent = `${observerFrame === "moving" ? "Moving observer" : "Rod rest"} frame simultaneous length = ${measuredLength.toFixed(3)}; proper rod length = ${properLength.toFixed(3)}. Markers show rod endpoints at rest-frame ct = ${(4*time-2).toFixed(2)}. The other frame’s simultaneity slice is tilted in this view.`;
  } else if (activeScene.id === "particle") {
    const state = particleMomentumState(rows.at(-1));
    readout.textContent = `Unit-mass four-momentum (E/c, p): [${state.components.map(value => value.toFixed(4)).join(", ")}]; invariant mass = ${state.invariantMass.toFixed(6)}. The trajectory is a numerical RK4 approximation.`;
  } else if (activeScene.id === "geodesic") {
    const rays = new Set(rows.map((row) => row.label));
    const maxNullDrift = Math.max(...rows.map((row) => Math.abs(row.metricNorm)), 0);
    readout.textContent = `${rays.size} null-ray geodesics in a softened weak-field metric; the ray bundle bends toward the central mass. Library check: maximum |g(u,u)| drift in this sample = ${maxNullDrift.toExponential(2)}. The model is pedagogical, not a calibrated lens.`;
  } else if (activeScene.id === "curvature") {
    const center = rows.find((row) => row.id === "grid-8-8"), sample = rows.find((row) => row.id === "grid-12-8");
    readout.textContent = `Surface metric overlay: radial gᵣᵣ = ${sample.radialMetric.toFixed(4)} at r = ${Math.hypot(sample.x, sample.z).toFixed(2)}; wire spacing represents the induced radial stretch. Surface color shows Gaussian curvature K; at the center K = ${center.gaussianCurvature.toFixed(4)}. This is a pedagogical embedded surface, not a spacetime metric.`;
  } else if (activeScene.id === "field-slices") {
    const surfaces = fieldIsosurfaces(beta, time);
    readout.textContent = `Normalized vacuum plane wave: E points along x and B along y, propagating along +z at c = 1. Surface height displays 0.45 Eₓ, not physical y position. One playback cycle spans a wave period (ct = 2π × time / (1 + parameter)). ${surfaces.length} translucent sheets mark Eₓ = ±0.6. The synthetic inertial observer paths show field height at each event, not physical y or field-driven trajectories.`;
  } else if (activeScene.id === "fields") {
    const fields = transformedFieldState(beta);
    const list = (values) => values.map((value) => value.toFixed(3)).join(", ");
    readout.textContent = `Library field tensor: E′ = [${list(fields.electric)}], B′ = [${list(fields.magnetic)}]. Invariants: B² − E²/c² = ${fields.invariants.magneticMinusElectricOverC2.toFixed(3)}, E·B = ${fields.invariants.electricDotMagnetic.toFixed(3)}.`;
  } else if (activeScene.id === "aberration") {
    const composed = composeCollinearVelocities(beta, beta);
    readout.textContent = `Library Lorentz composition: two successive collinear velocities of ${beta.toFixed(2)}c produce ${composed.toFixed(3)}c, below c. The star field shows how one of those boosts changes light directions.`;
  } else if (activeScene.id === "twins") {
    const clocks = twinClockReadings(beta, time);
    readout.textContent = `At ct = ${clocks.stayHome.toFixed(3)}: stay-home clock (blue) = ${clocks.stayHome.toFixed(3)}; traveler clock (pink) = ${clocks.traveler.toFixed(3)}. Reunion occurs at ct = ${clocks.reunion[0].toFixed(3)}.`;
  } else {
    readout.textContent = "Adjust time or the model parameter to update this bounded scene sample.";
  }
}

export function drawSceneOverlays(context, sceneId, rows, colors, frame = { time, beta, observerFrame }) {
  const { THREE, scene, resources } = context;
  const frameTime = frame.time ?? time, frameBeta = frame.beta ?? beta, frameObserver = frame.observerFrame ?? observerFrame;
  const scope = createOverlayScope(scene, resources);
  const add = scope.add;
  try {
  const pathScenes = new Set(["minkowski", "twins", "lorentz", "particle", "geodesic"]);
  if (pathScenes.has(sceneId)) {
    const groups = new Map();
    for (const row of rows) {
      const key = sceneId === "twins" || sceneId === "minkowski" || sceneId === "lorentz" || sceneId === "geodesic" ? row.label : "path";
      const group = groups.get(key) ?? []; group.push(row); groups.set(key, group);
    }
    let index = 0;
    for (const group of groups.values()) {
      const distinct = distinctPathPoints(group);
      if (distinct.length < 2) continue;
      const points = distinct.map(({ x, y, z }) => new THREE.Vector3(x, y, z));
      const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
      const geometry = new THREE.TubeGeometry(curve, Math.max(16, (points.length - 1) * 2), .022, 8, false);
      const color = sceneId === "twins" ? twinPathColor(group[0].label) : sceneId === "minkowski" && index === 1 ? 0xff8f9c : colors[sceneId];
      const materialColor = sceneId === "lorentz" ? [0x55b9ff, 0x76d8aa, 0xffd166, 0xff8f9c][index % 4]
        : sceneId === "geodesic" ? [0x55b9ff, 0xff8f9c, 0x76d8aa, 0xffd166, 0xbba1ff, 0xffaa5c, 0x5fc7ff, 0xe89dff][index % 8]
          : color;
      const material = createCarPaintMaterial(THREE, materialColor), mesh = new THREE.Mesh(geometry, material);
      mesh.name = `${sceneId}-trajectory-${index}`; add(mesh, geometry, material); index += 1;
    }
  }
  if (sceneId === "minkowski" || sceneId === "lorentz") {
    selectedWorldlineEvents(sceneId, frameBeta, frameTime, frameObserver).forEach((position, index) => {
      const geometry = new THREE.SphereGeometry(.085, 16, 12);
      const color = index === 0 ? 0x55b9ff : sceneId === "minkowski" ? 0xff8f9c : 0x76d8aa;
      const material = createCarPaintMaterial(THREE, color);
      const marker = new THREE.Mesh(geometry, material);
      marker.position.set(position.x, position.y, position.z);
      marker.name = `${sceneId}-selected-event-${index}`;
      add(marker, geometry, material);
    });
  }
  if (sceneId === "particle" || sceneId === "geodesic") {
    const endpoints = new Map();
    for (const row of rows) endpoints.set(sceneId === "particle" ? "particle" : row.label, row);
    const rayColors = [0x55b9ff, 0xff8f9c, 0x76d8aa, 0xffd166, 0xbba1ff, 0xffaa5c, 0x5fc7ff, 0xe89dff];
    [...endpoints.values()].forEach((position, index) => {
      const geometry = new THREE.SphereGeometry(.085, 16, 12);
      const material = createCarPaintMaterial(THREE, sceneId === "particle" ? colors[sceneId] : rayColors[index % rayColors.length]);
      const marker = new THREE.Mesh(geometry, material);
      marker.position.set(position.x, position.y, position.z);
      marker.name = `${sceneId}-selected-endpoint-${index}`;
      add(marker, geometry, material);
    });
  }
  if (sceneId === "geodesic") {
    const geometry = new THREE.SphereGeometry(.22, 28, 20);
    const material = createCarPaintMaterial(THREE, 0xf0c96d);
    const lens = new THREE.Mesh(geometry, material);
    lens.name = "softened weak-field lens center";
    add(lens, geometry, material);
  }
  if (sceneId === "twins") {
    twinClockPositions(frameBeta, frameTime).forEach((position, index) => {
      const geometry = new THREE.SphereGeometry(.09, 16, 12);
      const material = createCarPaintMaterial(THREE, twinPathColor(index === 0 ? "stay-home clock" : "traveler clock"));
      const marker = new THREE.Mesh(geometry, material);
      marker.position.set(position.x, position.y, position.z);
      marker.name = index === 0 ? "stay-home clock at selected time" : "traveler clock at selected time";
      add(marker, geometry, material);
    });
  }
  if (sceneId === "light-cone" || sceneId === "aberration") {
    const radius = sceneId === "aberration" ? 1 : Math.hypot(rows[0]?.x ?? 0, rows[0]?.y ?? 0, rows[0]?.z ?? 0);
    const geometry = new THREE.SphereGeometry(radius, 32, 18);
    const material = createCarPaintMaterial(THREE, colors[sceneId], { transparent: true, opacity: .14, side: THREE.DoubleSide, depthWrite: false });
    add(new THREE.Mesh(geometry, material), geometry, material);
  }
  if (sceneId === "fields" || sceneId === "field-slices") {
    const tails = new Map(rows.filter((row) => row.id.endsWith("-tail")).map((row) => [row.id.slice(0, -5), row]));
    for (const tip of rows.filter((row) => row.id.endsWith("-tip"))) {
      const tail = tails.get(tip.id.slice(0, -4));
      if (!tail) continue;
      const start = new THREE.Vector3(tail.x, tail.y, tail.z), end = new THREE.Vector3(tip.x, tip.y, tip.z);
      const direction = end.clone().sub(start), length = direction.length();
      if (!(length > 1e-8)) continue;
      direction.normalize(); const headLength = length * .24, shaftLength = length - headLength;
      const orientation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      const color = tip.id.startsWith("electric") ? 0x71e0c2 : 0xbba1ff;
      const shaftGeometry = new THREE.CylinderGeometry(.018, .018, shaftLength, 10);
      const shaftMaterial = createCarPaintMaterial(THREE, color);
      const shaft = new THREE.Mesh(shaftGeometry, shaftMaterial);
      shaft.position.copy(start).addScaledVector(direction, shaftLength / 2); shaft.quaternion.copy(orientation);
      add(shaft, shaftGeometry, shaftMaterial);
      const headGeometry = new THREE.ConeGeometry(.055, headLength, 12);
      const headMaterial = createCarPaintMaterial(THREE, color);
      const head = new THREE.Mesh(headGeometry, headMaterial);
      head.position.copy(end).addScaledVector(direction, -headLength / 2); head.quaternion.copy(orientation);
      add(head, headGeometry, headMaterial);
    }
  }
  if (sceneId === "curvature" || sceneId === "field-slices") {
    const indexed = new Map(); let count = 0;
    for (const row of rows) {
      const match = /^grid-(\d+)-(\d+)$/.exec(row.id);
      if (!match) continue;
      const i = Number(match[1]), j = Number(match[2]); count = Math.max(count, i + 1, j + 1); indexed.set(i * 1000 + j, row);
    }
    if (count > 1 && indexed.size === count * count) {
      const positions = new Float32Array(count * count * 3), indices = [];
      for (let i = 0; i < count; i++) for (let j = 0; j < count; j++) {
        const row = indexed.get(i * 1000 + j), offset = (i * count + j) * 3;
        positions[offset] = row.x; positions[offset + 1] = row.y; positions[offset + 2] = row.z;
      }
      for (let i = 0; i < count - 1; i++) for (let j = 0; j < count - 1; j++) {
        const a = i * count + j, b = (i + 1) * count + j, c = a + 1, d = b + 1;
        indices.push(a, b, c, b, d, c);
      }
      const geometry = new THREE.BufferGeometry(); geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3)); geometry.setIndex(indices);
      if (sceneId === "curvature") {
        const maxCurvature = Math.max(...rows.map((row) => Math.abs(row.gaussianCurvature)), 1e-12);
        const vertexColors = new Float32Array(count * count * 3);
        for (let i = 0; i < count; i++) for (let j = 0; j < count; j++) {
          const row = indexed.get(i * 1000 + j), strength = Math.min(1, Math.abs(row.gaussianCurvature) / maxCurvature);
          const offset = (i * count + j) * 3;
          vertexColors[offset] = row.gaussianCurvature < 0 ? .18 : .18 + .82 * strength;
          vertexColors[offset + 1] = .24 + .56 * (1 - strength);
          vertexColors[offset + 2] = row.gaussianCurvature < 0 ? .48 + .5 * strength : .9 - .72 * strength;
        }
        geometry.setAttribute("color", new THREE.Float32BufferAttribute(vertexColors, 3));
      }
      geometry.computeVertexNormals(); geometry.computeBoundingSphere();
      const material = createCarPaintMaterial(THREE, sceneId === "curvature" ? 0xffffff : colors[sceneId], { side: THREE.DoubleSide, flatShading: false, ...(sceneId === "curvature" ? { vertexColors: true } : {}) });
      add(new THREE.Mesh(geometry, material), geometry, material);
      if (sceneId === "curvature") {
        const wireGeometry = geometry.clone();
        const wireMaterial = createCarPaintMaterial(THREE, 0x91d6ff, { wireframe: true, transparent: true, opacity: .56, depthWrite: false, side: THREE.DoubleSide });
        add(new THREE.Mesh(wireGeometry, wireMaterial), wireGeometry, wireMaterial);
      }
    }
  }
  if (sceneId === "field-slices") {
    for (const [index, worldline] of fieldWorldlines(frameBeta, frameTime).entries()) {
      const points = worldline.path.map(({ x, y, z }) => new THREE.Vector3(x, y, z));
      const geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 48, .025, 8, false);
      const material = createCarPaintMaterial(THREE, index ? 0xff8f9c : 0xffd166);
      add(new THREE.Mesh(geometry, material), geometry, material);
      const markerGeometry = new THREE.SphereGeometry(.08, 16, 12);
      const markerMaterial = createCarPaintMaterial(THREE, index ? 0xff8f9c : 0xffd166);
      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.position.set(worldline.current.x, worldline.current.y, worldline.current.z);
      marker.name = `field-observer-${index}`;
      add(marker, markerGeometry, markerMaterial);
    }
    for (const surface of fieldIsosurfaces(frameBeta, frameTime)) {
      const geometry = new THREE.PlaneGeometry(4.4, 4.4);
      const color = surface.level > 0 ? 0x71e0c2 : 0xbba1ff;
      const material = createCarPaintMaterial(THREE, color, { transparent: true, opacity: .12, side: THREE.DoubleSide, depthWrite: false });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(0, 0, surface.z);
      mesh.name = `electric-field-isosurface-${surface.level}-${surface.z.toFixed(4)}`;
      add(mesh, geometry, material);
    }
  }
  return scope.dispose;
  } catch (error) {
    try { scope.dispose(); } catch (cleanupError) { throw new AggregateError([error, cleanupError], "Overlay construction and cleanup failed."); }
    throw error;
  }
}

export function makeVisualizationPlotter() {
  const colors = { minkowski: 0x55b9ff, "light-cone": 0x8be7ff, lorentz: 0xffd166, twins: 0xff8f9c, aberration: 0xbba1ff, fields: 0x71e0c2, particle: 0xffaa5c, geodesic: 0xe89dff, curvature: 0x76d8aa, "field-slices": 0x5fc7ff };
  const base = pointCloudPlotter;
  return Object.freeze({
    capabilities: Object.freeze(["selection", "time", "animation"]),
    create(context, definition, initialData) {
      const { THREE, scene, resources, renderer } = context;
      const releases = [];
      try {
      releases.push(retainCarPaintEnvironment(THREE, scene, renderer));
      const markerGeometry = resources.track(new THREE.SphereGeometry(.16, 24, 16));
      const markerMaterial = resources.track(createCarPaintMaterial(THREE, 0xf0c96d));
      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.name = "Einstein polished reference marker";
      marker.rotation.y = initialData.time * Math.PI * 2;
      scene.add(marker);
      const axes = new THREE.AxesHelper(2.2);
      axes.name = "labeled-coordinate-axes";
      resources.track(axes.geometry);
      for (const material of Array.isArray(axes.material) ? axes.material : [axes.material]) resources.track(material);
      scene.add(axes);
      const axisLabels = [["x", 0xff6f6f, [2.3, 0, 0]], ["y", 0x71e0a1, [0, 2.3, 0]], ["z", 0x71aaff, [0, 0, 2.3]]].map(([label, color, position]) => {
        const box = createLabeledBox({ THREE, resources }, label, color, { fontSize: 28, worldUnitsPerPixel: .012, depth: .025 });
        box.position.set(...position); box.name = `coordinate-axis-label-${label}`; scene.add(box);
        return box;
      });
      let model = scenes.find((entry) => entry.id === initialData.sceneId) ?? scenes[0];
      let currentTime = initialData.time;
      let currentBeta = initialData.beta ?? beta, currentFrame = initialData.observerFrame ?? observerFrame;
      let releaseOverlays = drawSceneOverlays(context, model.id, initialData.rows, colors, { time: currentTime, beta: initialData.beta, observerFrame: initialData.observerFrame });
      const pointDefinition = {
        ...definition,
        configuration: {
          ...definition.configuration,
          pointCloud: { color: colors[model.id], size: .075, sizeAttenuation: true, name: `${model.id}-samples` },
        },
      };
      const points = base.create(context, pointDefinition, initialData.rows);
      function applyFrame(nextModel, nextTime, nextBeta, nextFrame, rows, pointConfiguration = {}) {
        const previousRotation = marker.rotation.y;
        marker.rotation.y = nextTime * Math.PI * 2;
        try {
          releaseOverlays = replaceOverlay(releaseOverlays, () => drawSceneOverlays(context, nextModel.id, rows, colors, { time: nextTime, beta: nextBeta, observerFrame: nextFrame }), () => points.update(rows, pointConfiguration), context.callbacks.onError);
        } catch (error) { marker.rotation.y = previousRotation; throw error; }
        model = nextModel;
        currentTime = nextTime;
        currentBeta = nextBeta;
        currentFrame = nextFrame;
      }
      return {
        capabilities: ["selection", "time", "animation"],
        update(data) {
          const nextModel = scenes.find((entry) => entry.id === data.sceneId) ?? activeScene;
          applyFrame(nextModel, data.time, data.beta ?? currentBeta, data.observerFrame ?? currentFrame, data.rows, { pointCloud: { color: colors[nextModel.id] } });
          context.requestRender();
        },
        setTime(value) {
          const next = Math.max(0, Math.min(1, Number(value)));
          const rows = buildSceneData(model.id, { time: next, beta: currentBeta, observerFrame: currentFrame });
          applyFrame(model, next, currentBeta, currentFrame, rows);
          context.requestRender();
        },
        updateFrame({ elapsedSeconds }) {
          const next = (elapsedSeconds % 8) / 8;
          const rows = buildSceneData(model.id, { time: next, beta: currentBeta, observerFrame: currentFrame });
          applyFrame(model, next, currentBeta, currentFrame, rows);
          time = next;
          timeSlider.value = String(next);
          byId("time-output").value = next.toFixed(2);
        },
        describeSelection(hit) { return points.describeSelection(hit); },
        dispose() {
          scene.remove(marker);
          scene.remove(axes);
          axisLabels.forEach((label) => scene.remove(label));
          points.dispose();
          releaseOverlays();
          resources.release(markerGeometry);
          resources.release(markerMaterial);
          releases.forEach((release) => release());
        },
      };
      } catch (error) {
        releases.forEach((release) => release());
        throw error;
      }
    },
  });
}

async function mountScene() {
  if (viewer) {
    viewer.dispose();
    viewer = undefined;
  }
  host.replaceChildren();
  const runtime = globalThis.EINSTEIN_THREE_RUNTIME;
  if (!runtime?.THREE) {
    fallback.hidden = false;
    status.textContent = "3D rendering is unavailable offline.";
    return;
  }
  fallback.hidden = true;
  status.textContent = "Preparing local 3D renderer…";
  try {
    const mountedState = { sceneId: activeScene.id, time, beta, observerFrame };
    const rows = currentData();
    const scenePlotter = makeVisualizationPlotter();
    const mountedViewer = await createViewerWithRuntime(host, {
      plotter: scenePlotter,
      data: { ...mountedState, rows },
      configuration: { backgroundColor: 0x08131f, autoplay: false, pointCloud: { color: 0x55b9ff } },
      renderer: { antialias: true, maxPixelRatio: 1.5, powerPreference: "high-performance", toneMapping: "ACESFilmicToneMapping", toneMappingExposure: 1.05 },
      camera: { type: "perspective", position: [4.5, 3.6, 5.2], target: [0, 0, 0], fov: 45 },
      controls: { enabled: Boolean(runtime.OrbitControls), enableDamping: true, dampingFactor: .07, minDistance: 2.5, maxDistance: 15 },
      callbacks: {
        onError(error) { status.textContent = `Scene error: ${error.message}`; },
        onSelection(selection) {
          const index = currentRows.findIndex(row => row.id === selection.id);
          stopAnimation();
          const displayed = selection.values, current = currentRows[index];
          if (!current || ["x", "y", "z"].some(axis => current[axis] !== displayed[axis]) || String(current.label ?? current.id) !== String(displayed.label ?? displayed.id)) {
            byId("selection-readout").textContent = describeDisplayedSample(displayed);
            return;
          }
          byId("sample-number").value = String(index + 1);
          byId("selection-readout").textContent = describeSample(currentRows, index + 1);
        },
      },
    }, runtime);
    if (pageClosed) { mountedViewer.dispose(); return; }
    viewer = mountedViewer;
    if (mountedState.sceneId !== activeScene.id || mountedState.time !== time || mountedState.beta !== beta || mountedState.observerFrame !== observerFrame) {
      await viewer.update({ sceneId: activeScene.id, rows: currentData(), time, beta, observerFrame });
    }
    if (pageClosed) return;
    updateSceneLabel();
    status.textContent = runtime.OrbitControls ? "3D scene ready · library operations connected." : "3D scene ready · camera orbit controls unavailable.";
  } catch (error) {
    viewer?.dispose?.();
    viewer = undefined;
    if (pageClosed) return;
    updateSceneLabel();
    fallback.hidden = false;
    status.textContent = `3D renderer could not start: ${error.message}`;
  }
}

function updateRows() {
  const selectionReadout = byId("selection-readout");
  const selectionPrompt = "Select a sample point or use the sample inspector to view its display coordinates.";
  if (selectionReadout.textContent !== selectionPrompt) selectionReadout.textContent = selectionPrompt;
  byId("time-output").value = time.toFixed(2);
  byId("speed-output").value = beta.toFixed(2);
  const rows = currentData();
  currentRows = rows;
  const sampleNumber = byId("sample-number");
  sampleNumber.max = String(rows.length);
  if (!Number.isInteger(Number(sampleNumber.value)) || Number(sampleNumber.value) < 1 || Number(sampleNumber.value) > rows.length) sampleNumber.value = "1";
  updateModelReadout(rows);
  updateViewer(viewer, { sceneId: activeScene.id, rows, time, beta, observerFrame }, {
    isClosed: () => pageClosed,
    report: error => { status.textContent = `Scene update failed: ${error.message}`; },
  });
}

function selectScene(scene) {
  activeScene = scene;
  time = 0;
  playback.seek(0);
  observerFrame = "lab";
  timeSlider.value = "0";
  stopAnimation();
  updateDescription();
  updateRows();
}

function stopAnimation() {
  playback.stop();
  viewer?.pause?.();
}

function startAnimation() {
  const result = playback.play();
  if (result === "reduced-motion") {
    status.textContent = "Animation is off because reduced motion is enabled. Use the time slider to inspect frames.";
    return;
  }
  if (result === "unavailable") {
    status.textContent = "Animation is unavailable after this page closes.";
  }
}

for (const [index, scene] of scenes.entries()) {
  const item = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.scene = scene.id;
  button.setAttribute("aria-label", `Visualization ${index + 1}: ${scene.title}`);
  button.innerHTML = `<span class="scene-number">${String(index + 1).padStart(2, "0")}</span><span>${scene.title}</span>`;
  button.addEventListener("click", () => { history.replaceState(null, "", `#${scene.id}`); selectScene(scene); });
  item.append(button);
  sceneList.append(item);
}

timeSlider.addEventListener("input", () => { time = Number(timeSlider.value); playback.seek(time); updateRows(); });
byId("scene").addEventListener("keydown", event => {
  if (event.target !== event.currentTarget || event.altKey || event.ctrlKey || event.metaKey) return;
  try { if (moveCameraWithKey(viewer, event.key)) event.preventDefault(); }
  catch (error) { status.textContent = `Camera control failed: ${error.message}`; }
});
byId("skip-to-scene").addEventListener("click", event => {
  event.preventDefault();
  const target = byId("scene");
  target.focus({ preventScroll: true });
  target.scrollIntoView({ block: "start" });
});
window.addEventListener("hashchange", () => {
  if (location.hash !== "#scene") selectScene(sceneFromFragment(location.hash));
});
speedSlider.addEventListener("input", () => { beta = Number(speedSlider.value); updateRows(); });
observerFrameSelect.addEventListener("change", () => { observerFrame = observerFrameSelect.value; updateRows(); });
playButton.addEventListener("click", () => playButton.getAttribute("aria-pressed") === "true" ? stopAnimation() : startAnimation());
byId("inspect-sample").addEventListener("click", () => {
  stopAnimation();
  try { byId("selection-readout").textContent = describeSample(currentRows, Number(byId("sample-number").value)); }
  catch (error) { byId("selection-readout").textContent = error.message; }
});
byId("reset-button").addEventListener("click", () => {
  stopAnimation();
  time = 0;
  playback.seek(0);
  beta = .6;
  observerFrame = "lab";
  timeSlider.value = "0";
  speedSlider.value = ".6";
  observerFrameSelect.value = observerFrame;
  updateDescription();
  viewer?.resetView?.();
  updateRows();
});
prefersReducedMotion.addEventListener?.("change", (event) => { if (event.matches) stopAnimation(); });
window.addEventListener("beforeunload", () => { pageClosed = true; stopAnimation(); playback.dispose(); viewer?.dispose(); viewer = undefined; }, { once: true });

updateDescription();
updateRows();
await mountScene();
