import { validatePointRecords } from "../data/validation.js";
import { VisualizationError } from "../core/errors.js";

function buildGeometry(THREE, records) {
  const positions = new Float32Array(records.length * 3);
  records.forEach((point, index) => {
    positions[index * 3] = Math.fround(point.x);
    positions[index * 3 + 1] = Math.fround(point.y);
    positions[index * 3 + 2] = Math.fround(point.z);
  });
  const geometry = new THREE.BufferGeometry();
  try {
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.computeBoundingSphere?.();
    return geometry;
  } catch (error) {
    geometry.dispose?.();
    throw error;
  }
}

/** Basic selectable point-cloud plotter. It creates scene objects only; it creates no DOM/UI. */
export const pointCloudPlotter = Object.freeze({
  capabilities: Object.freeze(["selection"]),
  create(context, definition, initialData) {
    const { THREE, scene, resources } = context;
    const settings = definition.configuration.pointCloud ?? {};
    if (typeof THREE.BufferGeometry !== "function" || typeof THREE.Points !== "function" || typeof THREE.PointsMaterial !== "function") {
      throw new VisualizationError("POINTS_UNAVAILABLE", "The runtime does not provide the Three.js point-cloud classes.");
    }
    let records = validatePointRecords(initialData);
    let geometry = resources.track(buildGeometry(THREE, records));
    const material = resources.track(new THREE.PointsMaterial({
      color: settings.color ?? 0x4ea5ff,
      size: settings.size ?? 0.06,
      sizeAttenuation: settings.sizeAttenuation ?? true,
    }));
    const points = new THREE.Points(geometry, material);
    points.name = settings.name ?? "point-cloud";
    scene.add(points);

    return {
      capabilities: ["selection"],
      update(nextData, configuration = {}) {
        const nextRecords = validatePointRecords(nextData);
        const nextGeometry = resources.track(buildGeometry(THREE, nextRecords));
        const previousGeometry = geometry, previousRecords = records;
        const previousColor = material.color?.clone?.();
        try {
          if (configuration.pointCloud?.color !== undefined) material.color?.set?.(configuration.pointCloud.color);
          geometry = nextGeometry;
          points.geometry = nextGeometry;
          records = nextRecords;
          context.requestRender();
        } catch (error) {
          geometry = previousGeometry;
          points.geometry = previousGeometry;
          records = previousRecords;
          if (previousColor) material.color.copy(previousColor);
          resources.release(nextGeometry);
          throw error;
        }
        try { resources.release(previousGeometry); } catch (error) { context.callbacks?.onError?.(error); }
      },
      describeSelection(hit) {
        if (hit.object !== points) return undefined;
        const record = records[hit.index];
        if (!record) return undefined;
        return Object.freeze({ id: record.id, label: String(record.label ?? record.id), values: Object.freeze({ ...record }) });
      },
      dispose() {
        scene.remove(points);
        resources.release(geometry);
        resources.release(material);
      },
    };
  },
});
