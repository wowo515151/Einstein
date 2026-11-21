import assert from "node:assert/strict";
import test from "node:test";
import { buildSceneData, scenePlaybackCursor } from "../visualizations/scene-models.js";

test("aberration playback cursor visits different transformed sky directions", () => {
  const rows = buildSceneData("aberration", { beta: .6 });
  const start = scenePlaybackCursor("aberration", { time: 0, beta: .6, rows });
  const end = scenePlaybackCursor("aberration", { time: 1, beta: .6, rows });
  assert.deepEqual(start, { x: rows[0].x, y: rows[0].y, z: rows[0].z });
  assert.deepEqual(end, { x: rows.at(-1).x, y: rows.at(-1).y, z: rows.at(-1).z });
  assert.notDeepEqual(start, end);
});

test("field mixing playback cursor scans between the unchanged field vector tips", () => {
  const rows = buildSceneData("fields", { beta: .6 });
  const start = scenePlaybackCursor("fields", { time: 0, beta: .6, rows });
  const middle = scenePlaybackCursor("fields", { time: .5, beta: .6, rows });
  const end = scenePlaybackCursor("fields", { time: 1, beta: .6, rows });
  assert.deepEqual(start, { x: rows.find((row) => row.id === "electric-tip").x, y: rows.find((row) => row.id === "electric-tip").y, z: rows.find((row) => row.id === "electric-tip").z });
  assert.deepEqual(end, { x: rows.find((row) => row.id === "magnetic-tip").x, y: rows.find((row) => row.id === "magnetic-tip").y, z: rows.find((row) => row.id === "magnetic-tip").z });
  assert.notDeepEqual(start, middle);
  assert.notDeepEqual(middle, end);
});

test("curvature playback cursor moves radially across and back on the embedded surface", () => {
  const start = scenePlaybackCursor("curvature", { time: 0, beta: .6 });
  const middle = scenePlaybackCursor("curvature", { time: .5, beta: .6 });
  const end = scenePlaybackCursor("curvature", { time: 1, beta: .6 });
  assert.deepEqual(start, end);
  assert.equal(start.x, 0);
  assert.ok(middle.x > 1.7);
  assert.ok(middle.y < start.y);
});

test("static-model playback cursors reject unsupported inputs", () => {
  assert.throws(() => scenePlaybackCursor("minkowski", { time: 0 }), /no static-model playback cursor/);
  assert.throws(() => scenePlaybackCursor("fields", { time: 1.1 }), /time must be in/);
  assert.throws(() => scenePlaybackCursor("fields", { beta: 1 }), /beta must be in/);
});
