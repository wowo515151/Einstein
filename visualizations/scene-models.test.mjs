import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { buildSceneData, composeCollinearVelocities, curvatureEmbeddingValues, curvatureEmbeddingLibraryCheck, fieldIsosurfaces, fieldWorldlines, planeWaveState, particleMomentumState, sampleLimit, sceneAccessibleLabel, sceneFromFragment, scenes, sceneAxes, selectedWorldlineEvents, transformedFieldState, twinClockReadings, twinClockPositions } from "./scene-models.js";
import { LorentzTransform, Vector4, samplePlaneWaveBatch } from "../src/relativity/index.js";
import { updateViewer } from "./viewer-update.js";
import { distinctPathPoints, twinPathColor } from "./path-points.js";

test("stale viewer failures do not overwrite newer update status",async()=>{
  const pending=[],reports=[],viewer={update:()=>new Promise((resolve,reject)=>pending.push({resolve,reject}))};
  const first=updateViewer(viewer,{}, {isClosed:()=>false,report:error=>reports.push(error.message)});
  const second=updateViewer(viewer,{}, {isClosed:()=>false,report:error=>reports.push(error.message)});
  pending[1].resolve();await second;pending[0].reject(new Error("stale"));await first;assert.deepEqual(reports,[]);
  const latest=updateViewer(viewer,{}, {isClosed:()=>false,report:error=>reports.push(error.message)});
  pending[2].reject(new Error("current"));await latest;assert.deepEqual(reports,["current"]);
  await updateViewer({update(){throw new Error("synchronous");}}, {}, {isClosed:()=>false,report:error=>reports.push(error.message)});
  assert.deepEqual(reports,["current","synchronous"]);
  let recoveries=0;
  const oldSuccess=updateViewer(viewer,{}, {isClosed:()=>false,report:error=>reports.push(error.message),onSuccess:()=>recoveries++});
  const newFailure=updateViewer(viewer,{}, {isClosed:()=>false,report:error=>reports.push(error.message),onSuccess:()=>recoveries++});
  pending[4].reject(new Error("new failure"));await newFailure;pending[3].resolve();await oldSuccess;
  assert.equal(recoveries,0);assert.equal(reports.at(-1),"new failure");
});
test("curvature embedding approaches flat space at extreme finite distances",()=>{
  for(const beta of [0,.8])for(const [x,z] of [[1e200,0],[0,-1e200],[1e308,1e308]])
    assert.deepEqual(curvatureEmbeddingValues(beta,x,z),{radialMetric:1,gaussianCurvature:0});
});
test("stationary particle paths have no degenerate tube segments",()=>{
  const stationary=buildSceneData("particle",{beta:0,time:1});
  assert.equal(distinctPathPoints(stationary).length,1);
  const moving=buildSceneData("particle",{beta:.4,time:1});
  assert.ok(distinctPathPoints(moving).length>2);
});
test("twin paths and markers use distinct consistent colors",()=>{
  assert.equal(twinPathColor("stay-home clock"),0x55b9ff);
  assert.equal(twinPathColor("traveler clock"),0xff8f9c);
  assert.notEqual(twinPathColor("stay-home clock"),twinPathColor("traveler clock"));
});

test("viewer update reports failures only while the page is active", async () => {
  const reports = [], error = new Error("renderer unavailable"), viewer = { update: async () => { throw error; } };
  await updateViewer(viewer, {}, { isClosed: () => false, report: e => reports.push(e) });
  await updateViewer(viewer, {}, { isClosed: () => true, report: e => reports.push(e) });
  await updateViewer(undefined, {}, { isClosed: () => false, report: e => reports.push(e) });
  assert.deepEqual(reports, [error]);
});

test("visualization shell loads licensed local Three.js and controls before the application module", async () => {
  const [html, runtime, controls, license, app, playback] = await Promise.all([
    readFile(new URL("./index.html", import.meta.url), "utf8"),
    readFile(new URL("./vendor/three-r128.min.js", import.meta.url), "utf8"),
    readFile(new URL("./vendor/OrbitControls-r128.js", import.meta.url), "utf8"),
    readFile(new URL("./vendor/THREE-LICENSE.txt", import.meta.url), "utf8"),
    readFile(new URL("./visualizations.js", import.meta.url), "utf8"),
    readFile(new URL("./playback.js", import.meta.url), "utf8"),
  ]);
  const runtimeScript = html.indexOf('src="./vendor/three-r128.min.js"');
  const controlsScript = html.indexOf('src="./vendor/OrbitControls-r128.js"');
  const runtimeRegistration = html.indexOf("EINSTEIN_THREE_RUNTIME");
  const applicationModule = html.indexOf('type="module" src="./visualizations.js"');
  assert.ok(runtimeScript >= 0 && runtimeScript < controlsScript);
  assert.ok(controlsScript < runtimeRegistration && runtimeRegistration < applicationModule);
  assert.doesNotMatch(html, /<script\b[^>]*src=["']https?:/i);
  assert.match(runtime.slice(0, 512), /Three\.js Authors[\s\S]*SPDX-License-Identifier: MIT/);
  assert.match(runtime.slice(0, 512), /["']128["']/);
  assert.match(controls, /THREE\.OrbitControls = OrbitControls/);
  assert.match(license, /Three\.js[\s\S]*SPDX-License-Identifier: MIT/);
  assert.match(html, /id="observer-frame-control"[\s\S]*id="observer-frame"/);
  assert.match(html, /id="axis-key"[\s\S]*id="axis-x-meaning"[\s\S]*id="axis-y-meaning"[\s\S]*id="axis-z-meaning"/);
  assert.match(html, /id="scene-title" aria-live="polite"/);
  assert.match(html, /id="time-output"[^>]*aria-live="off"/);
  assert.doesNotMatch(html, /class="scene-description"[^>]*aria-live/);
  assert.match(app, /observerFrameSelect\.addEventListener\("change"/);
  assert.match(app, /observerFrameControl\.hidden = activeScene\.id !== "minkowski" && activeScene\.id !== "lorentz"/);
  assert.match(app, /vertexColors: true/);
  assert.match(app, /wireframe: true/);
  assert.match(app, /new THREE\.AxesHelper\(2\.2\)/);
  assert.match(app, /createLabeledBox\(\{ THREE, resources \}, label, color/);
  assert.match(app, /createPlaybackController/);
  assert.match(app, /prefersReducedMotion\.addEventListener/);
  assert.match(app, /beforeunload/);
  assert.match(app, /const rows = currentData\(\);\s*currentRows = rows;[\s\S]*?updateModelReadout\(rows\);\s*updateViewer\(viewer, \{ sceneId: activeScene\.id, rows, time, beta, observerFrame \}/);
  assert.match(app, /beforeunload"[\s\S]*?playback\.dispose\(\); viewer\?\.dispose\(\); viewer = undefined/);
  assert.match(app, /if \(pageClosed\) \{ mountedViewer\.dispose\(\); return; \}/);
  assert.match(app, /let model = scenes\.find\(\(entry\) => entry\.id === initialData\.sceneId\)/);
  assert.match(app, /let currentTime = initialData\.time;[\s\S]*?initialData\.rows, colors, \{ time: currentTime, beta: initialData\.beta, observerFrame: initialData\.observerFrame \}/);
  assert.match(app, /mountedState\.sceneId !== activeScene\.id[\s\S]*await viewer\.update\(\{ sceneId: activeScene\.id, rows: currentData\(\), time, beta, observerFrame \}\)/);
  assert.match(app, /viewer\?\.dispose\?\.\(\);\s*viewer = undefined;/);
  assert.match(playback, /export function createPlaybackController/);
});

test("the visualization catalog contains ten unique scenes", () => {
  assert.equal(scenes.length, 10);
  assert.equal(new Set(scenes.map((scene) => scene.id)).size, 10);
  assert.equal(scenes.find((scene) => scene.id === "aberration").title, "Velocity addition and aberration");
});
test("top-level scene links resolve to each scene and unknown fragments fall back",async()=>{
  const index=await readFile(new URL("../index.html",import.meta.url),"utf8");
  for(const scene of scenes){
    assert.ok(index.includes(`visualizations/index.html#${scene.id}`));
    assert.equal(sceneFromFragment(`#${scene.id}`),scene);
  }
  assert.equal(sceneFromFragment("#unknown"),scenes[0]);
  assert.match(sceneAccessibleLabel(sceneFromFragment("#twins"),true),/Twin-paradox worldlines.*Drag to orbit/);
  assert.doesNotMatch(sceneAccessibleLabel(sceneFromFragment("#particle"),false),/Drag to orbit/);
});
test("field explorer worldlines are subluminal and mark the selected time",()=>{
  for(const beta of [0,.9])for(const line of fieldWorldlines(beta,.5)){
    assert.equal(line.path.length,25);
    assert.deepEqual(line.current,line.path[12]);
    const elapsed=line.path.at(-1).ct-line.path[0].ct;
    assert.ok(Math.abs(line.path.at(-1).x-line.path[0].x)/elapsed<1);
    assert.notEqual(line.path.at(-1).x,line.path[0].x);
    assert.ok(Math.abs(line.current.ct-planeWaveState(beta,.5,line.current.z).ct)<1e-12);
    assert.ok(Math.abs(Math.abs(line.path.at(-1).x-line.path[0].x)/elapsed-(.2+.6*beta))<1e-12);
    assert.ok(line.path.every((event,index)=>Math.abs(event.y-.45*planeWaveState(beta,index/24,event.z).electric[0])<1e-12));
    assert.ok(line.path.every(event=>event.physicalY===line.current.physicalY));
  }
  assert.throws(()=>fieldWorldlines(1,.5),/supported ranges/);
});
test("all ten scenes declare x/y/z meanings for the accessible normalized-unit axis legend", () => {
  assert.equal(Object.keys(sceneAxes).length, scenes.length);
  for (const scene of scenes) {
    assert.deepEqual(Object.keys(sceneAxes[scene.id]), ["x", "y", "z"]);
    for (const meaning of Object.values(sceneAxes[scene.id])) assert.ok(meaning.length > 0, `${scene.id} has an unlabeled axis`);
  }
});

test("all scene samples are finite, deterministic, and bounded", () => {
  for (const scene of scenes) {
    const options = { time: .37, beta: .62 };
    const first = buildSceneData(scene.id, options);
    const second = buildSceneData(scene.id, options);
    assert.ok(first.length > 0, `${scene.id} returned no sample data`);
    assert.ok(first.length <= sampleLimit, `${scene.id} exceeded the sampling bound`);
    assert.deepEqual(first, second, `${scene.id} was not deterministic`);
    for (const row of first) {
      for (const axis of ["x", "y", "z"]) assert.ok(Number.isFinite(row[axis]), `${scene.id}.${axis} is not finite`);
    }
  }
});
test("curved spatial scenes expose complete rectangular grids for surface rendering", () => {
  const curvature = buildSceneData("curvature"), slices = buildSceneData("field-slices");
  assert.equal(curvature.length, 17 * 17);
  assert.equal(slices.filter((row) => row.id.startsWith("grid-")).length, 16 * 16);
  assert.equal(slices.length, sampleLimit);
  assert.equal(slices.filter((row) => row.id.startsWith("electric-slice-") && row.id.endsWith("-tip")).length, 16);
  assert.equal(slices.filter((row) => row.id.startsWith("magnetic-slice-") && row.id.endsWith("-tip")).length, 16);
  assert.equal(new Set(curvature.map((row) => row.id)).size, curvature.length);
});
test("curved embedding exposes its induced radial metric and Gaussian curvature", () => {
  const beta = .62, center = curvatureEmbeddingValues(beta, 0, 0), radius = curvatureEmbeddingValues(beta, 1, 0);
  const amplitude = .55 * beta, decay = .38;
  const expectedCenterCurvature = 4 * decay ** 2 * amplitude ** 2;
  const first = -2 * decay * amplitude * Math.exp(-decay);
  const second = 2 * decay * amplitude * Math.exp(-decay) * (2 * decay - 1);
  const expectedMetric = 1 + first ** 2;
  const expectedCurvature = first * second / expectedMetric ** 2;
  assert.equal(center.radialMetric, 1);
  assert.ok(Math.abs(center.gaussianCurvature - expectedCenterCurvature) < 1e-12);
  assert.ok(Math.abs(radius.radialMetric - expectedMetric) < 1e-12);
  assert.ok(Math.abs(radius.gaussianCurvature - expectedCurvature) < 1e-12);
  assert.ok(radius.radialMetric > center.radialMetric);
  const rows = buildSceneData("curvature", { beta });
  assert.ok(rows.every((row) => Number.isFinite(row.radialMetric) && Number.isFinite(row.gaussianCurvature)));
  assert.throws(() => curvatureEmbeddingValues(1, 0, 0), /beta/);
});
test("curvature scene matches the library curvature of its induced metric",()=>{
  for(const beta of [.1,.6,.95])for(const [x,z] of [[0,0],[1.1,0],[.7,-.4]]){
    const expected=curvatureEmbeddingValues(beta,x,z).gaussianCurvature;
    assert.ok(Math.abs(curvatureEmbeddingLibraryCheck(beta,x,z)-expected)<1e-7);
  }
});

test("animated field slice carries paired E/B vectors and analytic isosurfaces", () => {
  const beta = .62, time = .37, rows = buildSceneData("field-slices", { beta, time });
  const grid=rows.filter(row=>row.id.startsWith("grid-"));
  assert.equal(grid.length,256);
  for(const row of grid)assert.ok(Math.abs(row.y-.45*Math.cos((1+beta)*row.z-time*Math.PI*2))<1e-12);
  const tips = rows.filter((row) => row.vector);
  assert.equal(tips.length, 32);
  for (const row of tips) {
    const amplitude = Math.cos((1 + beta) * row.z - time * Math.PI * 2);
    if (row.id.startsWith("electric-slice-")) assert.ok(Math.hypot(...row.vector.map((v, i) => v - [amplitude, 0, 0][i])) < 1e-12);
    else assert.ok(Math.hypot(...row.vector.map((v, i) => v - [0, amplitude, 0][i])) < 1e-12);
  }
  const surfaces = fieldIsosurfaces(beta, time);
  assert.ok(surfaces.length > 0);
  for (const { z, level } of surfaces) {
    assert.ok(Math.abs(Math.cos((1 + beta) * z - time * Math.PI * 2) - level) < 1e-12);
    assert.ok(Math.abs(z) <= 2);
  }
  assert.throws(() => fieldIsosurfaces(1, time), /beta/);
});
test("plane-wave fields satisfy source-free Maxwell evolution in normalized coordinates",()=>{
  for(const beta of [0,.62,.9]){
    const k=1+beta,ct=.43,z=.71,h=1e-5;
    const at=(t,position)=>planeWaveState(beta,t*k/(2*Math.PI),position);
    const before=at(ct-h,z),after=at(ct+h,z),left=at(ct,z-h),right=at(ct,z+h),center=at(ct,z);
    const dEct=(after.electric[0]-before.electric[0])/(2*h),dBct=(after.magnetic[1]-before.magnetic[1])/(2*h);
    const dEz=(right.electric[0]-left.electric[0])/(2*h),dBz=(right.magnetic[1]-left.magnetic[1])/(2*h);
    assert.ok(Math.abs(dEct+dBz)<1e-9);assert.ok(Math.abs(dBct+dEz)<1e-9);
    assert.ok(Math.abs(center.ct-ct)<1e-15);
    assert.equal(center.electric[0],center.magnetic[1]);
    assert.equal(center.electric[1],0);assert.equal(center.magnetic[0],0);
  }
});

test("batch plane-wave CPU sampling agrees with the visualization field model",async()=>{
  for(const beta of [0,.62,.9])for(const time of [0,.37,1]){
    const z=.71,model=planeWaveState(beta,time,z);
    const [sample]=await samplePlaneWaveBatch([model.ct,0,0,z],{waveNumber:1+beta,backend:"cpu"});
    assert.deepEqual(sample.electric,model.electric);
    assert.deepEqual(sample.magnetic,model.magnetic);
  }
});

test("field isosurfaces cover large extents and periodic playback offsets",()=>{
  const beta=.4, time=.37, extent=40, level=.6, k=1+beta;
  const expected=[];
  for(const value of [-level,level])for(const sign of [-1,1])for(let n=-30;n<=30;n++){
    const z=(time*2*Math.PI+sign*Math.acos(value)+n*2*Math.PI)/k;
    if(Math.abs(z)<=extent)expected.push({z,level:value});
  }
  expected.sort((a,b)=>a.z-b.z);
  const actual=fieldIsosurfaces(beta,time,{extent,level});
  assert.equal(actual.length,expected.length);
  for(let i=0;i<actual.length;i++){
    assert.equal(actual[i].level,expected[i].level);
    assert.ok(Math.abs(actual[i].z-expected[i].z)<1e-12);
  }
  const shifted=fieldIsosurfaces(beta,time+100,{extent,level});
  assert.equal(shifted.length,actual.length);
  shifted.forEach((surface,i)=>assert.ok(Math.abs(surface.z-actual[i].z)<1e-12));
  assert.equal(fieldIsosurfaces(0,0,{level:1e-14}).length,4);
  assert.throws(()=>fieldIsosurfaces(0,0,{extent:1e8}),/count exceeds/);
});

test("light-cone radius is proportional to normalized coordinate time", () => {
  const small = buildSceneData("light-cone", { time: .6 })[0];
  const large = buildSceneData("light-cone", { time: .7 })[0];
  const smallRadius = Math.hypot(small.x, small.y, small.z);
  const largeRadius = Math.hypot(large.x, large.y, large.z);
  assert.ok(Math.abs(largeRadius / smallRadius - 2) < 1e-12);
  assert.ok(buildSceneData("light-cone", { time: .25 })[0].ct < 0);
  assert.equal(buildSceneData("light-cone", { time: .5 })[0].ct, 0);
  assert.ok(buildSceneData("light-cone", { time: .75 })[0].ct > 0);
});

test("causal worldline samples use the library's Minkowski interval", () => {
  const samples = buildSceneData("minkowski", { beta: .6 });
  for (const sample of samples) {
    const expected = new Vector4([sample.ct, sample.x, 0, 0]);
    assert.equal(sample.z, 0);
    assert.equal(sample.causalType, expected.causalType());
    assert.ok(Math.abs(sample.intervalSquared - expected.normSquared()) < 1e-12);
  }
  assert.equal(samples[0].causalType, Vector4.CAUSAL_TIMELIKE);
});

test("Minkowski observer-frame selection boosts both worldlines and preserves their intervals", () => {
  const beta = .6;
  const laboratory = buildSceneData("minkowski", { beta, observerFrame: "lab" });
  const movingFrame = buildSceneData("minkowski", { beta, observerFrame: "moving" });
  const labMovingLine = laboratory.filter((row) => row.label === "moving observer");
  const boostedMovingLine = movingFrame.filter((row) => row.label === "moving observer");
  const boostedLabLine = movingFrame.filter((row) => row.label === "laboratory observer");
  assert.equal(labMovingLine.length, 100);
  assert.equal(boostedMovingLine.length, 100);
  assert.equal(boostedLabLine.length, 100);
  for (let i = 0; i < 100; i++) {
    assert.ok(Math.abs(labMovingLine[i].x - beta * labMovingLine[i].ct) < 1e-12);
    assert.ok(Math.abs(boostedMovingLine[i].x) < 1e-12);
    assert.ok(Math.abs(laboratory.find((row) => row.id === boostedMovingLine[i].id).intervalSquared - boostedMovingLine[i].intervalSquared) < 1e-12);
    assert.ok(Math.abs(laboratory.find((row) => row.id === boostedLabLine[i].id).intervalSquared - boostedLabLine[i].intervalSquared) < 1e-12);
  }
  assert.throws(() => buildSceneData("minkowski", { observerFrame: "rest-of-universe" }), /observerFrame/);
});

test("light-cone samples come from the library and are null separated", () => {
  const samples = buildSceneData("light-cone", { time: .4 });
  assert.equal(samples.length, 220);
  for (const sample of samples) {
    assert.equal(sample.causalType, Vector4.CAUSAL_NULL);
    assert.ok(Math.abs(Math.abs(sample.ct) - Math.hypot(sample.x, sample.y, sample.z)) < 1e-12);
  }
});

test("Lorentz scene shows endpoint worldlines and both synchronized rod lengths", () => {
  const beta = .7, rest = buildSceneData("lorentz", { beta, observerFrame: "lab" }), moving = buildSceneData("lorentz", { beta, observerFrame: "moving" });
  const boost = LorentzTransform.boost([beta, 0, 0]);
  const properSlice = rest.filter((row) => row.label === "rod-rest simultaneity slice");
  const tiltedMovingSlice = rest.filter((row) => row.label === "moving-observer simultaneity slice");
  const contractedSlice = moving.filter((row) => row.label === "moving-observer simultaneity slice");
  const tiltedRestSlice = moving.filter((row) => row.label === "rod-rest simultaneity slice");
  const properLength = properSlice.at(-1).x - properSlice[0].x;
  const measuredLength = contractedSlice.at(-1).x - contractedSlice[0].x;
  assert.equal(rest.filter((row) => row.label.endsWith("endpoint worldline")).length, 96);
  assert.ok(Math.abs(properLength - 4) < 1e-12);
  assert.ok(Math.abs(measuredLength - 4 * Math.sqrt(1 - beta ** 2)) < 1e-12);
  assert.ok(properSlice.every((row) => Math.abs(row.ct) < 1e-12));
  assert.ok(contractedSlice.every((row) => Math.abs(row.ct) < 1e-12));
  assert.ok(Math.max(...tiltedMovingSlice.map((row) => row.ct)) - Math.min(...tiltedMovingSlice.map((row) => row.ct)) > .1);
  assert.ok(Math.max(...tiltedRestSlice.map((row) => row.ct)) - Math.min(...tiltedRestSlice.map((row) => row.ct)) > .1);
  const labEndpoint = rest.find((row) => row.label === "left rod endpoint worldline" && row.sourceCt > 0);
  const movingEndpoint = moving.find((row) => row.id === labEndpoint.id);
  const expected = boost.applyVector([labEndpoint.sourceCt, labEndpoint.sourceX, 0, 0]).components;
  assert.ok(Math.abs(movingEndpoint.ct - expected[0]) < 1e-12);
  assert.ok(Math.abs(movingEndpoint.x - expected[1]) < 1e-12);
  assert.ok(Math.abs(new Vector4([labEndpoint.sourceCt, labEndpoint.sourceX, 0, 0]).normSquared() - new Vector4([movingEndpoint.ct, movingEndpoint.x, 0, 0]).normSquared()) < 1e-12);
  assert.throws(() => buildSceneData("lorentz", { observerFrame: "invalid" }), /observerFrame/);
});
test("worldline markers follow selected source time in either observer frame",()=>{
  const beta=.6,time=.25,sourceCt=-1,transform=LorentzTransform.boost([beta,0,0]);
  for(const scene of ["minkowski","lorentz"])for(const frame of ["lab","moving"]){
    const positions=scene==="minkowski"?[0,beta*sourceCt]:[-2,2],markers=selectedWorldlineEvents(scene,beta,time,frame);
    for(let i=0;i<2;i++){
      const expected=frame==="moving"?transform.applyVector([sourceCt,positions[i],0,0]).components:[sourceCt,positions[i],0,0];
      assert.ok(Math.abs(markers[i].x-expected[1])<1e-12);
      assert.ok(Math.abs(markers[i].y-expected[0])<1e-12);
    }
  }
  assert.notDeepEqual(selectedWorldlineEvents("minkowski",beta,0),selectedWorldlineEvents("minkowski",beta,1));
});

test("aberration directions are derived from the library Lorentz boost", () => {
  const beta = .54, count = 140, u = 1 - 1 / count, r = Math.sqrt(1 - u * u);
  const boosted = LorentzTransform.boost([beta, 0, 0]).applyVector([1, r, u, 0]).components;
  const sample = buildSceneData("aberration", { beta })[0];
  assert.ok(Math.abs(sample.x - boosted[1] / boosted[0]) < 1e-12);
  assert.ok(Math.abs(sample.y - boosted[2] / boosted[0]) < 1e-12);
  assert.ok(Math.abs(sample.z - boosted[3] / boosted[0]) < 1e-12);
  assert.ok(Math.abs(Math.hypot(sample.x, sample.y, sample.z) - 1) < 1e-12);
});
test("aberration star field follows the closed-form angular transformation",()=>{
  for(const beta of [0,.2,.6,.95]){
    const rows=buildSceneData("aberration",{beta}),gamma=1/Math.sqrt(1-beta*beta);
    assert.equal(rows.length,140);
    for(let i=0;i<rows.length;i++){
      const y=1-2*(i+.5)/140,phi=Math.PI*(1+Math.sqrt(5))*i,r=Math.sqrt(1-y*y),x=r*Math.cos(phi),z=r*Math.sin(phi),denominator=1-beta*x;
      const expected=[(x-beta)/denominator,y/(gamma*denominator),z/(gamma*denominator)];
      for(let axis=0;axis<3;axis++)assert.ok(Math.abs([rows[i].x,rows[i].y,rows[i].z][axis]-expected[axis])<1e-12,`direction mismatch at beta ${beta}, star ${i}`);
    }
  }
});

test("velocity addition in the aberration scene composes four-velocities through Lorentz transforms", () => {
  for (const [first, relative] of [[0, 0], [.2, .4], [.6, .6], [-.35, .5]]) {
    const expected = (first + relative) / (1 + first * relative), actual = composeCollinearVelocities(first, relative);
    assert.ok(Math.abs(actual - expected) < 1e-12); assert.ok(Math.abs(actual) < 1);
  }
  assert.throws(() => composeCollinearVelocities(1, 0), /strictly subluminal/);
  assert.throws(() => composeCollinearVelocities(0, NaN), /finite/);
  assert.equal(scenes.find((scene) => scene.id === "aberration").title, "Velocity addition and aberration");
});

test("charged-particle path integrates the library Lorentz force and preserves four-velocity norm", () => {
  const rows = buildSceneData("particle", { beta: .4, time: 1 });
  const start = buildSceneData("particle", { beta: .4, time: 0 });
  assert.equal(start.length, 1);
  assert.equal(start[0].ct, 0);
  assert.ok(Math.abs(particleMomentumState(rows.at(-1)).invariantMass - 1) < 1e-9);
  assert.deepEqual(particleMomentumState(start[0]).components, start[0].fourVelocity);
  const expected = rows[1].fourVelocity[1];
  rows[1].fourVelocity[1] = 999;
  assert.equal(buildSceneData("particle", { beta: .4, time: .5 })[1].fourVelocity[1], expected);
  rows[1].fourVelocity[1] = expected;
  assert.notEqual(buildSceneData("particle", { beta: .2, time: 0 })[0].fourVelocity[1], start[0].fourVelocity[1]);
  assert.equal(rows.length, 241);
  assert.equal(rows[0].ct, 0);
  assert.ok(Math.hypot(rows.at(-1).x, rows.at(-1).y, rows.at(-1).z) < 1e-5);
  for (let i = 0; i < rows.length; i++) {
    assert.ok(Math.abs(new Vector4(rows[i].fourVelocity).normSquared() + 1) < 1e-9);
    if (i) assert.ok(rows[i].ct > rows[i - 1].ct);
  }
});

test("uniform magnetic-field particle paths match the analytic circular orbit",()=>{
  for(const beta of [.1,.6,.95]){
    const gamma=1/Math.sqrt(1-beta*beta),radius=gamma*beta;
    const rows=buildSceneData("particle",{beta,time:1});
    for(const row of rows){
      const tau=row.parameter,expected=[radius*Math.sin(tau),radius*(Math.cos(tau)-1),0],velocity=[gamma,radius*Math.cos(tau),-radius*Math.sin(tau),0];
      for(let i=0;i<3;i++)assert.ok(Math.abs([row.x,row.y,row.z][i]-expected[i])<1e-7,`position mismatch at beta ${beta}, parameter ${tau}`);
      for(let i=0;i<4;i++)assert.ok(Math.abs(row.fourVelocity[i]-velocity[i])<1e-7,`velocity mismatch at beta ${beta}, parameter ${tau}`);
      assert.ok(Math.abs(row.ct-gamma*tau)<1e-10);
    }
  }
});

test("weak-field lensing scene traces a bounded 3D bundle of bent null geodesics", () => {
  const start = buildSceneData("geodesic", { beta: .8, time: 0 });
  assert.equal(start.length, 8);
  assert.ok(start.every(row => row.parameter === 0 && row.x === -3));
  const rows = buildSceneData("geodesic", { beta: .8, time: 1 });
  const labels = new Set(rows.map((row) => row.label));
  assert.equal(rows.length, 8 * 39);
  assert.equal(labels.size, 8);
  assert.ok(rows.length <= sampleLimit);
  const radii = new Set();
  for (const label of labels) {
    const ray = rows.filter((row) => row.label === label);
    assert.equal(ray.length, 39);
    radii.add(ray[0].impactParameter);
    assert.ok(ray.every((row) => row.tangent.every(Number.isFinite)));
    assert.ok(ray.every((row) => Math.abs(row.metricNorm) < 3e-6));
    assert.ok(ray.at(-1).x > ray[0].x + 5);
    const [dy, dz] = ray[0].impactDirection;
    const inwardDeflection = ray.at(-1).tangent[2] * dy + ray.at(-1).tangent[3] * dz;
    assert.ok(inwardDeflection < -.05, `${label} did not bend toward the lens`);
  }
  assert.deepEqual([...radii].sort((a, b) => a - b), [.65, 1.3]);
  assert.ok(rows.some((row) => Math.abs(row.z) > .5), "ray bundle should extend in the third spatial dimension");
});
test("lensing cache reuses bounded paths without sharing mutable sample rows", () => {
  const first = buildSceneData("geodesic", { beta: .8, time: 1 });
  const repeat = buildSceneData("geodesic", { beta: .8, time: 1 });
  assert.deepEqual(first, repeat);
  assert.notEqual(first[0], repeat[0]);
  first[0].tangent[0] = Number.NaN;
  first[0].impactDirection[0] = Number.NaN;
  const isolated = buildSceneData("geodesic", { beta: .8, time: 1 });
  assert.deepEqual(isolated, repeat);
  const changedField = buildSceneData("geodesic", { beta: .7, time: 1 });
  assert.notDeepEqual(changedField, repeat);
  assert.deepEqual(buildSceneData("geodesic", { beta: .8, time: 1 }), repeat);
});

test("boosted electromagnetic fields preserve both library invariants", () => {
  const rest = transformedFieldState(0);
  const boosted = transformedFieldState(.65);
  assert.ok(Math.abs(rest.invariants.magneticMinusElectricOverC2 - boosted.invariants.magneticMinusElectricOverC2) < 1e-9);
  assert.ok(Math.abs(rest.invariants.electricDotMagnetic - boosted.invariants.electricDotMagnetic) < 1e-9);
  assert.ok(boosted.magnetic.some((component) => Math.abs(component) > 0));
});

test("twin paths reunite and proper times use the library interval", () => {
  const beta = .6;
  const clocks = twinClockReadings(beta);
  assert.equal(clocks.stayHome, 4);
  assert.ok(Math.abs(clocks.traveler - 3.2) < 1e-12);
  const halfway = twinClockReadings(beta, .5);
  assert.equal(halfway.stayHome, 2);
  assert.ok(Math.abs(halfway.traveler - 1.6) < 1e-12);
  assert.equal(twinClockReadings(beta, 0).traveler, 0);
  assert.throws(() => twinClockReadings(beta, 1.1), /time must be/);
  assert.deepEqual(twinClockPositions(beta, 0).map(p => p.y), [-2, -2]);
  assert.ok(Math.abs(twinClockPositions(beta, .5)[1].x) < 1e-12);
  assert.deepEqual(twinClockPositions(beta, 1).map(p => p.x), [-1.2, -1.2]);
  for(const t of [0,1]){const [home,traveler]=twinClockPositions(beta,t);assert.deepEqual(home,traveler);}
  const midpoint=twinClockPositions(beta,.5);assert.notEqual(midpoint[0].z,midpoint[1].z);
  const samples = buildSceneData("twins", { beta });
  const home = samples.filter((row) => row.id.startsWith("stay-home-"));
  const traveler = samples.filter((row) => row.id.startsWith("traveler-"));
  for(const index of [0,home.length-1]){for(const axis of ["x","y","z"])assert.equal(home[index][axis],traveler[index][axis]);}
  assert.notEqual(home[70].z,traveler[70].z);
  assert.equal(home.length, 140);
  assert.equal(traveler.length, 140);
  assert.equal(home[0].x, home.at(-1).x);
  assert.equal(traveler[0].x, traveler.at(-1).x);
  assert.equal(samples.filter((row) => row.y === -2).length, 2);
  assert.equal(samples.filter((row) => row.y === 2).length, 2);
});

test("input validation rejects superluminal speeds and unknown scenes", () => {
  assert.throws(() => buildSceneData("minkowski", { beta: 1 }), /beta must be in/);
  assert.throws(() => buildSceneData("minkowski", { time: Number.NaN }), /time must be finite/);
  assert.throws(() => buildSceneData("not-a-scene"), /Unknown scene/);
});
