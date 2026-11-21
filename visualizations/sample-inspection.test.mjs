import test from "node:test";
import assert from "node:assert/strict";
import { describeSample } from "./sample-inspection.js";
import { buildSceneData } from "./scene-models.js";

test("sample inspector describes the requested scene sample and rejects invalid indexes",()=>{
  const rows=[{id:"one",label:"first clock",x:1,y:2,z:3},{id:"two",x:-.1,y:0,z:4}];
  assert.equal(describeSample(rows,1),"Selected sample 1 of 2: first clock. Display coordinates: (1.000, 2.000, 3.000).");
  assert.match(describeSample(rows,2),/2 of 2: two.*-0\.100/);
  for(const value of [0,3,1.5,NaN])assert.throws(()=>describeSample(rows,value),/Choose a sample from 1 to 2/);
});


test("sample inspector distinguishes physical coordinates from display offsets",()=>{
  const row={id:"traveler",x:-.5,y:-1,z:.12,ct:1,worldlineX:.6,properTime:.8};
  assert.equal(describeSample([row],1),"Selected sample 1 of 1: traveler. Display coordinates: (-0.500, -1.000, 0.120). Coordinate time ct: 1.000. Worldline x: 0.600. Clock proper time: 0.800.");
  assert.match(describeSample([{id:"ray",x:0,y:0,z:0,ct:2,metricNorm:-1e-7}],1),/Coordinate time ct: 2.000. Metric tangent norm: -1.000e-7/);
});


test("sample inspector reports library causal types and squared intervals",()=>{
  const rows=buildSceneData("minkowski",{beta:.6,time:0}),label=describeSample(rows,1);
  assert.match(label,/Coordinate time ct: -2\.000/);assert.match(label,/Squared interval from origin: -4\.000e\+0/);assert.match(label,/Vector classification: timelike/);
  const surface=describeSample(buildSceneData("light-cone",{time:.5}),1);assert.match(surface,/Vector classification: null/);
  assert.doesNotMatch(describeSample([{id:"bad",x:0,y:0,z:0,causalType:99}],1),/Vector classification/);
});


test("field sample readout separates physical slice position from display height",()=>{
  const row=buildSceneData("field-slices",{beta:.4,time:.3})[0],message=describeSample([row],1);
  assert.match(message,/Coordinate time ct:/);assert.match(message,/Physical slice position: \(-2\.000, 0\.000, -2\.000\)/);
  assert.match(message,/Electric field: \(.*\)/);assert.match(message,/Magnetic field: \(.*\)/);
});
