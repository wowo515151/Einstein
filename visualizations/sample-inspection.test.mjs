import test from "node:test";
import assert from "node:assert/strict";
import { describeSample } from "./sample-inspection.js";

test("sample inspector describes the requested scene sample and rejects invalid indexes",()=>{
  const rows=[{id:"one",label:"first clock",x:1,y:2,z:3},{id:"two",x:-.1,y:0,z:4}];
  assert.equal(describeSample(rows,1),"Selected sample 1 of 2: first clock. Display coordinates: (1.000, 2.000, 3.000).");
  assert.match(describeSample(rows,2),/2 of 2: two.*-0\.100/);
  for(const value of [0,3,1.5,NaN])assert.throws(()=>describeSample(rows,value),/Choose a sample from 1 to 2/);
});
