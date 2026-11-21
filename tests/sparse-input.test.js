import test from "node:test";
import assert from "node:assert/strict";
import {matrix} from "../src/array.js";
import {DifferentialOperators4,FourVelocity,LightCone,SpatialSlice,Vector4,minkowskiDot} from "../src/relativity/index.js";

test("numeric parsers reject sparse rows, coordinates, batches and directions",()=>{
  const rows=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  delete rows[2];
  assert.throws(()=>matrix(rows),TypeError);
  const point=[0,0,0,0];delete point[1];
  assert.throws(()=>new DifferentialOperators4().partial(x=>x[0],point),/finite/);
  const points=[[0,0,0,0],[1,0,0,0]];delete points[1];
  assert.throws(()=>new DifferentialOperators4().partial(x=>x[0],points),/finite/);
  assert.throws(()=>new FourVelocity(point,{c:1}),/finite/);
  const direction=[1,0,0];delete direction[1];
  assert.throws(()=>new LightCone([0,0,0,0],{c:1,branch:"future"}).sample(direction,1),/finite/);
});
test("numeric parsers reject oversized dimensions before accessing elements",()=>{
  const oversized=length=>{const values=new Array(length);Object.defineProperty(values,0,{get(){throw new Error("oversized input was traversed");}});return values;};
  const large=oversized(1_000_001),geometry=oversized(65_537);
  assert.throws(()=>new Vector4(large),/component limit/);
  assert.throws(()=>minkowskiDot(large,[1,0,0,0]),/component limit/);
  assert.throws(()=>new DifferentialOperators4().partial(x=>x[0],geometry),/component limit/);
  assert.throws(()=>new SpatialSlice(large,[1,0,0,0]),/component limit/);
  assert.throws(()=>new FourVelocity(large,{c:1}),/component limit/);
  assert.throws(()=>new Vector4(new Uint8Array(1_000_001)),/component limit/);
});
