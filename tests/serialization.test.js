import test from "node:test";
import assert from "node:assert/strict";
import {Vector4,Covector4,Matrix4,Metric4,Tensor4,Bivector4,LeviCivita4,LorentzTransform,PoincareTransform,Event4,SpacetimeInterval,LightCone,InertialFrame,MaxwellEquations,FourVelocity,FourMomentum,FourCurrent,EMFieldTensor,serializeRelativity,deserializeRelativity} from "../src/relativity/index.js";

test("versioned value serialization preserves types, batches, variance and constants",()=>{
  const samples=[new Vector4([[1,2,3,4],[4,3,2,1]]),new Covector4([1,2,3,4]),Matrix4.identity(),Metric4.minkowski(),new Tensor4([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]],["up","down"]),new Bivector4([[0,1,0,0],[-1,0,0,0],[0,0,0,0],[0,0,0,0]]),LorentzTransform.boost([0.2,0,0]),new FourVelocity([2,0,0,0],{c:2}),new FourMomentum([3,1,0,0],{c:2}),new FourCurrent([4,1,2,3],{c:2}),EMFieldTensor.fromEB([1,2,3],[3,2,1],{c:2,epsilon0:3})];
  for(const value of samples){const json=serializeRelativity(value),restored=deserializeRelativity(json);assert.equal(restored.constructor,value.constructor);assert.equal(serializeRelativity(restored),json);}
  const tensor=deserializeRelativity(serializeRelativity(samples[4]));assert.deepEqual(tensor.indexTypes,["up","down"]);
  assert.equal(deserializeRelativity(serializeRelativity(samples[10])).epsilon0,3);
});

test("deserialization rejects malformed, unsupported and physically invalid data",()=>{
  assert.throws(()=>deserializeRelativity("{"),SyntaxError);
  assert.throws(()=>deserializeRelativity(JSON.stringify({format:"relativity-value",version:2,type:"Vector4",data:{components:[1,2,3,4]}})),/format/);
  assert.throws(()=>deserializeRelativity(JSON.stringify({format:"relativity-value",version:1,type:"Function",data:{}})),/type/);
  assert.throws(()=>deserializeRelativity(JSON.stringify({format:"relativity-value",version:1,type:"FourMomentum",data:{components:[1,2,3,4]}})),/fields/);
  assert.throws(()=>deserializeRelativity(JSON.stringify({format:"relativity-value",version:1,type:"Vector4",data:{components:[1,2,3,null]}})),/finite/);
  assert.throws(()=>serializeRelativity({components:[1,2,3,4]}),/unsupported/);
  const value=new Vector4([1,2,3,4]);value.components[0]=Infinity;assert.throws(()=>serializeRelativity(value),/finite/);
});


test("spacetime interchange preserves event metadata and cone configuration",()=>{
  const event=new Event4([[1,2,3,4],[5,6,7,8]],"laboratory",2);
  const values=[event,new SpacetimeInterval([4,1,0,0],{c:2,atol:1e-8}),new LightCone(event,{branch:"future",c:3,atol:1e-7})];
  for(const value of values){const json=serializeRelativity(value),copy=deserializeRelativity(json);assert.equal(copy.constructor,value.constructor);assert.equal(serializeRelativity(copy),json);}
  const cone=deserializeRelativity(serializeRelativity(values[2]));assert.equal(cone.apex.frame,"laboratory");assert.equal(cone.apex.c,2);assert.equal(cone.c,3);assert.equal(cone.branch,"future");
  assert.throws(()=>serializeRelativity(new Event4([0,0,0,0],{name:"frame"})),/frame/);
  const record=JSON.parse(serializeRelativity(values[2]));record.data.apex.frame={name:"frame"};assert.throws(()=>deserializeRelativity(JSON.stringify(record)),/frame/);
});


test("compound value interchange restores transforms, frame and field constants",()=>{
  const map=new PoincareTransform(LorentzTransform.boost([0.2,0,0]),[1,2,3,4]);
  const values=[new LeviCivita4(Metric4.minkowski(),-1),map,new InertialFrame({origin:new Event4([1,2,3,4],"lab",2),relativeVelocity:[0.5,0,0],c:2}),new MaxwellEquations({c:2,epsilon0:3})];
  for(const value of values){const json=serializeRelativity(value),restored=deserializeRelativity(json);assert.equal(restored.constructor,value.constructor);assert.equal(serializeRelativity(restored),json);}
  const restored=deserializeRelativity(serializeRelativity(map));assert.deepEqual(restored.applyEvent([4,1,2,3]),map.applyEvent([4,1,2,3]));
  assert.throws(()=>serializeRelativity(new InertialFrame({origin:new Event4([0,0,0,0],{})})),/frame/);
});
