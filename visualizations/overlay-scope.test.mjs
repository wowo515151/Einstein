import test from "node:test";
import assert from "node:assert/strict";
import { createOverlayScope, replaceOverlay } from "./overlay-scope.js";

test("overlay scope releases all attached resources once, including after scene insertion fails",()=>{
  const attached=new Set(),tracked=new Set(),released=[];
  const scene={add(object){attached.add(object);if(object.fail)throw new Error("insertion failed");},remove(object){attached.delete(object);}};
  const resources={track(resource){tracked.add(resource);return resource;},release(resource){assert.ok(tracked.delete(resource));released.push(resource);}};
  const scope=createOverlayScope(scene,resources),first={},second={fail:true},g1={},m1={},g2={},m2={};
  scope.add(first,g1,m1);
  assert.throws(()=>scope.add(second,g2,m2),/insertion failed/);
  scope.dispose();scope.dispose();
  assert.equal(attached.size,0);assert.equal(tracked.size,0);
  assert.deepEqual(released,[g1,m1,g2,m2]);
});

test("failed replacement keeps the previous overlay and cleans up the new one",()=>{
  const events=[],previous=()=>events.push("old disposed"),create=()=>{events.push("new created");return ()=>events.push("new disposed");};
  assert.throws(()=>replaceOverlay(previous,create,()=>{throw new Error("point update failed");}),/point update failed/);
  assert.deepEqual(events,["new created","new disposed"]);
  events.length=0;
  const next=replaceOverlay(previous,create,()=>events.push("points updated"));
  assert.deepEqual(events,["new created","points updated","old disposed"]);
  next();assert.equal(events.at(-1),"new disposed");
});

test("cleanup continues after a resource error and successful replacement keeps its new disposer",()=>{
  const calls=[],scene={add(){},remove(object){calls.push(`remove ${object.id}`);}},resources={track(value){return value;},release(value){calls.push(`release ${value.id}`);if(value.fail)throw new Error("dispose failed");}};
  const scope=createOverlayScope(scene,resources);
  scope.add({id:"a"},{id:"g1",fail:true},{id:"m1"});scope.add({id:"b"},{id:"g2"},{id:"m2"});
  let cleanupError;
  const next=replaceOverlay(scope.dispose,()=>()=>calls.push("new disposed"),()=>{},error=>{cleanupError=error;});
  assert.ok(cleanupError instanceof AggregateError);
  assert.deepEqual(calls,["remove a","release g1","release m1","remove b","release g2","release m2"]);
  next();assert.equal(calls.at(-1),"new disposed");
});

test("replacement preserves both update and cleanup errors",()=>{
  const updateError=new Error("update failed"),cleanupError=new Error("cleanup failed");
  assert.throws(()=>replaceOverlay(()=>{},()=>()=>{throw cleanupError;},()=>{throw updateError;}),error=>error instanceof AggregateError&&error.errors[0]===updateError&&error.errors[1]===cleanupError);
});
