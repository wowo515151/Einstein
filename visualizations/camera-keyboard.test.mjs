import test from "node:test";
import assert from "node:assert/strict";
import { moveCameraWithKey } from "./camera-keyboard.js";

test("keyboard camera controls orbit and zoom around the current target",()=>{
  let pose={mode:"perspective",position:[1,2,8],target:[1,2,3],up:[0,1,0]};
  const viewer={getCameraPose:()=>pose,setCameraPose:value=>{pose=value;}};
  assert.equal(moveCameraWithKey(viewer,"ArrowRight"),true);
  assert.ok(pose.position[0]>1);assert.ok(Math.abs(Math.hypot(...pose.position.map((v,i)=>v-pose.target[i]))-5)<1e-12);
  assert.equal(moveCameraWithKey(viewer,"+"),true);
  assert.ok(Math.hypot(...pose.position.map((v,i)=>v-pose.target[i]))<5);
  const saved=pose;assert.equal(moveCameraWithKey(viewer,"a"),false);assert.equal(pose,saved);
  assert.equal(moveCameraWithKey(undefined,"ArrowUp"),false);
});

test("keyboard camera motion keeps polar angle and zoom within bounds",()=>{
  let pose={mode:"perspective",position:[0,0,3],target:[0,0,0],up:[0,1,0]};
  const viewer={getCameraPose:()=>pose,setCameraPose:value=>{pose=value;}};
  for(let i=0;i<100;i++)moveCameraWithKey(viewer,"ArrowUp");
  assert.ok(pose.position[1]>0);assert.ok(Math.hypot(pose.position[0],pose.position[2])>.1);
  for(let i=0;i<100;i++)moveCameraWithKey(viewer,"+");
  assert.ok(Math.abs(Math.hypot(...pose.position)-2.5)<1e-12);
  for(let i=0;i<100;i++)moveCameraWithKey(viewer,"-");
  assert.ok(Math.abs(Math.hypot(...pose.position)-15)<1e-12);
});
