import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

test("documented JavaScript examples execute against the local library",async()=>{
  const source=await readFile(new URL("../docs/examples.md",import.meta.url),"utf8");
  const examples=[...source.matchAll(/^## ([^\n]+)[\s\S]*?```js\s*\n([\s\S]*?)\n```/gm)].map(match=>({title:match[1],code:match[2]}));
  assert.ok(examples.length>=6);
  const checks={
    "Euclidean and Minkowski products":"[dot(components,components),minkowskiDot(components,components)]",
    "Geodesic initial-value problem":"samples.at(-1).event.position",
    "Maxwell evolution and residuals":"{electricTimes,magneticTimes,ampere:residuals.ampere[0][0][0][0]}",
    "Plane-wave spatial grid":"{events:Array.from(grid.events),amplitudes:grid.fields.map(field=>field.electric[0])}",
    "Scalar RK4 integration":"solution.at(-1)",
    "Rectangular field derivatives":"derivative",
    "Forward geodesic boundary shooting":"result[3]",
    "Collocating Maxwell histories":"{time:times[aligned.frameIndices[0]],invariants:field.invariants()}",
    "Rectangular ODE parameter sensitivity":"result",
  };
  for(const {title,code} of examples){
    const expression=checks[title],probe=expression?`\nconsole.log('__EXAMPLE_RESULT__'+JSON.stringify(${expression}));`:"";
    const result=spawnSync(process.execPath,["--input-type=module","--eval",code+probe],{cwd:fileURLToPath(new URL("../docs/",import.meta.url)),encoding:"utf8",timeout:10000});
    assert.equal(result.status,0,`${title}: ${result.stderr || result.error?.message}`);
    if(!expression)continue;
    const line=result.stdout.split(/\r?\n/).find(value=>value.startsWith("__EXAMPLE_RESULT__"));
    assert.ok(line,`${title} did not return its checked result`);
    const actual=JSON.parse(line.slice("__EXAMPLE_RESULT__".length));
    if(title==="Euclidean and Minkowski products")assert.deepEqual(actual,[5,-3]);
    else if(title==="Geodesic initial-value problem")actual.forEach((value,i)=>assert.ok(Math.abs(value-[1,.5,0,0][i])<1e-12));
    else if(title==="Forward geodesic boundary shooting")actual.forEach((value,i)=>assert.ok(Math.abs(value-[1,.5,0,0][i])<1e-6));
    else if(title==="Collocating Maxwell histories"){assert.equal(actual.time,.1);assert.equal(actual.invariants.magneticMinusElectricOverC2,-1);assert.equal(actual.invariants.electricDotMagnetic,0);}
    else if(title==="Rectangular ODE parameter sensitivity"){assert.ok(Math.abs(actual.value[0][0]-2*Math.exp(.3))<1e-8);assert.equal(actual.value[0][1],1);assert.ok(Math.abs(actual.jacobian[0][0]-2*Math.exp(.3))<1e-8);assert.equal(actual.jacobian[1][0],0);}
    else if(title==="Scalar RK4 integration"){assert.equal(actual[0],1);assert.ok(Math.abs(actual[1]-Math.E)<1e-6);}
    else if(title==="Rectangular field derivatives")actual.forEach((row,i)=>row.forEach((value,j)=>assert.ok(Math.abs(value-[[1,1,0,0],[0,0,4,3]][i][j])<1e-9)));
    else if(title==="Plane-wave spatial grid"){
      assert.deepEqual(actual.events,[0,0,0,0,0,0,0,Math.PI/2]);
      assert.ok(Math.abs(actual.amplitudes[0]-1)<1e-15&&Math.abs(actual.amplitudes[1])<1e-15);
    }
    else {
      assert.deepEqual(actual.ampere,[0,0,0]);
      actual.electricTimes.forEach((value,i)=>assert.ok(Math.abs(value-[0,.1,.2][i])<1e-12));
      actual.magneticTimes.forEach((value,i)=>assert.ok(Math.abs(value-[.05,.15][i])<1e-12));
      assert.equal(actual.electricTimes.length,3);assert.equal(actual.magneticTimes.length,2);
    }
  }
  for(const title of Object.keys(checks))assert.ok(examples.some(example=>example.title===title),`missing checked example: ${title}`);
});
