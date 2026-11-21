import test from "node:test";
import assert from "node:assert/strict";
import {Dual4,forwardJacobian4,DifferentialOperators4} from "../src/relativity/index.js";

const field=x=>x[0].sin().multiply(x[1].exp()).add(x[2].pow(3).divide(x[3].pow(2).add(1)));
function expected([t,x,y,z]){const denominator=1+z*z;return [Math.cos(t)*Math.exp(x),Math.sin(t)*Math.exp(x),3*y*y/denominator,-2*z*y**3/denominator**2];}

test("forward-mode scalar derivatives match an analytic gradient",()=>{
  const point=[.3,-.2,1.2,.4],result=forwardJacobian4(field,point),want=expected(point);
  const value=Math.sin(point[0])*Math.exp(point[1])+point[2]**3/(1+point[3]**2);
  assert.ok(Math.abs(result.value-value)<1e-14);result.jacobian.forEach((v,i)=>assert.ok(Math.abs(v-want[i])<1e-12));
  const operators=new DifferentialOperators4(),gradient=operators.gradientForward(field,point);
  gradient.forEach((v,i)=>assert.ok(Math.abs(v-(i===0?-want[i]:want[i]))<1e-12));
  assert.deepEqual(operators.partialForward(field,point),operators.jacobianForward(field,point));
  const metric=[[-4,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]],raised=new DifferentialOperators4(metric).gradientForward(field,point);
  raised.forEach((v,i)=>assert.ok(Math.abs(v-want[i]/[-4,2,3,4][i])<1e-12));
});

test("forward derivatives handle explicit elementary operations and large coordinates",()=>{
  const p=[.2,.3,.4,.5],f=x=>x[0].cos().subtract(x[1].sinh()).add(x[2].cosh()).add(x[3].exp().log()).negate(),actual=forwardJacobian4(f,p).jacobian;
  const want=[Math.sin(p[0]),Math.cosh(p[1]),-Math.sinh(p[2]),-1];actual.forEach((v,i)=>assert.ok(Math.abs(v-want[i])<1e-12));
  const base=2**53,point=[base,0,0,0],linear=x=>x[0].subtract(base).multiply(2),op=new DifferentialOperators4();
  assert.throws(()=>op.partial(p=>2*(p[0]-base),point),/distinct finite stencil/);
  assert.deepEqual(op.partialForward(linear,point),[2,0,0,0]);
  const root=forwardJacobian4(x=>x[1].sqrt(),[0,4,0,0]);assert.equal(root.value,2);assert.equal(root.jacobian[1],.25);
});

test("forward-mode Jacobians preserve tensor and coordinate batch shapes",()=>{
  const f=x=>[[x[0].multiply(x[1]),3],[x[2].sin(),x[3].pow(2)]],p=[2,3,0,4],out=forwardJacobian4(f,p);
  assert.deepEqual(out.value,[[6,3],[0,16]]);assert.deepEqual(out.jacobian,[[[3,2,0,0],[0,0,0,0]],[[0,0,1,0],[0,0,0,8]]]);
  const op=new DifferentialOperators4(),batched=op.jacobianForward(f,[p,[1,5,0,2]]);
  assert.deepEqual(batched[0],out.jacobian);assert.deepEqual(batched[1][0][0],[5,1,0,0]);
  assert.throws(()=>op.gradientForward(f,p),/scalar/);
});

test("dual operations reject implicit conversion, invalid domains and malformed outputs",()=>{
  const x=Dual4.variable(-2,0);assert.equal(x.pow(3).value,-8);assert.equal(x.pow(3).derivatives[0],12);
  assert.equal(Dual4.variable(0,0).pow(2).derivatives[0],0);
  assert.throws(()=>x.pow(.5),/real/);assert.throws(()=>Dual4.variable(0,0).sqrt(),/positive/);assert.throws(()=>x.log(),/positive/);
  assert.throws(()=>x.divide(0),/zero/);assert.throws(()=>Math.sin(x),/implicitly converted/);
  assert.throws(()=>forwardJacobian4(a=>a[0]+1,[1,2,3,4]),/implicitly converted/);
  assert.throws(()=>forwardJacobian4(()=>[[1],[1,2]],[1,2,3,4]),/rectangular/);
  const cycle=[];cycle.push(cycle);assert.throws(()=>forwardJacobian4(()=>cycle,[1,2,3,4]),/cycles/);
  assert.throws(()=>forwardJacobian4(()=>new Array(16385).fill(1),[1,2,3,4]),/component limit/);
});
