import test from "node:test";
import assert from "node:assert/strict";
import {Dual4,forwardJacobian4,DifferentialOperators4,Connection4,Geodesic4,Metric4} from "../src/relativity/index.js";

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


test("forward-mode Christoffel symbols match an analytic time-varying metric",()=>{
  const base=2**53,dualMetric=p=>[[-1,0,0,0],[0,p[0].subtract(base).multiply(.2).exp(),0,0],[0,0,1,0],[0,0,0,1]];
  const numericMetric=p=>[[-1,0,0,0],[0,Math.exp(.2*(p[0]-base)),0,0],[0,0,1,0],[0,0,0,1]];
  const points=[[base,0,0,0],[base+4,0,0,0]],result=new Connection4(dualMetric).christoffelForward(points);
  for(let i=0;i<points.length;i++){const a=Math.exp(.2*(points[i][0]-base)),g=result[i];assert.ok(Math.abs(g[0][1][1]-.1*a)<1e-12);assert.ok(Math.abs(g[1][0][1]-.1)<1e-12);assert.ok(Math.abs(g[1][1][0]-.1)<1e-12);}
  assert.throws(()=>new Connection4(numericMetric).christoffel(points[0]),/distinct finite stencil/);
  assert.throws(()=>new Connection4(p=>[[-1,0,0,0],[0,Math.exp(p[0]),0,0],[0,0,1,0],[0,0,0,1]]).christoffelForward([0,0,0,0]),/implicitly converted/);
  assert.throws(()=>new Connection4(()=>[[-1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]]).christoffelForward([0,0,0,0]),/singular/);
});


test("forward-mode connection handles a varying off-diagonal metric",()=>{
  const metric=p=>{const a=p[0].multiply(.1).add(2),b=p[0].multiply(.2),c=p[0].multiply(.05).add(3);return [[-1,0,0,0],[0,a,b,0],[0,b,c,0],[0,0,0,1]];};
  const connection=new Connection4(metric);for(const t of [0,1,2]){
    const a=2+.1*t,b=.2*t,c=3+.05*t,det=a*c-b*b,g=connection.christoffelForward([t,0,0,0]);
    const cases=[[g[0][1][1],.05],[g[0][1][2],.1],[g[0][2][2],.025],[g[1][0][1],.5*(c*.1-b*.2)/det],[g[1][0][2],.5*(c*.2-b*.05)/det],[g[2][0][1],.5*(-b*.1+a*.2)/det],[g[2][0][2],.5*(-b*.2+a*.05)/det]];
    for(const [actual,expected] of cases)assert.ok(Math.abs(actual-expected)<1e-12);
  }
});


test("forward-mode connections support acceleration and parallel transport",()=>{
  const metric=p=>[[-1,0,0,0],[0,p[0].multiply(.2).exp(),0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(metric,{metricDerivative:"forward"});
  const g=connection.christoffel([0,0,0,0]);assert.ok(Math.abs(g[0][1][1]-.1)<1e-12);
  const acceleration=connection.geodesicAcceleration([0,0,0,0],[1,.5,0,0]);assert.ok(Math.abs(acceleration[0]+.025)<1e-12);assert.ok(Math.abs(acceleration[1]+.1)<1e-12);
  const path=[[0,0,0,0],[1,0,0,0]],transported=connection.parallelTransport([0,1,0,0],path,{stepsPerSegment:16});
  assert.ok(Math.abs(transported.at(-1)[1]-Math.exp(-.1))<1e-9);
  const spatialMetric=p=>[[-1,0,0,0],[0,p[1].multiply(2).exp(),0,0],[0,0,1,0],[0,0,0,1]],geo=new Geodesic4(new Connection4(spatialMetric,{metricDerivative:"forward"}),[0,0,0,0],[1,.4,0,0]);
  const endpoint=geo.integrate([0,1],{steps:16}).at(-1);assert.ok(Math.abs(endpoint.event.position[1]-Math.log(1.4))<1e-7);assert.ok(Math.abs(endpoint.tangent[1]-.4/1.4)<1e-7);
  assert.throws(()=>new Connection4(metric,{metricDerivative:"invalid"}),/metricDerivative/);
  connection.metricDerivative="invalid";assert.throws(()=>connection.christoffel([0,0,0,0]),/metricDerivative/);
});


test("forward covariant derivatives handle upper and lower indices and batches",()=>{
  const metric=p=>[[-1,0,0,0],[0,p[0].multiply(.2).exp(),0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(metric,{metricDerivative:"forward"});
  const upper=p=>[0,p[0].multiply(-.1).exp(),0,0],lower=p=>[0,p[0].multiply(.1).exp(),0,0],points=[[0,0,0,0],[1,0,0,0]];
  const upperResult=connection.covariantDerivativeForward(upper,points,["up"]),lowerResult=connection.covariantDerivativeForward(lower,points,["down"]);
  for(let i=0;i<points.length;i++){const t=points[i][0];assert.ok(Math.abs(upperResult[i][1][0])<1e-12);assert.ok(Math.abs(upperResult[i][0][1]-.1*Math.exp(.1*t))<1e-12);assert.ok(Math.abs(lowerResult[i][1][0])<1e-12);assert.ok(Math.abs(lowerResult[i][0][1]+.1*Math.exp(.1*t))<1e-12);}
  assert.deepEqual(connection.covariantDerivativeVectorForward(upper,points),upperResult);
  const compatibility=connection.covariantDerivativeForward(metric,points,["down","down"]);assert.ok(compatibility.flat(Infinity).every(value=>Math.abs(value)<1e-12));
  assert.throws(()=>connection.covariantDerivativeForward(p=>p[0].value===0?[0,0,0,0]:[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],points,["up"]),/shape/);
  assert.deepEqual(connection.covariantDerivativeForward(p=>p[0].pow(2),[2,0,0,0],[]),[4,0,0,0]);
  assert.throws(()=>connection.covariantDerivativeForward(upper,points,["up","down"]),/indexTypes length/);
  assert.throws(()=>connection.covariantDerivativeForward(upper,points,["invalid"]),/indexTypes/);
});


test("forward exterior derivatives preserve form signs and batches",()=>{
  const op=new DifferentialOperators4(),form=p=>[0,p[0].multiply(p[2]),p[1].sin(),p[3].pow(2)],points=[[2,.3,4,1],[3,.5,5,2]],result=op.exteriorDerivativeForward(form,points);
  for(let i=0;i<points.length;i++){const [t,x,y]=points[i],d=result[i];assert.ok(Math.abs(d[0][1]-y)<1e-12);assert.ok(Math.abs(d[2][1]-(t-Math.cos(x)))<1e-12);for(let a=0;a<4;a++)for(let b=0;b<4;b++)assert.ok(Math.abs(d[a][b]+d[b][a])<1e-12);}
  assert.deepEqual(op.exteriorDerivativeForward(p=>p[0].pow(2),[2,0,0,0]),[4,0,0,0]);
  const twoForm=p=>{const q=p[2].multiply(p[3]);return [[0,q,0,0],[q.negate(),0,0,0],[0,0,0,0],[0,0,0,0]];},d2=op.exteriorDerivativeForward(twoForm,[0,0,2,3]);
  assert.equal(d2[2][0][1],3);assert.equal(d2[0][2][1],-3);assert.equal(d2[3][0][1],2);
  assert.throws(()=>op.exteriorDerivativeForward(()=>[1,2,3],[0,0,0,0]),/length 4/);
});


test("forward divergence combines field and metric derivatives",()=>{
  const base=2**53,metric=p=>[[-1,0,0,0],[0,p[0].subtract(base).multiply(.2).exp(),0,0],[0,0,1,0],[0,0,0,1]],field=p=>[p[0].subtract(base),0,0,0],points=[[base,2,0,0],[base+4,2,0,0]];
  const op=new DifferentialOperators4(metric,{metricDerivative:"forward"}),result=op.divergenceForward(field,points);
  assert.ok(Math.abs(result[0]-1)<1e-12);assert.ok(Math.abs(result[1]-1.4)<1e-12);
  const raised=op.gradientForward(p=>p[1].pow(2),points[1]);assert.ok(Math.abs(raised[1]-4/Math.exp(.8))<1e-12);
  assert.equal(new DifferentialOperators4().divergenceForward(p=>[p[0].multiply(2),0,0,0],[1,0,0,0]),2);
  assert.equal(new DifferentialOperators4(Metric4.minkowski(),{metricDerivative:"forward"}).divergenceForward(p=>[p[0].multiply(2),0,0,0],[1,0,0,0]),2);
  assert.throws(()=>new DifferentialOperators4(metric,{metricDerivative:"bad"}),/metricDerivative/);
  assert.throws(()=>op.divergenceForward(p=>[p[0],0,0],points[0]),/4 components/);
  op.connection.metricDerivative="bad";assert.throws(()=>op.gradientForward(p=>p[0],points[0]),/metricDerivative/);
});
