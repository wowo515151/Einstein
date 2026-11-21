import test from "node:test";
import assert from "node:assert/strict";
import {Dual4,Jet4,forwardJacobian4,forwardHessian4,forwardSecondOrder4,DifferentialOperators4,Connection4,Geodesic4,Metric4,Curvature4,integrateRK4Forward4,integrateRK4Forward,ODEWorldline} from "../src/relativity/index.js";

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


test("second-order jets match analytic scalar Hessians",()=>{
  const field=x=>x[0].sin().multiply(x[1].exp()).add(x[2].pow(2).multiply(x[3])),point=[.3,-.2,1.2,.4],result=forwardHessian4(field,point);
  const [t,x,y,z]=point,s=Math.sin(t)*Math.exp(x),c=Math.cos(t)*Math.exp(x),gradient=[c,s,2*y*z,y*y],hessian=[[-s,c,0,0],[c,s,0,0],[0,0,2*z,2*y],[0,0,2*y,0]];
  for(let i=0;i<4;i++){assert.ok(Math.abs(result.gradient[i]-gradient[i])<1e-12);for(let j=0;j<4;j++)assert.ok(Math.abs(result.hessian[i][j]-hessian[i][j])<1e-12);}
  const op=new DifferentialOperators4(),batched=op.hessianForward(field,[point,point]);assert.deepEqual(batched[0],result.hessian);assert.deepEqual(batched[1],result.hessian);
  const base=2**53,large=forwardHessian4(p=>p[0].subtract(base).pow(2),[base,0,0,0]);assert.equal(large.gradient[0],0);assert.equal(large.hessian[0][0],2);
});

test("second-order division, logarithm and square root follow analytic derivatives",()=>{
  const quotient=forwardHessian4(x=>x[0].divide(x[1]),[3,2,0,0]);
  assert.equal(quotient.value,1.5);assert.deepEqual(quotient.gradient,[.5,-.75,0,0]);assert.equal(quotient.hessian[0][1],-.25);assert.equal(quotient.hessian[1][0],-.25);assert.equal(quotient.hessian[1][1],.75);
  for(const scale of [1e-200,1e200]){
    const identity=forwardHessian4(x=>x[0].divide(x[0]),[scale,0,0,0]);
    assert.equal(identity.value,1);assert.deepEqual(identity.gradient,[0,0,0,0]);assert.deepEqual(identity.hessian,Array.from({length:4},()=>[0,0,0,0]));
    const constant=forwardHessian4(()=>Jet4.constant(scale).divide(scale),[0,0,0,0]);assert.equal(constant.value,1);assert.deepEqual(constant.gradient,[0,0,0,0]);
  }
  const scaled=forwardHessian4(x=>x[0].divide(1e-200),[1e-200,0,0,0]);assert.equal(scaled.value,1);assert.equal(scaled.gradient[0],1e200);assert.equal(scaled.hessian[0][0],0);
  const elementary=forwardHessian4(x=>x[0].log().add(x[1].sqrt()),[2,4,0,0]);assert.equal(elementary.hessian[0][0],-.25);assert.equal(elementary.hessian[1][1],-1/32);
  const large=forwardHessian4(x=>x[0].log(),[1e155,0,0,0]);assert.ok(large.hessian[0][0]<0);assert.ok(Math.abs(large.hessian[0][0]/-1e-310-1)<1e-10);
});

test("forward wave operator handles flat and curved metrics",()=>{
  const flat=new DifferentialOperators4(),quadratic=x=>x[0].pow(2).add(x[1].pow(2)).add(x[2].pow(2)).add(x[3].pow(2));
  assert.ok(Math.abs(flat.dAlembertianForward(quadratic,[1,2,3,4])-4)<1e-12);
  const metric=p=>[[-1,0,0,0],[0,p[0].multiply(.2).exp(),0,0],[0,0,1,0],[0,0,0,1]],curved=new DifferentialOperators4(metric,{metricDerivative:"forward"}),field=x=>x[0].pow(2).add(x[1].pow(2)),points=[[1,2,0,0],[2,2,0,0]];
  const values=curved.dAlembertianForward(field,points);for(let i=0;i<points.length;i++){const t=points[i][0],expected=-2+2/Math.exp(.2*t)-.2*t;assert.ok(Math.abs(values[i]-expected)<1e-12);}
  const base=2**53,largeMetric=p=>[[-1,0,0,0],[0,p[0].subtract(base).multiply(.2).exp(),0,0],[0,0,1,0],[0,0,0,1]],largeField=x=>x[0].subtract(base).pow(2).add(x[1].pow(2)),largeOp=new DifferentialOperators4(largeMetric,{metricDerivative:"forward"});
  assert.ok(Math.abs(largeOp.dAlembertianForward(largeField,[base+4,2,0,0])-(-2+2/Math.exp(.8)-.8))<1e-12);
});

test("second-order jet domains and scalar output are validated",()=>{
  const x=Jet4.variable(0,0),square=x.pow(2);assert.equal(square.hessian[0][0],2);assert.equal(x.pow(3).hessian[0][0],0);
  assert.throws(()=>x.sqrt(),/positive/);assert.throws(()=>x.pow(1.5),/second derivative/);assert.throws(()=>x.divide(0),/zero/);assert.throws(()=>Math.sin(x),/implicitly converted/);
  assert.throws(()=>forwardHessian4(p=>[p[0],p[1]],[1,2,3,4]),/scalar/);
  assert.deepEqual(forwardHessian4(()=>5,[0,0,0,0]).hessian,Array.from({length:4},()=>[0,0,0,0]));
});


test("second-order rectangular fields expose coordinate derivative axes",()=>{
  const field=p=>[[p[0].pow(2),p[0].multiply(p[1])],[p[2].sin(),3]],point=[2,3,.4,0],result=forwardSecondOrder4(field,point);
  assert.deepEqual(result.value,[[4,6],[Math.sin(.4),3]]);
  assert.deepEqual(result.jacobian[0][1],[3,2,0,0]);
  assert.equal(result.hessian[0][0][0][0],2);assert.equal(result.hessian[0][1][0][1],1);assert.equal(result.hessian[0][1][1][0],1);
  assert.equal(result.hessian[1][0][2][2],-Math.sin(.4));assert.deepEqual(result.hessian[1][1],Array.from({length:4},()=>[0,0,0,0]));
  const batch=new DifferentialOperators4().tensorHessianForward(field,[point,point]);assert.deepEqual(batch,[result.hessian,result.hessian]);
});

test("second-order rectangular fields enforce output and batch bounds",()=>{
  const p=[0,0,0,0],cycle=[];cycle.push(cycle);
  assert.throws(()=>forwardSecondOrder4(()=>cycle,p),/cycles/);
  assert.throws(()=>forwardSecondOrder4(()=>[[1],[2,3]],p),/rectangular/);
  assert.throws(()=>forwardSecondOrder4(()=>Array(3121).fill(0),p),/component limit/);
  assert.throws(()=>forwardSecondOrder4(()=>[NaN],p),/finite/);
  assert.throws(()=>new DifferentialOperators4().tensorHessianForward(q=>q[0].value===0?[q[0]]:[q[0],q[1]],[p,[1,0,0,0]]),/shape/);
});


test("forward metric curvature reproduces analytic de Sitter invariants at large coordinates",()=>{
  const H=.2,base=2**53,metric=p=>{const a=p[0].subtract(base).multiply(2*H).exp();return [[-1,0,0,0],[0,a,0,0],[0,0,a,0],[0,0,0,a]];};
  const curvature=new Curvature4(metric,{metricDerivative:"forward"});
  for(const t of [base,base+2]){
    const point=[t,0,0,0],a=Math.exp(2*H*(t-base)),g=[[-1,0,0,0],[0,a,0,0],[0,0,a,0],[0,0,0,a]],R=curvature.riemann(point);
    for(let i=0;i<4;i++)for(let j=0;j<4;j++)for(let k=0;k<4;k++)for(let l=0;l<4;l++)assert.ok(Math.abs(R[i][j][k][l]-H*H*((i===k?g[j][l]:0)-(i===l?g[j][k]:0)))<1e-12);
    const ricci=curvature.ricci(point),einstein=curvature.einsteinTensor(point);
    for(let i=0;i<4;i++)for(let j=0;j<4;j++){assert.ok(Math.abs(ricci[i][j]-3*H*H*g[i][j])<1e-12);assert.ok(Math.abs(einstein[i][j]+3*H*H*g[i][j])<1e-12);}
    assert.ok(Math.abs(curvature.ricciScalar(point)-12*H*H)<1e-12);assert.ok(Math.abs(curvature.kretschmannScalar(point)-24*H**4)<1e-12);
  }
  assert.deepEqual(curvature.riemann([[base,0,0,0],[base,0,0,0]])[0],curvature.riemann([base,0,0,0]));
  assert.throws(()=>new Curvature4(metric,{metricDerivative:"other"}),/metricDerivative/);
});


test("forward curvature agrees with finite differences for a non-diagonal metric",()=>{
  const metric=p=>[[-1,p[2],0,0],[p[2],1,0,0],[0,0,1,0],[0,0,0,1]],point=[0,0,.3,0];
  const forward=new Curvature4(metric,{metricDerivative:"forward"}).riemann(point),numeric=new Curvature4(metric,{step:1e-4}).riemann(point);
  assert.ok(forward.flat(3).some(v=>Math.abs(v)>.01));
  forward.flat(3).forEach((v,i)=>assert.ok(Math.abs(v-numeric.flat(3)[i])<1e-6));
  const c=new Curvature4(metric,{metricDerivative:"forward"});c.metricDerivative="bad";assert.throws(()=>c.riemann(point),/metricDerivative/);
});


test("changing curvature derivative mode updates the complete calculation",()=>{
  let dualCalls=0,jetCalls=0,numericCalls=0;
  const metric=p=>{const t=p[0];if(t instanceof Dual4)dualCalls++;else if(t instanceof Jet4)jetCalls++;else numericCalls++;const a=typeof t==="number"?Math.exp(.2*t):t.multiply(.2).exp();return [[-1,0,0,0],[0,a,0,0],[0,0,a,0],[0,0,0,a]];};
  const curvature=new Curvature4(metric,{metricDerivative:"forward",step:1e-4}),point=[.3,0,0,0];
  const forward=curvature.riemann(point);assert.ok(jetCalls>0);
  dualCalls=0;jetCalls=0;numericCalls=0;curvature.metricDerivative="finite";const finite=curvature.riemann(point);
  assert.equal(dualCalls,0);assert.equal(jetCalls,0);assert.ok(numericCalls>0);
  forward.flat(3).forEach((v,i)=>assert.ok(Math.abs(v-finite.flat(3)[i])<1e-6));
  curvature.metricDerivative="forward";jetCalls=0;curvature.riemann(point);assert.ok(jetCalls>0);
});


test("forward Schwarzschild curvature is vacuum with analytic Kretschmann invariant",()=>{
  const mass=.5,metric=p=>{const r=p[1],theta=p[2],f=Jet4.constant(1).subtract(Jet4.constant(2*mass).divide(r)),r2=r.pow(2);return [[f.negate(),0,0,0],[0,Jet4.constant(1).divide(f),0,0],[0,0,r2,0],[0,0,0,r2.multiply(theta.sin().pow(2))]];};
  const curvature=new Curvature4(metric,{metricDerivative:"forward"});
  for(const point of [[0,3,Math.PI/2,0],[0,4,.8,.2]]){
    const ricci=curvature.ricci(point);for(const component of ricci.flat())assert.ok(Math.abs(component)<1e-12);
    assert.ok(Math.abs(curvature.ricciScalar(point))<1e-12);
    assert.ok(Math.abs(curvature.kretschmannScalar(point)-48*mass*mass/point[1]**6)<1e-12);
  }
});


test("forward connection derivatives match analytic expanding metric",()=>{
  const H=.2,base=2**53,metric=p=>{const a=p[0].subtract(base).multiply(2*H).exp();return [[-1,0,0,0],[0,a,0,0],[0,0,a,0],[0,0,0,a]];},connection=new Connection4(metric);
  const point=[base+2,0,0,0],derivative=connection.christoffelDerivativeForward(point),a=Math.exp(4*H);
  assert.ok(Math.abs(derivative[0][0][1][1]-2*H*H*a)<1e-14);
  assert.ok(Math.abs(derivative[0][1][0][1])<1e-14);
  for(let e=1;e<4;e++)for(const v of derivative[e].flat(2))assert.equal(v,0);
  const batch=connection.christoffelDerivativeForward([point,point]);assert.deepEqual(batch,[derivative,derivative]);
  assert.throws(()=>connection.christoffelDerivativeForward(Array.from({length:257},()=>point)),/limit/);
});


test("forward geodesic endpoint sensitivities match a curved-coordinate solution",()=>{
  const metric=p=>[[-1,0,0,0],[0,p[1].multiply(2).exp(),0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(metric,{metricDerivative:"forward"}),geodesic=new Geodesic4(connection,[0,0,0,0],[1,.5,0,0]);
  const result=geodesic.endpointSensitivityForward([0,1],{steps:64});
  assert.ok(Math.abs(result.event[1]-Math.log(1.5))<1e-8);
  const expected=[1,1/1.5,1,1];for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(result.jacobian[i][j]-(i===j?expected[i]:0))<1e-7);
  const tangentExpected=[1,1/(1.5*1.5),1,1];for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(result.tangentJacobian[i][j]-(i===j?tangentExpected[i]:0))<1e-7);
  const shot=new Geodesic4(connection,[0,0,0,0],[1,.1,0,0]).solveBoundary([1,Math.log(1.5),0,0],[0,1],{steps:32,jacobianMethod:"forward",format:"components",tolerance:1e-7});
  assert.ok(Math.abs(shot[3][1]-.5)<1e-6);assert.ok(Math.abs(shot[1].at(-1)[1]-Math.log(1.5))<1e-7);
  assert.throws(()=>geodesic.solveBoundary([1,0,0,0],[0,1],{jacobianMethod:"bad"}),/jacobianMethod/);
  assert.throws(()=>geodesic.endpointSensitivityForward([0,1],{steps:2001}),/steps/);
});


test("forward endpoint sensitivities agree with perturbations for a coupled metric",()=>{
  const metric=p=>[[-1,p[2],0,0],[p[2],1,0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(metric,{metricDerivative:"forward"}),initial=[0,.1,.3,0],velocity=[1,.2,.1,.05],steps=24,span=[0,.4],geodesic=new Geodesic4(connection,initial,velocity),jacobian=geodesic.endpointSensitivityForward(span,{steps}).jacobian,h=1e-5;
  for(let column=0;column<4;column++){
    const plus=velocity.slice(),minus=velocity.slice();plus[column]+=h;minus[column]-=h;
    const end=v=>new Geodesic4(connection,initial,v).integrate(span,{steps}).at(-1).event.position,p=end(plus),m=end(minus);
    for(let row=0;row<4;row++)assert.ok(Math.abs(jacobian[row][column]-(p[row]-m[row])/(2*h))<1e-8);
  }
});


test("forward boundary sensitivity works when tangent perturbations collapse",()=>{
  const flat=()=>[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],geodesic=new Geodesic4(new Connection4(flat,{metricDerivative:"forward"}),[0,0,0,0],[2**53,0,0,0]),span=[0,1e-15],options={steps:4,maxIterations:3,tolerance:1e-8,damping:1e-40,format:"components"};
  assert.throws(()=>geodesic.solveBoundary([10,0,0,0],span,{...options,jacobianMethod:"finite"}),/distinct finite stencil points/);
  const result=geodesic.solveBoundary([10,0,0,0],span,{...options,jacobianMethod:"forward"});assert.ok(Math.abs(result[1].at(-1)[0]-10)<1e-8);
});


test("forward endpoint sensitivity requires the same standard forward connection as the trajectory",()=>{
  const flat=()=>[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],event=[0,0,0,0],tangent=[1,0,0,0];
  const finite=new Geodesic4(new Connection4(flat),event,tangent);
  assert.throws(()=>finite.endpointSensitivityForward([0,1]),/metricDerivative forward/);
  assert.throws(()=>finite.solveBoundary([0,0,0,0],[0,1],{jacobianMethod:"forward"}),/metricDerivative forward/);
  const custom=new Connection4(flat,{metricDerivative:"forward"});custom.geodesicAcceleration=()=>[0,0,0,0];
  assert.throws(()=>new Geodesic4(custom,event,tangent).endpointSensitivityForward([0,1]),/standard metric connection/);
});


test("forward endpoint sensitivities preserve bounded geodesic batches",()=>{
  const flat=()=>[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(flat,{metricDerivative:"forward"}),events=[[[0,0,0,0],[1,0,0,0]]],tangents=[[[1,.1,0,0],[1,.2,0,0]]],span=[0,2];
  const result=new Geodesic4(connection,events,tangents).endpointSensitivityForward(span,{steps:4});
  for(let i=0;i<2;i++){const scalar=new Geodesic4(connection,events[0][i],tangents[0][i]).endpointSensitivityForward(span,{steps:4});assert.deepEqual(result.event[0][i],scalar.event);assert.deepEqual(result.tangent[0][i],scalar.tangent);assert.deepEqual(result.jacobian[0][i],scalar.jacobian);assert.deepEqual(result.tangentJacobian[0][i],scalar.tangentJacobian);}
  const points=Array.from({length:7},()=>[0,0,0,0]),velocities=Array.from({length:7},()=>[1,0,0,0]);
  assert.throws(()=>new Geodesic4(connection,points,velocities).endpointSensitivityForward([0,1],{steps:2000}),/work limit/);
  const oversized=Array.from({length:1639},()=>[0,0,0,0]),oversizedVelocity=Array.from({length:1639},()=>[1,0,0,0]);
  assert.throws(()=>new Geodesic4(connection,oversized,oversizedVelocity).endpointSensitivityForward([0,1],{steps:1}),/component or work limit/);
});


test("forward geodesic sensitivities can differentiate the initial event",()=>{
  const metric=p=>[[-1,p[2],0,0],[p[2],1,0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(metric,{metricDerivative:"forward"}),initial=[0,.1,.3,0],velocity=[1,.2,.1,.05],steps=24,span=[0,.4],geodesic=new Geodesic4(connection,initial,velocity),result=geodesic.endpointSensitivityForward(span,{steps,withRespectTo:"initialEvent"}),h=1e-5;
  for(let column=0;column<4;column++){
    const plus=initial.slice(),minus=initial.slice();plus[column]+=h;minus[column]-=h;
    const end=x=>new Geodesic4(connection,x,velocity).integrate(span,{steps}).at(-1),p=end(plus),m=end(minus);
    for(let row=0;row<4;row++){assert.ok(Math.abs(result.jacobian[row][column]-(p.event.position[row]-m.event.position[row])/(2*h))<1e-8);assert.ok(Math.abs(result.tangentJacobian[row][column]-(p.tangent[row]-m.tangent[row])/(2*h))<1e-8);}
  }
  const batched=new Geodesic4(connection,[initial,initial],[velocity,velocity]).endpointSensitivityForward(span,{steps,withRespectTo:"initialEvent"});assert.deepEqual(batched.jacobian,[result.jacobian,result.jacobian]);
  assert.throws(()=>geodesic.endpointSensitivityForward(span,{withRespectTo:"unknown"}),/withRespectTo/);
});


test("forward four-state RK4 propagates analytic oscillator sensitivities",()=>{
  const rhs=(_t,x)=>[x[1],x[0].negate(),0,0],rows=integrateRK4Forward4(rhs,[1,0,2,3],0,Math.PI/2,64),last=rows.at(-1);
  assert.ok(Math.abs(last.value[0])<1e-8);assert.ok(Math.abs(last.value[1]+1)<1e-8);assert.deepEqual(last.value.slice(2),[2,3]);
  const expected=[[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]];
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(last.jacobian[i][j]-expected[i][j])<1e-8);
  assert.equal(rows[0].parameter,0);assert.equal(last.parameter,Math.PI/2);
  const nonautonomous=integrateRK4Forward4((t,x)=>[x[0].multiply(t),0,0,0],[2,0,0,0],0,1,64).at(-1);
  assert.ok(Math.abs(nonautonomous.value[0]-2*Math.exp(.5))<1e-8);assert.ok(Math.abs(nonautonomous.jacobian[0][0]-Math.exp(.5))<1e-8);
  assert.throws(()=>integrateRK4Forward4(()=>[1,2],[0,0,0,0],0,1,4),/rhs result/);
  assert.throws(()=>integrateRK4Forward4((_t,x)=>[Math.sin(x[0]),0,0,0],[0,0,0,0],0,1,4),/implicitly converted/);
});


test("bounded forward RK4 handles coupled sensitivity directions beyond four states",()=>{
  const rhs=(_t,x)=>[x[4],0,0,0,x[0].negate(),0],rows=integrateRK4Forward(rhs,[1,2,3,4,0,6],0,Math.PI/2,64),last=rows.at(-1);
  assert.ok(Math.abs(last.value[0])<1e-8);assert.ok(Math.abs(last.value[4]+1)<1e-8);assert.ok(Math.abs(last.jacobian[0][4]-1)<1e-8);assert.ok(Math.abs(last.jacobian[4][0]+1)<1e-8);assert.equal(last.jacobian[5][5],1);
  assert.throws(()=>integrateRK4Forward(rhs,Array(33).fill(0),0,1,2),/1 to 32/);
  assert.throws(()=>integrateRK4Forward(rhs,Array(32).fill(0),0,1,1000),/work limit/);
  assert.throws(()=>integrateRK4Forward(()=>[1,2],[0],0,1,2),/shape/);
  let count=0;assert.throws(()=>integrateRK4Forward((_t,x)=>{count++;return x.map(()=>count);},Array(5).fill(0),0,1,2),/consistent/);
});


test("forward ODE differentiates explicit parameters across dual groups",()=>{
  const rhs=(_t,x,p)=>[x[0].multiply(p[0].add(p[4]))],parameters=[.1,0,0,0,.2],last=integrateRK4Forward(rhs,[2],0,2,64,{parameters,withRespectTo:"parameters"}).at(-1),expected=2*Math.exp(.6);
  assert.ok(Math.abs(last.value[0]-expected)<1e-8);
  for(let i=0;i<5;i++)assert.ok(Math.abs(last.jacobian[0][i]-([0,4].includes(i)?2*expected:0))<1e-8);
  const initial=integrateRK4Forward(rhs,[2],0,2,64,{parameters}).at(-1);assert.ok(Math.abs(initial.jacobian[0][0]-Math.exp(.6))<1e-8);
  assert.throws(()=>integrateRK4Forward(rhs,[2],0,1,4,{withRespectTo:"parameters"}),/at least one parameter/);
  assert.throws(()=>integrateRK4Forward(rhs,[2],0,1,4,{parameters,withRespectTo:"bad"}),/withRespectTo/);
});


test("ODE worldline exposes initial-event and parameter sensitivity histories",()=>{
  const velocity=(_parameter,x,p)=>[x[0].multiply(p[0]),0,0,0],worldline=new ODEWorldline([2,0,0,0],velocity),options={steps:48,parameters:[.3]},target=2*Math.exp(.3);
  const [times,events,parameterJacobian]=worldline.integrateForward([0,1],{...options,withRespectTo:"parameters"});
  assert.equal(times.length,49);assert.ok(Math.abs(events.at(-1)[0]-target)<1e-8);assert.ok(Math.abs(parameterJacobian.at(-1)[0][0]-target)<1e-8);
  const [,stateEvents,stateJacobian]=worldline.integrateForward([0,1],options);assert.ok(Math.abs(stateEvents.at(-1)[0]-target)<1e-8);assert.ok(Math.abs(stateJacobian.at(-1)[0][0]-Math.exp(.3))<1e-8);
  const initial=[[[2,0,0,0],[3,0,0,0]]],batch=new ODEWorldline(initial,velocity).integrateForward([0,1],{...options,withRespectTo:"parameters"});
  for(let i=0;i<2;i++){const scalar=new ODEWorldline(initial[0][i],velocity).integrateForward([0,1],{...options,withRespectTo:"parameters"});assert.deepEqual(batch[1].at(-1)[0][i],scalar[1].at(-1));assert.deepEqual(batch[2].at(-1)[0][i],scalar[2].at(-1));}
  let calls=0;const many=Array.from({length:3000},()=>[1,0,0,0]);assert.throws(()=>new ODEWorldline(many,()=>{calls++;return [0,0,0,0];}).integrateForward([0,1]),/component or work limit/);assert.equal(calls,0);
});


test("batched forward worldlines match analytic coupled trajectories and sensitivities",()=>{
  const rhs=(_t,x)=>[1,x[2],x[1].negate(),0],initial=[[0,1,0,0],[0,2,0,0]],worldline=new ODEWorldline(initial,rhs);
  const [times,events,jacobians]=worldline.integrateForward([0,Math.PI/2],{steps:64}),last=events.at(-1),J=jacobians.at(-1);
  assert.equal(times.at(-1),Math.PI/2);
  for(let i=0;i<2;i++){assert.ok(Math.abs(last[i][0]-Math.PI/2)<1e-12);assert.ok(Math.abs(last[i][1])<1e-8);assert.ok(Math.abs(last[i][2]+initial[i][1])<1e-8);assert.ok(Math.abs(J[i][1][2]-1)<1e-8);assert.ok(Math.abs(J[i][2][1]+1)<1e-8);}
});
