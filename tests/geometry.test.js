import test from "node:test";
import assert from "node:assert/strict";
import {Connection4,Curvature4,DifferentialOperators4,Geodesic4,Metric4,Event4} from "../src/relativity/index.js";
const flat=()=>new Metric4().components;

test("pointwise derivatives account for unequal representable stencil spacing",()=>{
  const base=2**53,operators=new DifferentialOperators4(null,{step:3}),point=[base,0,0,0],scalar=p=>(p[0]-base)**2;
  assert.deepEqual(operators.partial(scalar,point),[0,0,0,0]);
  const shifted=operators.partial(p=>scalar(p)+2*(p[0]-base),point);assert.ok(Math.abs(shifted[0]-2)<1e-14);assert.deepEqual(shifted.slice(1),[0,0,0]);
  assert.deepEqual(operators.gradient(scalar,point),[0,0,0,0]);
  assert.deepEqual(operators.jacobian(p=>[scalar(p),2*(p[0]-base)+p[1]],point),[[0,0,0,0],[2,1,0,0]]);
  assert.throws(()=>operators.partial(p=>p[0]===base?NaN:scalar(p),point),/finite/);
});
test("connection and form derivatives use unequal stencil center values",()=>{
  const base=2**53,point=[base,0,0,0],square=p=>(p[0]-base)**2,metric=p=>[[-1,0,0,0],[0,1+square(p)/100,0,0],[0,0,1,0],[0,0,0,1]];
  const connection=new Connection4(metric,{step:3}),gamma=connection.christoffel(point);
  assert.ok(Math.abs(gamma[1][0][1])<1e-14);
  const flatConnection=new Connection4(()=>[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],{step:3});
  const derivative=flatConnection.covariantDerivative(p=>[0,square(p),0,0],point,["up"]);assert.ok(Math.abs(derivative[1][0])<1e-14);
  const exterior=new DifferentialOperators4(null,{step:3}).exteriorDerivative(p=>[0,square(p),0,0],point);assert.ok(Math.abs(exterior[0][1])<1e-14);
});
test("curvature derivatives use the center connection on unequal stencils",()=>{
  const base=2**53,curvature=new Curvature4(()=>[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],{step:3});
  curvature.connection={christoffel:p=>{const gamma=Array.from({length:4},()=>Array.from({length:4},()=>Array(4).fill(0)));gamma[1][0][0]=(p[2]-base)**2;return gamma;}};
  assert.ok(Math.abs(curvature.riemann([0,0,base,0])[1][0][2][0])<1e-14);
});
test("curvature calculations reject non-finite products and contractions",()=>{
  const point=[0,0,0,0],flat=()=>[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],tensor4=()=>Array.from({length:4},()=>Array.from({length:4},()=>Array.from({length:4},()=>Array(4).fill(0))));
  const curvature=new Curvature4(flat),gamma=Array.from({length:4},()=>Array.from({length:4},()=>Array(4).fill(0)));
  gamma[0][1][1]=1e200;gamma[1][0][0]=1e200;curvature.connection={christoffel:()=>gamma};
  assert.throws(()=>curvature.riemann(point),/Riemann curvature is non-finite/);
  const r=tensor4();r[0][1][0][1]=1e308;r[2][1][2][1]=1e308;curvature._riemannScalar=()=>r;
  assert.throws(()=>curvature.ricci(point),/Ricci tensor is non-finite/);
  curvature._ricciTensorScalar=()=>[[0,0,0,0],[0,1e308,0,0],[0,0,1e308,0],[0,0,0,0]];
  assert.throws(()=>curvature.ricciScalar(point),/Ricci scalar is non-finite/);
  assert.throws(()=>curvature.einsteinTensor(point),/Einstein tensor curvature scalar is non-finite/);
  const squared=tensor4();squared[1][0][1][0]=1e200;curvature._riemannScalar=()=>squared;
  assert.throws(()=>curvature.kretschmannScalar(point),/Kretschmann scalar is non-finite/);
});
test("differential operators reject non-finite combined results",()=>{
  const point=[0,0,0,0],flat=new DifferentialOperators4(),metric=new Metric4([[-1e-308,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),scaled=new DifferentialOperators4(metric);
  assert.throws(()=>scaled.gradient(x=>10*x[0],point),/gradient is non-finite/);
  assert.throws(()=>flat.divergence(x=>x.map(value=>1e308*value),point),/divergence is non-finite/);
  assert.throws(()=>flat.exteriorDerivative(x=>[1e308*x[1],-1e308*x[0],0,0],point),/exterior derivative is non-finite/);
  assert.throws(()=>scaled.dAlembertian(x=>x[0]*x[0],point),/dAlembertian is non-finite/);
});
test("geometry rejects reshaped Metric4 components before use",()=>{
  const metric=Metric4.minkowski(),components=metric.components;
  metric.components=[components,components];
  const point=[0,0,0,0],twoForm=[[0,1,0,0],[-1,0,0,0],[0,0,0,0],[0,0,0,0]];
  for(const operation of [()=>new DifferentialOperators4(metric).hodgeDual(twoForm),()=>new Connection4(()=>metric).christoffel(point),()=>new Curvature4(()=>metric).ricciScalar(point),()=>new Curvature4(()=>metric).einsteinTensor(point),()=>new Curvature4(()=>metric).kretschmannScalar(point)])
    assert.throws(operation,/metric batch shape cannot change/);
});
test("exterior derivative of a three-form has the analytic four-form orientation",()=>{
  const parity=indices=>{
    if(new Set(indices).size!==indices.length)return 0;
    let inversions=0;
    for(let i=0;i<indices.length;i++)for(let j=i+1;j<indices.length;j++)if(indices[i]>indices[j])inversions++;
    return inversions%2?-1:1;
  };
  const form=x=>Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>Array.from({length:4},(_,k)=>[i,j,k].includes(0)?0:parity([i,j,k])*x[0])));
  const derivative=new DifferentialOperators4(null,{step:1e-5}).exteriorDerivative(form,[.3,.2,.1,0]);
  for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++)
    assert.ok(Math.abs(derivative[a][b][c][d]-parity([a,b,c,d]))<1e-9);
});

test("flat metric has zero connection and curvature",()=>{
  const conn=new Connection4(flat),gamma=conn.christoffel([0,0,0,0]);
  assert.ok(gamma.flat(2).every(x=>Math.abs(x)<1e-12));
  const curvature=new Curvature4(flat);assert.ok(curvature.ricciScalar([0,0,0,0])<1e-12);
});
test("pointwise derivatives support rectangular non-four-dimensional outputs",()=>{
  const ops=new DifferentialOperators4(null,{step:1e-4}),point=[1,2,3,4];
  const field=x=>[x[0]+x[1],x[2]*x[3]],expected=[[1,1,0,0],[0,0,4,3]];
  for(const method of ["partial","jacobian"]){
    const values=ops[method](field,point);
    values.forEach((row,i)=>row.forEach((value,j)=>assert.ok(Math.abs(value-expected[i][j])<1e-9)));
    const batched=ops[method](field,[point,[0,1,2,3]]);
    assert.equal(batched.length,2);assert.ok(Math.abs(batched[1][1][2]-3)<1e-9);
  }
  const matrixField=x=>[[x[0],2*x[1],x[2]+x[3]],[3*x[0],x[2],4*x[3]]];
  const jacobian=ops.jacobian(matrixField,point);
  assert.equal(jacobian.length,2);assert.equal(jacobian[0].length,3);
  assert.ok(Math.abs(jacobian[1][2][3]-4)<1e-9);
  assert.throws(()=>ops.jacobian(()=>[[1],[1,2]],point),/rectangular/);
  const cycle=[];cycle.push(cycle);assert.throws(()=>ops.partial(()=>cycle,point),/cyclic/);
  let calls=0;
  assert.throws(()=>ops.jacobian(()=>{calls++;return Array(6000).fill(1);},point),/bounded/);
  assert.equal(calls,2);
});
test("finite-difference geometry rejects zero and non-finite step sizes",()=>{
  for(const step of [0,-1,NaN,Infinity]){
    assert.throws(()=>new Connection4(flat,{step}),RangeError);
    assert.throws(()=>new Curvature4(flat,{step}),RangeError);
  }
});

test("geometry uses representable stencil spacing and rejects collapsed steps",()=>{
  const base=1e10,point=[base,0,0,0],field=x=>x[0]-base,ops=new DifferentialOperators4(null,{step:1e-5});
  assert.equal(ops.partial(field,point)[0],1);
  assert.equal(ops.jacobian(x=>[field(x),0,0,0],point)[0][0],1);
  assert.equal(ops.exteriorDerivative(x=>[0,field(x),0,0],point)[0][1],1);
  const metric=x=>[[-1,0,0,0],[0,1+field(x),0,0],[0,0,1,0],[0,0,0,1]];
  const connection=new Connection4(metric,{step:1e-5});
  assert.equal(connection.christoffel(point)[1][0][1],.5);
  const collapsed=[1e20,0,0,0];
  for(const operation of [()=>ops.partial(field,collapsed),()=>ops.jacobian(field,collapsed),()=>ops.exteriorDerivative(field,collapsed),()=>connection.christoffel(collapsed),()=>new Curvature4(flat).riemann(collapsed)])assert.throws(operation,/distinct finite stencil points/);
});
test("finite differences avoid overflowing a representable quotient and reject unrepresentable results",()=>{
  const point=[0,0,0,0],linear=x=>1e308*x[0],ops=new DifferentialOperators4(null,{step:1});
  assert.equal(ops.partial(linear,point)[0],1e308);
  assert.equal(ops.jacobian(x=>[linear(x),0,0,0],point)[0][0],1e308);
  assert.equal(ops.exteriorDerivative(x=>[0,linear(x),0,0],point)[0][1],1e308);
  assert.equal(new Connection4(flat,{step:1}).covariantDerivative(linear,point,[])[0],1e308);
  const sharp=x=>x[0]>0?1e308:x[0]<0?-1e308:0,small=new DifferentialOperators4(null,{step:.1});
  for(const operation of [()=>small.partial(sharp,point),()=>small.jacobian(sharp,point),()=>small.exteriorDerivative(sharp,point),()=>new Connection4(flat,{step:.1}).covariantDerivative(sharp,point,[])])assert.throws(operation,/derivative is non-finite/);
});
test("curvature tensors and invariants preserve bounded leading point batches",()=>{
  const metric=x=>{const a=1+x[0]*x[0];return [[-1,0,0,0],[0,a*a,0,0],[0,0,1,0],[0,0,0,1]];},curvature=new Curvature4(metric,{step:1e-4}),positions=[[.2,0,0,0],[.4,0,0,0]];
  for(const method of ["riemann","ricci","ricciScalar","einsteinTensor","kretschmannScalar"])assert.deepEqual(curvature[method](positions),positions.map(point=>curvature[method](point)));
  assert.throws(()=>curvature.riemann(Array.from({length:257},()=>[0,0,0,0])),/component limit/);
});
test("Riemann tensor from an off-diagonal metric obeys the declared index antisymmetry",()=>{
  const metric=x=>[[-1,x[2],0,0],[x[2],1,0,0],[0,0,1,0],[0,0,0,1]],riemann=new Curvature4(metric,{step:1e-4}).riemann([0,0,0,0]);
  for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++)assert.ok(Math.abs(riemann[a][b][c][d]+riemann[a][b][d][c])<1e-10);
  assert.ok(Math.abs(riemann[0][1][0][1]+.25)<1e-8);
});
test("Kretschmann batch work is rejected before metric callbacks",()=>{
  let calls=0;const curvature=new Curvature4(()=>{calls++;return flat();});
  assert.throws(()=>curvature.kretschmannScalar(Array.from({length:58},()=>[0,0,0,0])),/bounded work limit/);
  assert.equal(calls,0);
});
test("Kretschmann scalar lowers the Riemann index before raising and contraction",()=>{
  const metric=x=>{const a=1+x[0]*x[0];return [[-1,0,0,0],[0,a*a,0,0],[0,0,1,0],[0,0,0,1]];};
  const point=[.4,0,0,0],curvature=new Curvature4(metric,{step:1e-4}),riemann=curvature.riemann(point),g=metric(point),gi=new Metric4(g).inverseComponents;
  const lower=Array.from({length:4},()=>Array.from({length:4},()=>Array.from({length:4},()=>Array(4).fill(0))));
  for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++)for(let e=0;e<4;e++)lower[a][b][c][d]+=g[a][e]*riemann[e][b][c][d];
  let expected=0;for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++)for(let A=0;A<4;A++)for(let B=0;B<4;B++)for(let C=0;C<4;C++)for(let D=0;D<4;D++)expected+=gi[a][A]*gi[b][B]*gi[c][C]*gi[d][D]*lower[a][b][c][d]*lower[A][B][C][D];
  const actual=curvature.kretschmannScalar(point);
  assert.ok(Math.abs(actual-expected)<1e-10*Math.max(1,Math.abs(expected)));
  assert.ok(Math.abs(riemann[0][1][0][1]-2*(1+point[0]**2))<1e-6);
  assert.ok(Math.abs(actual-4*(2/(1+point[0]**2))**2)<1e-5);
  let symmetryError=0,bianchiError=0;
  for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++){
    symmetryError=Math.max(symmetryError,Math.abs(lower[a][b][c][d]+lower[b][a][c][d]),Math.abs(lower[a][b][c][d]+lower[a][b][d][c]),Math.abs(lower[a][b][c][d]-lower[c][d][a][b]));
    bianchiError=Math.max(bianchiError,Math.abs(lower[a][b][c][d]+lower[a][c][d][b]+lower[a][d][b][c]));
  }
  assert.ok(symmetryError<2e-8);assert.ok(bianchiError<1e-10);
  const exactK=4*(2/(1+point[0]**2))**2;
  const errors=[4e-4,2e-4,1e-4].map(step=>Math.abs(new Curvature4(metric,{step}).kretschmannScalar(point)-exactK));
  assert.ok(errors[0]>errors[1]&&errors[1]>errors[2]);
  const f=1+point[0]**2,ricci=curvature.ricci(point),scalar=curvature.ricciScalar(point),einstein=curvature.einsteinTensor(point);
  assert.ok(Math.abs(ricci[0][0]+2/f)<2e-7);assert.ok(Math.abs(ricci[1][1]-2*f)<2e-7);
  assert.ok(Math.abs(scalar-4/f)<2e-7);
  assert.ok(Math.abs(einstein[0][0])<1e-7);assert.ok(Math.abs(einstein[1][1])<1e-7);
  assert.ok(Math.abs(einstein[2][2]+2/f)<2e-7);assert.ok(Math.abs(einstein[3][3]+2/f)<2e-7);
  const metricValues=new Curvature4(x=>new Metric4(metric(x)),{step:1e-4});
  assert.ok(Math.abs(metricValues.kretschmannScalar(point)-actual)<1e-10*Math.max(1,Math.abs(actual)));
});
test("finite-difference gradient and divergence match linear fields",()=>{
  const ops=new DifferentialOperators4(flat,{step:1e-4});
  const gradient=ops.gradient(x=>2*x[0]+3*x[1]-x[2], [0,0,0,0]);
  assert.ok(Math.abs(gradient[0]+2)<1e-8);assert.ok(Math.abs(gradient[1]-3)<1e-8);assert.ok(Math.abs(gradient[2]+1)<1e-8);
  const div=ops.divergence(x=>[x[0],x[1],x[2],x[3]],[1,2,3,4]);assert.ok(Math.abs(div-4)<1e-8);
});
test("pointwise differential operators preserve leading coordinate batches",()=>{
  const metric=x=>[[-1,0,0,0],[0,Math.exp(2*x[0]),0,0],[0,0,1,0],[0,0,0,1]],ops=new DifferentialOperators4(metric,{step:1e-4}),points=[[.1,0,0,0],[.3,0,0,0]],scalar=x=>x[0]**2+3*x[1],vectorField=x=>x.map((value,index)=>value*(index+1)),form=x=>[x[1],-x[0],x[2],0];
  for(const [method,field] of [["partial",scalar],["jacobian",vectorField],["gradient",scalar],["divergence",vectorField],["exteriorDerivative",form],["dAlembertian",scalar]])assert.deepEqual(ops[method](field,points),points.map(point=>ops[method](field,point)));
  assert.throws(()=>ops.jacobian(vectorField,Array.from({length:5000},()=>[0,0,0,0])),/bounded component limit/);
});
test("d'Alembertian batch work is rejected before callbacks",()=>{
  let fieldCalls=0,metricCalls=0;const ops=new DifferentialOperators4(()=>{metricCalls++;return flat();}),field=()=>{fieldCalls++;return 0;};
  assert.throws(()=>ops.dAlembertian(field,Array.from({length:3334},()=>[0,0,0,0])),/bounded work limit/);
  assert.equal(fieldCalls,0);assert.equal(metricCalls,0);
});
test("finite differences converge on smooth fields and reject samples outside the coordinate domain",()=>{
  const errors=[.4,.2,.1].map(step=>Math.abs(new DifferentialOperators4(null,{step}).partial(x=>x[0]**3,[2,0,0,0])[0]-12));
  assert.ok(errors[0]>errors[1]&&errors[1]>errors[2]);
  assert.ok(Math.abs(errors[0]/errors[1]-4)<1e-10);
  assert.throws(()=>new DifferentialOperators4(null,{step:.02}).partial(x=>Math.log(x[0]),[.01,0,0,0]),/samples must be finite/);
});
test("divergence and d'Alembertian match their metric formulas",()=>{
  const metric=x=>[[-1,0,0,0],[0,(1+x[0])**2,0,0],[0,0,1,0],[0,0,0,1]];
  const curved=new DifferentialOperators4(metric,{step:1e-4});
  const divergence=curved.divergence(x=>[x[0],0,0,0],[2,0,0,0]);
  assert.ok(Math.abs(divergence-5/3)<1e-7);
  const flatOps=new DifferentialOperators4(null,{step:1e-4});
  const box=flatOps.dAlembertian(x=>x[0]**2+x[1]**2+x[2]**2+x[3]**2,[.3,-.2,.4,.1]);
  assert.ok(Math.abs(box-4)<1e-6);
});
test("jacobian evaluates each central-difference side once and preserves row/column order",()=>{
  const samples=[];
  const ops=new DifferentialOperators4(flat,{step:1e-5});
  const jacobian=ops.jacobian(x=>{samples.push(x);return [2*x[0]+3*x[1],-x[1]+4*x[3],x[2]*x[2],x[0]-x[3]];},[1,2,3,4]);
  assert.equal(samples.length,9);assert.equal(samples.filter(x=>x.every((v,i)=>v===[1,2,3,4][i])).length,1);
  assert.ok(Math.abs(jacobian[0][0]-2)<1e-8);assert.ok(Math.abs(jacobian[0][1]-3)<1e-8);
  assert.ok(Math.abs(jacobian[1][1]+1)<1e-8);assert.ok(Math.abs(jacobian[1][3]-4)<1e-8);
  assert.ok(Math.abs(jacobian[2][2]-6)<1e-8);assert.ok(Math.abs(jacobian[3][0]-1)<1e-8);
});
test("jacobian preserves scalar and higher-rank field axes",()=>{
  const ops=new DifferentialOperators4(null,{step:1e-5}),scalar=ops.jacobian(x=>x[0]*x[0]+x[1],[2,0,0,0]);
  assert.ok(scalar.every((value,i)=>Math.abs(value-[4,1,0,0][i])<1e-8));
  const tensor=ops.jacobian(x=>{
    const value=Array.from({length:4},()=>Array(4).fill(0));value[0][1]=x[0]*x[1];return value;
  },[2,3,0,0]);
  assert.ok(Math.abs(tensor[0][1][0]-3)<1e-8);assert.ok(Math.abs(tensor[0][1][1]-2)<1e-8);
});
test("tensor-valued finite differences reject cyclic and over-nested arrays",()=>{
  const ops=new DifferentialOperators4(flat),cyclic=Array(4).fill(0);cyclic[0]=cyclic;
  assert.throws(()=>ops.jacobian(()=>cyclic,[0,0,0,0]),/cannot be cyclic/);
  let nested=0;for(let i=0;i<9;i++)nested=[nested,0,0,0];
  assert.throws(()=>ops.jacobian(()=>nested,[0,0,0,0]),/rank exceeds/);
});
test("flat-space geodesic integrates a straight line",()=>{
  const conn=new Connection4(flat),geo=new Geodesic4(conn,[0,0,0,0],[1,2,0,0]);
  const rows=geo.integrate([0,1],{steps:10}),end=rows.at(-1),[parameters,events,tangents]=geo.integrate([0,1],{steps:10,format:"components"});
  assert.deepEqual(end.event.position,[1,2,0,0]);assert.deepEqual(parameters,rows.map(row=>row.parameter));
  assert.deepEqual(events,rows.map(row=>row.event.position));assert.deepEqual(tangents,rows.map(row=>row.tangent));
  assert.throws(()=>geo.integrate([0,1],{format:"unknown"}),/format must be/);
});
test("geodesic samples preserve initial Event4 metadata through boundary shooting",()=>{
  const initial=new Event4([0,0,0,0],"observer",2);
  const geo=new Geodesic4(new Connection4(flat),initial,[1,0,0,0]);
  for(const samples of [geo.integrate([0,1],{steps:2}),geo.solveBoundary([1,0,0,0],[0,1],{steps:2})]){
    for(const sample of samples){assert.equal(sample.event.frame,"observer");assert.equal(sample.event.c,2);}
  }
});
test("geodesic integration supports matching event and tangent batches",()=>{
  const conn=new Connection4(flat),events=[[0,0,0,0],[1,2,3,4]],tangents=[[1,0,0,0],[0,2,0,0]],rows=new Geodesic4(conn,events,tangents).integrate([0,1],{steps:8});
  assert.deepEqual(rows.at(-1).event.position,[[1,0,0,0],[1,4,3,4]]);
  assert.deepEqual(rows.at(-1).tangent,tangents);
  assert.deepEqual(rows[3].event.position[1],new Geodesic4(conn,events[1],tangents[1]).integrate([0,1],{steps:8})[3].event.position);
  assert.throws(()=>new Geodesic4(conn,events,[[1,0,0,0]]),/matching/);
  assert.throws(()=>new Geodesic4(conn,events,tangents).solveBoundary([1,0,0,0],[0,1]),/unbatched/);
});
test("finite-difference connection responds to a varying metric",()=>{
  const metric=x=>[[-1,0,0,0],[0,(1+x[0])**2,0,0],[0,0,1,0],[0,0,0,1]];
  const gamma=new Connection4(metric,{step:1e-5}).christoffel([0,0,0,0]);
  assert.ok(Math.abs(gamma[1][0][1]-1)<1e-8);assert.ok(Math.abs(gamma[1][1][0]-1)<1e-8);
});
test("Christoffel symbols preserve lower-index symmetry for off-diagonal metric variation",()=>{
  const metric=x=>[[-1,x[2],0,0],[x[2],1,0,0],[0,0,1,0],[0,0,0,1]],gamma=new Connection4(metric,{step:1e-5}).christoffel([0,0,0,0]);
  for(const [a,b,c,expected] of [[0,1,2,-.5],[1,0,2,.5],[2,0,1,-.5]])assert.ok(Math.abs(gamma[a][b][c]-expected)<1e-9);
  for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)assert.ok(Math.abs(gamma[a][b][c]-gamma[a][c][b])<1e-12);
});
test("connection and geodesic acceleration reject non-finite results",()=>{
  const steep=x=>[[-1,0,0,0],[0,1e-308+4*x[0],0,0],[0,0,1,0],[0,0,0,1]];
  assert.throws(()=>new Connection4(steep).christoffel([0,0,0,0]),/Christoffel symbols are non-finite/);
  const metric=x=>[[-1,0,0,0],[0,Math.exp(4*x[0]),0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(metric);
  assert.throws(()=>connection.geodesicAcceleration([0,0,0,0],[1e200,1e200,0,0]),/geodesic acceleration is non-finite/);
});
test("Christoffel symbols preserve leading point batches within the output bound",()=>{
  const metric=x=>[[-1,0,0,0],[0,Math.exp(2*x[0]),0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(metric,{step:1e-4}),positions=[[0,0,0,0],[.2,0,0,0]],batch=connection.christoffel(positions);
  assert.deepEqual(batch,positions.map(point=>connection.christoffel(point)));
  assert.throws(()=>connection.christoffel(Array.from({length:1025},()=>[0,0,0,0])),/component limit/);
});
test("geodesic acceleration broadcasts compatible position and tangent batches",()=>{
  const metric=x=>[[-1,0,0,0],[0,Math.exp(2*x[0]),0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(metric,{step:1e-4}),positions=[[0,0,0,0],[.2,0,0,0]],tangent=[1,2,0,0],batch=connection.geodesicAcceleration(positions,tangent);
  assert.deepEqual(batch.map(acceleration=>acceleration.map(value=>Math.round(value*1e6)/1e6)),[[-4,-4,0,0],[-5.967299,-4,0,0]]);
  assert.deepEqual(connection.geodesicAcceleration(positions[1],tangent),batch[1]);
  assert.throws(()=>connection.geodesicAcceleration(positions,[[1,2,0,0],[1,2,0,0],[1,2,0,0]]),/broadcast-compatible/);
});
test("connection finite differences converge and reject metric samples outside their domain",()=>{
  const metric=x=>[[-1,0,0,0],[0,Math.exp(2*x[0]),0,0],[0,0,1,0],[0,0,0,1]],point=[.3,0,0,0];
  const errors=[.4,.2,.1].map(step=>Math.abs(new Connection4(metric,{step}).christoffel(point)[1][0][1]-1));
  assert.ok(errors[0]>errors[1]&&errors[1]>errors[2]);
  assert.ok(Math.abs(errors[0]/errors[1]-4)<.2);
  const domainMetric=x=>[[-1,0,0,0],[0,Math.log(x[0]),0,0],[0,0,1,0],[0,0,0,1]];
  assert.throws(()=>new Connection4(domainMetric,{step:.02}).christoffel([.01,0,0,0]),TypeError);
});
test("covariant derivative supports scalar, vector, covector, and mixed tensor fields",()=>{
  const flatConnection=new Connection4(flat,{step:1e-5});
  const scalar=flatConnection.covariantDerivative(x=>2*x[0]+x[1], [0,0,0,0], []);
  assert.ok(Math.abs(scalar[0]-2)<1e-8);assert.ok(Math.abs(scalar[1]-1)<1e-8);
  const vectorDerivative=flatConnection.covariantDerivative(x=>[x[0],2*x[1],0,0],[0,0,0,0],["up"]);
  assert.deepEqual(flatConnection.covariantDerivativeVector(x=>[x[0],2*x[1],0,0],[0,0,0,0]),vectorDerivative);
  assert.ok(Math.abs(vectorDerivative[0][0]-1)<1e-8);assert.ok(Math.abs(vectorDerivative[1][1]-2)<1e-8);

  const metric=x=>[[-1,0,0,0],[0,(1+x[0])**2,0,0],[0,0,1,0],[0,0,0,1]];
  const curvedConnection=new Connection4(metric,{step:1e-5});
  const covector=curvedConnection.covariantDerivative(()=>[0,2,0,0],[0,0,0,0],["down"]);
  assert.ok(Math.abs(covector[1][0]+2)<1e-8);
  const mixed=curvedConnection.covariantDerivative(()=>{const tensor=Array.from({length:4},()=>Array(4).fill(0));tensor[1][0]=3;return tensor;},[0,0,0,0],["up","down"]);
  assert.ok(Math.abs(mixed[1][0][0]-3)<1e-8);
  const positions=[[0,0,0,0],[.25,0,0,0]],scalarField=x=>2*x[0]+x[1],vectorField=x=>[x[0],2*x[1],0,0],covectorField=x=>[0,2+x[0],0,0],tensorField=x=>{const tensor=Array.from({length:4},()=>Array(4).fill(0));tensor[1][0]=3+x[0];return tensor;};
  for(const [field,types] of [[scalarField,[]],[vectorField,["up"]],[covectorField,["down"]],[tensorField,["up","down"]]])assert.deepEqual(curvedConnection.covariantDerivative(field,positions,types),positions.map(point=>curvedConnection.covariantDerivative(field,point,types)));
  assert.throws(()=>curvedConnection.covariantDerivative(vectorField,Array.from({length:5000},()=>[0,0,0,0]),["up"]),/bounded component limit/);
});
test("covariant derivative rejects missing index variance before evaluating the field",()=>{
  let calls=0;const field=()=>{calls++;return [1,0,0,0];};
  assert.throws(()=>new Connection4(flat).covariantDerivative(field,[0,0,0,0],new Array(1)),/indexTypes/);
  assert.equal(calls,0);
});
test("covariant derivative rejects non-finite connection corrections",()=>{
  const metric=x=>[[-1,0,0,0],[0,Math.exp(4*x[0]),0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(metric);
  const field=()=>[0,1e308,0,0];
  assert.throws(()=>connection.covariantDerivativeVector(field,[0,0,0,0]),/covariant derivative is non-finite/);
  assert.throws(()=>connection.covariantDerivative(field,[[0,0,0,0],[.1,0,0,0]],["up"]),/covariant derivative is non-finite/);
});
test("parallel transport uses RK4 and returns the value at every path point",()=>{
  const flatValues=new Connection4(flat).parallelTransport([1,2,3,4],[[0,0,0,0],[1,0,0,0],[2,0,0,0]]);
  assert.deepEqual(flatValues,[[1,2,3,4],[1,2,3,4],[1,2,3,4]]);
  const metric=x=>[[-1,0,0,0],[0,(1+x[0])**2,0,0],[0,0,1,0],[0,0,0,1]];
  const values=new Connection4(metric).parallelTransport([0,1,0,0],[[0,0,0,0],[.5,0,0,0]],{stepsPerSegment:20});
  assert.equal(values.length,2);assert.ok(Math.abs(values[1][1]-2/3)<1e-7);
  const batchPaths=[[[0,0,0,0],[.5,0,0,0]],[[.1,0,0,0],[.7,0,0,0]]],batchVectors=[[0,1,0,0],[0,2,0,0]],batched=new Connection4(metric).parallelTransport(batchVectors,batchPaths,{stepsPerSegment:20});
  assert.equal(batched.length,2);
  for(let sample=0;sample<batchPaths.length;sample++){
    const scalar=new Connection4(metric).parallelTransport(batchVectors[sample],batchPaths[sample],{stepsPerSegment:20});
    batched[sample].forEach((transported,index)=>transported.forEach((component,axis)=>assert.ok(Math.abs(component-scalar[index][axis])<1e-12)));
  }
  assert.throws(()=>new Connection4(metric).parallelTransport([[1,0,0,0],[0,1,0,0],[0,0,1,0]],batchPaths),/broadcast-compatible/);
  assert.throws(()=>new Connection4(flat).parallelTransport([1,0,0,0],[0,0,0,0]),/at least two coordinate points/);
  const longPath=new Array(1252).fill([0,0,0,0]);
  assert.throws(()=>new Connection4(flat).parallelTransport([1,0,0,0],longPath),RangeError);
  const geo=new Geodesic4(new Connection4(flat),[0,0,0,0],[1,0,0,0]);
  assert.throws(()=>geo.solveBoundary([1,0,0,0],[0,1],{iterations:13}),RangeError);
  assert.throws(()=>geo.solveBoundary([1,0,0,0],[0,1],{steps:2001}),RangeError);
});
test("connection is metric-compatible for a varying off-diagonal metric",()=>{
  const metric=x=>[[-1,.1*x[2],0,0],[.1*x[2],1+.2*x[0],0,0],[0,0,1+.1*x[1],0],[0,0,0,1]],points=[[.2,.3,.4,0],[-.1,.1,-.2,.3]],connection=new Connection4(metric,{step:1e-4});
  const covariant=connection.covariantDerivative(metric,points,["down","down"]),contravariant=connection.covariantDerivative(x=>new Metric4(metric(x)).inverseComponents,points,["up","up"]);
  for(const value of [...covariant.flat(3),...contravariant.flat(3)])assert.ok(Math.abs(value)<1e-8);
});
test("curved parallel transport preserves the metric norm and reverses along the path",()=>{
  const metric=x=>[[-1,0,0,0],[0,(1+x[0])**2,0,0],[0,0,1,0],[0,0,0,1]],connection=new Connection4(metric);
  const path=[[0,0,0,0],[.5,0,0,0],[1,0,0,0]],initial=[1,2,.5,0];
  const values=connection.parallelTransport(initial,path,{stepsPerSegment:20});
  for(let i=0;i<path.length;i++){
    assert.ok(Math.abs(values[i][1]-2/(1+path[i][0]))<1e-7);
    assert.ok(Math.abs(new Metric4(metric(path[i])).dot(values[i],values[i])-3.25)<1e-6);
  }
  const reversed=connection.parallelTransport(values.at(-1),path.slice().reverse(),{stepsPerSegment:20}).at(-1);
  reversed.forEach((value,i)=>assert.ok(Math.abs(value-initial[i])<1e-6));
});
test("geodesic boundary shooting uses damped least squares and returns a matching endpoint",()=>{
  const geo=new Geodesic4(new Connection4(flat),[0,0,0,0],[.5,.25,0,0]);
  const rows=geo.solveBoundary([2,1,0,0],[0,2],{steps:16,tolerance:1e-9});
  assert.ok(Math.hypot(...rows.at(-1).event.position.map((value,i)=>value-[2,1,0,0][i]))<1e-9);
  assert.ok(Math.hypot(...rows[0].tangent.map((value,i)=>value-[1,.5,0,0][i]))<1e-8);
  const [parameters,events,tangents,fitted]=geo.solveBoundary([2,1,0,0],[0,2],{steps:16,tolerance:1e-9,format:"components"});
  assert.equal(parameters.length,17);events.at(-1).forEach((value,i)=>assert.ok(Math.abs(value-[2,1,0,0][i])<1e-9));assert.deepEqual(fitted,tangents[0]);
  assert.throws(()=>geo.solveBoundary([2,1,0,0],[0,2],{format:"unknown"}),/format must be/);
  assert.throws(()=>geo.solveBoundary([2,1,0,0],[0,2],{steps:2000,maxIterations:1}),/bounded integration-step limit/);
  assert.throws(()=>geo.solveBoundary([2,1,0,0],[0,2],{damping:0}),/damping must be positive/);
});
test("geodesic boundary defaults use the Lib4 tolerance and permit its iteration budget",()=>{
  const connection={geodesicAcceleration:()=>[0,0,0,0]},target=[1,0,0,0];
  const near=new Geodesic4(connection,[0,0,0,0],[1-5e-8,0,0,0]);
  assert.throws(()=>near.solveBoundary(target,[0,1],{steps:1,maxIterations:0}),/did not converge/);
  assert.equal(near.solveBoundary(target,[0,1],{steps:1,maxIterations:0,tolerance:1e-7}).length,2);
  const exact=new Geodesic4(connection,[0,0,0,0],[1,0,0,0]);
  assert.equal(exact.solveBoundary(target,[0,1],{steps:100,maxIterations:12}).length,101);
  const damped=new Geodesic4(connection,[0,0,0,0],[0,0,0,0]);
  assert.throws(()=>damped.solveBoundary(target,[0,1],{steps:1,maxIterations:8,damping:.1}),/did not converge/);
  const solved=damped.solveBoundary(target,[0,1],{steps:1,damping:.1});
  assert.ok(Math.abs(solved.at(-1).event.position[0]-1)<1e-9);
});

test("boundary shooting reuses perturbed trajectories across Jacobian rows",()=>{
  let calls=0;
  const connection={geodesicAcceleration:()=>{calls++;return [0,0,0,0];}};
  const solver=new Geodesic4(connection,[0,0,0,0],[0,0,0,0]);
  const result=solver.solveBoundary([1,2,3,4],[0,1],{steps:1,maxIterations:1,tolerance:1e-7});
  assert.ok(Math.hypot(...result.at(-1).event.position.map((v,i)=>v-[1,2,3,4][i]))<1e-7);
  assert.equal(calls,44); // residual, eight perturbed paths, residual, final path; four RK4 stages each
});
test("boundary shooting uses representable tangent perturbations",()=>{
  const connection={geodesicAcceleration:()=>[0,0,0,0]};
  const solver=new Geodesic4(connection,[0,0,0,0],[1e10,0,0,0]);
  const target=[1e10+1,0,0,0];
  const result=solver.solveBoundary(target,[0,1],{steps:1,maxIterations:1,tolerance:1e-5,damping:1e-12});
  assert.ok(Math.abs(result.at(-1).event.position[0]-target[0])<1e-5);
  const collapsed=new Geodesic4(connection,[0,0,0,0],[1e20,0,0,0]);
  assert.throws(()=>collapsed.solveBoundary([1e20+1e6,0,0,0],[0,1],{steps:1,maxIterations:1}),/distinct finite stencil points/);
});
test("geodesic boundary shooting converges on a curved metric from a perturbed tangent",()=>{
  const metric=x=>[[-1,0,0,0],[0,Math.exp(2*x[0]),0,0],[0,0,1,0],[0,0,0,1]];
  const connection=new Connection4(metric,{step:1e-4}),initial=[0,0,0,0],span=[0,.5];
  const targetPath=new Geodesic4(connection,initial,[1,.5,0,0]).integrate(span,{steps:32});
  const target=targetPath.at(-1).event.position;
  const result=new Geodesic4(connection,initial,[.8,.7,0,0]).solveBoundary(target,span,{steps:32,maxIterations:8,tolerance:1e-6,damping:1e-8});
  assert.ok(Math.hypot(...result.at(-1).event.position.map((value,i)=>value-target[i]))<1e-6);
  assert.ok(Math.hypot(...result[0].tangent.map((value,i)=>value-[1,.5,0,0][i]))<1e-5);
});
test("curved-coordinate geodesic agrees with its analytic path under step refinement",()=>{
  const metric=x=>[[-1,0,0,0],[0,Math.exp(2*x[1]),0,0],[0,0,1,0],[0,0,0,1]];
  const connection=new Connection4(metric,{step:1e-4}),initial=[0,0,0,0],tangent=[1,.5,0,0],span=[0,.5];
  const exact=[.5,Math.log(1+.5*.5),0,0];
  const errors=[8,16,32].map(steps=>{
    const end=new Geodesic4(connection,initial,tangent).integrate(span,{steps}).at(-1).event.position;
    return Math.hypot(...end.map((value,i)=>value-exact[i]));
  });
  assert.ok(errors[0]>errors[1]&&errors[1]>errors[2]);
  assert.ok(errors[2]<1e-9);
});

test("Rindler coordinates preserve flat curvature and the mapped inertial geodesic",()=>{
  const rindler=x=>{
    const rho=1+x[1];
    return [[-rho*rho,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  };
  const point=[.2,.3,0,0],connection=new Connection4(rindler,{step:1e-4}),gamma=connection.christoffel(point);
  const rho=1+point[1];
  assert.ok(Math.abs(gamma[0][0][1]-1/rho)<1e-9);
  assert.ok(Math.abs(gamma[0][1][0]-1/rho)<1e-9);
  assert.ok(Math.abs(gamma[1][0][0]-rho)<1e-8);

  const curvature=new Curvature4(rindler,{step:1e-4});
  assert.ok(Math.max(...curvature.riemann(point).flat(3).map(Math.abs))<1e-5);
  assert.ok(Math.abs(curvature.ricciScalar(point))<1e-5);

  const exact=parameter=>[Math.atanh(parameter),Math.sqrt(1-parameter*parameter)-1,0,0];
  const errors=[8,16,32].map(steps=>{
    const rows=new Geodesic4(connection,[0,0,0,0],[1,0,0,0]).integrate([0,.3],{steps});
    const actual=rows.at(-1).event.position;
    return Math.hypot(...actual.map((value,index)=>value-exact(.3)[index]));
  });
  assert.ok(errors[0]>errors[1]&&errors[1]>errors[2]);
  assert.ok(errors[2]<1e-7);
});

test("de Sitter FLRW metric reproduces analytic curvature and comoving geodesics",()=>{
  const H=.2,deSitter=x=>{
    const scale=Math.exp(H*x[0]);
    return [[-1,0,0,0],[0,scale*scale,0,0],[0,0,scale*scale,0],[0,0,0,scale*scale]];
  };
  const curvature=new Curvature4(deSitter,{step:2e-4}),points=[[-.3,0,0,0],[.3,.1,-.2,.4]];
  for(const value of curvature.ricciScalar(points))assert.ok(Math.abs(value-12*H*H)<1e-6);
  for(const value of curvature.kretschmannScalar(points))assert.ok(Math.abs(value-24*H**4)<1e-6);
  const point=points[1],rho=deSitter(point),ricci=curvature.ricci(point);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(ricci[i][j]-3*H*H*rho[i][j])<1e-6);
  const gamma=new Connection4(deSitter,{step:2e-4}).christoffel(point);
  assert.ok(Math.abs(gamma[1][0][1]-H)<1e-8);
  assert.ok(Math.abs(gamma[0][1][1]-H*Math.exp(2*H*point[0]))<1e-7);
  const endpoint=new Geodesic4(new Connection4(deSitter),[0,0,0,0],[1,0,0,0]).integrate([0,.5],{steps:32}).at(-1).event.position;
  assert.deepEqual(endpoint,[.5,0,0,0]);
});

test("de Sitter differential operators match analytic curved-metric formulas",()=>{
  const H=.2,metric=x=>{const a=Math.exp(2*H*x[0]);return [[-1,0,0,0],[0,a,0,0],[0,0,a,0],[0,0,0,a]];};
  const ops=new DifferentialOperators4(metric,{step:2e-4}),field=x=>x[0]**2+x[1]**2+.5*x[2]**2,vector=x=>[x[0],x[1],0,0];
  for(const t of [-.3,.3]){
    const point=[t,.2,-.1,0],factor=Math.exp(-2*H*t),gradient=ops.gradient(field,point),expected=[-2*t,.4*factor,-.1*factor,0];
    for(let i=0;i<4;i++)assert.ok(Math.abs(gradient[i]-expected[i])<1e-8);
    assert.ok(Math.abs(ops.divergence(vector,point)-(2+3*H*t))<1e-8);
    assert.ok(Math.abs(ops.dAlembertian(field,point)-(-2-6*H*t+3*factor))<1e-7);
  }
});
test("de Sitter Riemann components obey the constant-curvature identity",()=>{
  const H=.2,metric=x=>{const a=Math.exp(2*H*x[0]);return [[-1,0,0,0],[0,a,0,0],[0,0,a,0],[0,0,0,a]];};
  const curvature=new Curvature4(metric,{step:2e-4});
  for(const time of [-.3,.3]){
    const point=[time,.1,-.2,.4],g=metric(point),R=curvature.riemann(point);
    for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++){
      const expected=H*H*((a===c?g[b][d]:0)-(a===d?g[b][c]:0));
      assert.ok(Math.abs(R[a][b][c][d]-expected)<2e-8,`R^${a}_${b}${c}${d} at ${time}`);
    }
  }
});
test("Schwarzschild exterior is vacuum and matches its analytic Kretschmann scalar",()=>{
  const mass=.5,radius=3,theta=Math.PI/2,point=[0,radius,theta,.4];
  const schwarzschild=x=>{
    const r=x[1],angle=x[2],f=1-2*mass/r;
    return [[-f,0,0,0],[0,1/f,0,0],[0,0,r*r,0],[0,0,0,r*r*Math.sin(angle)**2]];
  };
  const expected=48*mass*mass/radius**6,steps=[4e-3,2e-3,1e-3];
  const errors=steps.map(step=>{
    const curvature=new Curvature4(schwarzschild,{step}),ricci=curvature.ricci(point);
    return {ricci:Math.max(...ricci.flat().map(Math.abs)),scalar:Math.abs(curvature.ricciScalar(point)),kretschmann:Math.abs(curvature.kretschmannScalar(point)-expected)};
  });
  assert.ok(errors[0].ricci>errors[1].ricci&&errors[1].ricci>errors[2].ricci);
  assert.ok(errors[0].kretschmann>errors[1].kretschmann&&errors[1].kretschmann>errors[2].kretschmann);
  assert.ok(errors[2].ricci<2e-6&&errors[2].scalar<2e-7&&errors[2].kretschmann<1e-6);
  const gamma=new Connection4(schwarzschild,{step:1e-4}).christoffel(point);
  const f=1-2*mass/radius;
  assert.ok(Math.abs(gamma[1][0][0]-f*mass/radius**2)<1e-9);
});
test("exterior derivative returns an antisymmetric two-form",()=>{
  const ops=new DifferentialOperators4(flat,{step:1e-5}),f=ops.exteriorDerivative(x=>[x[1],-x[0],0,0],[0,0,0,0]);
  assert.equal(f[0][1],-2);assert.equal(f[1][0],2);
});
test("exterior derivative supports scalar and higher-degree forms",()=>{
  const ops=new DifferentialOperators4(null,{step:1e-5}),df=ops.exteriorDerivative(x=>2*x[0]+3*x[1],[0,0,0,0]);
  assert.ok(df.every((value,i)=>Math.abs(value-[2,3,0,0][i])<1e-9));
  const dA=ops.exteriorDerivative(x=>{
    const form=Array.from({length:4},()=>Array(4).fill(0));form[0][1]=x[2];form[1][0]=-x[2];return form;
  },[0,0,0,0]);
  assert.ok(Math.abs(dA[2][0][1]-1)<1e-9);assert.ok(Math.abs(dA[0][2][1]+1)<1e-9);
  assert.throws(()=>new DifferentialOperators4(null,{step:0}),RangeError);
});
test("differential-operator Hodge dual follows the Lorentzian covariant-form convention",()=>{
  const form=Array.from({length:4},()=>Array(4).fill(0));form[0][1]=1;form[1][0]=-1;
  const flatOps=new DifferentialOperators4(new Metric4()),dual=flatOps.hodgeDual(form),twice=flatOps.hodgeDual(dual);
  assert.ok(Math.abs(dual[2][3]-1)<1e-12);assert.ok(Math.abs(dual[3][2]+1)<1e-12);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(twice[i][j]+form[i][j])<1e-12);
  assert.deepEqual(flatOps.hodgeDual([form,form]),[flatOps.hodgeDual(form),flatOps.hodgeDual(form)]);
  const curved=new DifferentialOperators4(x=>[[-1,0,0,0],[0,1+x[0],0,0],[0,0,1,0],[0,0,0,1]]);
  assert.throws(()=>curved.hodgeDual(form),/x is required/);assert.ok(curved.hodgeDual(form,[0,0,0,0]).flat().every(Number.isFinite));
  const forms=[form,form.map(row=>row.map(value=>2*value))],positions=[[0,0,0,0],[.4,0,0,0]];
  assert.deepEqual(curved.hodgeDual(forms,positions),forms.map((value,index)=>curved.hodgeDual(value,positions[index])));
  const euclidean=new DifferentialOperators4(new Metric4([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]));
  assert.equal(euclidean.hodgeDual(form)[2][3],-1);
  const tiny=new DifferentialOperators4(new Metric4([[-1e-200,0,0,0],[0,1e-200,0,0],[0,0,1,0],[0,0,0,1]]));
  assert.throws(()=>tiny.hodgeDual(form),/raised components are non-finite/);
});
