import test from "node:test";
import assert from "node:assert/strict";
import {C,Vector4,Covector4,Tensor4,FourVelocity,FourMomentum,FourCurrent,EMFieldTensor,MaxwellEquations,RetardedField,LorentzTransform,RelativityContext,Metric4} from "../src/relativity/index.js";

test("Maxwell vacuum plane wave converges under grid refinement",()=>{
  // E_x=cos(2π(y-t)), B_z=-E_x, with B sampled half a cell ahead.
  function error(n){
    const electric=Array.from({length:2},()=>Array.from({length:n},(_,j)=>Array.from({length:2},()=>[Math.cos(2*Math.PI*j/n),0,0])));
    const magnetic=Array.from({length:2},()=>Array.from({length:n},(_,j)=>Array.from({length:2},()=>[0,0,-Math.cos(2*Math.PI*(j+.5)/n)])));
    const [,history]=new MaxwellEquations({c:1,epsilon0:1}).evolve(electric,magnetic,{spacing:[1,1/n,1],dt:.25/n,steps:2*n});
    const last=history.at(-1);
    return Math.sqrt(Array.from({length:n},(_,j)=>(last[0][j][0][0]-Math.cos(2*Math.PI*(j/n-.5)))**2).reduce((a,b)=>a+b,0)/n);
  }
  const coarse=error(8),medium=error(16),fine=error(32);
  assert.ok(coarse<.06 && fine<.004, `wave errors: ${coarse}, ${medium}, ${fine}`);
  assert.ok(coarse/medium>3.8 && medium/fine>3.8, `refinement ratios: ${coarse/medium}, ${medium/fine}`);
});

test("Maxwell evolution and residuals reject a missing grid spacing",()=>{
  const grid=Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,0])));
  const solver=new MaxwellEquations({c:1,epsilon0:1}),spacing=[1,1,1];delete spacing[1];
  assert.throws(()=>solver.evolve(grid,grid,{spacing,dt:.1,steps:1}),/spacing/);
  const [,electric,,magnetic]=solver.evolve(grid,grid,{spacing:[1,1,1],dt:.1,steps:1});
  const residuals=solver.maxwellResiduals(electric,magnetic,{spacing:[1,1,1],dt:.1});
  assert.equal(residuals.ampere.length,1);assert.deepEqual(residuals.faraday,[]);
  const constraints=solver.constraintResiduals(electric,magnetic,{spacing:[1,1,1],dt:.1});
  assert.equal(constraints.gauss_e.length,2);assert.equal(constraints.gauss_b.length,1);
  assert.throws(()=>solver.maxwellResiduals(electric,magnetic,{spacing,dt:.1}),/spacing/);
  assert.throws(()=>solver.constraintResiduals(electric,magnetic,{spacing,dt:.1}),/spacing/);
});
test("Maxwell CFL check handles very small coordinate scales",()=>{
  const grid=Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,0])));
  const solver=new MaxwellEquations({c:1,epsilon0:1}),spacing=[1e-200,1e-200,1e-200];
  assert.doesNotThrow(()=>solver.evolve(grid,grid,{spacing,dt:1e-210,steps:1}));
  assert.throws(()=>solver.evolve(grid,grid,{spacing,dt:1e-200,steps:1}),/CFL/);
});
test("Maxwell residuals handle subnormal timesteps without an infinite reciprocal",()=>{
  const grid=value=>Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[value,0,0])));
  const dt=1e-310,solver=new MaxwellEquations({c:1,epsilon0:1});
  const residual=solver.maxwellResiduals([grid(0),grid(dt),grid(2*dt)],[grid(0),grid(dt)],{spacing:[1,1,1],dt});
  for(const frame of residual.ampere)assert.equal(frame[0][0][0][0],1);
  assert.equal(residual.faraday[0][0][0][0][0],1);
  assert.equal(residual.ampere[0][0][0][0][1],0);
  assert.equal(residual.faraday[0][0][0][0][1],0);
});
test("Maxwell spatial differences recover finite derivatives when subtraction overflows",()=>{
  const zero=Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,0])));
  const electric=Array.from({length:2},()=>Array.from({length:2},(_,j)=>Array.from({length:2},()=>[0,0,j?1e308:-1e308])));
  const solver=new MaxwellEquations({c:1e-308,epsilon0:1});
  const [,history,,magnetic]=solver.evolve(electric,zero,{spacing:[4,4,4],dt:.1,steps:2});
  assert.ok(Math.abs(magnetic[0][0][0][0][0]/-2.5e306-1)<1e-12);
  const residual=solver.maxwellResiduals(history,magnetic,{spacing:[4,4,4],dt:.1});
  assert.ok(Math.abs(residual.faraday[0][0][0][0][0])/5e307<1e-12);
  assert.throws(()=>solver.evolve(electric,zero,{spacing:[.1,.1,.1],dt:.01,steps:1}),/spatial derivative is non-finite/);
});
test("Maxwell residual time differences recover finite derivatives after subtraction overflow",()=>{
  const grid=value=>Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[value,0,0])));
  const solver=new MaxwellEquations({c:.001,epsilon0:1});
  const electric=[grid(-1e308),grid(1e308),grid(1e308)],magnetic=[grid(-1e308),grid(1e308)];
  const residual=solver.maxwellResiduals(electric,magnetic,{spacing:[10,10,10],dt:4});
  assert.equal(residual.ampere[0][0][0][0][0],5e307);
  assert.equal(residual.ampere[1][0][0][0][0],0);
  assert.equal(residual.faraday[0][0][0][0][0],5e307);
  assert.throws(()=>solver.maxwellResiduals(electric,magnetic,{spacing:[10,10,10],dt:.1}),/time derivative is non-finite/);
});
test("Maxwell update preserves representable fields when c squared overflows or underflows",()=>{
  const electric=Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,0])));
  for(const [c,dt,amplitude,expected] of [[1e200,1e-300,1e-100,1],[1e-200,1e-100,1e300,1e-200]]){
    const magnetic=Array.from({length:2},()=>Array.from({length:2},(_,j)=>Array.from({length:2},()=>[0,0,j*amplitude])));
    const solver=new MaxwellEquations({c,epsilon0:1}),[,E,,B]=solver.evolve(electric,magnetic,{spacing:[1,1,1],dt,steps:1});
    assert.ok(Math.abs(E[1][0][0][0][0]/expected+1)<1e-12);
    assert.ok(Math.abs(E[1][0][1][0][0]/expected-1)<1e-12);
    const residual=solver.maxwellResiduals(E,B,{spacing:[1,1,1],dt});
    const error=Math.abs(residual.ampere[0][0][0][0][0]);
    assert.ok(error < Math.abs(expected/dt)*1e-12);
  }
});
test("Maxwell current coupling survives an unrepresentable inverse permittivity",()=>{
  const epsilon0=1e-320,grid=Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,0])));
  const current=Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[epsilon0,0,0])));
  const solver=new MaxwellEquations({c:1,epsilon0}),options={spacing:[1,1,1],dt:.1,steps:1};
  const [,zeroE,,zeroB]=solver.evolve(grid,grid,options);
  assert.equal(zeroE[1][0][0][0][0],0);
  assert.equal(solver.maxwellResiduals(zeroE,zeroB,options).ampere[0][0][0][0][0],0);
  const [,E,,B]=solver.evolve(grid,grid,{...options,current});
  assert.ok(Math.abs(E[1][0][0][0][0]+.1)<1e-12);
  assert.ok(Math.abs(solver.maxwellResiduals(E,B,{...options,current}).ampere[0][0][0][0][0])<1e-12);
});
test("Maxwell Gauss residual preserves subnormal charge continuity",()=>{
  const epsilon0=Number.MIN_VALUE,grid=Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,0])));
  const current=Array.from({length:2},()=>Array.from({length:2},(_,j)=>Array.from({length:2},()=>[0,j*epsilon0,0])));
  const solver=new MaxwellEquations({c:1,epsilon0}),options={spacing:[1,1,1],dt:.1,steps:1,current};
  const [,electric,,magnetic]=solver.evolve(grid,grid,options);
  const residual=solver.constraintResiduals(electric,magnetic,options);
  for(const frame of residual.gauss_e)for(const plane of frame)for(const row of plane)for(const value of row)assert.ok(Math.abs(value)<1e-12);
  assert.ok(Math.abs(electric[1][0][1][0][1]+.1)<1e-12);
});

test("Lib4 frozen mechanics value objects reject reassignment",()=>{
  const cases=[
    [new FourVelocity([1,0,0,0],{c:1}),"c"],
    [new FourMomentum([1,0,0,0],{c:1}),"c"],
    [new FourCurrent([1,0,0,0],{c:1}),"c"],
    [EMFieldTensor.fromEB([0,0,0],[0,0,0],{c:1}),"epsilon0"],
    [new MaxwellEquations({c:1}),"epsilon0"],
  ];
  for(const [value,field] of cases){assert.ok(Object.isFrozen(value));assert.equal(Reflect.set(value,field,"changed"),false);}
});

test("four-velocity and four-momentum preserve invariant mass",()=>{
  const u=FourVelocity.fromVelocity3([.6*C,0,0]);assert.ok(u.validate());assert.ok(Math.abs(u.gamma()-1.25)<1e-12);
  assert.ok(Math.abs(u.toVelocity3()[0]-.6*C)<1e-6);
  assert.equal(new FourVelocity([2,0,0,0],{c:1}).validate(),false);
  assert.throws(()=>new FourVelocity([-1,0,0,0],{c:1}).toVelocity3(),/future-directed/);
  const p=FourMomentum.fromMassVelocity(2,[.6*C,0,0]);assert.ok(Math.abs(p.invariantMass()-2)<1e-12);
  assert.throws(()=>FourVelocity.fromVelocity3([C,0,0]),RangeError);
});
test("four-momentum broadcasts mass and velocity batches through observables and boosts",()=>{
  const p=FourMomentum.fromMassVelocity([[1],[2]],[[[.3,0,0],[0,.4,0]]],1);
  for(const [i,mass] of [1,2].entries())for(const value of p.invariantMass()[i])assert.ok(Math.abs(value-mass)<1e-14);
  assert.ok(Math.abs(p.energy()[0][0]-1/Math.sqrt(1-.3**2))<1e-12);
  for(const [i,value] of p.momentum3()[0][1].entries())assert.ok(Math.abs(value-[0,.4/Math.sqrt(1-.4**2),0][i])<1e-14);
  const matrices=[LorentzTransform.boost([.1,0,0]).matrix,LorentzTransform.boost([0,.2,0]).matrix].map(matrix=>[matrix]);
  const boosted=p.boost(matrices);
  for(let i=0;i<2;i++)for(let j=0;j<2;j++)assert.deepEqual(boosted.components[i][j],new LorentzTransform(matrices[i][0]).applyVector(p.components[i][j]).components);
  assert.throws(()=>FourMomentum.fromMassVelocity(-1,[0,0,0],1),/nonnegative/);
});
test("retarded-time rejects a bracket inconsistent with source motion",()=>{
  const velocity=()=>[.5,0,0],position=t=>[.5*t,0,0],event=[0,1,0,0];
  const bad=new RetardedField(position,velocity,1,0,{c:1,epsilon0:1,tolerance:1e-12});
  assert.throws(()=>bad.retardedTime(event),/bracket does not contain a root/);
  const bounded=new RetardedField(position,velocity,1,.5,{c:1,epsilon0:1,tolerance:1e-12});
  assert.ok(Math.abs(bounded.retardedTime(event)+2)<1e-12);
});
test("retarded-time bisection detects unrefinable time brackets",()=>{
  const stationary=()=>[0,0,0],field=new RetardedField(stationary,stationary,1,0,{c:1,epsilon0:1,tolerance:1e-12});
  assert.equal(field.retardedTime([1e16,4,0,0]),1e16-4);
  assert.throws(()=>field.retardedTime([1e16,1,0,0]),/cannot be refined/);
});
test("retarded fields resolve small nonzero separations near the origin",()=>{
  const stationary=()=>[0,0,0];
  for(const distance of [1e-15,1e-200]){
    const field=new RetardedField(stationary,stationary,1,0,{c:1,epsilon0:1,tolerance:distance*1e-6});
    const event=[0,distance,0,0],root=field.retardedTime(event),potential=field.retardedPotential(event);
    assert.ok(Math.abs(root/distance+1)<1e-6);
    assert.ok(Math.abs(potential[0]*distance*4*Math.PI-1)<1e-12);
  }
});
test("electromagnetic dual accepts the constructor's relative antisymmetry tolerance",()=>{
  const value=1e6,offset=1e-5,field=new EMFieldTensor([[0,value,0,0],[-value+offset,0,0,0],[0,0,0,0],[0,0,0,0]],{c:1});
  const dual=field.dual().components,expected=-value+offset/2;
  assert.ok(Math.abs(dual[2][3]-expected)<1e-9);assert.equal(dual[3][2],-dual[2][3]);
  assert.equal(field.dual({orientation:-1}).components[2][3],-dual[2][3]);
});
test("four-momentum and four-velocity accessors validate mutable data and outputs",()=>{
  assert.throws(()=>new FourMomentum([1e308,0,0,0],{c:2}).energy(),/energy is non-finite/);
  assert.throws(()=>new FourVelocity([1,0,0,0],{c:1e-309}).gamma(),/gamma is non-finite/);
  const momentum=new FourMomentum([1,2,3,4],{c:1});momentum.components[1]=Infinity;
  for(const operation of [()=>momentum.energy(),()=>momentum.momentum3(),()=>momentum.invariantMass()])assert.throws(operation,/finite/);
  const velocity=new FourVelocity([1,0,0,0],{c:1});velocity.components.pop();
  for(const operation of [()=>velocity.gamma(),()=>velocity.toVelocity3(),()=>velocity.validate()])assert.throws(operation,/shape|dimension|four-velocity/i);
});
test("four-current extraction validates mutable components and finite density",()=>{
  const tinyC=new FourCurrent([1,2,3,4],{c:1e-309});assert.throws(()=>tinyC.chargeDensity(),/charge density is non-finite/);
  const current=new FourCurrent([[1,2,3,4],[2,3,4,5]],{c:1});current.components[1][1]=Infinity;
  assert.throws(()=>current.chargeDensity(),/finite/);assert.throws(()=>current.spatialCurrent(),/finite/);
  const malformed=new FourCurrent([1,2,3,4],{c:1});malformed.components.pop();
  assert.throws(()=>malformed.spatialCurrent(),/shape|dimension|component/i);
});
test("batched electromagnetic four-forces are orthogonal to four-velocity",()=>{
  const field=EMFieldTensor.fromEB([[1,.2,-.3],[-.2,.7,.1]],[[.4,-.3,.5],[.6,.1,-.2]],{c:1});
  const velocity=FourVelocity.fromVelocity3([[.2,.1,0],[-.1,.3,.2]],1).components,covariant=velocity.map(row=>[-row[0],...row.slice(1)]);
  const force=field.lorentzForce([1.2,-.7],covariant);
  for(let n=0;n<2;n++){const contraction=force[n].reduce((sum,value,i)=>sum+value*covariant[n][i],0);assert.ok(Math.abs(contraction)<1e-13);}
});
test("electromagnetic outputs handle overflow and invalid metric batches",()=>{
  const field=new EMFieldTensor([[0,1e200,0,0],[-1e200,0,0,0],[0,0,0,0],[0,0,0,0]],{c:1});
  const force=field.lorentzForce(1e-200,[0,1e200,0,0]);assert.ok(Math.abs(force[0]/1e200-1)<1e-14);assert.deepEqual(force.slice(1),[0,0,0]);
  assert.deepEqual(field.lorentzForce(0,[0,1e200,0,0]),[0,0,0,0]);
  assert.throws(()=>field.lorentzForce(1,[0,1e200,0,0]),/Lorentz force is non-finite/);
  assert.throws(()=>field.electricField(1e200),/electric field is non-finite/);
  const metric=Metric4.minkowski({batchShape:[2]});metric.components.push(Metric4.minkowski().components);
  assert.throws(()=>field.energyMomentumTensor(metric),/metric batch shape cannot change/);
});
test("mechanics transforms reject reshaped Lorentz matrices",()=>{
  const transform=LorentzTransform.identity({batchShape:[2]});transform.matrix.push(LorentzTransform.identity().matrix);
  const current=FourCurrent.fromChargeCurrent(1,[0,0,0],{c:1}),field=EMFieldTensor.fromEB([1,0,0],[0,1,0],{c:1});
  assert.throws(()=>current.transform(transform),/Lorentz matrix batch shape cannot change/);
  assert.throws(()=>field.transform(transform),/Lorentz matrix batch shape cannot change/);
});
test("four-current construction preserves charge and current components",()=>{
  const j=FourCurrent.fromChargeCurrent(2,[3,4,5],{c:10});assert.equal(j.chargeDensity(),2);assert.deepEqual(j.spatialCurrent(),[3,4,5]);
  assert.ok(j.transform(LorentzTransform.boost([.2,0,0])).components.every(Number.isFinite));
  assert.throws(()=>j.transform([[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),RangeError);
  assert.throws(()=>new FourCurrent([1,0,0,0],{c:0}),/c must be positive/);
});
test("four-current broadcasts leading dimensions and transforms each sample",()=>{
  const j=FourCurrent.fromChargeCurrent([[1,2]],[[[3,0,0]],[[0,4,0]]],{c:10});
  assert.deepEqual(j.components,[[[10,3,0,0],[20,3,0,0]],[[10,0,4,0],[20,0,4,0]]]);
  assert.deepEqual(j.chargeDensity(),[[1,2],[1,2]]);
  assert.deepEqual(j.spatialCurrent(),[[[3,0,0],[3,0,0]],[[0,4,0],[0,4,0]]]);
  const boosted=j.transform(LorentzTransform.boost([.2,0,0]));
  assert.equal(boosted.components.length,2);assert.equal(boosted.components[0].length,2);
  assert.ok(boosted.components.flat(2).every(Number.isFinite));
  assert.throws(()=>FourCurrent.fromChargeCurrent([[1,2],[3,4]],[[[1,0,0]],[[0,1,0]],[[0,0,1]]]),/broadcast-compatible/);
  const cyclic=[];cyclic.push(cyclic);assert.throws(()=>new FourCurrent(cyclic),/cycles/);
});
test("four-current broadcasts batched Lorentz matrices across current samples",()=>{
  const current=FourCurrent.fromChargeCurrent([[1,2,3]],[[[0,1,0]],[[0,0,1]]],{c:1});
  const transforms=[LorentzTransform.boost([.1,0,0]),LorentzTransform.boost([0,-.2,0])];
  const result=current.transform(transforms.map(item=>[item.matrix]));
  for(let i=0;i<2;i++)for(let j=0;j<3;j++)assert.deepEqual(result.components[i][j],transforms[i].applyVector(current.components[i][j]).components);
  assert.throws(()=>current.transform([[[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]),/preserve the Minkowski metric/);
  assert.throws(()=>current.transform([[transforms[0].matrix],[[[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]]),/preserve the Minkowski metric/);
  assert.throws(()=>current.transform([transforms[0].matrix,transforms[1].matrix,transforms[0].matrix,transforms[1].matrix]),/broadcast-compatible/);
});
test("four-momentum roundoff clamping scales with momentum magnitude",()=>{
  const nearNull=new FourMomentum([1e8,1e8+1e-7,0,0],{c:1});
  assert.equal(nearNull.invariantMass(),0);
  assert.throws(()=>nearNull.invariantMass({clampRoundoff:false}),/spacelike/);
  const spacelike=new FourMomentum([1e8,1e8+1e-4,0,0],{c:1});
  assert.throws(()=>spacelike.invariantMass(),/spacelike/);
  assert.throws(()=>new FourMomentum([1,0,0,0],{c:0}),/c must be positive/);
});
test("relativity context propagates unit constants and validates shared configuration",()=>{
  const natural=new RelativityContext({units:"natural",c:1});
  assert.equal(natural.createEMFieldTensor([0,0,0],[0,0,1]).epsilon0,1/(4*Math.PI));
  const si=new RelativityContext({epsilon0:2});
  assert.equal(si.createEMFieldTensor([0,0,0],[0,0,1]).epsilon0,2);
  const naturalFromSI=si.setUnits("natural");
  assert.equal(naturalFromSI.c,1);assert.equal(naturalFromSI.epsilon0,1/(4*Math.PI));assert.equal(naturalFromSI.units,"natural");
  const customSI=naturalFromSI.setUnits("SI",{c:2,epsilon0:3});
  assert.equal(customSI.c,2);assert.equal(customSI.epsilon0,3);assert.equal(customSI.units,"SI");
  assert.throws(()=>si.setUnits("unsupported"),/units must be SI or natural/);
  const deviceView=customSI.toDevice("webgpu");
  assert.notEqual(deviceView,customSI);assert.equal(deviceView.device,"webgpu");
  assert.equal(deviceView.c,customSI.c);assert.equal(deviceView.epsilon0,customSI.epsilon0);assert.equal(deviceView.metric,customSI.metric);
  const metric=new Metric4([[-1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]]);
  assert.equal(new RelativityContext({metric}).metric,metric);
  assert.throws(()=>new RelativityContext({c:Number.NaN}),/finite/);
  assert.throws(()=>new RelativityContext({epsilon0:0}),/epsilon0 must be positive/);
  assert.throws(()=>new RelativityContext({metric:[[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]]}),/singular/);
});
test("relativity context factories construct algebra and mechanics values with shared constants",()=>{
  const context=new RelativityContext({c:2,epsilon0:3});
  assert.ok(context.createVector([1,2,3,4]) instanceof Vector4);
  assert.ok(context.createCovector([1,2,3,4]) instanceof Covector4);
  const values=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>i===j?1:0));
  assert.ok(context.createTensor(values,["up","down"]) instanceof Tensor4);
  const directVelocity=context.createFourVelocity([2,0,0,0]);assert.equal(directVelocity.c,2);assert.equal(directVelocity.validate(),true);
  const derivedVelocity=context.fourVelocityFromVelocity3([.5,0,0]);assert.equal(derivedVelocity.c,2);assert.ok(Math.abs(derivedVelocity.components[0]-2/Math.sqrt(1-.25**2))<1e-12);
  const directMomentum=context.createFourMomentum([6,0,0,0]);assert.equal(directMomentum.invariantMass(),3);
  assert.equal(context.fourMomentumFromMassVelocity(3,[0,0,0]).invariantMass(),3);
  const current=context.createFourCurrent(2,[1,2,3]);assert.equal(current.c,2);assert.deepEqual(current.components,[4,1,2,3]);
  const maxwell=context.createMaxwellEquations();assert.equal(maxwell.c,2);assert.equal(maxwell.epsilon0,3);
  const retarded=context.createRetardedField(()=>[0,0,0],()=>[0,0,0],1,.5);assert.equal(retarded.c,2);assert.equal(retarded.epsilon0,3);
});
test("relativity context propagates c through spacetime factories",()=>{
  const context=new RelativityContext({c:2});
  const event=context.createEvent([0,0,0,0]);
  const frame=context.createInertialFrame([0,0,0,0],{relativeVelocity:[1,0,0]});
  const cone=context.createLightCone(event);
  const slice=context.createSpatialSlice([0,0,0,0],[1,0,0,0]);
  assert.equal(event.c,2);
  assert.equal(frame.c,2);
  assert.equal(frame.origin.c,2);
  assert.equal(cone.c,2);
  assert.equal(context.createLightCone([0,0,0,0]).apex.c,2);
  assert.equal(cone.apex,event);
  assert.equal(slice.c,2);
  assert.deepEqual(context.createSpacetimeInterval([2,1,0,0]).properTime(),Math.sqrt(3)/2);
});
test("four-velocity relative velocity is a rest-frame three-vector and broadcasts",()=>{
  const a=FourVelocity.fromVelocity3([.3*C,0,0]),b=FourVelocity.fromVelocity3([-.2*C,0,0]),relative=a.relativeVelocity(b);
  assert.ok(Math.abs(relative[0]+(.5/1.06)*C)<1e-7);assert.ok(Math.abs(relative[1])<1e-9&&Math.abs(relative[2])<1e-9);
  const observers=FourVelocity.fromVelocity3([[.3,0,0],[0,.2,0]],1),targets=FourVelocity.fromVelocity3([[0,.1,0]],1),batch=observers.relativeVelocity(targets);
  assert.equal(batch.length,2);assert.equal(batch[0].length,3);
  for(let i=0;i<2;i++)assert.deepEqual(batch[i],new FourVelocity(observers.components[i],{c:1}).relativeVelocity(FourVelocity.fromVelocity3([0,.1,0],1)));
  assert.throws(()=>a.relativeVelocity([1,0,0,0]),/other must be a FourVelocity/);
  assert.throws(()=>a.relativeVelocity(FourVelocity.fromVelocity3([0,0,0],1)),/same speed of light/);
});
test("relative velocity matches analytic longitudinal and transverse components",()=>{
  for(const {c,u,target} of [
    {c:1,u:.4,target:[.3,.5,.1]},
    {c:2,u:.8,target:[.6,.4,-.2]},
    {c:1,u:-.6,target:[-.2,.3,.4]},
  ]){
    const observer=FourVelocity.fromVelocity3([u,0,0],c),moving=FourVelocity.fromVelocity3(target,c);
    const actual=observer.relativeVelocity(moving),denominator=1-u*target[0]/(c*c),gamma=1/Math.sqrt(1-(u/c)**2);
    const expected=[(target[0]-u)/denominator,target[1]/(gamma*denominator),target[2]/(gamma*denominator)];
    actual.forEach((value,i)=>assert.ok(Math.abs(value-expected[i])<1e-12));
  }
});
test("four-velocity conversion preserves finite values at extreme scales",()=>{
  for(const c of [1e-200,1e-160,1e200]){
    const velocity=.5*c,moving=FourVelocity.fromVelocity3([velocity,0,0],c),observer=FourVelocity.fromVelocity3([0,0,0],c);
    assert.ok(Math.abs(moving.toVelocity3()[0]/velocity-1)<1e-14);
    assert.ok(Math.abs(observer.relativeVelocity(moving)[0]/velocity-1)<1e-14);
  }
});
test("four-momentum invariant mass handles c-squared overflow and underflow",()=>{
  for(const c of [1e-200,1e-160,1e200]){
    const momentum=FourMomentum.fromMassVelocity(2,[0,0,0],c);
    assert.ok(Math.abs(momentum.invariantMass()-2)<1e-14);
    assert.throws(()=>new FourMomentum([0,c,0,0],{c}).invariantMass(),/spacelike/);
  }
});
test("four-momentum preserves tiny representable invariant masses",()=>{
  for(const mass of [1e-160,1e-200,Number.MIN_VALUE]){
    assert.equal(new FourMomentum([mass,0,0,0],{c:1}).invariantMass(),mass);
    assert.equal(new FourMomentum([mass,mass,0,0],{c:1}).invariantMass(),0);
    const spacelike=new FourMomentum([0,mass,0,0],{c:1});
    assert.equal(spacelike.invariantMass(),0);
    assert.throws(()=>spacelike.invariantMass({clampRoundoff:false}),/spacelike/);
  }
});
test("four-momentum recovers large masses without squaring overflow",()=>{
  const mass=new FourMomentum([1e200,0,0,0],{c:1}).invariantMass();
  assert.equal(mass,1e200);
  const moving=new FourMomentum([2e200,1e200,0,0],{c:1}).invariantMass();
  assert.ok(Math.abs(moving/1e200-Math.sqrt(3))<1e-14);
  assert.equal(new FourMomentum([1e200,1e200,0,0],{c:1}).invariantMass(),0);
  assert.throws(()=>new FourMomentum([0,1e200,0,0],{c:1}).invariantMass(),/spacelike/);
});
test("four-velocity mass-shell validation handles extreme c scales",()=>{
  for(const c of [1e-200,1e-160,1e200]){
    const options={atol:0,rtol:1e-10};
    assert.equal(FourVelocity.fromVelocity3([.5*c,0,0],c).validate(options),true);
    assert.equal(new FourVelocity([2*c,0,0,0],{c}).validate(options),false);
    assert.equal(new FourVelocity([-c,0,0,0],{c}).validate(options),false);
  }
});
test("electromagnetic tensor round trips fields and invariants under boost",()=>{
  const f=EMFieldTensor.fromEB([1,2,3],[.5,.25,-1],{c:1}),g=f.transform(LorentzTransform.boost([.3,.1,0]));
  assert.deepEqual(f.electricField(),[1,2,3]);assert.deepEqual(f.magneticField(),[.5,.25,-1]);
  const a=f.invariants(),b=g.invariants();assert.ok(Math.abs(a.magneticMinusElectricOverC2-b.magneticMinusElectricOverC2)<1e-9);assert.ok(Math.abs(a.electricDotMagnetic-b.electricDotMagnetic)<1e-9);
});
test("electromagnetic invariants avoid unnecessary c-squared overflow and underflow",()=>{
  for(const c of [1e-200,1e200]){
    const field=EMFieldTensor.fromEB([c,0,0],[0,0,0],{c});
    assert.deepEqual(field.invariants(),{magneticMinusElectricOverC2:-1,electricDotMagnetic:0});
  }
  const huge=EMFieldTensor.fromEB([1e200,0,0],[0,0,0],{c:1});
  assert.throws(()=>huge.invariants(),/invariants are non-finite/);
  const balanced=EMFieldTensor.fromEB([1e200,0,0],[0,1e200,0],{c:1});
  assert.deepEqual(balanced.invariants(),{magneticMinusElectricOverC2:0,electricDotMagnetic:0});
});
test("electromagnetic pseudoscalar survives unrepresentable electric intermediates",()=>{
  const a=1e200,b=1e200,orthogonal=new EMFieldTensor([[0,a,0,0],[-a,0,0,-b],[0,0,0,0],[0,b,0,0]],{c:1e200});
  assert.deepEqual(orthogonal.invariants(),{magneticMinusElectricOverC2:0,electricDotMagnetic:0});
  const tiny=1e-200,strong=1e150,parallel=new EMFieldTensor([[0,tiny,0,0],[-tiny,0,0,0],[0,0,0,strong],[0,0,-strong,0]],{c:1e-200});
  const result=parallel.invariants();
  assert.ok(Math.abs(result.electricDotMagnetic/1e-250-1)<1e-12);
  assert.ok(Math.abs(result.magneticMinusElectricOverC2/1e300-1)<1e-12);
  const disparate=new EMFieldTensor([[0,1e200,1e-200,0],[-1e200,0,0,-1e200],[-1e-200,0,0,0],[0,1e200,0,0]],{c:1e-200});
  const mixed=disparate.invariants();
  assert.equal(mixed.magneticMinusElectricOverC2,0);
  assert.ok(Math.abs(mixed.electricDotMagnetic/1e-200-1)<1e-12);
});
test("stress-energy lowering follows the declared two-index contraction",()=>{
  const field=EMFieldTensor.fromEB([1,0,0],[0,0,0],{c:1,epsilon0:1});
  const matrix=a=>[[-1,a,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const expected=a=>[[.5,a/2,0,0],[-a,-.5,0,0],[0,0,.5,0],[0,0,0,.5]];
  assert.deepEqual(field.energyMomentumTensor(matrix(.5)),expected(.5));
  assert.deepEqual(field.energyMomentumTensor([matrix(.5),matrix(-.25)]),[expected(.5),expected(-.25)]);
});
test("stress energy scales representable results without forming c squared",()=>{
  const zero=EMFieldTensor.fromEB([0,0,0],[0,0,0],{c:1e200,epsilon0:1});
  assert.ok(zero.energyMomentumTensor().flat().every(value=>value===0));
  const large=EMFieldTensor.fromEB([0,0,0],[0,0,1e-100],{c:1e200,epsilon0:1});
  const small=EMFieldTensor.fromEB([0,0,0],[0,0,1e100],{c:1e-200,epsilon0:1});
  assert.ok(Math.abs(large.energyMomentumTensor()[0][0]/1e200-.5)<1e-13);
  assert.ok(Math.abs(small.energyMomentumTensor()[0][0]/1e-200-.5)<1e-13);
  const overflowing=EMFieldTensor.fromEB([0,0,0],[0,0,1e200],{c:1,epsilon0:1});
  assert.throws(()=>overflowing.energyMomentumTensor(),/stress-energy.*non-finite/);
});
test("electromagnetic tensor batches construction, invariants, dual, force, and stress energy",()=>{
  const field=EMFieldTensor.fromEB([[1,0,0],[0,2,0]],[[0,0,1],[1,0,0]],{c:1,epsilon0:2});
  assert.deepEqual(field.electricField(),[[1,0,0],[0,2,0]]);
  assert.deepEqual(field.magneticField(),[[0,0,1],[1,0,0]]);
  const invariant=field.invariants();assert.deepEqual(invariant.electricDotMagnetic,[0,0]);
  const twiceDual=field.dual().dual().components;for(let n=0;n<2;n++)for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(twiceDual[n][i][j]+field.components[n][i][j])<1e-12);
  const transforms=[LorentzTransform.boost([.1,0,0]).matrix,LorentzTransform.boost([0,.2,0]).matrix],transformed=field.transform(transforms);
  for(let n=0;n<2;n++)assert.deepEqual(transformed.components[n],new EMFieldTensor(field.components[n],{c:1,epsilon0:2}).transform(transforms[n]).components);
  const force=field.lorentzForce([2,3],[[-1,1,0,0],[-2,0,1,0]]);assert.equal(force.length,2);assert.deepEqual(force[0],new EMFieldTensor(field.components[0],{c:1}).lorentzForce(2,[-1,1,0,0]));
  const stress=field.energyMomentumTensor();for(let n=0;n<2;n++)assert.deepEqual(stress[n],new EMFieldTensor(field.components[n],{c:1,epsilon0:2}).energyMomentumTensor());
  const metrics=[[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[-1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]]],curvedStress=field.energyMomentumTensor(metrics);
  for(let n=0;n<2;n++)assert.deepEqual(curvedStress[n],new EMFieldTensor(field.components[n],{c:1,epsilon0:2}).energyMomentumTensor(metrics[n]));
  assert.throws(()=>field.transform([transforms[0],[[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]),/Lorentz transformation/);
  assert.throws(()=>new EMFieldTensor([field.components[0],[[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]]),/antisymmetric/);
});
test("electromagnetic invariant components preserve multidimensional field batches",()=>{
  const electric=[[[1,0,0]],[[0,2,0]]],magnetic=[[1,0,0],[0,1,0],[0,0,1]],field=EMFieldTensor.fromEB(electric,magnetic,{c:1});
  const values=field.invariants();
  assert.deepEqual(values.magneticMinusElectricOverC2,[[0,0,0],[-3,-3,-3]]);
  assert.deepEqual(values.electricDotMagnetic,[[1,0,0],[0,2,0]]);
  for(let i=0;i<2;i++)for(let j=0;j<3;j++){
    const scalar=new EMFieldTensor(field.components[i][j],{c:1}).invariants();
    assert.equal(values.magneticMinusElectricOverC2[i][j],scalar.magneticMinusElectricOverC2);
    assert.equal(values.electricDotMagnetic[i][j],scalar.electricDotMagnetic);
  }
});
test("electromagnetic dual, Lorentz force, and stress-energy obey the declared convention",()=>{
  const field=EMFieldTensor.fromEB([1,0,0],[0,0,0],{c:1,epsilon0:3});
  const twiceDual=field.dual().dual().components;
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(twiceDual[i][j]+field.components[i][j])<1e-12);
  assert.equal(field.dual().epsilon0,3);
  assert.deepEqual(field.lorentzForce(2,[-2,1,0,0]),[2,4,0,0]);
  const stress=field.energyMomentumTensor();
  assert.ok(Math.abs(stress[0][0]-1.5)<1e-12);
  assert.ok(Math.abs(stress[1][1]+1.5)<1e-12);
  assert.ok(Math.abs(stress[2][2]-1.5)<1e-12);
  assert.throws(()=>field.transform([[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),RangeError);
  assert.throws(()=>field.lorentzForce(Infinity,[-2,1,0,0]),TypeError);
});
test("Maxwell stress-energy delegates with matching physical constants",()=>{
  const field=EMFieldTensor.fromEB([1,0,0],[0,0,0],{c:2,epsilon0:3});
  const solver=new MaxwellEquations({c:2,epsilon0:3});
  assert.deepEqual(solver.energyMomentumTensor(field),field.energyMomentumTensor());
  assert.throws(()=>solver.energyMomentumTensor([0,0,0,0]),/fieldTensor must be an EMFieldTensor/);
  assert.throws(()=>new MaxwellEquations({c:1,epsilon0:3}).energyMomentumTensor(field),/share physical constants/);
});
test("Maxwell Yee evolution staggers fields in time and residuals match the update",()=>{
  const fill=Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,0])));
  const solver=new MaxwellEquations({c:1,epsilon0:1}),[tE,electric,tB,magnetic]=solver.evolve(fill,fill,{spacing:[1,1,1],dt:.1,steps:2});
  assert.deepEqual(tE,[0,.1,.2]);assert.ok(Math.abs(tB[0]-.05)<1e-15&&Math.abs(tB[1]-.15)<1e-15);
  assert.equal(electric.length,3);assert.equal(magnetic.length,2);assert.deepEqual(electric.at(-1),fill);assert.deepEqual(magnetic.at(-1),fill);
  const residuals=solver.maxwellResiduals(electric,magnetic,{spacing:[1,1,1],dt:.1});
  assert.ok(residuals.ampere.flat(5).every(value=>Math.abs(value)<1e-12));
  assert.ok(residuals.faraday.flat(5).every(value=>Math.abs(value)<1e-12));
  const constraints=solver.constraintResiduals(electric,magnetic,{spacing:[1,1,1],dt:.1});
  assert.ok(constraints.gauss_e.flat(3).every(value=>Math.abs(value)<1e-12));
  assert.ok(constraints.gauss_b.flat(3).every(value=>Math.abs(value)<1e-12));
  assert.throws(()=>solver.evolve(fill,fill,{spacing:[1,1,1],dt:1,steps:1}),RangeError);
  assert.throws(()=>solver.evolve(fill,fill,{spacing:[1,1,1],dt:.1,steps:21_000}),RangeError);
  const oversized=Array.from({length:33},()=>Array.from({length:33},()=>Array.from({length:33},()=>[0,0,0])));
  assert.throws(()=>solver.maxwellResiduals([oversized,oversized],[oversized],{spacing:[1,1,1],dt:.1}),RangeError);
});
test("Maxwell Yee update uses forward/backward periodic differences and spatial current",()=>{
  const electric=Array.from({length:3},(_,i)=>Array.from({length:3},()=>Array.from({length:3},()=>[0,0,i])));
  const magnetic=Array.from({length:3},()=>Array.from({length:3},()=>Array.from({length:3},()=>[0,0,0])));
  const current=Array.from({length:3},()=>Array.from({length:3},()=>Array.from({length:3},()=>[0,0,0,0])));
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)for(let k=0;k<3;k++)current[i][j][k]=[0,0,0,2];
  const [timesE,historyE,timesB,historyB]=new MaxwellEquations({c:1,epsilon0:1}).evolve(electric,magnetic,{spacing:[1,1,1],dt:.1,steps:2,current});
  assert.deepEqual(timesE,[0,.1,.2]);assert.ok(Math.abs(timesB[0]-.05)<1e-15&&Math.abs(timesB[1]-.15)<1e-15);
  assert.ok(Math.abs(historyB[0][0][0][0][1]-.05)<1e-12);
  assert.ok(Math.abs(historyE[1][0][0][0][2]+.185)<1e-12);
  const residuals=new MaxwellEquations({c:1,epsilon0:1}).maxwellResiduals(historyE,historyB,{spacing:[1,1,1],dt:.1,current});
  assert.ok(residuals.ampere.flat(5).every(value=>Math.abs(value)<1e-12));
  const constraints=new MaxwellEquations({c:1,epsilon0:1}).constraintResiduals(historyE,historyB,{spacing:[1,1,1],dt:.1,current});
  assert.ok(constraints.gauss_e.flat(4).every(value=>Math.abs(value)<1e-12));
  assert.ok(constraints.gauss_b.flat(4).every(value=>Math.abs(value)<1e-12));
});
test("single-step batched Maxwell constraints preserve continuity for nonuniform current",()=>{
  const field=()=>Array.from({length:3},()=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,0])));
  const source=scale=>Array.from({length:3},(_,i)=>Array.from({length:2},()=>Array.from({length:2},()=>[0,scale*i,0,0])));
  const electric=[field(),field()],magnetic=[field(),field()],current=[source(1),source(-2)];
  const solver=new MaxwellEquations({c:1,epsilon0:1}),options={spacing:[1,1,1],dt:.1,steps:1,current};
  const [,historyE,,historyB]=solver.evolve(electric,magnetic,options);
  const residuals=solver.maxwellResiduals(historyE,historyB,options),constraints=solver.constraintResiduals(historyE,historyB,options);
  assert.deepEqual(residuals.faraday,[]);
  assert.ok(residuals.ampere.flat(Infinity).every(value=>Math.abs(value)<1e-12));
  assert.ok(constraints.gauss_e.flat(Infinity).every(value=>Math.abs(value)<1e-12));
  assert.ok(constraints.gauss_b.flat(Infinity).every(value=>Math.abs(value)<1e-12));
  assert.notEqual(historyE[1][0][0][0][0][0],historyE[1][0][1][0][0][0]);
});
test("Maxwell evolution and residuals preserve multiple leading batch dimensions",()=>{
  const field=(slope)=>Array.from({length:2},(_,i)=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,slope*i])));
  const current=(jz)=>Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,0,jz])));
  const electric=[[field(0),field(1)],[field(2),field(0)]],magnetic=[[field(0),field(0)],[field(0),field(0)]];
  const currents=[[current(0),current(2)],[current(1),current(0)]];
  const solver=new MaxwellEquations({c:1,epsilon0:1}),options={spacing:[1,1,1],dt:.1,steps:2,current:currents};
  const batched=solver.evolve(electric,magnetic,options);
  assert.deepEqual(batched[0],[0,.1,.2]);assert.equal(batched[2].length,2);assert.ok(Math.abs(batched[2][0]-.05)<1e-15&&Math.abs(batched[2][1]-.15)<1e-15);
  assert.deepEqual(solver.evolve(electric,magnetic,{...options,current:new FourCurrent(currents,{c:1})}),batched);
  for(let a=0;a<2;a++)for(let b=0;b<2;b++){
    const single=solver.evolve(electric[a][b],magnetic[a][b],{...options,current:currents[a][b]});
    for(let t=0;t<single[1].length;t++)assert.deepEqual(batched[1][t][a][b],single[1][t]);
    for(let t=0;t<single[3].length;t++)assert.deepEqual(batched[3][t][a][b],single[3][t]);
  }
  const residuals=solver.maxwellResiduals(batched[1],batched[3],options);
  const constraints=solver.constraintResiduals(batched[1],batched[3],options);
  for(let a=0;a<2;a++)for(let b=0;b<2;b++){
    const single=solver.evolve(electric[a][b],magnetic[a][b],{...options,current:currents[a][b]});
    const expectedResiduals=solver.maxwellResiduals(single[1],single[3],{...options,current:currents[a][b]});
    const expectedConstraints=solver.constraintResiduals(single[1],single[3],{...options,current:currents[a][b]});
    for(let t=0;t<expectedResiduals.ampere.length;t++)assert.deepEqual(residuals.ampere[t][a][b],expectedResiduals.ampere[t]);
    for(let t=0;t<expectedResiduals.faraday.length;t++)assert.deepEqual(residuals.faraday[t][a][b],expectedResiduals.faraday[t]);
    for(let t=0;t<expectedConstraints.gauss_e.length;t++)assert.deepEqual(constraints.gauss_e[t][a][b],expectedConstraints.gauss_e[t]);
    for(let t=0;t<expectedConstraints.gauss_b.length;t++)assert.deepEqual(constraints.gauss_b[t][a][b],expectedConstraints.gauss_b[t]);
  }
  assert.throws(()=>solver.evolve(electric,magnetic,{...options,current:[[current(0)]]}),/batched current grid/);
});
test("Maxwell history output bound is checked before scanning frame payloads",()=>{
  const frame=Array.from({length:2},()=>Array.from({length:2},()=>Array.from({length:2},()=>[0,0,0])));
  const electric=new Array(30_000),magnetic=new Array(29_999);electric[0]=frame;magnetic[0]=frame;
  Object.defineProperty(electric,1,{get(){throw new Error("history payload was traversed before the bound check");}});
  assert.throws(()=>new MaxwellEquations({c:1,epsilon0:1}).maxwellResiduals(electric,magnetic,{spacing:[1,1,1],dt:.1}),/bounded output-component limit/);
  const largeGrid=Array.from({length:16},()=>Array.from({length:16},()=>Array.from({length:16},()=>[0,0,0])));
  const manyBatches=new Array(100);manyBatches[0]=largeGrid;Object.defineProperty(manyBatches,1,{get(){throw new Error("batch payload was traversed before aggregate bound check");}});
  assert.throws(()=>new MaxwellEquations({c:1,epsilon0:1}).evolve(manyBatches,manyBatches,{spacing:[1,1,1],dt:.1,steps:2}),/bounded output-component limit/);
  const cyclic=[];cyclic.push(cyclic);
  assert.throws(()=>new MaxwellEquations({c:1,epsilon0:1}).evolve(cyclic,cyclic,{spacing:[1,1,1],dt:.1}),/cannot be cyclic/);
});
test("retarded field handles large representable spatial distances",()=>{
  const field=new RetardedField(()=>[0,0,0],()=>[0,0,0],1,0,{c:1,epsilon0:1,tolerance:1e185});
  const event=[2e200,1e200,0,0],time=field.retardedTime(event),potential=field.retardedPotential(event);
  assert.ok(Math.abs(time/1e200-1)<1e-14);
  assert.ok(Math.abs(potential[0]*1e200-1/(4*Math.PI))<1e-14);
  assert.deepEqual(potential.slice(1),[0,0,0]);
});
test("retarded potentials avoid overflow and underflow in c squared",()=>{
  const tiny=1e-160,stationary=new RetardedField(()=>[0,0,0],()=>[0,0,0],1,0,{c:tiny,epsilon0:1,tolerance:1e145});
  const staticPotential=stationary.retardedPotential([0,1,0,0]);
  assert.ok(Math.abs(staticPotential[0]*tiny-1/(4*Math.PI))<1e-14);
  assert.deepEqual(staticPotential.slice(1),[0,0,0]);
  const large=1e160,speed=.2*large,moving=new RetardedField(t=>[speed*t,0,0],()=>[speed,0,0],1,speed,{c:large,epsilon0:1,tolerance:1e-175});
  const movingPotential=moving.retardedPotential([0,1,0,0]);
  assert.ok(movingPotential[0]>0&&movingPotential[1]>0);
  assert.ok(Math.abs(movingPotential[1]/movingPotential[0]-.2)<1e-14);
});
test("retarded potential avoids overflow or underflow in the Coulomb divisor",()=>{
  const stationary=()=>[0,0,0];
  const large=new RetardedField(stationary,stationary,1e308,0,{c:1,epsilon0:1,tolerance:1e293});
  assert.ok(Math.abs(large.retardedPotential([0,1e308,0,0])[0]-1/(4*Math.PI))<1e-13);
  const tiny=new RetardedField(stationary,stationary,1e-320,0,{c:1,epsilon0:1e-320,tolerance:1e-15});
  assert.ok(Math.abs(tiny.retardedPotential([0,1e-13,0,0])[0]*1e-13-1/(4*Math.PI))<1e-13);
  const subnormal=new RetardedField(stationary,stationary,1e-310,0,{c:1,epsilon0:1e-310,tolerance:1e-15});
  assert.ok(Math.abs(subnormal.retardedPotential([0,1e-13,0,0])[0]*1e-13-1/(4*Math.PI))<1e-13);
});
test("retarded four-potential survives an unrepresentable scalar intermediate",()=>{
  const stationary=()=>[0,0,0];
  const tiny=new RetardedField(stationary,stationary,1e-200,0,{c:1e-200,epsilon0:1e200,tolerance:1e185});
  const large=new RetardedField(stationary,stationary,1e308,0,{c:1e200,epsilon0:1e-100,tolerance:1e-215});
  const smallValue=tiny.retardedPotential([0,1,0,0]),largeValue=large.retardedPotential([0,1,0,0]);
  assert.ok(Math.abs(smallValue[0]/1e-200-1/(4*Math.PI))<1e-13);
  assert.ok(Math.abs(largeValue[0]/1e208-1/(4*Math.PI))<1e-13);
  assert.deepEqual(smallValue.slice(1),[0,0,0]);assert.deepEqual(largeValue.slice(1),[0,0,0]);
});
test("retarded fields avoid overflowing root midpoints and reject non-finite results",()=>{
  const source=()=>[0,0,0],stationary=()=>[0,0,0];
  const field=new RetardedField(source,stationary,1,0,{c:1,epsilon0:1,tolerance:1e293});
  const root=field.retardedTime([1.5e308,1e307,0,0]);
  assert.ok(Number.isFinite(root));assert.ok(Math.abs(root/1e308-1.4)<1e-14);
  assert.throws(()=>new RetardedField(source,stationary,1,0,{c:.5,epsilon0:1}).retardedTime([1e308,1,0,0]),/observation time/);
  assert.throws(()=>new RetardedField(source,stationary,1e308,0,{c:1,epsilon0:1}).retardedPotential([1,1e-13,0,0]),/non-finite/);
});
test("retarded potential matches an oblique uniformly moving point charge",()=>{
  for(const beta of [-.6,0,.6]){
    const field=new RetardedField(t=>[beta*t,0,0],()=>[beta,0,0],1,Math.abs(beta),{c:1,epsilon0:1,tolerance:1e-13});
    const event=[0,0,1,0],gamma=1/Math.sqrt(1-beta*beta),time=field.retardedTime(event),potential=field.retardedPotential(event);
    assert.ok(Math.abs(time+gamma)<1e-12);
    assert.ok(Math.abs(potential[0]-gamma/(4*Math.PI))<1e-12);
    assert.ok(Math.abs(potential[1]-beta*gamma/(4*Math.PI))<1e-12);
    assert.deepEqual(potential.slice(2),[0,0]);
  }
});
test("retarded point-charge potential matches Lib4 and rejects invalid trajectories",()=>{
  const rf=new RetardedField(()=>[0,0,0],()=>[0,0,0],1,0,{c:1,epsilon0:1,tolerance:1e-10});
  const event=[3,2,0,0],potential=rf.retardedPotential(event);
  assert.ok(Math.abs(rf.retardedTime(event)-1)<1e-8);
  assert.ok(Math.abs(potential[0]-1/(8*Math.PI))<1e-10);
  assert.deepEqual(rf.evaluate(event),potential);assert.deepEqual(rf.sample([event]),[potential]);
  const events=[[3,2,0,0],[3,0,2,0]],potentials=rf.retardedPotential(events);
  assert.ok(rf.retardedTime(events).every(time=>Math.abs(time-1)<1e-8));
  assert.deepEqual(potentials,events.map(point=>rf.retardedPotential(point)));
  assert.deepEqual(rf.evaluate(events),potentials);assert.deepEqual(rf.sample(events),potentials);
  assert.deepEqual(rf.sample([[events[0]],[events[1]]]),[[potentials[0]],[potentials[1]]]);
  assert.throws(()=>rf.evaluate([[3,2,0,0],[3,2,0]]),/rectangular/);
  let trajectoryCalls=0;const bounded=new RetardedField(()=>{trajectoryCalls++;return [0,0,0];},()=>[0,0,0],1,0,{c:1,epsilon0:1}),largeEventBatch=Array.from({length:10_205},()=>event);
  assert.throws(()=>bounded.retardedPotential(largeEventBatch),/bounded work limit/);assert.equal(trajectoryCalls,0);
  const moving=new RetardedField(t=>[.2*t,0,0],()=>[.2,0,0],1,.2,{c:1,epsilon0:1});
  const movingPotential=moving.evaluate([3,2,0,0]);assert.ok(movingPotential.every(Number.isFinite));
  const eventOnMovingSource=[3,.6,0,0];
  assert.equal(moving.retardedTime(eventOnMovingSource),3);
  assert.throws(()=>moving.evaluate(eventOnMovingSource),/singular/);
  assert.throws(()=>new RetardedField(()=>[0,0,0],()=>[0,0,0],1,0,{c:1,epsilon0:1,maxIterations:1,tolerance:1e-14}).retardedTime(event),/did not converge/);
  assert.throws(()=>new RetardedField(()=>[0,0,0],()=>[.1,0,0],1,0,{c:1,epsilon0:1}).evaluate(event),/exceeded the declared maxSpeed/);
  assert.throws(()=>rf.evaluate([0,0,0,0]),/singular/);
  assert.throws(()=>new RetardedField(()=>[0,0,0],()=>[0,0,0],1,0,{maxIterations:4097}),RangeError);
});
