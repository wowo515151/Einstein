import test from "node:test";
import assert from "node:assert/strict";
import {Event4,SpacetimeInterval,LightCone,SpatialSlice,WorldlineSegment,Worldline,integrateRK4,ODEWorldline,LorentzTransform,PoincareTransform,InertialFrame} from "../src/relativity/index.js";

test("piecewise worldline evaluates a long sequence at joins and beyond its endpoints",()=>{
  const segments=Array.from({length:128},(_,i)=>new WorldlineSegment([i,i*i,0,0],[i+1,(i+1)**2,0,0],[1,2*i,0,0],[1,2*(i+1),0,0],[i,i+1]));
  const worldline=new Worldline(segments),queries=[-1,0,0.5,31,63.25,64,127.5,128,129];
  const events=worldline.evaluate(queries).position,tangents=worldline.tangent(queries).components;
  for(let i=0;i<queries.length;i++){
    assert.ok(Math.abs(events[i][0]-queries[i])<1e-10);
    assert.ok(Math.abs(events[i][1]-queries[i]**2)<1e-9);
    assert.ok(Math.abs(tangents[i][1]-2*queries[i])<1e-9);
  }
  const left=new WorldlineSegment([0,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1]);
  const right=new WorldlineSegment([1,1e-9,0,0],[2,1e-9,0,0],[1,0,0,0],[1,0,0,0],[1,2]);
  assert.equal(new Worldline([left,right]).evaluate(1).position[1],1e-9);
});

test("spatial slice preflights broadcast dimensions before normalizing",()=>{
  const origins=Array.from({length:251},()=>[[0,0,0,0]]);
  const normals=[Array.from({length:251},()=>[1,0,0,0])];
  assert.throws(()=>new SpatialSlice(origins,normals),/component limit/);
  const small=new SpatialSlice([[[0,0,0,0]],[[1,0,0,0]]],[[[1,0,0,0],[1,0,0,0]]]);
  small.normal[0][0][0]=2;
  assert.equal(small.normal[1][0][0],1);
  assert.equal(small.normal[0][1][0],1);
});
test("spatial slice broadcast origins and explicit bases are independent",()=>{
  const origins=[[[0,0,0,0]],[[1,0,0,0]]],normals=[[[1,0,0,0],[1,0,0,0]]];
  const basis=[[0,1,0,0],[0,0,1,0],[0,0,0,1]],slice=new SpatialSlice(origins,normals,{basis});
  slice.originPosition[0][0][0]=8;
  assert.equal(slice.originPosition[0][1][0],0);
  slice.basis[0][0][0][1]=4;
  assert.equal(slice.basis[1][0][0][1],1);
  assert.equal(slice.basis[0][1][0][1],1);
  assert.equal(basis[0][1],1);
});
test("spatial slice normalizes timelike directions across finite scales",()=>{
  for(const magnitude of [1e-200,1,1e200]){
    const slice=new SpatialSlice([0,0,0,0],[magnitude,0,0,0]);
    assert.deepEqual(slice.normal,[1,0,0,0]);
    assert.deepEqual(slice.projectTo3D([0,2,3,4]),[2,3,4]);
  }
  assert.throws(()=>new SpatialSlice([0,0,0,0],[0,0,0,0]),/timelike/);
});

test("interval classification follows changes to its mutable displacement",()=>{
  const interval=new SpacetimeInterval([2,0,0,0],{c:1});
  assert.equal(interval.isTimelike(),true);
  interval.displacement[0]=0;interval.displacement[1]=3;
  assert.equal(interval.intervalSquared(),9);
  assert.equal(interval.isSpacelike(),true);
  assert.equal(interval.properDistance(),3);
  assert.throws(()=>interval.properTime(),/timelike/);
});

test("events classify oriented causal separation and calculate proper time/distance",()=>{
  const origin=new Event4([0,0,0,0]);
  assert.equal(origin.causalRelation(new Event4([5,3,0,0])),"future_timelike");
  assert.equal(origin.causalRelation(new Event4([-5,3,0,0])),"past_timelike");
  assert.equal(origin.causalRelation(new Event4([3,3,0,0])),"future_null");
  assert.equal(origin.causalRelation(new Event4([-3,3,0,0])),"past_null");
  assert.equal(origin.causalRelation(new Event4([1,3,0,0])),"spacelike");
  assert.equal(origin.causalRelation(new Event4([0,0,0,0])),"coincident");
  assert.equal(new SpacetimeInterval([5,3,0,0],{c:1}).properTime(),4);
  assert.equal(new SpacetimeInterval([1,3,0,0],{c:1}).properDistance(),Math.sqrt(8));
});
test("Lib4 frozen spacetime value objects reject reassignment",()=>{
  const cases=[
    [new Event4([0,0,0,0]),"c"],
    [new SpacetimeInterval([1,0,0,0]),"atol"],
    [new LightCone([0,0,0,0]),"branch"],
    [new SpatialSlice([0,0,0,0],[1,0,0,0]),"c"],
    [new InertialFrame(),"c"],
  ];
  for(const [value,field] of cases){assert.ok(Object.isFrozen(value));assert.equal(Reflect.set(value,field,"changed"),false);}
});
test("spacetime intervals use scale-aware tolerance and event constants propagate",()=>{
  const origin=new Event4([0,0,0,0],null,2),timelike=new Event4([2,0,0,0],null,2),interval=origin.intervalTo(timelike);
  assert.deepEqual(origin.displacementTo(timelike),[2,0,0,0]);assert.throws(()=>origin.displacementTo([2,0,0,0]),TypeError);
  assert.equal(interval.c,2);assert.equal(interval.properTime(),1);assert.equal(timelike.transform(LorentzTransform.identity()).c,2);
  const translated=timelike.transform(new PoincareTransform(LorentzTransform.identity(),[1,0,0,0]));
  assert.deepEqual(translated.position,[3,0,0,0]);assert.equal(translated.c,2);
  const nearlyNull=new SpacetimeInterval([1e6,1e6+1e-7,0,0],{atol:1e-10});
  assert.equal(nearlyNull.isNull(),true);assert.throws(()=>nearlyNull.properTime(),RangeError);
  assert.throws(()=>new SpacetimeInterval([0,0,0,0],{c:0}),RangeError);
  const extremeNull=new SpacetimeInterval([1e308,1e308,0,0]);
  assert.equal(extremeNull.squared,0);
  assert.throws(()=>extremeNull.isNull(),/classification scale is non-finite/);
  assert.throws(()=>new Event4([0,0,0,0],null,Number.NaN),RangeError);
  assert.throws(()=>new Event4([0,0,0,0]).causalRelation(new Event4([1e308,1e308,0,0])),RangeError);
});
test("interval and event classification scale tolerances before extreme squaring",()=>{
  const origin=new Event4([0,0,0,0]),large=[1e200,1e200,0,0],atol=1e-200;
  const interval=new SpacetimeInterval(large,{atol});
  assert.ok(Math.abs(interval.tolerance()/2e200-1)<1e-15);
  assert.equal(interval.isNull(),true);
  assert.equal(origin.causalRelation(new Event4(large),{atol}),"future_null");
  assert.equal(origin.causalRelation(new Event4(large.map(x=>-x)),{atol}),"past_null");
  assert.equal(new SpacetimeInterval([1e308,1e308,0,0],{atol:0}).isNull(),true);
  assert.equal(origin.causalRelation(new Event4([1.7e308,1.7e308,0,0]),{atol:0}),"future_null");
});
test("events and intervals preserve batch shapes through classification and scalar/batched transforms",()=>{
  const origin=new Event4([0,0,0,0]),events=new Event4([[5,3,0,0],[-5,3,0,0],[3,3,0,0],[-3,3,0,0],[1,3,0,0],[0,0,0,0]]);
  assert.deepEqual(origin.causalRelation(events),["future_timelike","past_timelike","future_null","past_null","spacelike","coincident"]);
  const intervals=new SpacetimeInterval([[5,3,0,0],[1,3,0,0],[3,3,0,0]],{c:1});
  assert.deepEqual(intervals.intervalSquared(),[-16,8,0]);
  assert.deepEqual(intervals.isTimelike(),[true,false,false]);
  assert.deepEqual(intervals.isSpacelike(),[false,true,false]);
  assert.deepEqual(intervals.isNull(),[false,false,true]);
  assert.throws(()=>intervals.properTime(),RangeError);
  assert.deepEqual(new SpacetimeInterval([[5,3,0,0],[10,6,0,0]],{c:1}).properTime(),[4,8]);
  assert.deepEqual(new SpacetimeInterval([[1,3,0,0],[2,6,0,0]]).properDistance(),[Math.sqrt(8),Math.sqrt(32)]);
  const boost=LorentzTransform.boost([0.2,0,0]),mapped=events.transform(boost);
  assert.deepEqual(mapped.position,events.position.map(position=>boost.applyVector(position).components));
  assert.equal(mapped.frame,null);
  const shifted=events.transform(new PoincareTransform(boost,[1,0,0,0]));
  assert.deepEqual(shifted.position,mapped.position.map(position=>[position[0]+1,...position.slice(1)]));
  const samples=new Event4([[1,2,0,0],[3,0,1,0]],null,2),batchedLorentz=LorentzTransform.boost([[.1,0,0],[0,.2,0]]),batchedPoincare=new PoincareTransform(batchedLorentz,[[1,0,0,0],[0,1,0,0]]),affine=samples.transform(batchedPoincare);
  assert.equal(affine.c,2);
  affine.position.forEach((position,index)=>{const scalar=new PoincareTransform(LorentzTransform.boost(index===0?[.1,0,0]:[0,.2,0]),index===0?[1,0,0,0]:[0,1,0,0]).applyEvent(samples.position[index]).position;position.forEach((value,axis)=>assert.ok(Math.abs(value-scalar[axis])<1e-12));});
});
test("light cone contains causal events and samples null points",()=>{
  const cone=new LightCone([0,0,0,0],{branch:"future"});
  assert.equal(cone.contains([2,1,0,0]),true);assert.equal(cone.contains([-2,1,0,0]),false);
  const points=cone.sample([[1,0,0],[0,1,0]],2);
  for(const p of points)assert.ok(Math.abs(p[0]**2-p[1]**2-p[2]**2-p[3]**2)<1e-12);
  assert.deepEqual(cone.intersectWorldline([0,2,0,0],[1,0,0,0]),[[2,2,0,0]]);
  assert.throws(()=>cone.intersectWorldline([0,0,0,0],[1,1,0,0]),RangeError);
});
test("light-cone classifications retain representable extreme tolerances",()=>{
  const future=new LightCone([0,0,0,0],{branch:"future",atol:1e-200});
  const positive=[1e200,1e200,0,0],negative=[-1e200,1e200,0,0];
  assert.deepEqual(future.onSurface([positive,negative]),[true,false]);
  assert.deepEqual(future.contains([positive,negative]),[true,false]);
  const exact=new LightCone([0,0,0,0],{branch:"future",atol:0});
  assert.equal(exact.onSurface([1e308,1e308,0,0]),true);
  assert.equal(exact.contains([1e308,1e308,0,0]),true);
  assert.throws(()=>new LightCone([0,0,0,0]).contains([1e308,1e308,0,0]),/classification is non-finite/);
});
test("light-cone directions normalize consistently at extreme finite scales",()=>{
  const cone=new LightCone([0,0,0,0],{branch:"future"});
  const expected=cone.sample([1,1,1],2);
  const hit=cone.intersectHypersurface([1,0,0,0],2,[1,1,1]);
  for(const scale of [Number.MIN_VALUE,1.7e308]){
    const sample=cone.sample([scale,scale,scale],2);
    const intersection=cone.intersectHypersurface([1,0,0,0],2,[scale,scale,scale]);
    sample.forEach((value,index)=>assert.ok(Math.abs(value-expected[index])<1e-14));
    assert.equal(intersection.hitMask,true);
    intersection.events.forEach((value,index)=>assert.ok(Math.abs(value-hit.events[index])<1e-14));
  }
  assert.throws(()=>cone.sample([0,0,0],1),/nonzero/);
});
test("light-cone worldline intersections scale extreme finite coefficients",()=>{
  for(const branch of ["future","past","both"]){
    const cone=new LightCone([0,0,0,0],{branch});
    const hits=cone.intersectWorldline([0,1e200,0,0],[1,0,0,0]);
    assert.equal(hits.length,branch==="both"?2:1);
    for(const hit of hits)assert.ok(Math.abs(Math.abs(hit[0])/1e200-1)<1e-14);
  }
  const cone=new LightCone([0,0,0,0],{branch:"future",atol:0});
  const hit=cone.intersectWorldline([0,1e200,0,0],[1e200,0,0,0]);
  assert.equal(hit.length,1);assert.ok(Math.abs(hit[0][0]/1e200-1)<1e-14);
  for(const radius of [1e-160,1e-200,1e-308]){
    const single=cone.intersectWorldline([0,radius,0,0],[1,0,0,0]);
    const smallDirection=cone.intersectWorldline([0,radius,0,0],[radius,0,0,0]);
    assert.equal(single.length,1);assert.equal(smallDirection.length,1);
    assert.ok(Math.abs(single[0][0]/radius-1)<1e-14);
    assert.ok(Math.abs(smallDirection[0][0]/radius-1)<1e-14);
  }
  assert.equal(cone.intersectWorldline([0,1e308,0,0],[1e-308,0,0,0])[0][0],1e308);
  for(const radius of [1e-300,1e-200,1,1e200,1e300])for(const speed of [1e-100,1,1e100]){
    const result=cone.intersectWorldline([0,radius,0,0],[speed,.25*speed,0,0]);
    assert.equal(result.length,1);
    assert.ok(Math.abs(result[0][0]/(radius/.75)-1)<1e-12,`radius ${radius}, speed ${speed}`);
  }
});
test("light-cone quadratic roots retain the nearby intersection",()=>{
  const cone=new LightCone([0,0,0,0],{branch:"both",atol:0}),speed=1-1e-8;
  for(const direction of [speed,-speed]){
    const events=cone.intersectWorldline([0,1,0,0],[1,direction,0,0]);
    assert.equal(events.length,2);
    const near=events.reduce((a,b)=>Math.abs(a[0])<Math.abs(b[0])?a:b);
    const expected=-1/(direction+Math.sign(direction));
    assert.ok(Math.abs(near[0]-expected)<1e-15);
    assert.ok(Math.abs(Math.abs(near[0])-Math.abs(near[1]))<1e-15);
  }
});
test("translated light cones retain analytic future and past intersections across scales",()=>{
  for(const radius of [1e-200,1,1e200]){
    const apex=[3*radius,-2*radius,0,0],origin=[3*radius,-radius,0,0],direction=[2,.5,0,0];
    for(const [branch,expected] of [["future",3+1/.75],["past",3-1/1.25]]){
      const hits=new LightCone(apex,{branch,atol:0}).intersectWorldline(origin,direction);
      assert.equal(hits.length,1);
      assert.ok(Math.abs(hits[0][0]/radius-expected)<1e-12);
      assert.ok(Math.abs(Math.abs((hits[0][0]-apex[0])/radius)-Math.abs((hits[0][1]-apex[1])/radius))<1e-12);
    }
  }
});
test("light-cone branch, scaled tolerances, and sampling match Lib4 semantics",()=>{
  const future=new LightCone([0,0,0,0],{branch:"future",atol:1e-10});
  assert.equal(future.onSurface([2,2,0,0]),true);assert.equal(future.onSurface([-2,2,0,0]),false);
  assert.equal(future.contains([1e6,1e6+1e-7,0,0]),true);
  assert.deepEqual(new LightCone([0,0,0,0],{branch:"past"}).sample([[1,0,0]],2)[0],[-2,2,0,0]);
  assert.throws(()=>new LightCone([0,0,0,0]).sample([[1,0,0]],2),/requires a future or past branch/);
  const tooManyDirections=new Array(65_537);
  assert.throws(()=>future.sample(tooManyDirections,1),/direction limit/);
  assert.throws(()=>future.intersectHypersurface([1,0,0,0],1,tooManyDirections),/direction limit/);
  const nearlyParallel=new LightCone([0,0,0,0],{branch:"future",atol:1e-6});
  assert.deepEqual(nearlyParallel.intersectHypersurface([1,1+1e-7,0,0],-1,[[-1,0,0]]).hitMask,[false]);
  assert.throws(()=>new LightCone([0,0,0,0],{c:0}),RangeError);
});
test("light-cone containment, surface checks, sampling, and hyperplane hits broadcast batches",()=>{
  const apex=new Event4([[0,0,0,0],[10,0,0,0]]),events=new Event4([[2,2,0,0],[8,2,0,0]]);
  const both=new LightCone(apex,{branch:"both"});
  assert.deepEqual(both.onSurface(events),[true,true]);
  assert.deepEqual(both.contains(new Event4([[1,0,0,0],[11,0,0,0]])),[true,true]);
  const future=new LightCone(apex,{branch:"future"});
  assert.deepEqual(future.contains(events),[true,false]);
  const apexGrid=new Event4([[[0,0,0,0]],[[10,0,0,0]]]);
  const directions=[[[1,0,0],[0,1,0],[0,0,1]]],radii=[[1,2,3],[4,5,6]];
  const sampled=new LightCone(apexGrid,{branch:"past"}).sample(directions,radii);
  assert.deepEqual(sampled.map(row=>row.length),[3,3]);
  assert.deepEqual(sampled[1][2], [4,0,0,6]);
  const futureGrid=new LightCone(apexGrid,{branch:"future"});
  const hits=futureGrid.intersectHypersurface([1,0,0,0],2,[[[1,0,0],[0,1,0]]]);
  assert.deepEqual(hits.hitMask,[[true,true],[true,true]]);
  assert.deepEqual(hits.events[1][1],[12,0,2,0]);
  const misses=futureGrid.intersectHypersurface([1,1,0,0],1,[[-1,0,0]]);
  assert.deepEqual(misses.hitMask,[[false],[false]]);
  assert.ok(misses.events.every(row=>row[0].every(Number.isNaN)));
  assert.throws(()=>futureGrid.intersectWorldline([0,0,0,0],[1,0,0,0]),/unbatched cone apex/);
});
test("light-cone worldline branch filtering uses time-coordinate tolerance",()=>{
  const future=new LightCone([0,0,0,0],{branch:"future",atol:1e-10});
  const past=new LightCone([0,0,0,0],{branch:"past",atol:1e-10});
  assert.deepEqual(future.intersectWorldline([0,1e10,0,0],[1,0,0,0]),[[1e10,1e10,0,0]]);
  assert.deepEqual(past.intersectWorldline([0,1e10,0,0],[1,0,0,0]),[[-1e10,1e10,0,0]]);
});
test("spatial slice projects and embeds an orthonormal frame",()=>{
  const slice=new SpatialSlice([0,0,0,0],[1,0,0,0]);
  const p=slice.projectTo3D([0,2,3,4]);assert.deepEqual(p,[2,3,4]);
  assert.deepEqual(slice.embed3D(p).position,[0,2,3,4]);
});
test("spatial slices broadcast batched frames, events, and intersections",()=>{
  const origins=[[0,0,0,0],[1,1,0,0]],normals=[[1,0,0,0],[Math.sqrt(2),1,0,0]];
  const slice=new SpatialSlice(origins,normals),coordinates=[[2,3,4],[1,0,0]],embedded=slice.embed3D(coordinates);
  const projected=slice.projectTo3D(embedded);assert.deepEqual(projected[0],coordinates[0]);assert.ok(Math.abs(projected[1][0]-coordinates[1][0])<1e-12);assert.deepEqual(projected[1].slice(1),coordinates[1].slice(1));
  assert.deepEqual(slice.contains(embedded),[true,true]);
  const lines=origins.map((origin,index)=>origin.map((value,axis)=>value+normals[index][axis]));
  const hit=slice.intersectWorldline(lines,normals);
  assert.ok(hit.parameter.every(value=>Math.abs(value+1)<1e-12));
  assert.ok(hit.event.position.every((event,index)=>event.every((value,axis)=>Math.abs(value-origins[index][axis])<1e-12)));
  const cones=slice.intersectLightcone(new LightCone([[0,0,0,0],[0,0,0,0]],{branch:"future"}));
  assert.deepEqual(cones.hit,[true,true]);
});
test("spatial slice embedding preserves batched Event4 origin metadata",()=>{
  const frame={name:"observer"},origin=new Event4([[0,0,0,0],[1,1,0,0]],frame,2),normals=[[1,0,0,0],[Math.sqrt(2),1,0,0]];
  const slice=new SpatialSlice(origin,normals,{c:3}),embedded=slice.embed3D([[2,3,4],[1,0,0]]);
  assert.ok(embedded instanceof Event4);assert.equal(embedded.c,2);assert.equal(embedded.frame,frame);
  assert.deepEqual(embedded.position[0],[0,2,3,4]);
  assert.ok(Math.abs(embedded.position[1][0]-2)<1e-12);assert.ok(Math.abs(embedded.position[1][1]-(1+Math.sqrt(2)))<1e-12);assert.deepEqual(embedded.position[1].slice(2),[0,0]);
});
test("spatial slice validates future normal and scales containment tolerance",()=>{
  assert.throws(()=>new SpatialSlice([0,0,0,0],[-1,0,0,0]),/future-directed/);
  assert.throws(()=>new SpatialSlice([0,0,0,0],[0,1,0,0]),/timelike/);
  const slice=new SpatialSlice([0,0,0,0],[1,0,0,0],{atol:1e-9});
  assert.equal(slice.contains([5e-4,1e6,0,0]),true);
  assert.throws(()=>slice.intersectWorldline([1,0,0,0],[0,1,0,0]),/parallel to or contained/);
  const hit=slice.intersectWorldline([1,2,0,0],[1,0,0,0]);
  assert.equal(hit.parameter,-1);assert.deepEqual(hit.event.position,[0,2,0,0]);
});
test("spatial slice containment scales an overflowing coordinate norm",()=>{
  const slice=new SpatialSlice([0,0,0,0],[1,0,0,0]),event=[1e308,1.7e308,1.7e308,0];
  assert.equal(slice.contains(event,{atol:.1}),false);
  assert.equal(slice.contains(event,{atol:.5}),true);
  assert.equal(slice.contains(event,{atol:0}),false);
  assert.deepEqual(slice.contains([event,[0,1.7e308,1.7e308,0]],{atol:.1}),[false,true]);
  assert.throws(()=>slice.contains(event,{atol:1}),/classification is non-finite/);
});
test("spatial-slice light-cone intersection returns branch-aware sphere data",()=>{
  const slice=new SpatialSlice([2,0,0,0],[1,0,0,0]);
  const future=slice.intersectLightcone(new LightCone([0,0,0,0],{branch:"future"}));
  assert.deepEqual(future,{center:[0,0,0],radius:2,hit:true});
  const past=slice.intersectLightcone(new LightCone([0,0,0,0],{branch:"past"}));
  assert.equal(past.hit,false);assert.ok(Number.isNaN(past.radius));assert.ok(past.center.every(Number.isNaN));
  const apexSlice=new SpatialSlice([0,0,0,0],[1,0,0,0]);
  assert.deepEqual(apexSlice.intersectLightcone(new LightCone([0,0,0,0])),{center:[0,0,0],radius:0,hit:true});
});
test("Hermite worldline reproduces a straight path and tangent",()=>{
  const seg=new WorldlineSegment([0,0,0,0],[2,2,0,0],[1,1,0,0],[1,1,0,0],[0,2]);
  assert.deepEqual(seg.evaluate(1).position,[1,1,0,0]);assert.deepEqual(seg.tangent(1).components,[1,1,0,0]);
  const line=new Worldline([seg]);assert.deepEqual(line.evaluate(2).position,[2,2,0,0]);
  assert.deepEqual(line.evaluate(-1).position,[-1,-1,0,0]);assert.deepEqual(line.evaluate(3).position,[3,3,0,0]);
});
test("worldlines require continuous joined segments",()=>{
  const first=new WorldlineSegment([0,0,0,0],[1,1,0,0],[1,1,0,0],[1,1,0,0],[0,1]);
  const next=new WorldlineSegment([1,1,0,0],[2,2,0,0],[1,1,0,0],[1,1,0,0],[1,2]);
  assert.equal(new Worldline([first,next]).evaluate(1.5).position[0],1.5);
  assert.throws(()=>new Worldline([first,new WorldlineSegment([2,2,0,0],[3,3,0,0],[1,1,0,0],[1,1,0,0],[2,3])]),/parameter spans must meet/);
  assert.throws(()=>new Worldline([first,new WorldlineSegment([1.1,1,0,0],[2,2,0,0],[1,1,0,0],[1,1,0,0],[1,2])]),/share their event/);
  assert.throws(()=>new Worldline([first,new WorldlineSegment([1,1,0,0],[2,2,0,0],[0,1,0,0],[1,1,0,0],[1,2])]),/share their event/);
  assert.throws(()=>new Worldline(new Array(1)),/WorldlineSegment/);
  const sparse=[first,first];delete sparse[1];
  assert.throws(()=>new Worldline(sparse),/WorldlineSegment/);
  assert.throws(()=>new WorldlineSegment([0,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[-1e308,1e308]),/finite duration/);
});
test("worldline Hermite interpolation reproduces a cubic and its derivative",()=>{
  const segment=new WorldlineSegment([8,0,0,0],[64,0,0,0],[12,0,0,0],[48,0,0,0],[2,4]);
  for(const parameter of [1,2,2.5,3.5,4,5]){
    const event=segment.evaluate(parameter).position,tangent=segment.tangent(parameter).components;
    assert.ok(Math.abs(event[0]-parameter**3)<1e-12);
    assert.ok(Math.abs(tangent[0]-3*parameter**2)<1e-12);
    assert.deepEqual(event.slice(1),[0,0,0]);assert.deepEqual(tangent.slice(1),[0,0,0]);
  }
});
test("worldline interpolation preserves speed-of-light and frame metadata",()=>{
  const frame={name:"observer"},start=new Event4([0,0,0,0],frame,2),end=new Event4([2,2,0,0],frame,2),segment=new WorldlineSegment(start,end,[1,1,0,0],[1,1,0,0],[0,2]);
  const sample=segment.evaluate(1),lineSample=new Worldline([segment]).evaluate(1);
  assert.equal(sample.c,2);assert.equal(sample.frame,frame);assert.equal(lineSample.c,2);assert.equal(lineSample.frame,frame);
  const mixed=new WorldlineSegment(start,[2,2,0,0],[1,1,0,0],[1,1,0,0],[0,2]);
  assert.equal(mixed.evaluate(1).c,2);assert.equal(mixed.evaluate(1).frame,frame);
  assert.throws(()=>new WorldlineSegment(start,new Event4([2,2,0,0],frame,3),[1,1,0,0],[1,1,0,0],[0,2]),/share c and frame/);
  assert.throws(()=>new WorldlineSegment(start,new Event4([2,2,0,0],{name:"other"},2),[1,1,0,0],[1,1,0,0],[0,2]),/share c and frame/);
  const next=new WorldlineSegment(new Event4([2,2,0,0],frame,3),[4,4,0,0],[1,1,0,0],[1,1,0,0],[2,4]);
  assert.throws(()=>new Worldline([segment,next]),/same c and frame/);
});
test("worldline segments and piecewise paths evaluate bounded parameter batches",()=>{
  const first=new WorldlineSegment([0,0,0,0],[1,1,0,0],[1,1,0,0],[1,1,0,0],[0,1]),second=new WorldlineSegment([1,1,0,0],[2,2,0,0],[1,1,0,0],[1,1,0,0],[1,2]),line=new Worldline([first,second]),parameters=[[-1,0],[.5,1],[1.5,2],[2.5,3]],events=line.evaluate(parameters).position,tangents=line.tangent(parameters).components;
  events.forEach((row,i)=>row.forEach((event,j)=>event.forEach((value,axis)=>assert.ok(Math.abs(value-(axis<2?parameters[i][j]:0))<1e-12))));
  tangents.forEach(row=>row.forEach(tangent=>assert.ok(Math.abs(tangent[0]-1)<1e-12&&Math.abs(tangent[1]-1)<1e-12&&tangent[2]===0&&tangent[3]===0)));
  const segmentBatch=first.evaluate([0,.5,1]).position;assert.equal(segmentBatch.length,3);assert.deepEqual(segmentBatch[1].slice(2),[0,0]);
});
test("RK4 integrates exponential growth and ODE worldline constant motion",()=>{
  const rows=integrateRK4((t,y)=>[y[0]], [1],0,1,20);
  assert.ok(Math.abs(rows.at(-1)[1][0]-Math.E)<1e-6);
  const line=new ODEWorldline([0,0,0,0],(parameter,event)=>[1,0,0,0]),[times,events]=line.integrate([0,2],{steps:2});
  assert.deepEqual(times,[0,1,2]);assert.deepEqual(events,[[0,0,0,0],[1,0,0,0],[2,0,0,0]]);
  assert.throws(()=>new ODEWorldline([0,0,0,0],[1,0,0,0]),TypeError);
  assert.throws(()=>integrateRK4(()=>[0],[0],0,1,10001),RangeError);
});
test("RK4 integrates scalar states and checks derivative shape",()=>{
  const forward=integrateRK4((_t,y)=>y,1,0,1,20);
  assert.equal(typeof forward[0][1],"number");
  assert.ok(Math.abs(forward.at(-1)[1]-Math.E)<1e-6);
  const backward=integrateRK4((_t,y)=>y,Math.E,1,0,20);
  assert.ok(Math.abs(backward.at(-1)[1]-1)<1e-6);
  assert.throws(()=>integrateRK4(()=>[1],1,0,1,1),/rhs result shape/);
  assert.throws(()=>integrateRK4(()=>1,NaN,0,1,1),/finite/);
});
test("RK4 avoids overflowing an intermediate weighted derivative sum",()=>{
  const rows=integrateRK4(()=>[1e308],[0],0,1e-308,1);
  assert.equal(rows.length,2);
  assert.ok(Math.abs(rows[1][1][0]-1)<1e-14);
});
test("RK4 rejects unrepresentable time grids before evaluating the derivative",()=>{
  let calls=0;const rhs=()=>{calls++;return [1];};
  assert.throws(()=>integrateRK4(rhs,[0],1,1+Number.EPSILON,4),/time grid/);
  assert.throws(()=>integrateRK4(rhs,[0],1+Number.EPSILON,1,4),/time grid/);
  assert.throws(()=>integrateRK4(rhs,[0],0,Number.MIN_VALUE,2),/time grid/);
  assert.equal(calls,0);
  assert.deepEqual(integrateRK4(()=>[1],[0],1,1,2),[[1,[0]],[1,[0]],[1,[0]]]);
});
test("RK4 uses actual step endpoints and rejects collapsed midpoints",()=>{
  const base=1,stages=[];
  const rows=integrateRK4(t=>{stages.push(t);return [1];},[0],base,1.1,3);
  assert.equal(rows.at(-1)[1][0],1.1-base);
  for(let i=0;i<3;i++)assert.equal(stages[4*i+3],rows[i+1][0]);
  assert.notEqual(rows[1][0]-rows[0][0],rows[3][0]-rows[2][0]);
  let calls=0;
  assert.throws(()=>integrateRK4(()=>{calls++;return [1];},[0],1e16,1e16+2,1),/distinct midpoint/);
  assert.throws(()=>integrateRK4(()=>{calls++;return [1];},[0],1e16,1e16+6,1),/midpoint is too asymmetric/);
  assert.throws(()=>integrateRK4(()=>{calls++;return [1];},[0],1e16+6,1e16,1),/midpoint is too asymmetric/);
  assert.equal(calls,0);
});
test("RK4 and ODEWorldline preserve bounded leading batches",()=>{
  const initial=[[1,2],[3,4]],batched=integrateRK4((t,state)=>state,initial,0,1,16);
  assert.deepEqual(batched[0][1],initial);
  const final=batched.at(-1)[1];
  for(let sample=0;sample<initial.length;sample++){
    const scalar=integrateRK4((t,state)=>state,initial[sample],0,1,16).at(-1)[1];
    final[sample].forEach((value,index)=>assert.ok(Math.abs(value-scalar[index])<1e-12));
  }
  assert.throws(()=>integrateRK4(()=>[1],[[1,2],[3,4]],0,1,2),/rhs result shape/);
  const line=new ODEWorldline([[0,0,0,0],[2,2,0,0]],(t,events)=>events.map(()=>[1,0,0,0])),[times,events]=line.integrate([0,2],{steps:2});
  assert.deepEqual(times,[0,1,2]);assert.deepEqual(events,[[[0,0,0,0],[2,2,0,0]],[[1,0,0,0],[3,2,0,0]],[[2,0,0,0],[4,2,0,0]]]);
});
test("RK4 rejects non-finite bounds, derivative output, and intermediate states",()=>{
  assert.throws(()=>integrateRK4((t,[y])=>[y],[1],NaN,1,4),/bounds must be finite/);
  assert.throws(()=>integrateRK4((t,[y])=>[y],[1],-1e308,1e308,1),/step size must be finite/);
  assert.throws(()=>integrateRK4(()=>[Infinity],[0],0,1,1),/rhs result/);
  assert.throws(()=>integrateRK4(()=>[1e308],[1e308],0,1,1),/integration state/);
});
test("inertial frame applies its spatial rotation and relativistic boost",()=>{
  const frame=new InertialFrame({relativeVelocity:[.5*299792458,0,0]});
  assert.ok(LorentzTransform.rotation(frame.orientation).validate());
  const rotation=[[0,-1,0],[1,0,0],[0,0,1]],rotated=new InertialFrame({orientation:rotation,origin:new Event4([1,2,3,4],null,2),c:2});
  assert.ok(rotated.toGlobalTransform instanceof PoincareTransform);assert.ok(rotated.fromGlobalTransform instanceof PoincareTransform);
  const transformRoundTrip=rotated.fromGlobalTransform.compose(rotated.toGlobalTransform).applyEvent({position:[0,1,0,0]}).position;
  transformRoundTrip.forEach((value,index)=>assert.ok(Math.abs(value-[0,1,0,0][index])<1e-12));
  const global=rotated.toGlobal([0,1,0,0]);assert.deepEqual(global.position,[1,2,4,4]);
  assert.ok(rotated.fromGlobal(global).position.every((value,i)=>Math.abs(value-[0,1,0,0][i])<1e-12));
  const globalEvent=new Event4([2,3,4,5],null,2),globalVector=[1,2,3,4];
  assert.deepEqual(rotated.transformEvent(globalEvent).position,rotated.fromGlobalTransform.applyEvent(globalEvent).position);
  assert.deepEqual(rotated.transformVector(globalVector).components,rotated.fromGlobalTransform.applyVector(globalVector).components);
  const other=new InertialFrame({relativeVelocity:[.2,0,0],c:1}),local=[0,1,2,0],expected=other.fromGlobal(frame.toGlobal(local).position).position;
  const relative=frame.relativeTransform(other).applyEvent({position:local}).position;
  assert.ok(relative.every((value,i)=>Math.abs(value-expected[i])<1e-12));
  assert.throws(()=>new InertialFrame({relativeVelocity:[299792458,0,0]}),RangeError);
  assert.throws(()=>new InertialFrame({orientation:[[1,0,0],[0,-1,0],[0,0,1]]}),RangeError);
});
test("inertial frames broadcast origins, orientations, velocities, and events",()=>{
  const origins=[[0,0,0,0],[1,2,3,4]],orientations=[[[1,0,0],[0,1,0],[0,0,1]],[[0,-1,0],[1,0,0],[0,0,1]]],velocities=[[0,0,0],[0,.1,0]],local=[[2,3,4,5],[1,0,2,3]];
  const frame=new InertialFrame({origin:new Event4(origins,null,1),orientation:orientations,relativeVelocity:velocities,c:1}),global=frame.toGlobal(local),roundTrip=frame.fromGlobal(global.position);
  for(let i=0;i<2;i++){
    const scalar=new InertialFrame({origin:new Event4(origins[i],null,1),orientation:orientations[i],relativeVelocity:velocities[i],c:1}).toGlobal(local[i]).position;
    global.position[i].forEach((value,axis)=>assert.ok(Math.abs(value-scalar[axis])<1e-12));
    roundTrip.position[i].forEach((value,axis)=>assert.ok(Math.abs(value-local[i][axis])<1e-12));
  }
  const otherOrigins=[[3,0,0,0],[0,1,0,0]],other=new InertialFrame({origin:new Event4(otherOrigins,null,1),orientation:orientations,relativeVelocity:velocities,c:1}),relative=frame.relativeTransform(other),moved=relative.applyEvent(local).position;
  assert.equal(moved.length,2);
  for(let i=0;i<2;i++){
    const first=new InertialFrame({origin:new Event4(origins[i],null,1),orientation:orientations[i],relativeVelocity:velocities[i],c:1}),second=new InertialFrame({origin:new Event4(otherOrigins[i],null,1),orientation:orientations[i],relativeVelocity:velocities[i],c:1}),expected=first.relativeTransform(second).applyEvent(local[i]).position;
    moved[i].forEach((value,axis)=>assert.ok(Math.abs(value-expected[axis])<1e-12));
  }
  assert.throws(()=>new InertialFrame({relativeVelocity:[[0,0,0],[1,0,0]],c:1}),/magnitude must be less than c/);
});
test("light cone intersects an affine hyperplane and filters its branch",()=>{
  const cone=new LightCone([0,0,0,0],{branch:"future"});
  const result=cone.intersectHypersurface([1,0,0,0],2,[[1,0,0],[0,1,0]]);
  assert.deepEqual(result.hitMask,[true,true]);assert.deepEqual(result.events[0],[2,2,0,0]);
  assert.throws(()=>new LightCone([0,0,0,0]).intersectHypersurface([1,0,0,0],2,[[1,0,0]]),RangeError);
});
test("light-cone hyperplane offset is relative to a translated apex",()=>{
  const apex=[7,3,-2,1],normal=[2,.5,-.25,0],directions=[[1,1,0],[-1,2,1]];
  for(const [branch,offset] of [["future",3],["past",-3]]){
    const {events,hitMask}=new LightCone(apex,{branch}).intersectHypersurface(normal,offset,directions);
    assert.deepEqual(hitMask,[true,true]);
    for(const event of events){
      const delta=event.map((value,index)=>value-apex[index]);
      assert.ok(Math.abs(delta.reduce((sum,value,index)=>sum+normal[index]*value,0)-offset)<1e-12);
      assert.ok(Math.abs(-(delta[0]**2)+delta[1]**2+delta[2]**2+delta[3]**2)<1e-12);
      assert.ok(branch==="future"?delta[0]>0:delta[0]<0);
    }
  }
});
test("invalid spatial bases and non-finite worldline parameters reject",()=>{
  assert.throws(()=>new SpatialSlice([0,0,0,0],[1,0,0,0],{basis:[[0,1,0,0],[0,1,0,0],[0,0,0,1]]}),RangeError);
  const seg=new WorldlineSegment([0,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]);assert.throws(()=>seg.evaluate(Number.NaN),RangeError);
  const batchedEvent=new Event4([[0,0,0,0],[1,0,0,0]]);
  assert.throws(()=>new WorldlineSegment(batchedEvent,[1,0,0,0],[1,0,0,0],[1,0,0,0]),/start event/);
  assert.throws(()=>new WorldlineSegment([0,0,0,0],batchedEvent,[1,0,0,0],[1,0,0,0]),/end event/);
});
