import test from "node:test";
import assert from "node:assert/strict";
import {Vector4,Covector4,Matrix4,Metric4,Tensor4,Bivector4,LeviCivita4,LorentzTransform,PoincareTransform,minkowskiDot,fromThreeVector3,toThreeVector3,fromThreeVector4,toThreeVector4,fromThreeMatrix4,toThreeMatrix4} from "../src/relativity/index.js";
import * as publicApi from "../src/relativity/index.js";

test("public entry point exposes validated products without internal primitives",()=>{
  assert.equal(typeof publicApi.minkowskiDot,"function");
  assert.equal("weightedDot4" in publicApi,false);
  assert.equal("metricDot4" in publicApi,false);
});

test("algebra inputs reject sparse components and sparse batches",()=>{
  const vector=[1,2,3,4];delete vector[2];
  const rows=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];delete rows[1][2];
  const batch=[[1,0,0,0],[0,1,0,0]];delete batch[1];
  assert.throws(()=>new Vector4(vector),/finite/);
  assert.throws(()=>new Matrix4(rows),/finite/);
  assert.throws(()=>new Vector4(batch),/finite/);
  assert.throws(()=>minkowskiDot(vector,[1,0,0,0]),/finite/);
  assert.throws(()=>minkowskiDot(batch,[1,0,0,0]),/finite/);
});

test("identity and Minkowski factories preflight sparse batch shapes",()=>{
  const sparse=[62500,1,2];delete sparse[1];
  for(const factory of [options=>Matrix4.identity(options),options=>Metric4.minkowski(options),options=>LorentzTransform.identity(options),options=>PoincareTransform.identity(options)]){
    assert.throws(()=>factory({batchShape:sparse}),/batchShape/);
    assert.doesNotThrow(()=>factory({batchShape:[2,1]}));
  }
});
test("broadcast algebra rejects oversized outputs before allocation",()=>{
  const column=Array.from({length:1001},()=>[[1,0,0,0]]);
  const row=[Array.from({length:1001},()=>[1,0,0,0])];
  const vectors=new Vector4(column),other=new Vector4(row),covectors=new Covector4(column);
  assert.throws(()=>vectors.add(other),/component limit/);
  assert.throws(()=>vectors.subtract(other),/component limit/);
  assert.throws(()=>vectors.scale([Array(1001).fill(1)]),/component limit/);
  assert.throws(()=>vectors.dot(other),/component limit/);
  assert.throws(()=>covectors.add(new Covector4(row)),/component limit/);
  assert.throws(()=>covectors.scale([Array(1001).fill(1)]),/component limit/);
  assert.throws(()=>covectors.pair(other),/component limit/);
  const smallColumn=new Vector4(Array.from({length:501},()=>[[2,0,0,0]]));
  const metricGrid=Metric4.minkowski({batchShape:[1,501]});
  assert.throws(()=>smallColumn.normalizeTimelike(metricGrid),/component limit/);
});
test("bivector invariants and rapidity boosts bound broadcast outputs",()=>{
  const field=Array.from({length:708},()=>[[[0,1,0,0],[-1,0,0,0],[0,0,0,0],[0,0,0,0]]]);
  const metrics=Metric4.minkowski({batchShape:[1,708]});
  assert.throws(()=>new Bivector4(field).invariants(metrics),/component limit/);
  const directions=Array.from({length:251},()=>[[1,0,0]]),rapidities=[Array(251).fill(.1)];
  assert.throws(()=>LorentzTransform.fromRapidity(rapidities,directions),/component limit/);
});
test("bivector invariants reject non-finite contractions",()=>{
  const field=new Bivector4([[0,1e200,0,0],[-1e200,0,0,0],[0,0,0,0],[0,0,0,0]]);
  assert.throws(()=>field.invariants(),/invariants are non-finite/);
});

test("tensor variance and permutation require every index entry",()=>{
  const components=Array.from({length:4},()=>[1,0,0,0]);
  const sparseTypes=["up","down"];delete sparseTypes[0];
  const sparseOrder=[0,1];delete sparseOrder[0];
  assert.throws(()=>new Tensor4(components,sparseTypes),/up\/down/);
  assert.throws(()=>new Tensor4(components,["up","down"]).permuteIndices(sparseOrder),/permutation/);
  assert.throws(()=>LorentzTransform.boost([0,0,0]).applyTensor(components,sparseTypes),/index types/);
});
test("vector normalization scales overflowing squared norms",()=>{
  assert.deepEqual(new Vector4([1e200,0,0,0]).normalizeTimelike().components,[1,0,0,0]);
  assert.deepEqual(new Vector4([0,1e200,0,0]).normalizeSpacelike().components,[0,1,0,0]);
  assert.deepEqual(new Vector4([[2,0,0,0],[1e200,0,0,0]]).normalizeTimelike().components,[[1,0,0,0],[1,0,0,0]]);
  const metric=new Metric4([[-4,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
  assert.deepEqual(new Vector4([1e200,0,0,0]).normalizeTimelike(metric).components,[.5,0,0,0]);
  assert.throws(()=>new Vector4([1e200,1e200,0,0]).normalizeTimelike(),/not timelike/);
});
test("vector normalization recovers underflowed squared norms at zero tolerance",()=>{
  const metric=new Metric4();
  assert.deepEqual(new Vector4([1e-200,0,0,0]).normalizeTimelike(metric,0).components,[1,0,0,0]);
  assert.deepEqual(new Vector4([0,1e-200,0,0]).normalizeSpacelike(metric,0).components,[0,1,0,0]);
  assert.deepEqual(new Vector4([[1e-200,0,0,0],[2,0,0,0]]).normalizeTimelike(metric,0).components,[[1,0,0,0],[1,0,0,0]]);
  assert.throws(()=>new Vector4([1e-200,0,0,0]).normalizeTimelike(),/not timelike/);
  assert.throws(()=>new Vector4([0,0,0,0]).normalizeTimelike(metric,0),/not timelike/);
  assert.throws(()=>new Vector4([1e-200,1e-200,0,0]).normalizeTimelike(metric,0),/indeterminate/);
});
test("vector normalization avoids loss of precision in subnormal squared norms",()=>{
  const metric=new Metric4();
  assert.deepEqual(new Vector4([1e-160,0,0,0]).normalizeTimelike(metric,0).components,[1,0,0,0]);
  assert.deepEqual(new Vector4([0,1e-160,0,0]).normalizeSpacelike(metric,0).components,[0,1,0,0]);
  assert.throws(()=>new Vector4([1e-160,0,0,0]).normalizeTimelike(metric,1e-319),/not timelike/);
  assert.deepEqual(new Vector4([1e-160,0,0,0]).normalizeTimelike(metric,1e-321).components,[1,0,0,0]);
});

test("metric inverse and raising reflect mutable metric components",()=>{
  const metric=new Metric4();
  metric.g[1][1]=4;
  assert.equal(metric.inverse.components[1][1],.25);
  assert.deepEqual(metric.raiseCovector(new Covector4([0,4,0,0])).components,[0,1,0,0]);
  metric.g[1][1]=0;
  assert.throws(()=>metric.inverse,/singular/);
  metric.g=[[-1,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,1]];
  assert.equal(metric.components,metric.g);
  assert.equal(metric.inverse.components[1][1],.5);
  assert.throws(()=>{metric.g=[metric.g];},/batch shape cannot change/);
});

test("small coordinate scales do not make a nonsingular metric singular",()=>{
  const metric=new Metric4([[-1,0,0,0],[0,1e-16,0,0],[0,0,1,0],[0,0,0,1]]);
  assert.equal(metric.determinant(),-1e-16);
  assert.equal(metric.inverse.components[1][1],1e16);
  assert.deepEqual(metric.raiseCovector(new Covector4([0,1e-16,0,0])).components,[0,1,0,0]);
  const tiny=new Metric4([[-1e-100,0,0,0],[0,1e-100,0,0],[0,0,1e-100,0],[0,0,0,1e-100]]);
  assert.equal(Math.abs(tiny.determinant()),0); // The product underflows even though each pivot is nonzero.
  assert.equal(tiny.inverse.components[1][1],1e100);
  const epsilon=new LeviCivita4(tiny);
  assert.ok(Math.abs(epsilon.epsilon("down")[0][1][2][3] / -1e-200 - 1)<1e-12);
  assert.ok(Math.abs(epsilon.epsilon("up")[0][1][2][3] / 1e200 - 1)<1e-12);
});

test("determinant avoids intermediate overflow and underflow when the final value is finite",()=>{
  for(const diagonal of [[1e200,1e200,1e-200,1e-200],[1e-200,1e-200,1e200,1e200]]){
    const matrix=new Matrix4(diagonal.map((value,i)=>diagonal.map((_,j)=>i===j?value:0)));
    assert.ok(Math.abs(matrix.determinant()-1)<1e-12);
  }
});

test("Three.js adapters copy vector values and transpose Matrix4 storage conventions",()=>{
  class Vector3 {constructor(x,y,z){Object.assign(this,{x,y,z});}}
  class ThreeVector4 {constructor(x,y,z,w){Object.assign(this,{x,y,z,w});}}
  class ThreeMatrix4 {set(...values){this.rowMajor=values;this.elements=Array.from({length:16},(_,index)=>values[(index%4)*4+Math.floor(index/4)]);return this;}}
  const THREE={Vector3,Vector4:ThreeVector4,Matrix4:ThreeMatrix4};
  const three3=new Vector3(2,3,4),spatial=fromThreeVector3(three3);
  assert.deepEqual(spatial,[2,3,4]);assert.deepEqual(toThreeVector3(spatial,THREE),three3);
  const four=fromThreeVector4(new ThreeVector4(1,2,3,4));
  assert.deepEqual(four.components,[1,2,3,4]);assert.deepEqual(toThreeVector4(four,THREE),new ThreeVector4(1,2,3,4));
  const rows=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],threeMatrix={elements:[1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16]};
  assert.deepEqual(fromThreeMatrix4(threeMatrix).components,rows);
  const converted=toThreeMatrix4(new Matrix4(rows),THREE);
  assert.deepEqual(converted.rowMajor,rows.flat());assert.deepEqual(converted.elements,threeMatrix.elements);
  assert.throws(()=>toThreeMatrix4(new Matrix4([rows,rows]),THREE),/single 4 by 4/);
});
test("Three.js adapters reject malformed and non-finite components",()=>{
  assert.throws(()=>fromThreeVector3({x:1,y:2,z:Infinity}),/finite/);
  assert.throws(()=>toThreeVector3([1,2],{}),/Vector3/);
  assert.throws(()=>fromThreeVector4({x:1,y:2,z:3,w:NaN}),/finite/);
  assert.throws(()=>fromThreeMatrix4({elements:Array(15).fill(0)}),/16 finite/);
  assert.throws(()=>toThreeMatrix4([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],{}),/Matrix4/);
  const THREE={Vector3:class {constructor(x=0,y=0,z=0){Object.assign(this,{x,y,z});}},Vector4:class {constructor(x=0,y=0,z=0,w=1){Object.assign(this,{x,y,z,w});}}};
  const sparse3=[1,2,3],sparse4=[1,2,3,4],sparseMatrix=Array(16).fill(0);delete sparse3[1];delete sparse4[3];delete sparseMatrix[5];
  assert.throws(()=>toThreeVector3(sparse3,THREE),/finite/);
  assert.throws(()=>toThreeVector4(sparse4,THREE),/finite/);
  assert.throws(()=>fromThreeMatrix4({elements:sparseMatrix}),/finite/);
  assert.deepEqual(toThreeVector3(new Float64Array([1,2,3]),THREE),new THREE.Vector3(1,2,3));
});

test("Minkowski norm classifies timelike, null, and spacelike vectors",()=>{
  assert.equal(Vector4.CAUSAL_TIMELIKE,-1);
  assert.equal(Vector4.CAUSAL_NULL,0);
  assert.equal(Vector4.CAUSAL_SPACELIKE,1);
  assert.equal(new Vector4([2,1,0,0]).causalType(),Vector4.CAUSAL_TIMELIKE);
  assert.equal(new Vector4([1,1,0,0]).causalType(),Vector4.CAUSAL_NULL);
  assert.equal(new Vector4([1,2,0,0]).causalType(),Vector4.CAUSAL_SPACELIKE);
  assert.equal(new Vector4([2,1,0,0]).causalName(),"timelike");
  assert.equal(new Vector4([1,1,0,0]).causalName(),"null");
  assert.equal(new Vector4([1,2,0,0]).causalName(),"spacelike");
  assert.equal(new Vector4([1e150,9e149,0,0]).causalType(),Vector4.CAUSAL_TIMELIKE);
  assert.throws(()=>new Vector4([1,0,0,0]).causalType(undefined,-1),RangeError);
  assert.equal(new Vector4([1e308,1e308,0,0]).causalType(),Vector4.CAUSAL_NULL);
  assert.throws(()=>new Vector4([1e308,0,0,0]).causalType(),TypeError);
});
test("Minkowski dot broadcasts bounded vector batches",()=>{
  assert.deepEqual(minkowskiDot([[1,2,3,4],[2,0,0,1]],[2,1,0,0]),[0,-4]);
  const a=[[[1,0,0,0]],[[0,1,0,0]]],b=[[1,1,0,0],[2,0,0,0],[0,0,3,0]];
  assert.deepEqual(minkowskiDot(a,b),[[-1,-2,0],[1,0,0]]);
  assert.throws(()=>minkowskiDot([[1,0,0,0],[1,0,0,0],[1,0,0,0]],[[1,0,0,0],[1,0,0,0]]),/broadcast-compatible/);
  assert.throws(()=>minkowskiDot([[1,0,0,0],[1,0]], [1,0,0,0]),/rectangular/);
});
test("Lorentz boosts preserve inner products and reject superluminal beta",()=>{
  const a=new Vector4([4,1,2,0]),b=new Vector4([3,-1,0,2]),L=LorentzTransform.boost([.6,.2,0]);
  assert.ok(L.validate());assert.ok(Math.abs(a.dot(b)-L.applyVector(a).dot(L.applyVector(b)))<1e-10);
  assert.throws(()=>LorentzTransform.boost([1,0,0]),RangeError);
  const tiny=LorentzTransform.boost([1e-8,1e-8,0]);
  assert.ok(Math.abs(tiny.matrix[1][2]-5e-17)<1e-30);
});
test("Lorentz boosts retain linear terms when squared speed underflows",()=>{
  for(const beta of [1e-200,-1e-200,Number.MIN_VALUE]){
    const boost=LorentzTransform.boost([beta,0,0]);
    assert.equal(boost.matrix[0][1],-beta);
    assert.equal(boost.matrix[1][0],-beta);
    assert.equal(boost.applyVector([1,0,0,0]).components[1],-beta);
  }
  assert.deepEqual(LorentzTransform.boost([0,0,0]).matrix,LorentzTransform.identity().matrix);
});
test("matrix inverse and metric index raising/lowering round-trip",()=>{
  const m=new Matrix4([[2,0,0,0],[0,1,0,0],[0,0,3,0],[0,0,0,1]]),x=[2,3,4,5];
  assert.deepEqual(m.inverse().apply(m.apply(x)).components,x);
  const g=Metric4.minkowski(),v=new Vector4([2,1,2,3]),w=g.raiseCovector(g.lowerVector(v));
  assert.deepEqual(w.components,v.components);
});
test("Matrix4 addition, multiplication, transpose, determinant, and application",()=>{
  const a=new Matrix4([[2,0,0,0],[0,3,0,0],[0,0,4,0],[0,0,0,5]]),b=Matrix4.identity();
  assert.deepEqual(a.add(b).components,[[3,0,0,0],[0,4,0,0],[0,0,5,0],[0,0,0,6]]);
  assert.deepEqual(a.multiply(b).components,a.components);
  assert.deepEqual(a.transpose().components,a.components);
  assert.equal(a.determinant(),120);
  assert.deepEqual(a.apply([1,2,3,4]).components,[2,6,12,20]);
});
test("Matrix4 broadcasts bounded matrix/vector batches through operations",()=>{
  const matrices=new Matrix4([
    [[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    [[1,0,0,0],[0,3,0,0],[0,0,1,0],[0,0,0,1]]
  ]);
  const identityBatch=Matrix4.identity({batchShape:[1]});
  assert.deepEqual(matrices.add(identityBatch).components,matrices.components.map(matrix=>new Matrix4(matrix).add(Matrix4.identity()).components));
  assert.deepEqual(matrices.multiply(identityBatch).components,matrices.components);
  assert.deepEqual(matrices.transpose().components,matrices.components);
  assert.deepEqual(matrices.determinant(),[2,3]);
  assert.deepEqual(matrices.inverse().components,matrices.components.map(matrix=>new Matrix4(matrix).inverse().components));
  const vectors=new Vector4([[1,2,3,4],[4,3,2,1]]);
  assert.deepEqual(matrices.apply(vectors).components,matrices.components.map((matrix,index)=>new Matrix4(matrix).apply(vectors.components[index]).components));
  assert.deepEqual(matrices.apply(new Vector4([1,0,0,0])).components,[[2,0,0,0],[1,0,0,0]]);
  assert.deepEqual(Matrix4.identity({batchShape:[2,1]}).batchShape,[2,1]);
  assert.throws(()=>matrices.multiply(new Matrix4([[],[]])),/shape/);
});
test("metrics reject nonsymmetric and degenerate components",()=>{
  assert.throws(()=>new Metric4([[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),/symmetric/);
  assert.throws(()=>new Metric4([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]]),/singular/);
});
test("rank-two tensor index conversion and permutation retain values",()=>{
  const t=new Tensor4([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],["up","down"]);
  assert.equal(t.permuteIndices([1,0]).components[1][0],2);
  const q=t.lowerIndex(0).raiseIndex(0);assert.deepEqual(q.components,t.components);
});
test("covectors and mixed tensors transform with the correct index variance",()=>{
  const L=LorentzTransform.boost([.35,-.1,0]);
  const alpha=new Covector4([2,1,-3,.5]),v=new Vector4([4,-1,2,3]);
  assert.ok(Math.abs(alpha.transform(L.matrix).pair(L.applyVector(v))-alpha.pair(v))<1e-10);
  const identityMixed=new Tensor4([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],["up","down"]);
  const transformed=identityMixed.transform(L.matrix);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(transformed.components[i][j]-(i===j?1:0))<1e-10);
});
test("vector and tensor outer products plus tensor symmetrization",()=>{
  const a=new Vector4([1,2,0,0]),b=new Vector4([3,4,0,0]);
  const outer=a.outer(b);
  assert.equal(outer.components[0][1],4);
  assert.deepEqual(outer.indexTypes,["up","up"]);
  const t=new Tensor4([[1,2,0,0],[4,5,0,0],[0,0,0,0],[0,0,0,0]],["down","down"]);
  assert.equal(t.symmetrize().components[0][1],3);
  assert.equal(t.antisymmetrize().components[0][1],-1);
});
test("antisymmetric bivector validates and wedge is antisymmetric",()=>{
  const a=new Vector4([1,2,0,1]),b=new Vector4([0,1,3,0]),w=a.wedge(b);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(w.components[i][j]+w.components[j][i])<1e-12);
  assert.throws(()=>new Bivector4([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),RangeError);
});
test("bivector dual and pseudoscalar use the metric volume form",()=>{
  const field=new Bivector4([[0,1,0,0],[-1,0,0,0],[0,0,0,2],[0,0,-2,0]]);
  const invariants=field.invariants();
  assert.equal(invariants.quadratic,3);
  assert.equal(invariants.pseudoscalar,-4);
  const squaredDual=field.dual().dual().components;
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(squaredDual[i][j]+field.components[i][j])<1e-12);
});
test("bivector dual and invariant lower both indices with the supplied matrix",()=>{
  const field=new Bivector4([[0,1,0,0],[-1,0,0,0],[0,0,0,0],[0,0,0,0]]);
  const matrix=a=>[[-1,a,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const metric=new Metric4([matrix(.5),matrix(-.25)],{validate:false});
  const invariants=field.invariants(metric),dual=field.dual(metric).components;
  assert.deepEqual(invariants.quadratic,[-1,-1]);
  assert.deepEqual(invariants.pseudoscalar,[0,0]);
  for(const value of dual){assert.equal(value[2][3],-1);assert.equal(value[3][2],1);}
});
test("bivector contraction and Hodge dual use custom Lorentz metrics",()=>{
  const field=new Bivector4([[0,1,0,0],[-1,0,0,0],[0,0,0,2],[0,0,-2,0]]);
  assert.deepEqual(field.contract([1,2,0,0]).components,[2,-1,0,0]);
  const metric=new Metric4([[-1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]]);
  const twice=field.dual(metric).dual(metric).components;
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(twice[i][j]+field.components[i][j])<1e-12);
});
test("bivector batches broadcast through wedge, contraction, dual, transform, and invariants",()=>{
  const a=new Vector4([[1,0,0,0],[0,1,0,0]]),b=new Vector4([[0,1,0,0],[0,0,1,0]]),field=a.wedge(b);
  assert.equal(field.components.length,2);
  const contracted=field.contract(new Covector4([[2,3,0,0],[0,4,5,0]]));assert.deepEqual(contracted.components,[[3,-2,0,0],[0,5,-4,0]]);
  const twice=field.dual().dual().components;for(let n=0;n<2;n++)for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(twice[n][i][j]+field.components[n][i][j])<1e-12);
  const matrices=[LorentzTransform.boost([.1,0,0]).matrix,LorentzTransform.boost([0,.2,0]).matrix],transformed=field.transform(matrices);
  for(let n=0;n<2;n++)assert.deepEqual(transformed.components[n],new Bivector4(field.components[n]).transform(matrices[n]).components);
  const invariants=field.invariants();for(let n=0;n<2;n++){const scalar=new Bivector4(field.components[n]).invariants();assert.equal(invariants.quadratic[n],scalar.quadratic);assert.equal(invariants.pseudoscalar[n],scalar.pseudoscalar);}
  assert.throws(()=>new Bivector4([field.components[0],[[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]]),/antisymmetric/);
  const cyclic=[];cyclic.push(cyclic);assert.throws(()=>new Bivector4(cyclic),/cycles/);
});
test("Minkowski dot accepts plain arrays",()=>assert.equal(minkowskiDot([2,1,0,0],[3,0,0,0]),-6));
test("Minkowski dot recovers extreme products and preserves overflow signaling",()=>{
  assert.equal(minkowskiDot([1e308,1e308,1,0],[1e308,1e308,1,0]),1);
  assert.deepEqual(minkowskiDot([[1e308,1e308,1,0],[0,0,2,0]],[1e308,1e308,1,0]),[1,2]);
  assert.equal(minkowskiDot([1e308,1e308,1,0],[1e308,1e308,1,0],[-2,2,3,1]),3);
  assert.ok(Math.abs(minkowskiDot([1e-100,0,0,0],[1e100,0,0,0],[1e-300,1,1,1])/1e-300-1)<1e-15);
  assert.ok(Math.abs(minkowskiDot([1e200,0,0,0],[1e-200,0,0,0],[1e200,1,1,1])/1e200-1)<1e-15);
  assert.equal(minkowskiDot([0,Number.MIN_VALUE,Number.MIN_VALUE,Number.MIN_VALUE],[0,.75,.75,.75]),2*Number.MIN_VALUE);
  assert.equal(minkowskiDot([0,1e308,0,0],[0,1e308,0,0]),Infinity);
});
test("Vector4 diagonal-metric products recover representable extreme cancellation",()=>{
  const vector=new Vector4([1e308,1e308,1,0]);
  assert.equal(vector.dot(vector),1);
  assert.equal(vector.normSquared(),1);
  assert.equal(vector.causalType(),Vector4.CAUSAL_SPACELIKE);
  const diagonal=entries=>entries.map((value,i)=>entries.map((_,j)=>i===j?value:0));
  const metrics=new Metric4([diagonal([-1,1,1,1]),diagonal([-2,2,3,1])]);
  assert.deepEqual(vector.dot(vector,metrics),[1,3]);
  assert.equal(new Vector4([0,1e308,0,0]).dot(new Vector4([0,1e308,0,0])),Infinity);
});
test("Vector4 non-diagonal metric products recover extreme cancellations and underflow",()=>{
  const matrix=[[-1,.5,0,0],[.5,0,0,0],[0,0,1,0],[0,0,0,1]],metric=new Metric4(matrix);
  const vector=new Vector4([1e308,1e308,1,0]);
  assert.equal(vector.dot(vector,metric),1);
  assert.equal(metric.dot(vector,vector),1);
  const batch=new Metric4([matrix,[[-2,1,0,0],[1,0,0,0],[0,0,3,0],[0,0,0,1]]]);
  assert.deepEqual(vector.normSquared(batch),[1,3]);
  const value=new Vector4([0,1e100,0,0]).dot(new Vector4([Number.MIN_VALUE,0,0,0]),metric);
  assert.ok(Math.abs(value/(Number.MIN_VALUE*1e100*.5)-1)<1e-15);
});
test("Vector4 shape and Covector4 broadcast algebra, causal classification, and index maps",()=>{
  const values=new Vector4([[2,1,0,0],[0,2,0,0]]),offset=new Vector4([1,0,0,0]);
  assert.deepEqual(values.shape,[2,4]);assert.deepEqual(offset.shape,[4]);assert.ok(Object.isFrozen(values.shape));
  assert.deepEqual(values.add(offset).components,[[3,1,0,0],[1,2,0,0]]);
  assert.deepEqual(values.subtract(offset).components,[[1,1,0,0],[-1,2,0,0]]);
  assert.deepEqual(values.scale([2,.5]).components,[[4,2,0,0],[0,1,0,0]]);
  assert.deepEqual(values.dot(values),[-3,4]);
  assert.deepEqual(values.causalType(),[Vector4.CAUSAL_TIMELIKE,Vector4.CAUSAL_SPACELIKE]);
  assert.deepEqual(values.causalName(),["timelike","spacelike"]);
  const timelike=new Vector4([[2,1,0,0],[3,1,0,0]]).normalizeTimelike(),spacelike=new Vector4([[0,2,0,0],[0,3,0,0]]).normalizeSpacelike();
  assert.ok(timelike.normSquared().every(value=>Math.abs(value+1)<1e-12));assert.ok(spacelike.normSquared().every(value=>Math.abs(value-1)<1e-12));
  assert.throws(()=>values.normalizeTimelike(),/not timelike/);assert.throws(()=>values.normalizeSpacelike(),/not spacelike/);
  const lower=values.lowerIndex(),raised=lower.raiseIndex();assert.deepEqual(raised.components,values.components);
  assert.deepEqual(lower.pair(values),[-3,4]);
  const geometry=Metric4.minkowski(),metricLower=geometry.lowerVector(values);assert.deepEqual(metricLower.components,lower.components);assert.deepEqual(geometry.raiseCovector(lower).components,values.components);assert.deepEqual(geometry.dot(values,values),[-3,4]);
  const transforms=[LorentzTransform.boost([.1,0,0]).matrix,LorentzTransform.boost([0,.2,0]).matrix],mapped=values.transform(transforms);
  for(let i=0;i<2;i++)assert.deepEqual(mapped.components[i],new LorentzTransform(transforms[i]).applyVector(values.components[i]).components);
  assert.equal(lower.transform(transforms[0]).components.length,2);
  const outer=values.outer(offset);assert.deepEqual(outer.batchShape,[2]);assert.deepEqual(outer.components[0][0][0],2);
});
test("proper spatial rotations embed in Lorentz group and reflections reject",()=>{
  const angle=.4,c=Math.cos(angle),s=Math.sin(angle),L=LorentzTransform.rotation([[c,-s,0],[s,c,0],[0,0,1]]);
  assert.ok(L.validate());assert.throws(()=>LorentzTransform.rotation([[-1,0,0],[0,1,0],[0,0,1]]),RangeError);
  const nearRotation=[[1+0.75e-10,0,0],[0,1,0],[0,0,1]];
  assert.ok(LorentzTransform.rotation(nearRotation).validate());
  assert.throws(()=>LorentzTransform.rotation(nearRotation,{atol:0}),RangeError);
  assert.throws(()=>LorentzTransform.rotation(nearRotation,{atol:-1}),RangeError);
});
test("rapidity boosts agree with beta boosts",()=>{
  const beta=.4,rapid=Math.atanh(beta);
  const fromRapidity=LorentzTransform.fromRapidity(rapid,[1,0,0]),fromBeta=LorentzTransform.boost([beta,0,0]);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(fromRapidity.matrix[i][j]-fromBeta.matrix[i][j])<1e-12);
  assert.throws(()=>LorentzTransform.fromRapidity([1,0,0],rapid),/direction must end in shape/);
});
test("rapidity direction normalization handles extreme finite scales",()=>{
  const reference=LorentzTransform.fromRapidity(.5,[1,1,1]).matrix;
  for(const magnitude of [1.7e308,Number.MIN_VALUE]){
    const actual=LorentzTransform.fromRapidity(.5,[magnitude,magnitude,magnitude]).matrix;
    for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(actual[i][j]-reference[i][j])<1e-15);
  }
  assert.throws(()=>LorentzTransform.fromRapidity(.5,[0,0,0]),/nonzero/);
});
test("raw Lorentz tensor transforms use the source inverse convention",()=>{
  const L=new LorentzTransform([[2,3,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]]);
  const input=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  assert.equal(L.validate(),false);
  assert.deepEqual(L.applyTensor(input,["up","up"]),[[13,3,0,0],[3,1,0,0],[0,0,1,0],[0,0,0,0]]);
  assert.deepEqual(L.applyTensor(input,["down","down"]),[[13,-3,0,0],[-3,1,0,0],[0,0,1,0],[0,0,0,0]]);
  assert.deepEqual(L.applyTensor(input,["down","up"]),[[-5,-3,0,0],[3,1,0,0],[0,0,1,0],[0,0,0,0]]);
  assert.throws(()=>L.applyTensor(new Tensor4(input,["up","up"])),/singular/);
});
test("Poincare transforms preserve affine composition and inverse",()=>{
  const t=new PoincareTransform(LorentzTransform.boost([.2,0,0]),[3,4,0,0]),p=[1,2,3,4];
  const roundTrip=t.inverse().applyEvent(t.applyEvent({position:p})).position;
  roundTrip.forEach((x,i)=>assert.ok(Math.abs(x-p[i])<1e-10));
  const translation=new PoincareTransform(LorentzTransform.identity(),[3,-1,2,0]);
  assert.deepEqual(translation.applyEvent({position:p}).position,[4,1,5,4]);
  assert.deepEqual(translation.applyEvent(p).position,[4,1,5,4]);
  assert.deepEqual(translation.applyVector(p).components,p);
  assert.throws(()=>new PoincareTransform([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),/lorentz must be a LorentzTransform/);
});
test("Lib4 frozen transformation value objects reject reassignment",()=>{
  for(const value of [LorentzTransform.identity(),new PoincareTransform()]){
    const field=value instanceof LorentzTransform?"batchRank":"translation";
    assert.ok(Object.isFrozen(value));assert.equal(Reflect.set(value,field,"changed"),false);
  }
});
test("Lorentz and Poincare transforms broadcast bounded leading batches",()=>{
  const matrices=LorentzTransform.boost([[.2,0,0],[0,.3,0]]),events=[[1,2,3,4],[5,6,7,8]];
  assert.deepEqual(matrices.batchShape,[2]);assert.equal(matrices.validate(),true);
  const mapped=matrices.applyVector(events).components;
  for(let i=0;i<2;i++)LorentzTransform.boost(i===0?[.2,0,0]:[0,.3,0]).applyVector(events[i]).components.forEach((value,j)=>assert.ok(Math.abs(mapped[i][j]-value)<1e-12));
  const rapidity=LorentzTransform.fromRapidity([Math.atanh(.2),Math.atanh(.3)],[[1,0,0],[0,1,0]]),rotation=LorentzTransform.rotation([[[1,0,0],[0,1,0],[0,0,1]],[[0,-1,0],[1,0,0],[0,0,1]]]);
  assert.deepEqual(rapidity.batchShape,[2]);assert.equal(rotation.validate(),true);
  const tensors=[Metric4.minkowski().components,Metric4.minkowski().components],tensorResult=matrices.applyTensor(tensors,["down","up"]);
  for(let i=0;i<2;i++)LorentzTransform.boost(i===0?[.2,0,0]:[0,.3,0]).applyTensor(tensors[i],["down","up"]).forEach((row,r)=>row.forEach((value,c)=>assert.ok(Math.abs(tensorResult[i][r][c]-value)<1e-12)));
  assert.deepEqual(matrices.inverse().compose(matrices).matrix.map((m)=>m.map((row,i)=>row.map((x,j)=>Math.abs(x-(i===j?1:0))<1e-12))),[[[true,true,true,true],[true,true,true,true],[true,true,true,true],[true,true,true,true]],[[true,true,true,true],[true,true,true,true],[true,true,true,true],[true,true,true,true]]]);
  const a=new PoincareTransform(matrices,[[1,0,0,0],[0,2,0,0]]),b=new PoincareTransform(LorentzTransform.identity({batchShape:[2]}),[[0,1,0,0],[0,0,1,0]]),affine=a.compose(b),seqA=a.applyEvent(b.applyEvent(events)).position,seqB=affine.applyEvent(events).position,roundTrip=a.inverse().applyEvent(a.applyEvent(events)).position;
  seqB.forEach((sample,i)=>sample.forEach((value,j)=>assert.ok(Math.abs(value-seqA[i][j])<1e-12)));
  roundTrip.forEach((sample,i)=>sample.forEach((value,j)=>assert.ok(Math.abs(value-events[i][j])<1e-12)));
  assert.equal(LorentzTransform.identity({batchShape:[2]}).validate(),true);
  const invalid=[[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],unvalidated=new LorentzTransform(invalid);
  assert.equal(unvalidated.validate(),false);
  assert.throws(()=>new LorentzTransform(invalid,{validate:true}),/not a Lorentz transformation/);
  const nearIdentity=LorentzTransform.identity().matrix;nearIdentity[0][0]=1+4e-7;
  const near=new LorentzTransform(nearIdentity);
  assert.equal(near.validate({atol:0,rtol:1e-6}),true);
  assert.equal(near.validate({atol:0,rtol:1e-8}),false);
  const inverseBoost=LorentzTransform.boost([.6,0,0]).inverse().matrix,expectedInverse=LorentzTransform.boost([-.6,0,0]).matrix;
  inverseBoost.forEach((row,i)=>row.forEach((value,j)=>assert.ok(Math.abs(value-expectedInverse[i][j])<1e-12)));
});
test("mixed tensor contraction and index symmetries behave correctly",()=>{
  const identityMixed=new Tensor4([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],["up","down"]);
  assert.equal(identityMixed.contract(0,1),4);
  const symmetric=new Tensor4([[1,2,0,0],[2,3,0,0],[0,0,0,0],[0,0,0,0]],["down","down"]);
  assert.deepEqual(symmetric.antisymmetrize().components[0][1],0);
  assert.deepEqual(symmetric.antisymmetrize().indexTypes,["down","down"]);
});

test("mixed rank-three tensors preserve contraction under a non-orthogonal basis change",()=>{
  const vector=[1,2,3,4],components=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>vector.map(value=>i===j?value:0)));
  const tensor=new Tensor4(components,["up","down","up"]),shear=[[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const transformed=tensor.transform(shear),expectedVector=[3,2,3,4];
  for(let i=0;i<4;i++)for(let j=0;j<4;j++)for(let k=0;k<4;k++)assert.equal(transformed.components[i][j][k],i===j?expectedVector[k]:0);
  assert.deepEqual(transformed.contract(0,1).components,expectedVector.map(value=>4*value));
});
test("non-scalar tensor contraction preserves remaining axes and variance",()=>{
  const components=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>Array.from({length:4},(_,k)=>(i===j?k+1:0))));
  const contracted=new Tensor4(components,["up","down","up"]).contract(0,1);
  assert.ok(contracted instanceof Tensor4);
  assert.deepEqual(contracted.indexTypes,["up"]);
  assert.deepEqual(contracted.components,[4,8,12,16]);
});

test("Metric4 exposes its inverse as a metric value",()=>{
  const metric=new Metric4([[-1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]]);
  assert.ok(metric.inverse instanceof Metric4);
  assert.ok(metric.inverseMetric() instanceof Metric4);assert.deepEqual(metric.inverseMetric().components,metric.inverse.components);
  assert.ok(Math.abs(metric.inverse.components[2][2]-1/3)<1e-12);
  assert.deepEqual(metric.inverse.inverse.components,metric.components);
});
test("Metric4 broadcasts batched metrics through contractions, index maps, and determinant/inverse access",()=>{
  const metrics=new Metric4([
    [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    [[-1,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,1]]
  ]);
  assert.deepEqual(metrics.batchShape,[2]);
  assert.deepEqual(metrics.determinant(),[-1,-2]);
  assert.deepEqual(metrics.inverse.components[1][1][1],0.5);
  assert.deepEqual(metrics.inverseMetric().components,metrics.inverse.components);
  const vector=new Vector4([0,1,0,0]);
  assert.deepEqual(metrics.dot(vector,vector),[1,2]);
  const lowered=metrics.lowerVector(vector);
  assert.deepEqual(lowered.components,[[0,1,0,0],[0,2,0,0]]);
  assert.deepEqual(metrics.raiseCovector(lowered).components,[[0,1,0,0],[0,1,0,0]]);
  assert.deepEqual(metrics.intervalSquared(vector),[1,2]);
  assert.deepEqual(Metric4.minkowski({batchShape:[2,1]}).batchShape,[2,1]);
});
test("batched metrics broadcast with vector and tensor batch axes",()=>{
  const metrics=new Metric4([
    [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    [[-1,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,1]]
  ]);
  const vectors=new Vector4([[0,1,0,0],[0,3,0,0]]);
  assert.deepEqual(vectors.normSquared(metrics),[1,18]);
  assert.deepEqual(vectors.lowerIndex(metrics).components,[[0,1,0,0],[0,6,0,0]]);
  const components=[0,1].map(batch=>[[0,1,0,0]]);
  const tensor=new Tensor4(components,["up"]);
  const metricGrid=Metric4.minkowski({batchShape:[1,3]});
  const mapped=tensor.lowerIndex(0,metricGrid);
  assert.deepEqual(mapped.batchShape,[2,3]);
  assert.deepEqual(mapped.components[0][0],[0,1,0,0]);
  assert.deepEqual(mapped.components[0][2],[0,1,0,0]);
  assert.deepEqual(mapped.components[1][1],[0,1,0,0]);
});
test("batched metrics propagate through Levi-Civita and bivector metric operations",()=>{
  const metrics=new Metric4([
    [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    [[-1,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,1]]
  ]);
  const epsilon=new LeviCivita4(metrics).epsilon("down");
  assert.equal(epsilon.length,2);
  assert.equal(epsilon[0][0][1][2][3],-1);
  assert.ok(Math.abs(epsilon[1][0][1][2][3]+Math.sqrt(2))<1e-12);
  const field=new Bivector4([[0,1,0,0],[-1,0,0,0],[0,0,0,2],[0,0,-2,0]]);
  const dual=field.dual(metrics),roundTrip=dual.dual(metrics);
  assert.equal(dual.components.length,2);
  for(let n=0;n<2;n++)for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.ok(Math.abs(roundTrip.components[n][i][j]+field.components[i][j])<1e-12);
  const invariants=field.invariants(metrics);
  assert.deepEqual(invariants.quadratic,[3,2]);
  assert.equal(invariants.pseudoscalar.length,2);
});
test("bivector invariant fields retain multidimensional broadcast batch shapes",()=>{
  const f=[[0,1,0,0],[-1,0,0,0],[0,0,0,2],[0,0,-2,0]],zero=Array.from({length:4},()=>[0,0,0,0]);
  const fields=new Bivector4([[f],[zero]]),metrics=new Metric4([[[ -1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[-1,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,1]]]);
  const actual=fields.invariants(metrics);
  assert.deepEqual(actual.quadratic,[[3,2],[0,0]]);
  assert.equal(actual.pseudoscalar.length,2);
  for(let i=0;i<2;i++)for(let j=0;j<2;j++){
    const expected=new Bivector4(fields.components[i][0]).invariants(new Metric4(metrics.components[j]));
    assert.equal(actual.pseudoscalar[i][j],expected.pseudoscalar);
  }
});
test("covector transformations broadcast batched Lorentz matrices per sample",()=>{
  const values=new Covector4([[1,2,3,4],[5,6,7,8]]);
  const matrices=[LorentzTransform.boost([0.2,0,0]).matrix,LorentzTransform.boost([0,0.3,0]).matrix];
  const mapped=values.transform(matrices);
  assert.deepEqual(mapped.components,values.components.map((components,index)=>new Covector4(components).transform(matrices[index]).components));
  const broadcast=values.transform([matrices[0]]);
  assert.deepEqual(broadcast.components,values.components.map(components=>new Covector4(components).transform(matrices[0]).components));
  assert.throws(()=>values.transform([[[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]]]),/singular/);
});
test("Tensor4 resolves negative axes and rejects invalid duplicate axes",()=>{
  const t=new Tensor4([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],["up","down"]);
  assert.deepEqual(t.permuteIndices([1,0]).components[1][0],2);
  assert.deepEqual(t.lowerIndex(-2).raiseIndex(-2).components,t.components);
  assert.equal(t.contract(-2,-1),34);
  assert.throws(()=>t.contract(-1,-1),RangeError);
  assert.throws(()=>t.symmetrize(0,0),RangeError);
  assert.throws(()=>t.antisymmetrize(-1,-1),RangeError);
  assert.throws(()=>t.raiseIndex(0),TypeError);
});
test("Tensor4 supports bounded rank nine dense values and rejects rank ten before allocation",()=>{
  const nested=(rank,value=0)=>rank===1?Array(4).fill(value):Array.from({length:4},()=>nested(rank-1,value));
  const rankFive=new Tensor4(nested(5),Array(5).fill("up"));
  assert.equal(rankFive.rank,5);
  assert.equal(rankFive.permuteIndices([4,3,2,1,0]).rank,5);
  const rankNine=new Tensor4(nested(9),["up","down",...Array(7).fill("up")]);assert.equal(rankNine.rank,9);assert.equal(rankNine.contract(0,1).rank,7);
  assert.throws(()=>new Tensor4(nested(10),Array(10).fill("up")),/rank must not exceed nine/);
});
test("Tensor4 keeps batch axes separate across contractions, index operations, transforms, and outer products",()=>{
  const make=(factor)=>Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>i===j?factor*(i+1):factor*(i-j)));
  const values=new Tensor4([make(1),make(2)],["up","down"]);assert.equal(values.rank,2);assert.deepEqual(values.batchShape,[2]);
  assert.deepEqual(values.contract(0,1),[10,20]);
  const swapped=values.permuteIndices([1,0]);assert.deepEqual(swapped.indexTypes,["down","up"]);assert.deepEqual(swapped.batchShape,[2]);
  const symmetric=values.symmetrize(),antisymmetric=values.antisymmetrize();
  for(let n=0;n<2;n++)for(let i=0;i<4;i++)for(let j=0;j<4;j++){assert.equal(symmetric.components[n][i][j],symmetric.components[n][j][i]);assert.equal(Math.abs(antisymmetric.components[n][i][j]+antisymmetric.components[n][j][i]),0);}
  const raised=values.raiseIndex(1),roundtrip=raised.lowerIndex(1);assert.deepEqual(roundtrip.components,values.components);
  const matrices=[LorentzTransform.boost([.1,0,0]).matrix,LorentzTransform.boost([0,.2,0]).matrix],transformed=values.transform(matrices);
  for(let n=0;n<2;n++)assert.deepEqual(transformed.components[n],new Tensor4(values.components[n],values.indexTypes).transform(matrices[n]).components);
  const left=new Tensor4([[[1,0,0,0]],[[2,0,0,0]]],["up"]),right=new Tensor4([[[1,1,0,0],[2,0,0,0],[3,0,0,0]]],["down"]),outer=left.outer(right);
  assert.deepEqual(outer.batchShape,[2,3]);assert.deepEqual(outer.indexTypes,["up","down"]);assert.deepEqual(outer.components[1][2][0][0],6);
  const rankThree=new Tensor4(Array.from({length:2},(_,n)=>Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>Array.from({length:4},(_,k)=>i===j?(n+1)*(k+1):0)))),["up","down","up"]);
  const contracted=rankThree.contract(0,1);assert.deepEqual(contracted.batchShape,[2]);assert.deepEqual(contracted.components,[[4,8,12,16],[8,16,24,32]]);
  assert.throws(()=>new Tensor4([make(1),make(2)]),/indexTypes is required/);
  assert.throws(()=>new Tensor4([make(1),make(2)],["up","down","up"]),/end in shape/);
});

test("Tensor4 transformation requires an invertible matrix for every variance",()=>{
  const singular=[[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],identity=Matrix4.identity().components;
  const upper=new Tensor4([1,2,3,4],["up"]);
  assert.throws(()=>upper.transform(singular),/singular/);
  assert.throws(()=>upper.transform([identity,singular]),/singular/);
  const data=Array.from({length:4},()=>Array.from({length:4},()=>Array(4).fill(0)));data[1][2][3]=120;
  const matrix=[[2,0,0,0],[0,3,0,0],[0,0,4,0],[0,0,0,5]];
  const transformed=new Tensor4(data,["up","down","down"]).transform(matrix);
  assert.ok(Math.abs(transformed.components[1][2][3]-18)<1e-12);
  assert.deepEqual(transformed.indexTypes,["up","down","down"]);
});

test("Tensor4 rank is read-only and follows its index types",()=>{
  const tensor=new Tensor4([[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]],["up","down"]);
  assert.equal(tensor.rank,2);assert.throws(()=>{tensor.rank=9;},TypeError);
  assert.equal(tensor.rank,2);assert.equal(tensor.contract(0,1),10);
});

test("tensor symmetrization avoids overflowing representable averages",()=>{
  const matrix=[[1e308,1e308,0,0],[-1e308,-1e308,0,0],[0,0,0,0],[0,0,0,0]];
  const tensor=new Tensor4(matrix,["down","down"]),symmetric=tensor.symmetrize().components,antisymmetric=tensor.antisymmetrize().components;
  assert.equal(symmetric[0][0],1e308);assert.equal(symmetric[1][1],-1e308);assert.equal(symmetric[0][1],0);
  assert.equal(antisymmetric[0][1],1e308);assert.equal(antisymmetric[1][0],-1e308);
});
test("Levi-Civita orientation and tensor shape validation",()=>{
  const epsilon=new LeviCivita4().epsilon("down"),reversed=new LeviCivita4(undefined,-1).epsilon("down");
  assert.equal(epsilon[0][1][2][3],-1);assert.equal(reversed[0][1][2][3],1);
  assert.equal(new LeviCivita4().epsilon("up")[0][1][2][3],1);
  assert.throws(()=>new Tensor4([[1,2],[3,4]], ["up","down"]),RangeError);
});
test("Levi-Civita tensor scales with a custom metric determinant",()=>{
  const symbol=LeviCivita4.symbol();assert.equal(symbol[0][1][2][3],1);assert.equal(symbol[1][0][2][3],-1);assert.equal(symbol[0][0][2][3],0);
  const metric=new Metric4([[-1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]]),epsilon=new LeviCivita4(metric);
  const up=epsilon.epsilon("up")[0][1][2][3],down=epsilon.epsilon("down")[0][1][2][3];
  assert.ok(Math.abs(up-1/Math.sqrt(24))<1e-12);
  assert.ok(Math.abs(down+Math.sqrt(24))<1e-12);
  assert.ok(Math.abs(up*down+1)<1e-12);
  assert.deepEqual(epsilon.volumeForm(),epsilon.epsilon("down"));
  const bivector=new Bivector4([[0,1,2,3],[-1,0,4,5],[-2,-4,0,6],[-3,-5,-6,0]]);
  assert.deepEqual(epsilon.dual(bivector).components,bivector.dual(metric).components);
  assert.deepEqual(new LeviCivita4(metric,-1).dual(bivector).components,bivector.dual(metric,-1).components);
  assert.throws(()=>epsilon.epsilon("side"),TypeError);
  assert.throws(()=>new LeviCivita4(metric,0),RangeError);
});
