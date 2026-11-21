import test from "node:test";
import assert from "node:assert/strict";
import {LorentzTransform,Matrix4,Tensor4,Vector4,FourVelocity,FourMomentum,EMFieldTensor,minkowskiDot} from "../src/relativity/index.js";

function seededRandom(seed){
  let state=seed>>>0;
  return ()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return state/0x1_0000_0000;};
}

test("seeded Lorentz transforms preserve random Minkowski products",()=>{
  const random=seededRandom(0x51e1a57),sample=()=>Array.from({length:4},()=>2*random()-1);
  for(let trial=0;trial<64;trial++){
    const beta=sample().slice(1).map(value=>value*.25),left=new Vector4(sample()),right=new Vector4(sample()),transform=LorentzTransform.boost(beta);
    const before=minkowskiDot(left.components,right.components),after=minkowskiDot(transform.applyVector(left).components,transform.applyVector(right).components);
    assert.ok(transform.validate({atol:2e-12,rtol:2e-12}));
    assert.ok(Math.abs(after-before)<=2e-12*Math.max(1,Math.abs(before)));
  }
});

test("seeded nonsingular diagonal matrices invert and round-trip random vectors",()=>{
  const random=seededRandom(0x4d415452),sample=()=>Array.from({length:4},()=>2*random()-1);
  for(let trial=0;trial<64;trial++){
    const diagonal=Array.from({length:4},()=>.25+random()*3),matrix=new Matrix4(diagonal.map((value,index)=>diagonal.map((_,column)=>index===column?value:0))),input=sample(),output=matrix.inverse().apply(matrix.apply(input)).components;
    output.forEach((value,index)=>assert.ok(Math.abs(value-input[index])<2e-12));
  }
});

test("seeded rank-three tensors survive reversed axis permutation twice",()=>{
  const random=seededRandom(0x54454e53),values=Array.from({length:4},()=>Array.from({length:4},()=>Array.from({length:4},()=>2*random()-1))),tensor=new Tensor4(values,["up","down","up"]),restored=tensor.permuteIndices([2,1,0]).permuteIndices([2,1,0]);
  assert.deepEqual(restored.indexTypes,tensor.indexTypes);
  assert.deepEqual(restored.components,tensor.components);
});

test("seeded four-velocity and momentum factories preserve invariant mass",()=>{
  const random=seededRandom(0x4d4f4d45),sample3=()=>Array.from({length:3},()=>.6*random()-.3);
  for(let trial=0;trial<64;trial++){
    const velocity=sample3(),mass=.1+random()*4,fourVelocity=FourVelocity.fromVelocity3(velocity,1),momentum=FourMomentum.fromMassVelocity(mass,velocity,1);
    assert.equal(fourVelocity.validate(),true);
    assert.ok(Math.abs(momentum.invariantMass()-mass)<2e-12*Math.max(1,mass));
  }
});

test("seeded general boosts preserve four-momentum invariant mass",()=>{
  const random=seededRandom(0x4d415353),sample3=()=>Array.from({length:3},()=>.5*random()-.25);
  for(const c of [1e-100,1,1e100])for(let trial=0;trial<32;trial++){
    const mass=.2+random()*3,velocity=sample3().map(value=>value*c),transform=LorentzTransform.boost(sample3());
    const momentum=FourMomentum.fromMassVelocity(mass,velocity,c),boosted=momentum.boost(transform),restored=boosted.boost(transform.inverse());
    assert.ok(Math.abs(boosted.invariantMass()-mass)<2e-11*Math.max(1,mass));
    restored.components.forEach((value,index)=>assert.ok(Math.abs((value-momentum.components[index])/c)<2e-11*Math.max(1,mass)));
  }
});

test("seeded electromagnetic fields retain both invariants under Lorentz boosts",()=>{
  const random=seededRandom(0x454d4649),sample3=()=>Array.from({length:3},()=>2*random()-1);
  for(let trial=0;trial<64;trial++){
    const field=EMFieldTensor.fromEB(sample3(),sample3(),{c:1}),transform=LorentzTransform.boost(sample3().map(value=>value*.2)),before=field.invariants(),after=field.transform(transform.matrix).invariants();
    for(const name of ["magneticMinusElectricOverC2","electricDotMagnetic"])assert.ok(Math.abs(before[name]-after[name])<2e-11*Math.max(1,Math.abs(before[name])));
  }
});
