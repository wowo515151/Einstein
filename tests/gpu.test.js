import test from "node:test";
import assert from "node:assert/strict";
import {minkowskiDotBatch,samplePlaneWaveBatch,samplePlaneWaveGrid,WebGPUBackend} from "../src/relativity/index.js";

test("CPU batch dot product handles multiple events",async()=>{
  assert.deepEqual(await minkowskiDotBatch([1,2,0,0,2,1,0,0],[1,2,0,0,2,1,0,0],{backend:"cpu"}),[3,-3]);
});
test("CPU batch dot recovers extreme cancellation and underflowed products",async()=>{
  const cancellation=await minkowskiDotBatch([1e200,1e200,1,0],[1e200,1e200,1,0],{backend:"cpu"});
  assert.deepEqual(cancellation,[1]);
  const addition=await minkowskiDotBatch([1e308,1e308,1e308,0],[1,-1,1,0],{backend:"cpu"});
  assert.deepEqual(addition,[-1e308]);
  const small=await minkowskiDotBatch([0,Number.MIN_VALUE,Number.MIN_VALUE,Number.MIN_VALUE],[0,.5,.5,.5],{backend:"cpu"});
  assert.deepEqual(small,[2*Number.MIN_VALUE]);
  const subnormal=await minkowskiDotBatch([0,Number.MIN_VALUE,Number.MIN_VALUE,Number.MIN_VALUE],[0,.75,.75,.75],{backend:"cpu"});
  assert.deepEqual(subnormal,[2*Number.MIN_VALUE]);
  await assert.rejects(()=>minkowskiDotBatch([0,1e200,0,0],[0,1e200,0,0],{backend:"cpu"}),/dot product is non-finite/);
  const gpu=new WebGPUBackend();assert.deepEqual(await gpu.minkowskiDotBatch([1e200,1e200,1,0],[1e200,1e200,1,0]),[1]);assert.equal(gpu.lastExecution,"cpu");
});
test("auto backend safely falls back when WebGPU is absent",async()=>{
  if(WebGPUBackend.supported())return;
  assert.deepEqual(await minkowskiDotBatch([1,1,0,0],[1,1,0,0]),[0]);
});
test("batch shapes must match four-component groups",async()=>{
  await assert.rejects(()=>minkowskiDotBatch([1,2,3],[1,2,3],{backend:"cpu"}),RangeError);
  const oversized=new Float32Array(262_145*4);
  await assert.rejects(()=>minkowskiDotBatch(oversized,oversized,{backend:"cpu"}),RangeError);
});
test("empty batches are valid and non-finite components reject",async()=>{
  assert.deepEqual(await minkowskiDotBatch([],[],{backend:"cpu"}),[]);
  await assert.rejects(()=>minkowskiDotBatch([1,NaN,0,0],[1,2,0,0],{backend:"cpu"}),TypeError);
});
test("WebGPU falls back to double precision when float32 products may overflow",async()=>{
  const gpu=new WebGPUBackend();gpu.initialize=async()=>{throw new Error("unsafe float32 magnitudes must fall back before device initialization");};
  const value=await minkowskiDotBatch([1e20,1e20,1e20,1e20],[1e20,1e20,1e20,1e20],{backend:"webgpu",gpu});
  assert.equal(value[0],2e40);assert.ok(Number.isFinite(value[0]));assert.notEqual(gpu.status,"ready");
});
test("WebGPU preserves terms that would become zero or subnormal in float32",async()=>{
  const gpu=new WebGPUBackend();gpu.initialize=async()=>{throw new Error("unsafe float32 terms must fall back before device initialization");};
  for(const [a,b] of [[1e-50,1e30],[1e-40,1],[1e-30,1e-10]]){
    const actual=await gpu.minkowskiDotBatch([0,a,0,0],[0,b,0,0]);
    assert.equal(actual[0],a*b);
    assert.notEqual(actual[0],0);
  }
});
test("WebGPU dot products fall back for near-null cancellation",async()=>{
  const gpu=new WebGPUBackend();gpu.initialize=async()=>{throw new Error("cancellation must fall back before device initialization");};
  const left=[1,1,0,0,0,2,0,0],right=[1,1+1e-8,0,0,0,3,0,0];
  const actual=await gpu.minkowskiDotBatch(left,right);
  assert.deepEqual(actual,[right[1]-1,6]);assert.notEqual(actual[0],0);assert.equal(gpu.lastExecution,"cpu");
  assert.deepEqual(await gpu.minkowskiDotBatch([1,1,0,0],[1,1,0,0]),[0]);
});

function deferred(){let resolve,reject;const promise=new Promise((r,j)=>{resolve=r;reject=j;});return {promise,resolve,reject};}
function mockDevice({limits={},messages=[],pipelineError=null,lost=deferred()}={}){
  const device={
    limits:{maxBufferSize:1e8,maxStorageBufferBindingSize:1e8,maxComputeWorkgroupsPerDimension:4096,maxStorageBuffersPerShaderStage:8,maxComputeWorkgroupSizeX:256,...limits},
    lost:lost.promise,destroyed:0,destroy(){this.destroyed++;},
    createShaderModule(){return {getCompilationInfo:async()=>({messages})};},
    createComputePipeline(){if(pipelineError)throw pipelineError;return {getBindGroupLayout(){return {};}};},
    createBuffer(){throw new Error("test expects limits to prevent buffer creation");}
  };
  return {device,lost};
}
async function withMockGpu(gpu,run){
  const names=["navigator","GPUBufferUsage","GPUMapMode"],previous=new Map(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  Object.defineProperty(globalThis,"navigator",{configurable:true,value:{gpu}});
  Object.defineProperty(globalThis,"GPUBufferUsage",{configurable:true,value:{STORAGE:1,COPY_DST:2,COPY_SRC:4,MAP_READ:8}});
  Object.defineProperty(globalThis,"GPUMapMode",{configurable:true,value:{READ:1}});
  try{return await run();}
  finally{for(const name of names){const descriptor=previous.get(name);if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];}}
}
function mockExecutionDevice(onMapAsync=async()=>{},lost=deferred()){
  const {device}=mockDevice({lost});let destroyedBuffers=0;
  device.destroy=()=>{device.destroyed++;};
  device.queue={writeBuffer(){},submit(){}};
  device.createBuffer=({size})=>({size,async mapAsync(){await onMapAsync();},getMappedRange(){return new ArrayBuffer(size);},unmap(){},destroy(){destroyedBuffers++;}});
  device.createBindGroup=()=>({});
  device.createCommandEncoder=()=>({beginComputePass:()=>({setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}}),copyBufferToBuffer(){},finish(){return {};}});
  return {device,lost,destroyedBuffers:()=>destroyedBuffers};
}

test("plane-wave sampling provides CPU fields and validates inputs",async()=>{
  const fields=await samplePlaneWaveBatch([0,0,0,0,0,0,0,Math.PI/2],{backend:"cpu"});
  assert.deepEqual(fields[0],{electric:[1,0,0],magnetic:[0,1,0]});
  assert.ok(Math.abs(fields[1].electric[0])<1e-15);
  assert.equal(fields[1].magnetic[1],fields[1].electric[0]);
  await assert.rejects(()=>samplePlaneWaveBatch([0,0,0],{backend:"cpu"}),RangeError);
  await assert.rejects(()=>samplePlaneWaveBatch([0,0,0,0],{waveNumber:0}),RangeError);
  await assert.rejects(()=>samplePlaneWaveBatch([0,0,0,Infinity]),TypeError);
  await assert.rejects(()=>samplePlaneWaveBatch([1e308,0,0,-1e308],{backend:"cpu"}),/phase/);
  assert.deepEqual(await samplePlaneWaveBatch([],{backend:"cpu"}),[]);
});
test("plane-wave grid samples ordered spatial coordinates and preflights bounds",async()=>{
  const grid=await samplePlaneWaveGrid({shape:[2,1,2],origin:[1,2,3],spacing:[.5,1,.25],ct:.5,waveNumber:2,backend:"cpu"});
  assert.deepEqual(grid.shape,[2,1,2]);
  assert.deepEqual(Array.from(grid.events),[.5,1,2,3,.5,1,2,3.25,.5,1.5,2,3,.5,1.5,2,3.25]);
  for(let i=0;i<grid.fields.length;i++)assert.ok(Math.abs(grid.fields[i].electric[0]-Math.cos(2*(grid.events[4*i+3]-.5)))<1e-15);
  let called=false;const gpu={samplePlaneWaveBatch(){called=true;throw new Error("unexpected dispatch");}};
  for(const options of [{shape:[0,1,1]},{shape:[65537,4,1]},{shape:[2,1,1],origin:[1e20,0,0]},{shape:[4,1,1],origin:[2**53-2,0,0]},{shape:[1,1,1],spacing:[1,0,1]},{shape:[1,1,1],backend:"invalid"}])
    await assert.rejects(()=>samplePlaneWaveGrid({...options,gpu}),/shape|limit|coordinates|spacing|backend/);
  assert.equal(called,false);
});

test("plane-wave GPU dispatch uses the wave pipeline and matches the CPU reference",async()=>{
  const {device}=mockDevice();let selected,bindings,dispatches=0;
  device.createComputePipeline=({compute})=>({entryPoint:compute.entryPoint,getBindGroupLayout(){return {};}});
  device.createBuffer=({size})=>({data:new ArrayBuffer(size),async mapAsync(){},getMappedRange(){return this.data;},unmap(){},destroy(){}});
  device.createBindGroup=({entries})=>entries.map(entry=>entry.resource.buffer);
  device.queue={writeBuffer(buffer,offset,values){new Uint8Array(buffer.data).set(new Uint8Array(values.buffer,values.byteOffset,values.byteLength),offset);},submit(){
    assert.equal(selected.entryPoint,"plane_wave");dispatches++;
    const events=new Float32Array(bindings[0].data),parameters=new Float32Array(bindings[1].data),output=new Float32Array(bindings[2].data);
    for(let i=0;i<output.length;i++)output[i]=Math.cos(Math.fround(parameters[0]*Math.fround(events[4*i+3]-events[4*i])));
  }};
  let copy;
  device.createCommandEncoder=()=>({beginComputePass:()=>({setPipeline(p){selected=p;},setBindGroup(_i,b){bindings=b;},dispatchWorkgroups(){},end(){}}),copyBufferToBuffer(a,_a,b,_b){copy=()=>new Uint8Array(b.data).set(new Uint8Array(a.data));},finish(){return {};}});
  const submit=device.queue.submit;device.queue.submit=()=>{submit();copy();};
  const events=[0,0,0,0,.25,1,2,1.5],options={waveNumber:1.5};
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>device})},async()=>{
    const backend=new WebGPUBackend(),actual=await samplePlaneWaveBatch(events,{...options,backend:"webgpu",gpu:backend}),expected=await samplePlaneWaveBatch(events,{...options,backend:"cpu"});
    assert.equal(dispatches,1);assert.equal(backend.status,"ready");assert.equal(backend.lastExecution,"webgpu");
    for(let i=0;i<actual.length;i++)assert.ok(Math.abs(actual[i].electric[0]-expected[i].electric[0])<1e-6);
    const grid=await samplePlaneWaveGrid({shape:[2,1,2],origin:[0,0,.5],spacing:[1,1,.25],ct:.25,...options,backend:"webgpu",gpu:backend});
    const reference=await samplePlaneWaveGrid({shape:[2,1,2],origin:[0,0,.5],spacing:[1,1,.25],ct:.25,...options,backend:"cpu"});
    assert.equal(dispatches,2);assert.equal(backend.lastExecution,"webgpu");
    grid.fields.forEach((field,i)=>assert.ok(Math.abs(field.electric[0]-reference.fields[i].electric[0])<1e-6));
    backend.dispose();assert.equal(device.destroyed,1);
  });
});

test("plane-wave GPU path falls back before initialization outside its precision envelope",async()=>{
  const gpu=new WebGPUBackend();gpu.initialize=async()=>{throw new Error("must fall back before GPU initialization");};
  const events=[100,0,0,101];
  assert.deepEqual(await samplePlaneWaveBatch(events,{backend:"webgpu",gpu}),await samplePlaneWaveBatch(events,{backend:"cpu"}));
  assert.equal(gpu.lastExecution,"cpu");
  assert.deepEqual(await gpu.samplePlaneWaveBatch([],1),[]);assert.equal(gpu.lastExecution,"none");
});
test("plane-wave sampling falls back on device limits and releases owned devices",async()=>{
  const {device}=mockDevice({limits:{maxStorageBufferBindingSize:4}});
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>device})},async()=>{
    const events=[.25,0,0,1.5],expected=await samplePlaneWaveBatch(events,{backend:"cpu"});
    assert.deepEqual(await samplePlaneWaveBatch(events,{backend:"webgpu"}),expected);
    assert.equal(device.destroyed,1);
  });
});

test("wave shader failure falls back without disabling GPU dot products",async()=>{
  const {device}=mockExecutionDevice();let waveCompilations=0;
  device.createShaderModule=({code})=>({getCompilationInfo:async()=>({messages:code.includes("fn plane_wave")?(waveCompilations++,[{type:"error",message:"wave unavailable"}]):[]})});
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>device})},async()=>{
    const backend=new WebGPUBackend();
    assert.equal(await backend.initialize(),true);assert.equal(waveCompilations,0);
    const fields=await backend.samplePlaneWaveBatch([0,0,0,0]);
    assert.equal(fields[0].electric[0],1);assert.equal(backend.lastExecution,"cpu");
    assert.match(backend.waveError.message,/wave unavailable/);assert.equal(backend.status,"ready");
    assert.deepEqual(await backend.minkowskiDotBatch([0,0,0,0],[0,0,0,0]),[0]);
    assert.equal(backend.lastExecution,"webgpu");assert.equal(device.destroyed,0);
    await backend.samplePlaneWaveBatch([0,0,0,0]);assert.equal(waveCompilations,1);
    backend.dispose();
  });
});
test("stale wave compilation cannot block or replace a reinitialized device pipeline",async()=>{
  const first=mockExecutionDevice(),second=mockExecutionDevice(),started=deferred(),compilation=deferred();let requests=0;
  first.device.createShaderModule=({code})=>({getCompilationInfo:()=>code.includes("fn plane_wave")?(started.resolve(),compilation.promise):Promise.resolve({messages:[]})});
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>++requests===1?first.device:second.device})},async()=>{
    const backend=new WebGPUBackend(),old=backend.samplePlaneWaveBatch([0,0,0,0]);
    await started.promise;first.lost.resolve({reason:"unknown"});assert.equal((await old)[0].electric[0],1);
    assert.equal(await backend.initialize(),true);
    await backend.samplePlaneWaveBatch([0,0,0,0]);const newPipeline=backend.wavePipeline;
    assert.ok(newPipeline);assert.equal(backend.device,second.device);
    compilation.resolve({messages:[]});await Promise.resolve();await Promise.resolve();
    assert.equal(backend.wavePipeline,newPipeline);assert.equal(backend.device,second.device);
    backend.dispose();
  });
});

test("convenience calls release internally owned GPU devices but preserve caller-owned backends",async()=>{
  const first=mockExecutionDevice().device,second=mockExecutionDevice().device;
  let requests=0;
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>requests++===0?first:second})},async()=>{
    assert.deepEqual(await minkowskiDotBatch([0,0,0,0],[0,0,0,0]),[0]);
    assert.equal(first.destroyed,1);
    const external=new WebGPUBackend();
    assert.deepEqual(await minkowskiDotBatch([0,0,0,0],[0,0,0,0],{gpu:external}),[0]);
    assert.equal(second.destroyed,0);
    external.dispose();assert.equal(second.destroyed,1);
  });
});

test("device-limit checks use CPU before allocating unsupported batches",async()=>withMockGpu({requestAdapter:async()=>({requestDevice:async()=>mockDevice({limits:{maxStorageBufferBindingSize:4}}).device})},async()=>{
  const backend=new WebGPUBackend();
  assert.deepEqual(await backend.minkowskiDotBatch([1,2,0,0],[1,2,0,0]),[3]);
  assert.equal(backend.status,"ready");
  assert.equal(backend.lastExecution,"cpu");
  backend.dispose();
}));

test("shader validation failure releases the initialized device",async()=>{
  const {device}=mockDevice({messages:[{type:"error",message:"bad shader"}]});
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>device})},async()=>{
    const backend=new WebGPUBackend();
    assert.equal(await backend.initialize(),false);
    assert.equal(backend.status,"failed");
    assert.equal(backend.device,null);
    assert.equal(backend.pipeline,null);
    assert.equal(device.destroyed,1);
  });
});

test("pipeline creation failure also destroys and clears the device",async()=>{
  const {device}=mockDevice({pipelineError:new Error("pipeline creation failed")});
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>device})},async()=>{
    const backend=new WebGPUBackend();
    assert.equal(await backend.initialize(),false);
    assert.equal(backend.status,"failed");
    assert.equal(backend.device,null);
    assert.equal(device.destroyed,1);
  });
});

test("asynchronous pipeline failures trigger the correct fallback",async()=>{
  const first=mockDevice().device,second=mockExecutionDevice().device;let requests=0;
  first.createComputePipelineAsync=async()=>{throw new Error("async dot pipeline failed");};
  second.createComputePipelineAsync=async({compute})=>{if(compute.entryPoint==="plane_wave")throw new Error("async wave pipeline failed");return {getBindGroupLayout(){return {};}};};
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>++requests===1?first:second})},async()=>{
    const dot=new WebGPUBackend();assert.equal(await dot.initialize(),false);assert.match(dot.error.message,/async dot/);assert.equal(first.destroyed,1);
    const wave=new WebGPUBackend();assert.equal((await wave.samplePlaneWaveBatch([0,0,0,0]))[0].electric[0],1);
    assert.match(wave.waveError.message,/async wave/);assert.equal(wave.status,"ready");
    assert.deepEqual(await wave.minkowskiDotBatch([0,0,0,0],[0,0,0,0]),[0]);assert.equal(wave.lastExecution,"webgpu");wave.dispose();
  });
});

test("disposing during asynchronous pipeline creation releases the device once",async()=>{
  const {device}=mockDevice(),started=deferred(),pipeline=deferred();
  device.createComputePipelineAsync=()=>{started.resolve();return pipeline.promise;};
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>device})},async()=>{
    const backend=new WebGPUBackend(),pending=backend.initialize();await started.promise;
    backend.dispose();pipeline.resolve({getBindGroupLayout(){return {};}});
    assert.equal(await pending,false);assert.equal(backend.status,"disposed");assert.equal(device.destroyed,1);
  });
});
test("device loss during pending initialization falls back promptly and permits replacement",async()=>{
  const first=mockDevice(),second=mockExecutionDevice(),started=deferred(),pipeline=deferred();let requests=0;
  first.device.createComputePipelineAsync=()=>{started.resolve();return pipeline.promise;};
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>++requests===1?first.device:second.device})},async()=>{
    const backend=new WebGPUBackend(),old=backend.minkowskiDotBatch([2,1,0,0],[2,1,0,0]);
    await started.promise;first.lost.resolve({reason:"unknown"});
    assert.deepEqual(await old,[-3]);assert.equal(backend.status,"lost");assert.equal(backend.lastExecution,"cpu");
    assert.equal(await backend.initialize(),true);const replacement=backend.pipeline;assert.equal(backend.device,second.device);
    pipeline.resolve({getBindGroupLayout(){return {};}});await Promise.resolve();await Promise.resolve();
    assert.equal(backend.device,second.device);assert.equal(backend.pipeline,replacement);assert.equal(backend.status,"ready");
    backend.dispose();
  });
});

test("device loss clears stale handles and safely falls back to CPU",async()=>{
  const first=mockDevice();let requests=0;
  await withMockGpu({requestAdapter:async()=>{requests++;return requests===1?{requestDevice:async()=>first.device}:null;}},async()=>{
    const backend=new WebGPUBackend();
    assert.equal(await backend.initialize(),true);
    first.lost.resolve({reason:"unknown"});
    await Promise.resolve();
    assert.equal(backend.status,"lost");
    assert.equal(backend.device,null);
    assert.equal(backend.pipeline,null);
    assert.deepEqual(await backend.minkowskiDotBatch([2,1,0,0],[2,1,0,0]),[-3]);
    assert.equal(backend.status,"unavailable");
    assert.equal(requests,2);
    backend.dispose();
  });
});

test("device loss during readback falls back and preserves the lost state",async()=>{
  const lost=deferred();
  const execution=mockExecutionDevice(async()=>{lost.resolve({reason:"unknown"});await Promise.resolve();throw new Error("device lost during mapping");},lost);
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>execution.device})},async()=>{
    const backend=new WebGPUBackend();
    const result=await backend.minkowskiDotBatch([2,1,0,0],[2,1,0,0]);
    assert.deepEqual(result,[-3]);
    assert.equal(backend.status,"lost");
    assert.equal(backend.lastExecution,"cpu");
    assert.equal(backend.device,null);
    assert.equal(backend.pipeline,null);
    assert.equal(execution.destroyedBuffers(),4);
    backend.dispose();
  });
});

test("plane-wave readback loss falls back before a replacement device is used",async()=>{
  const started=deferred(),readback=deferred(),first=mockExecutionDevice(()=>{started.resolve();return readback.promise;}),second=mockExecutionDevice();let requests=0;
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>++requests===1?first.device:second.device})},async()=>{
    const backend=new WebGPUBackend(),pending=backend.samplePlaneWaveBatch([0,0,0,0]);
    await started.promise;first.lost.resolve({reason:"unknown"});
    assert.deepEqual(await pending,[{electric:[1,0,0],magnetic:[0,1,0]}]);assert.equal(backend.lastExecution,"cpu");assert.equal(first.destroyedBuffers(),4);
    assert.equal(await backend.initialize(),true);assert.equal(backend.device,second.device);
    readback.resolve();await Promise.resolve();await Promise.resolve();assert.equal(backend.status,"ready");assert.equal(backend.device,second.device);
    backend.dispose();
  });
});
test("device loss resolves a pending readback through CPU fallback",async()=>{
  const started=deferred(),readback=deferred(),execution=mockExecutionDevice(()=>{started.resolve();return readback.promise;});
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>execution.device})},async()=>{
    const backend=new WebGPUBackend(),pending=backend.minkowskiDotBatch([2,1,0,0],[2,1,0,0]);
    await started.promise;execution.lost.resolve({reason:"unknown"});
    assert.deepEqual(await pending,[-3]);assert.equal(backend.status,"lost");assert.equal(backend.lastExecution,"cpu");assert.equal(execution.destroyedBuffers(),4);
    readback.reject(new Error("late mapping failure"));await Promise.resolve();backend.dispose();
  });
});
test("stale readback failure cannot dispose a reinitialized device",async()=>{
  const lost=deferred(),started=deferred(),readback=deferred();
  const first=mockExecutionDevice(async()=>{started.resolve();return readback.promise;},lost),second=mockDevice();let requests=0;
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>++requests===1?first.device:second.device})},async()=>{
    const backend=new WebGPUBackend();
    const pending=backend.minkowskiDotBatch([2,1,0,0],[2,1,0,0]);
    await started.promise;
    lost.resolve({reason:"unknown"});
    await Promise.resolve();await Promise.resolve();
    assert.equal(backend.status,"lost");
    assert.equal(await backend.initialize(),true);
    assert.equal(backend.device,second.device);
    readback.reject(new Error("late readback failure from the old device"));
    assert.deepEqual(await pending,[-3]);
    assert.equal(backend.status,"ready");
    assert.equal(backend.device,second.device);
    assert.equal(second.device.destroyed,0);
    backend.dispose();
  });
});

test("dispose releases the device and prevents later reinitialization",async()=>{
  const {device}=mockDevice();
  await withMockGpu({requestAdapter:async()=>({requestDevice:async()=>device})},async()=>{
    const backend=new WebGPUBackend();
    assert.equal(await backend.initialize(),true);
    backend.dispose();
    backend.dispose();
    assert.equal(device.destroyed,1);
    assert.equal(backend.status,"disposed");
    assert.deepEqual(await backend.minkowskiDotBatch([1,1,0,0],[1,1,0,0]),[0]);
    assert.equal(backend.status,"disposed");
    assert.equal(backend.lastExecution,"cpu");
  });
});
