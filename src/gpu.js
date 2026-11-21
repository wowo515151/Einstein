const MAX_BATCH_VECTORS=262_144;
const FLOAT32_MAX=3.402823466e38;
const FLOAT32_MIN_NORMAL=1.1754943508222875e-38;
const DOT_RELATIVE_ERROR_LIMIT=1e-5;

const SHADER_BINDINGS = `
struct Values { data: array<f32> };
@group(0) @binding(0) var<storage, read> left: Values;
@group(0) @binding(1) var<storage, read> right: Values;
@group(0) @binding(2) var<storage, read_write> result: Values;
`;
const DOT_SHADER = SHADER_BINDINGS + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let b = i * 4u;
  if (b + 3u >= arrayLength(&left.data)) { return; }
  result.data[i] = -left.data[b] * right.data[b] + left.data[b+1u] * right.data[b+1u] + left.data[b+2u] * right.data[b+2u] + left.data[b+3u] * right.data[b+3u];
}`;
const WAVE_SHADER = SHADER_BINDINGS + `
@compute @workgroup_size(64)
fn plane_wave(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let b = i * 4u;
  if (b + 3u >= arrayLength(&left.data)) { return; }
  result.data[i] = cos(right.data[0] * (left.data[b+3u] - left.data[b]));
}`;

export class WebGPUBackend {
  constructor(){this.adapter=null;this.device=null;this.pipeline=null;this.wavePipeline=null;this.waveError=null;this.status="uninitialized";this.lastExecution="none";this.error=null;this._initializing=null;this._waveInitializing=null;this._disposed=false;this._retiredFailureDevice=null;}
  static supported(){return typeof navigator!=="undefined"&&!!navigator.gpu;}
  async initialize(){
    if(this.status==="ready")return true;
    if(this._disposed)return false;
    if(this._initializing)return this._initializing;
    const gpu=globalThis.navigator?.gpu;
    if(!gpu){this.status="unavailable";return false;}
    const pending=(async()=>{
      let device;
      try{
        this.adapter=await gpu.requestAdapter();
        if(this._disposed)return false;
        if(!this.adapter){this.status="unavailable";return false;}
        device=await this.adapter.requestDevice();
        if(this._disposed){device.destroy?.();return false;}
        this.device=device;
        const lost=device.lost?.then(info=>{
          if(this._disposed)return false;
          if(this.device===device){this.device=null;this.pipeline=null;this.wavePipeline=null;this._waveInitializing=null;this.adapter=null;this.status="lost";}
          else if(info?.reason!=="destroyed"&&this._retiredFailureDevice===device&&this.status==="failed"){this.status="lost";this._retiredFailureDevice=null;}
          return false;
        });
        const build=(async()=>{
          const module=device.createShaderModule({code:DOT_SHADER});
          if(typeof module.getCompilationInfo==="function"){
            const info=await module.getCompilationInfo();
            const errors=info.messages?.filter(message=>message.type==="error")??[];
            if(errors.length)throw new Error(`WebGPU shader validation failed: ${errors.map(message=>message.message).join("; ")}`);
          }
          if(this._disposed||this.device!==device)return false;
          const pipeline=await createPipeline(device,module,"main");
          if(this._disposed||this.device!==device)return false;
          this.pipeline=pipeline;this.wavePipeline=null;this.waveError=null;this.status="ready";this.error=null;
          return true;
        })();
        return await (lost?Promise.race([build,lost]):build);
      }catch(error){if(!this._disposed&&(!device||this.device===device)){this.error=error;this._releaseDevice(device);this.status="failed";}return false;}
    })();
    this._initializing=pending;
    try{return await pending;}finally{if(this._initializing===pending)this._initializing=null;}
  }
  async _initializeWave(){
    if(this.wavePipeline)return true;
    if(this.waveError||this._disposed||!this.device)return false;
    if(this._waveInitializing)return this._waveInitializing;
    const device=this.device;
    const work=(async()=>{
      try{
        const module=device.createShaderModule({code:WAVE_SHADER});
        if(typeof module.getCompilationInfo==="function"){
          const info=await module.getCompilationInfo(),errors=info.messages?.filter(message=>message.type==="error")??[];
          if(errors.length)throw new Error(`WebGPU wave shader validation failed: ${errors.map(message=>message.message).join("; ")}`);
        }
        if(this._disposed||this.device!==device)return false;
        const pipeline=await createPipeline(device,module,"plane_wave");
        if(this._disposed||this.device!==device)return false;
        this.wavePipeline=pipeline;return true;
      }catch(error){if(this.device===device&&!this._disposed)this.waveError=error;return false;}
    })();
    const pending=device.lost?Promise.race([work,device.lost.then(()=>false)]):work;
    this._waveInitializing=pending;
    try{return await pending;}finally{if(this._waveInitializing===pending)this._waveInitializing=null;}
  }
  async minkowskiDotBatch(left,right){
    validateBatch(left,right);if(left.length===0){this.lastExecution="none";return [];}
    const fallback=()=>this._cpu(()=>cpuDots(left,right));
    if(this._disposed)return fallback();
    const left32=new Float32Array(left),right32=new Float32Array(right);let maxLeft=0,maxRight=0;
    for(let i=0;i<left32.length;i++){
      const a=Math.abs(left32[i]),b=Math.abs(right32[i]);
      if(!Number.isFinite(a)||!Number.isFinite(b))return fallback();
      if(left[i]!==0&&right[i]!==0&&(a<FLOAT32_MIN_NORMAL||b<FLOAT32_MIN_NORMAL||a*b<FLOAT32_MIN_NORMAL))return fallback();
      maxLeft=Math.max(maxLeft,a);maxRight=Math.max(maxRight,b);
    }
    if(maxLeft>0&&maxRight>FLOAT32_MAX/(4*maxLeft))return fallback();
    for(let base=0;base<left.length;base+=4){
      let signed=0,magnitude=0,conversionError=0;
      for(let axis=0;axis<4;axis++){
        const i=base+axis,product=left32[i]*right32[i],reference=left[i]*right[i];
        signed+=(axis===0?-1:1)*product;magnitude+=Math.abs(product);conversionError+=Math.abs(reference-product);
      }
      const errorEstimate=conversionError+8*2**-23*magnitude,lowerBound=Math.max(0,Math.abs(signed)-errorEstimate);
      if(errorEstimate>DOT_RELATIVE_ERROR_LIMIT*lowerBound)return fallback();
    }
    if(this.status!=="ready"&&!await this.initialize())return fallback();
    return this._compute(left32,right32,this.pipeline,fallback);
  }
  async samplePlaneWaveBatch(events,waveNumber=1){
    validateEvents(events,waveNumber);if(events.length===0){this.lastExecution="none";return [];}
    const fallback=()=>this._cpu(()=>cpuPlaneWave(events,waveNumber));
    if(this._disposed||waveNumber<.125||waveNumber>2)return fallback();
    const values=new Float32Array(events),parameters=new Float32Array(events.length);parameters[0]=waveNumber;
    for(let i=0;i<events.length;i+=4)if(Math.abs(events[i])>16||Math.abs(events[i+3])>16)return fallback();
    if(this.status!=="ready"&&!await this.initialize())return fallback();
    if(!await this._initializeWave())return fallback();
    const amplitudes=await this._compute(values,parameters,this.wavePipeline,()=>null);
    return amplitudes===null?fallback():fieldsFromAmplitudes(amplitudes);
  }
  async _compute(left,right,pipeline,fallback){
    const n=left.length/4,dev=this.device;
    if(!dev||!pipeline||!supportsBatchLimits(dev,n,left.length*4,right.length*4,n*4))return fallback();
    let a,b,out,read,mapped=false;
    try{
      a=dev.createBuffer({size:left.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      b=dev.createBuffer({size:right.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      out=dev.createBuffer({size:n*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      read=dev.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      dev.queue.writeBuffer(a,0,left);dev.queue.writeBuffer(b,0,right);
      const bind=dev.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:a}},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:out}}]});
      const encoder=dev.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(n/64));pass.end();
      encoder.copyBufferToBuffer(out,0,read,0,n*4);dev.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);mapped=true;
      const values=Array.from(new Float32Array(read.getMappedRange()).slice());
      if(!values.every(Number.isFinite))return fallback();
      this.lastExecution="webgpu";
      return values;
    }catch(error){
      if(this.device===dev){
        this.error=error;this._releaseDevice(dev);
        if(!this._disposed){this.status="failed";this._retiredFailureDevice=dev;}
      }
      return fallback();
    }
    finally{if(mapped)try{read?.unmap();}catch{}this._destroy(a,b,out,read);}
  }
  _cpu(callback){const result=callback();this.lastExecution="cpu";return result;}
  _destroy(...buffers){for(const b of buffers)try{b?.destroy();}catch{}}
  _releaseDevice(device=this.device){if(!device)return;if(this.device===device){this.device=null;this.pipeline=null;this.wavePipeline=null;this._waveInitializing=null;this.adapter=null;}try{device.destroy();}catch{}}
  dispose(){if(this._disposed)return;this._disposed=true;this._releaseDevice();this.status="disposed";}
}
function createPipeline(device,module,entryPoint){const descriptor={layout:"auto",compute:{module,entryPoint}};return typeof device.createComputePipelineAsync==="function"?device.createComputePipelineAsync(descriptor):device.createComputePipeline(descriptor);}
function supportsBatchLimits(device,vectorCount,leftBytes,rightBytes,resultBytes){
  const limits=device.limits??{};
  const maxBufferSize=limits.maxBufferSize,maxStorageBufferBindingSize=limits.maxStorageBufferBindingSize,maxGroups=limits.maxComputeWorkgroupsPerDimension;
  return Number.isFinite(maxBufferSize)&&Number.isFinite(maxStorageBufferBindingSize)&&Number.isFinite(maxGroups)
    &&Math.max(leftBytes,rightBytes,resultBytes)<=maxBufferSize
    &&Math.max(leftBytes,rightBytes,resultBytes)<=maxStorageBufferBindingSize
    &&Math.ceil(vectorCount/64)<=maxGroups
    &&(limits.maxStorageBuffersPerShaderStage??0)>=3
    &&(limits.maxComputeWorkgroupSizeX??0)>=64;
}
function validateBatch(left,right){if(!Array.isArray(left)&&!ArrayBuffer.isView(left)||!Array.isArray(right)&&!ArrayBuffer.isView(right))throw new TypeError("batch inputs must be numeric arrays");if(left.length!==right.length||left.length%4!==0)throw new RangeError("batched vectors must have matching lengths divisible by four");if(left.length/4>MAX_BATCH_VECTORS)throw new RangeError(`batch exceeds the ${MAX_BATCH_VECTORS}-vector limit`);for(let i=0;i<left.length;i++)if(typeof left[i]!=="number"||!Number.isFinite(left[i])||typeof right[i]!=="number"||!Number.isFinite(right[i]))throw new TypeError("batch inputs must contain finite numbers");}
function cpuDots(left,right){
  validateBatch(left,right);const out=[];
  for(let i=0;i<left.length;i+=4){
    const terms=Array.from({length:4},(_,axis)=>left[i+axis]*right[i+axis]);
    const extreme=terms.some((value,axis)=>!Number.isFinite(value)||(Math.abs(value)<2**-1022&&left[i+axis]!==0&&right[i+axis]!==0));
    const direct=-terms[0]+terms[1]+terms[2]+terms[3];
    const value=extreme||!Number.isFinite(direct)?exactDot(left,right,i):direct;
    if(!Number.isFinite(value))throw new RangeError("batch dot product is non-finite");
    out.push(value);
  }
  return out;
}
const floatBits=new DataView(new ArrayBuffer(8));
function binaryParts(value){
  floatBits.setFloat64(0,value,false);const bits=floatBits.getBigUint64(0,false),exponent=Number((bits>>52n)&2047n),fraction=bits&((1n<<52n)-1n);
  return {mantissa:(bits>>63n?-1n:1n)*(exponent?((1n<<52n)|fraction):fraction),exponent:exponent?exponent-1075:-1074};
}
function roundRight(value,shift){if(shift<=0)return value<<BigInt(-shift);const s=BigInt(shift),whole=value>>s,remainder=value-(whole<<s),half=1n<<(s-1n);return remainder>half||(remainder===half&&(whole&1n)!==0n)?whole+1n:whole;}
function exactDot(left,right,offset){
  const parts=Array.from({length:4},(_,axis)=>{const a=binaryParts(left[offset+axis]),b=binaryParts(right[offset+axis]);return {mantissa:(axis===0?-1n:1n)*a.mantissa*b.mantissa,exponent:a.exponent+b.exponent};}).filter(part=>part.mantissa!==0n);
  if(parts.length===0)return 0;
  const exponent=Math.min(...parts.map(part=>part.exponent));let sum=0n;
  for(const part of parts)sum+=part.mantissa<<BigInt(part.exponent-exponent);
  if(sum===0n)return 0;
  const sign=sum<0n?-1:1,magnitude=sum<0n?-sum:sum,bits=magnitude.toString(2).length,topExponent=bits-1+exponent;
  if(topExponent>1023)return sign*Infinity;
  if(topExponent < -1022)return sign*Number(roundRight(magnitude,-1074-exponent))*Number.MIN_VALUE;
  return sign*Number(roundRight(magnitude,bits-53))*2**(topExponent-52);
}

export async function minkowskiDotBatch(left,right,{backend="auto",gpu=null}={}){if(backend==="cpu")return cpuDots(left,right);if(backend!=="webgpu"&&backend!=="auto")throw new TypeError("backend must be auto, cpu, or webgpu");const instance=gpu??new WebGPUBackend();try{return await instance.minkowskiDotBatch(left,right);}finally{if(gpu==null)instance.dispose();}}

function validateEvents(events,waveNumber){if(!Array.isArray(events)&&!ArrayBuffer.isView(events))throw new TypeError("events must be a numeric array");if(events.length%4!==0||events.length/4>MAX_BATCH_VECTORS)throw new RangeError("events must be bounded groups of four coordinates");if(!Number.isFinite(waveNumber)||waveNumber<=0)throw new RangeError("waveNumber must be positive and finite");for(const value of events)if(typeof value!=="number"||!Number.isFinite(value))throw new TypeError("events must contain finite coordinates");}
function fieldsFromAmplitudes(amplitudes){return amplitudes.map(amplitude=>({electric:[amplitude,0,0],magnetic:[0,amplitude,0]}));}
function cpuPlaneWave(events,waveNumber){const amplitudes=[];for(let i=0;i<events.length;i+=4){const phase=waveNumber*(events[i+3]-events[i]);if(!Number.isFinite(phase))throw new RangeError("plane-wave phase is non-finite");amplitudes.push(Math.cos(phase));}return fieldsFromAmplitudes(amplitudes);}
export async function samplePlaneWaveBatch(events,{waveNumber=1,backend="auto",gpu=null}={}){validateEvents(events,waveNumber);if(backend==="cpu")return cpuPlaneWave(events,waveNumber);if(backend!=="auto"&&backend!=="webgpu")throw new TypeError("backend must be auto, cpu, or webgpu");const instance=gpu??new WebGPUBackend();try{return await instance.samplePlaneWaveBatch(events,waveNumber);}finally{if(gpu==null)instance.dispose();}}
