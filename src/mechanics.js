import {weightedDot,weightedDot4} from "./weighted-dot-internal.js";
import {C, vector, add, sub, scale, dot, cross, norm, zeros, identity, transpose, mmul, mvec, inverse, minkowskiDot, finiteNumber} from "./array.js";
import {Bivector4, Covector4, Metric4, LorentzTransform} from "./algebra.js";
import {Event4} from "./spacetime.js";

const MAX_GRID_CELLS=32_768;
const MAX_FIELD_HISTORY_COMPONENTS=1_000_000;
const MAX_MECHANICS_TENSOR_COMPONENTS=1_000_000;
const MAX_RETARDED_FIELD_BATCH_WORK=1_000_000;

function tensorShape(value,name,{trailingSize=null,trailingShape=null}={}){
  const active=new Set();let leaves=0;
  function visit(node,depth,path){
    if(ArrayBuffer.isView(node)&&!(node instanceof DataView))node=Array.from(node);
    if(Array.isArray(node)){
      if(depth>=8)throw new RangeError(`${name} rank exceeds 8`);
      if(active.has(node))throw new TypeError(`${name} cannot contain cycles`);
      active.add(node);const length=node.length;let childShape=null;
      for(let i=0;i<length;i++){const shape=visit(node[i],depth+1,`${path}[${i}]`);if(childShape===null)childShape=shape;else if(shape.length!==childShape.length||shape.some((n,j)=>n!==childShape[j]))throw new RangeError(`${name} must be rectangular`);}
      active.delete(node);return [length,...(childShape??[])];
    }
    finiteNumber(node,`${path}`);leaves++;if(leaves>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError(`${name} exceeds the component limit`);return [];
  }
  const normalized=normalizeTensor(value);
  const shape=visit(normalized,0,name);
  const suffix=trailingShape??(trailingSize===null?null:[trailingSize]);
  if(suffix&&(shape.length<suffix.length||suffix.some((size,index)=>shape[shape.length-suffix.length+index]!==size)))throw new RangeError(`${name} must end in shape ${suffix.join(" by ")}`);
  return {value:normalized,shape,leaves};
}
function normalizeTensor(value,state={active:new Set(),leaves:0},depth=0){
  if(ArrayBuffer.isView(value)&&!(value instanceof DataView)){if(value.length>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError("tensor exceeds the component limit");value=Array.from(value);}
  if(Array.isArray(value)){if(value.length>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError("tensor exceeds the component limit");if(depth>=8)throw new RangeError("tensor rank exceeds 8");if(state.active.has(value))throw new TypeError("tensor cannot contain cycles");state.active.add(value);const copy=Array.from(value,item=>normalizeTensor(item,state,depth+1));state.active.delete(value);return copy;}
  finiteNumber(value,"tensor component");state.leaves++;if(state.leaves>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError("tensor exceeds the component limit");
  return value;
}
function broadcastShape(a,b){
  const n=Math.max(a.length,b.length),out=Array(n);for(let i=0;i<n;i++){const x=a[a.length-n+i]??1,y=b[b.length-n+i]??1;if(x!==y&&x!==1&&y!==1)throw new RangeError("batch shapes are not broadcast-compatible");out[i]=x===1?y:y===1?x:x;}return out;
}
function broadcastGet(value,shape,targetIndex,targetShape){
  let node=value;const offset=targetShape.length-shape.length;
  for(let i=0;i<shape.length;i++)node=node[shape[i]===1?0:targetIndex[offset+i]];
  return node;
}
function broadcastSourceIndex(shape,targetIndex,targetShape){const offset=targetShape.length-shape.length;return shape.map((size,i)=>size===1?0:targetIndex[offset+i]);}
function nestedFrom(shape,build,prefix=[]){
  if(prefix.length===shape.length)return build(prefix);
  return Array.from({length:shape[prefix.length]},(_,i)=>nestedFrom(shape,build,[...prefix,i]));
}
function mapLastAxis(value,fn){if(value.length===0)return [];return Array.isArray(value[0])?value.map(item=>mapLastAxis(item,fn)):fn(value);}
function mapBatches(value,batchRank,fn){if(batchRank===0)return fn(value);return value.map(item=>mapBatches(item,batchRank-1,fn));}
function transformFourVectors(components,L,name){
  const current=tensorShape(components,`${name} components`,{trailingSize:4}),currentBatchShape=current.shape.slice(0,-1);
  const raw=L instanceof LorentzTransform?L._checkedMatrix():L,parsed=tensorShape(raw,"Lorentz matrix",{trailingShape:[4,4]}),matrixBatchShape=parsed.shape.slice(0,-2),batchShape=broadcastShape(currentBatchShape,matrixBatchShape);
  if(batchShape.reduce((p,n)=>p*n,1)*4>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError(`${name} transform exceeds the component limit`);
  const cache=new Map();
  return nestedFrom(batchShape,index=>{
    const sourceIndex=broadcastSourceIndex(matrixBatchShape,index,batchShape),key=sourceIndex.join(",");let transform=cache.get(key);
    if(!transform){transform=new LorentzTransform(broadcastGet(parsed.value,matrixBatchShape,index,batchShape),{validate:false});if(!transform.validate())throw new RangeError(`${name} transform must preserve the Minkowski metric`);cache.set(key,transform);}
    return transform.applyVector(broadcastGet(current.value,currentBatchShape,index,batchShape)).components;
  });
}
function transformRankTwo(tensor,L,name){
  const data=tensorShape(tensor,`${name} tensor`,{trailingShape:[4,4]}),batch=data.shape.slice(0,-2),parsed=tensorShape(L instanceof LorentzTransform?L._checkedMatrix():L,"Lorentz matrix",{trailingShape:[4,4]}),matrixBatch=parsed.shape.slice(0,-2),shape=broadcastShape(batch,matrixBatch);
  if(shape.reduce((p,n)=>p*n,1)*16>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError(`${name} transform exceeds the component limit`);
  const cache=new Map();return nestedFrom(shape,index=>{const sourceIndex=broadcastSourceIndex(matrixBatch,index,shape),key=sourceIndex.join(",");let transform=cache.get(key);if(!transform){transform=new LorentzTransform(broadcastGet(parsed.value,matrixBatch,index,shape),{validate:false});if(!transform.validate())throw new RangeError(`${name} requires a Lorentz transformation`);cache.set(key,transform);}const matrix=transform.matrix,value=broadcastGet(data.value,batch,index,shape);return mmul(mmul(matrix,value),transpose(matrix));});
}

function velocityComponent(c,spatial,time){if(spatial===0)return spatial;const product=c*spatial,direct=product/time;if(Number.isFinite(direct)&&Math.abs(product)>=2**-1022&&direct!==0)return direct;const alternate=spatial/time*c;if(Number.isFinite(alternate)&&alternate!==0)return alternate;const result=Math.sign(spatial)*Math.exp(Math.log(c)+Math.log(Math.abs(spatial))-Math.log(time));if(!Number.isFinite(result))throw new RangeError("three-velocity is non-finite");return result;}

export class FourVelocity {
  constructor(components,{c=C}={}){this.components=tensorShape(components,"four-velocity",{trailingSize:4}).value;this.c=finiteNumber(c,"c");if(c<=0)throw new RangeError("c must be positive");Object.freeze(this);}
  get u(){return this.components;}
  static fromVelocity3(v,c=C){c=finiteNumber(c,"c");if(c<=0)throw new RangeError("c must be positive");const parsed=tensorShape(v,"velocity",{trailingSize:3});return new FourVelocity(mapLastAxis(parsed.value,w=>{const speed=Math.hypot(...w);if(speed>=c)throw new RangeError("ordinary velocity must be strictly subluminal");const gamma=1/Math.sqrt(1-(speed/c)**2);return [gamma*c,...w.map(x=>gamma*x)];}),{c});}
  gamma(){const parsed=tensorShape(this.components,"four-velocity",{trailingSize:4});return mapLastAxis(parsed.value,u=>{const value=u[0]/this.c;if(!Number.isFinite(value))throw new RangeError("gamma is non-finite");return value;});}
  toVelocity3(){const parsed=tensorShape(this.components,"four-velocity",{trailingSize:4});return mapLastAxis(parsed.value,u=>{if(u[0]<=0)throw new RangeError("four-velocity must be future-directed");return u.slice(1).map(x=>velocityComponent(this.c,x,u[0]));});}
  relativeVelocity(other){
    if(!(other instanceof FourVelocity))throw new TypeError("other must be a FourVelocity");
    if(this.c!==other.c)throw new RangeError("four-velocities must use the same speed of light");
    const a=tensorShape(this.components,"four-velocity",{trailingSize:4}),b=tensorShape(other.components,"other four-velocity",{trailingSize:4}),aShape=a.shape.slice(0,-1),bShape=b.shape.slice(0,-1),shape=broadcastShape(aShape,bShape);
    if(shape.reduce((product,size)=>product*size,1)*3>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError("relative-velocity batch exceeds the component limit");
    return nestedFrom(shape,index=>{const observer=broadcastGet(a.value,aShape,index,shape),target=broadcastGet(b.value,bShape,index,shape);if(observer[0]<=0)throw new RangeError("four-velocity must be future-directed");const boost=LorentzTransform.boost(observer.slice(1).map(component=>component/observer[0])),rest=boost.applyVector(target).components;if(rest[0]<=0)throw new RangeError("relative frame transform did not preserve future direction");return rest.slice(1).map(component=>velocityComponent(this.c,component,rest[0]));});
  }
  validate({atol=1e-10,rtol=1e-10}={}){finiteNumber(atol,"atol");finiteNumber(rtol,"rtol");if(atol<0||rtol<0)throw new RangeError("validation tolerances must be nonnegative");const parsed=tensorShape(this.components,"four-velocity",{trailingSize:4}),c2=this.c*this.c,tolerance=atol+rtol*c2;let valid=true;mapLastAxis(parsed.value,u=>{const norm2=minkowskiDot(u,u);let matches;if(Number.isFinite(c2)&&c2>=2**-1022&&Number.isFinite(norm2)&&Number.isFinite(tolerance))matches=Math.abs(norm2+c2)<=tolerance;else{const scaled=u.map(value=>value/this.c),scaledNorm=scaled.every(Number.isFinite)?minkowskiDot(scaled,scaled):NaN,scaledTolerance=rtol+(atol/this.c)/this.c;matches=Number.isFinite(scaledNorm)?Math.abs(scaledNorm+1)<=scaledTolerance:c2===0&&atol>=2**-1022&&Number.isFinite(norm2)&&Math.abs(norm2)<=atol;}if(!(u[0]>0&&matches))valid=false;return null;});return valid;}
}

export class FourMomentum {
  constructor(components,{c=C}={}){this.components=tensorShape(components,"four-momentum",{trailingSize:4}).value;this.c=finiteNumber(c,"c");if(this.c<=0)throw new RangeError("c must be positive");Object.freeze(this);}
  get p(){return this.components;}
  static fromMassVelocity(mass,velocity,c=C){c=finiteNumber(c,"c");if(c<=0)throw new RangeError("c must be positive");const m=tensorShape(mass,"mass"),u=FourVelocity.fromVelocity3(velocity,c),v=tensorShape(u.components,"four-velocity",{trailingSize:4}),massBatchShape=m.shape,velocityBatchShape=v.shape.slice(0,-1),batchShape=broadcastShape(massBatchShape,velocityBatchShape);if(batchShape.reduce((p,n)=>p*n,1)*4>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError("four-momentum exceeds the component limit");return new FourMomentum(nestedFrom(batchShape,index=>{const value=broadcastGet(m.value,massBatchShape,index,batchShape);if(value<0)throw new RangeError("mass must be nonnegative");return broadcastGet(v.value,velocityBatchShape,index,batchShape).map(component=>component*value);}),{c});}
  energy(){const parsed=tensorShape(this.components,"four-momentum",{trailingSize:4});return mapLastAxis(parsed.value,row=>{const value=row[0]*this.c;if(!Number.isFinite(value))throw new RangeError("energy is non-finite");return value;});}
  momentum3(){const parsed=tensorShape(this.components,"four-momentum",{trailingSize:4});return mapLastAxis(parsed.value,row=>row.slice(1));}
  invariantMass({clampRoundoff=true}={}){const parsed=tensorShape(this.components,"four-momentum",{trailingSize:4});return mapLastAxis(parsed.value,row=>{const norm=-minkowskiDot(row,row),c2=this.c*this.c,scaled=row.map(value=>value/this.c);if(scaled.some(value=>!Number.isFinite(value))){const scale=Math.max(...row.map(Math.abs)),normalized=row.map(value=>value/scale),relative=-minkowskiDot(normalized,normalized),relativeMagnitude=normalized.reduce((sum,value)=>sum+value**2,0),ratio=this.c/scale,tolerance=64*Number.EPSILON*Math.max(ratio*ratio,relativeMagnitude);if(relative< -tolerance||relative<0&&!clampRoundoff)throw new RangeError("four-momentum is spacelike");if(relative<=0)return 0;const mass=(scale*Math.sqrt(relative))/this.c;if(!Number.isFinite(mass))throw new RangeError("four-momentum invariant is non-finite");return mass;}let q=norm/c2;if(!Number.isFinite(q)||!Number.isFinite(c2)||c2<2**-1022||(norm!==0&&Math.abs(norm)<2**-1022)||(norm===0&&row.some(value=>value!==0)))q=-minkowskiDot(scaled,scaled);const magnitude=scaled.reduce((sum,value)=>sum+value**2,0);if(!Number.isFinite(q)||!Number.isFinite(magnitude)||magnitude<2**-1022){const scale=Math.max(...scaled.map(Math.abs));if(scale===0)return 0;const normalized=scaled.map(value=>value/scale),relative=-minkowskiDot(normalized,normalized),normalizedMagnitude=normalized.reduce((sum,value)=>sum+value**2,0),inverseScale=1/scale,relativeTolerance=64*Number.EPSILON*Math.max(inverseScale*inverseScale,normalizedMagnitude);if(relative< -relativeTolerance||relative<0&&!clampRoundoff)throw new RangeError("four-momentum is spacelike");if(relative<=0)return 0;const mass=scale*Math.sqrt(relative);if(!Number.isFinite(mass))throw new RangeError("four-momentum invariant is non-finite");return mass;}const tolerance=64*Number.EPSILON*Math.max(1,magnitude);if(q< -tolerance)throw new RangeError("four-momentum is spacelike");if(clampRoundoff&&q<0)q=0;if(q<0)throw new RangeError("four-momentum is spacelike");return q===0?0:Math.sqrt(q);});}
  boost(transform){return new FourMomentum(transformFourVectors(this.components,transform,"four-momentum"),{c:this.c});}
}

export class FourCurrent {
  constructor(components,{c=C}={}){const parsed=tensorShape(components,"four-current components",{trailingSize:4});this.components=parsed.value;this.c=finiteNumber(c,"c");if(this.c<=0)throw new RangeError("c must be positive");Object.freeze(this);}
  static fromChargeCurrent(rho,j,{c=C}={}){c=finiteNumber(c,"c");if(c<=0)throw new RangeError("c must be positive");const density=tensorShape(rho,"charge density"),current=tensorShape(j,"spatial current",{trailingSize:3}),densityShape=density.shape,currentBatchShape=current.shape.slice(0,-1),batchShape=broadcastShape(densityShape,currentBatchShape);if(batchShape.reduce((p,n)=>p*n,1)*4>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError("four-current exceeds the component limit");const components=nestedFrom(batchShape,index=>[finiteNumber(broadcastGet(density.value,densityShape,index,batchShape))*c,...broadcastGet(current.value,currentBatchShape,index,batchShape)]);return new FourCurrent(components,{c});}
  chargeDensity(){const parsed=tensorShape(this.components,"four-current components",{trailingSize:4});return mapLastAxis(parsed.value,row=>{const density=row[0]/this.c;if(!Number.isFinite(density))throw new RangeError("charge density is non-finite");return density;});}
  spatialCurrent(){const parsed=tensorShape(this.components,"four-current components",{trailingSize:4});return mapLastAxis(parsed.value,row=>row.slice(1));}
  transform(L){return new FourCurrent(transformFourVectors(this.components,L,"four-current"),{c:this.c});}
}

function scaleStressTerm(term,c,epsilon0){if(!Number.isFinite(term))throw new RangeError("stress-energy contraction is non-finite");if(term===0)return 0;const direct=epsilon0*c*c*term,result=Number.isFinite(direct)&&Math.abs(direct)>=2**-1022?direct:Math.sign(term)*Math.exp(Math.log(epsilon0)+2*Math.log(c)+Math.log(Math.abs(term)));if(!Number.isFinite(result))throw new RangeError("stress-energy component is non-finite");return result;}

function scaledElectricMagneticDot(electricOverC,magnetic,c){
  const terms=[];
  for(let i=0;i<3;i++)if(electricOverC[i]!==0&&magnetic[i]!==0)terms.push({sign:Math.sign(electricOverC[i])*Math.sign(magnetic[i]),log:Math.log(Math.abs(electricOverC[i]))+Math.log(Math.abs(magnetic[i]))+Math.log(c)});
  if(!terms.length)return 0;
  const largest=Math.max(...terms.map(term=>term.log));
  const relative=terms.reduce((sum,term)=>sum+term.sign*Math.exp(term.log-largest),0);
  if(relative===0)return 0;
  const result=Math.sign(relative)*Math.exp(Math.log(Math.abs(relative))+largest);
  if(!Number.isFinite(result))throw new RangeError("electromagnetic invariants are non-finite");
  return result;
}

export class EMFieldTensor {
  constructor(components,{c=C,epsilon0=8.8541878128e-12,validate=true}={}){const parsed=tensorShape(components,"electromagnetic tensor",{trailingShape:[4,4]});this.components=parsed.value;this.c=finiteNumber(c,"c");this.epsilon0=finiteNumber(epsilon0,"epsilon0");if(this.c<=0||this.epsilon0<=0)throw new RangeError("c and epsilon0 must be positive");if(validate){const batchRank=parsed.shape.length-2;mapBatches(this.components,batchRank,f=>{for(let i=0;i<4;i++)for(let j=0;j<4;j++)if(Math.abs(f[i][j]+f[j][i])>1e-10+1e-10*Math.abs(f[j][i]))throw new RangeError("electromagnetic tensor must be antisymmetric");});}Object.freeze(this);}
  static fromEB(electric,magnetic,{c=C,epsilon0=8.8541878128e-12}={}){c=finiteNumber(c,"c");epsilon0=finiteNumber(epsilon0,"epsilon0");if(c<=0||epsilon0<=0)throw new RangeError("c and epsilon0 must be positive");const e=tensorShape(electric,"electric field",{trailingSize:3}),b=tensorShape(magnetic,"magnetic field",{trailingSize:3}),eShape=e.shape.slice(0,-1),bShape=b.shape.slice(0,-1),shape=broadcastShape(eShape,bShape);if(shape.reduce((p,n)=>p*n,1)*16>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError("electromagnetic field batch exceeds the component limit");const components=nestedFrom(shape,index=>{const E=broadcastGet(e.value,eShape,index,shape),B=broadcastGet(b.value,bShape,index,shape),f=zeros(4);for(let i=0;i<3;i++){f[0][i+1]=E[i]/c;f[i+1][0]=-E[i]/c;}f[1][2]=B[2];f[2][1]=-B[2];f[1][3]=-B[1];f[3][1]=B[1];f[2][3]=B[0];f[3][2]=-B[0];return f;});return new EMFieldTensor(components,{c,epsilon0});}
  electricField(c=this.c){c=finiteNumber(c??this.c,"c");if(c<=0)throw new RangeError("c must be positive");const fields=tensorShape(this.components,"electromagnetic tensor",{trailingShape:[4,4]}),batchRank=fields.shape.length-2;return mapBatches(fields.value,batchRank,f=>{const result=f[0].slice(1).map(x=>x*c);if(result.some(value=>!Number.isFinite(value)))throw new RangeError("electric field is non-finite");return result;});}
  magneticField(){const batchRank=tensorShape(this.components,"electromagnetic tensor",{trailingShape:[4,4]}).shape.length-2;return mapBatches(this.components,batchRank,f=>[f[2][3],-f[1][3],f[1][2]]);}
  invariants(c=this.c){c=finiteNumber(c??this.c,"c");if(c<=0)throw new RangeError("c must be positive");const fields=tensorShape(this.components,"electromagnetic tensor",{trailingShape:[4,4]}),batchRank=fields.shape.length-2;const values=mapBatches(fields.value,batchRank,f=>{const electricOverC=f[0].slice(1),E=electricOverC.map(x=>x*c),B=[f[2][3],-f[1][3],f[1][2]];const invariantComponents=[...B,...electricOverC];let magneticMinusElectricOverC2=weightedDot(invariantComponents,invariantComponents,[1,1,1,-1,-1,-1]);if(magneticMinusElectricOverC2===0)magneticMinusElectricOverC2=0;let electricDotMagnetic=dot(E,B);if(!Number.isFinite(electricDotMagnetic)||E.some((value,index)=>!Number.isFinite(value)||(value===0&&electricOverC[index]!==0)||(value!==0&&B[index]!==0&&value*B[index]===0)))electricDotMagnetic=scaledElectricMagneticDot(electricOverC,B,c);if(!Number.isFinite(magneticMinusElectricOverC2)||!Number.isFinite(electricDotMagnetic))throw new RangeError("electromagnetic invariants are non-finite");return {magneticMinusElectricOverC2,electricDotMagnetic};});return {magneticMinusElectricOverC2:mapBatches(values,batchRank,value=>value.magneticMinusElectricOverC2),electricDotMagnetic:mapBatches(values,batchRank,value=>value.electricDotMagnetic)};}
  dual({orientation=1}={}){const fields=tensorShape(this.components,"electromagnetic tensor",{trailingShape:[4,4]}),batchRank=fields.shape.length-2;return new EMFieldTensor(mapBatches(fields.value,batchRank,f=>new Bivector4(f,{validate:false}).dual(new Metric4(),orientation).components),{c:this.c,epsilon0:this.epsilon0});}
  transform(L){return new EMFieldTensor(transformRankTwo(this.components,L,"electromagnetic tensor"),{c:this.c,epsilon0:this.epsilon0});}
  lorentzForce(charge,uCovariant){const fields=tensorShape(this.components,"electromagnetic tensor",{trailingShape:[4,4]}),u=tensorShape(uCovariant instanceof Covector4?uCovariant.components:uCovariant,"covariant four-velocity",{trailingSize:4}),q=tensorShape(charge,"charge"),fieldBatch=fields.shape.slice(0,-2),velocityBatch=u.shape.slice(0,-1),shape=broadcastShape(broadcastShape(fieldBatch,velocityBatch),q.shape);if(shape.reduce((p,n)=>p*n,1)*4>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError("Lorentz-force batch exceeds the component limit");return nestedFrom(shape,index=>{const f=broadcastGet(fields.value,fieldBatch,index,shape),v=broadcastGet(u.value,velocityBatch,index,shape),chargeValue=broadcastGet(q.value,q.shape,index,shape);const result=f.map(row=>weightedDot4(row,v,[chargeValue,chargeValue,chargeValue,chargeValue]));if(result.some(value=>!Number.isFinite(value)))throw new RangeError("Lorentz force is non-finite");return result;});}
  energyMomentumTensor(metric=null,{epsilon0=this.epsilon0}={}){epsilon0=finiteNumber(epsilon0,"epsilon0");if(epsilon0<=0)throw new RangeError("epsilon0 must be positive");const fields=tensorShape(this.components,"electromagnetic tensor",{trailingShape:[4,4]}),fBatch=fields.shape.slice(0,-2),defaultMetric=identity(4).map((r,i)=>r.map((x,j)=>x*(i===0?-1:1))),g=tensorShape(metric instanceof Metric4?metric._checkedComponents().data:metric?.components??metric??defaultMetric,"metric",{trailingShape:[4,4]}),shape=broadcastShape(fBatch,g.shape.slice(0,-2));if(shape.reduce((p,n)=>p*n,1)*16>MAX_MECHANICS_TENSOR_COMPONENTS)throw new RangeError("stress-energy batch exceeds the component limit");return nestedFrom(shape,index=>{const F=broadcastGet(fields.value,fBatch,index,shape),G=broadcastGet(g.value,g.shape.slice(0,-2),index,shape),Gi=inverse(G),lower=mmul(mmul(G,F),transpose(G));let contraction=0;for(let i=0;i<4;i++)for(let j=0;j<4;j++)contraction+=F[i][j]*lower[i][j];const out=zeros(4);for(let mu=0;mu<4;mu++)for(let nu=0;nu<4;nu++){let sum=0;for(let a=0;a<4;a++)for(let b=0;b<4;b++)sum+=F[mu][a]*G[a][b]*F[nu][b];out[mu][nu]=scaleStressTerm(sum-.25*Gi[mu][nu]*contraction,this.c,epsilon0);}return out;});}
}

export class MaxwellEquations {
  constructor({c=C,epsilon0=8.8541878128e-12}={}){finiteNumber(c,"c");finiteNumber(epsilon0,"epsilon0");if(c<=0||epsilon0<=0)throw new RangeError("c and epsilon0 must be positive");this.c=c;this.epsilon0=epsilon0;Object.freeze(this);}
  /** Coordinates for the conventional periodic Yee placement used by the discrete curls. */
  gridCoordinates({shape,origin=[0,0,0],spacing=[1,1,1]}={}){
    const dimensions=vector(shape,3,"shape"),start=vector(origin,3,"origin"),delta=vector(spacing,3,"spacing");
    validateCellShape(dimensions,3);
    if(dimensions.some(size=>size<2))throw new RangeError("each spatial grid dimension must have at least 2 points");
    if(delta.some(value=>value<=0))throw new RangeError("spacing must contain positive finite values");
    for(let axis=0;axis<3;axis++)if(!Number.isFinite(start[axis]+(dimensions[axis]-.5)*delta[axis]))throw new RangeError("grid coordinate is non-finite");
    const make=offset=>nestedFrom(dimensions,index=>index.map((value,axis)=>start[axis]+(value+offset[axis])*delta[axis]));
    return {electric:[[.5,0,0],[0,.5,0],[0,0,.5]].map(make),magnetic:[[0,.5,.5],[.5,0,.5],[.5,.5,0]].map(make),charge:make([0,0,0]),magneticDivergence:make([.5,.5,.5])};
  }
  /** Interpolate staggered fields to charge nodes at the electric sample time. */
  collocateFields(electric,magneticBefore,magneticAfter){
    const eShape=shapeOf(electric),beforeShape=shapeOf(magneticBefore),afterShape=shapeOf(magneticAfter);
    if(eShape.length>4){
      const batchShape=eShape.slice(0,-4),spatial=eShape.slice(-4,-1),cells=validateCellShape(spatial,3),count=shapeProduct(batchShape);
      if(eShape.at(-1)!==3||JSON.stringify(eShape)!==JSON.stringify(beforeShape)||JSON.stringify(eShape)!==JSON.stringify(afterShape))throw new RangeError("collocation grids must have matching (...,Nx,Ny,Nz,3) shapes");
      if(spatial.some(size=>size<2))throw new RangeError("each spatial grid dimension must have at least 2 points");
      if(count*cells*6>MAX_FIELD_HISTORY_COMPONENTS)throw new RangeError("collocation batch exceeds the bounded output-component limit");
      validateNestedShape(electric,eShape);validateNestedShape(magneticBefore,eShape);validateNestedShape(magneticAfter,eShape);
      const result=nestedFrom(batchShape,index=>this.collocateFields(prefixSlice(electric,index),prefixSlice(magneticBefore,index),prefixSlice(magneticAfter,index)));
      return {electric:mapBatches(result,batchShape.length,value=>value.electric),magnetic:mapBatches(result,batchShape.length,value=>value.magnetic)};
    }
    const e=grid(electric),before=grid(magneticBefore),after=grid(magneticAfter);sameGrid(e,before);sameGrid(e,after);
    const shape=e.shape;if(shape.some(size=>size<2))throw new RangeError("each spatial grid dimension must have at least 2 points");
    const sample=(field,index,component)=>field.values[(index[0]*shape[1]+index[1])*shape[2]+index[2]][component];
    const shifted=(index,axis)=>index.map((value,i)=>i===axis?(value+shape[i]-1)%shape[i]:value);
    const average2=(a,b)=>weightedDot4([a,b,0,0],[1,1,0,0],[.5,.5,0,0]);
    const average4=values=>weightedDot4(values,[1,1,1,1],[.25,.25,.25,.25]);
    const electricNodes=nestedFrom(shape,index=>Array.from({length:3},(_,axis)=>average2(sample(e,index,axis),sample(e,shifted(index,axis),axis))));
    const magneticNodes=nestedFrom(shape,index=>Array.from({length:3},(_,axis)=>{
      const other=[0,1,2].filter(i=>i!==axis),a=shifted(index,other[0]),b=shifted(index,other[1]),ab=shifted(a,other[1]);
      const spatial=field=>average4([sample(field,index,axis),sample(field,a,axis),sample(field,b,axis),sample(field,ab,axis)]);
      return average2(spatial(before),spatial(after));
    }));
    if([...electricNodes.flat(3),...magneticNodes.flat(3)].some(value=>!Number.isFinite(value)))throw new RangeError("collocated field is non-finite");
    return {electric:electricNodes,magnetic:magneticNodes};
  }
  /** Collocate each interior electric frame with its adjacent magnetic half-frames. */
  collocateHistory(electricHistory,magneticHalfHistory){
    const eShape=shapeOf(electricHistory),bShape=shapeOf(magneticHalfHistory);
    if(eShape.length<5||eShape.at(-1)!==3||eShape[0]<3||bShape.length!==eShape.length||bShape[0]!==eShape[0]-1||bShape.slice(1).some((size,i)=>size!==eShape[i+1]))throw new RangeError("collocation histories must have matching (frames,...,Nx,Ny,Nz,3) shapes with one more electric frame");
    const spatial=eShape.slice(-4,-1),cells=validateCellShape(spatial,3),batchShape=eShape.slice(1,-4),count=shapeProduct(batchShape),frames=eShape[0]-2;
    if(spatial.some(size=>size<2))throw new RangeError("each spatial grid dimension must have at least 2 points");
    if(frames*count*cells*6>MAX_FIELD_HISTORY_COMPONENTS)throw new RangeError("collocation history exceeds the bounded output-component limit");
    validateNestedShape(electricHistory,eShape);validateNestedShape(magneticHalfHistory,bShape);
    const samples=Array.from({length:frames},(_,index)=>this.collocateFields(electricHistory[index+1],magneticHalfHistory[index],magneticHalfHistory[index+1]));
    return {frameIndices:Array.from({length:frames},(_,index)=>index+1),electric:samples.map(value=>value.electric),magnetic:samples.map(value=>value.magnetic)};
  }
  maxwellResiduals(electricHistory,magneticHalfHistory,{spacing=[1,1,1],dt,current=null}={}){
    const eShape=shapeOf(electricHistory),rawBShape=shapeOf(magneticHalfHistory),emptyMagnetic=eShape[0]===1&&rawBShape.length===1&&rawBShape[0]===0,bShape=emptyMagnetic?[0,...eShape.slice(1)]:rawBShape;
    if(eShape.length>5){
      const batchShape=eShape.slice(1,-4),gridShape=eShape.slice(-4,-1),steps=eShape[0]-1,cells=validateCellShape(gridShape,3),batchCount=shapeProduct(batchShape);
      if(eShape.length!==batchShape.length+5||eShape.at(-1)!==3||(eShape[0]<2&&!emptyMagnetic)||bShape.length!==eShape.length||bShape[0]!==steps||bShape.slice(1).some((size,i)=>size!==eShape[i+1]))throw new RangeError("batched histories must have matching (frames,...,Nx,Ny,Nz,3) shapes with one more electric frame");
      if(gridShape.some(size=>size<2))throw new RangeError("each spatial grid dimension must have at least 2 points");
      if((2*steps+1)*cells*batchCount*3>MAX_FIELD_HISTORY_COMPONENTS)throw new RangeError("Maxwell history exceeds the bounded output-component limit");
      validateMaxwellControls(this.c,spacing,dt,steps,cells,{allowZeroSteps:emptyMagnetic});
      validateBatchedCurrentShape(current,batchShape,gridShape,this.c);
      const indices=batchIndices(batchShape),results=indices.map(index=>this.maxwellResiduals(electricHistory.map(frame=>prefixSlice(frame,index)),magneticHalfHistory.map(frame=>prefixSlice(frame,index)),{spacing,dt,current:currentBatchSlice(current,index)}));
      return {ampere:packBatchedHistory(results,"ampere",batchShape),faraday:packBatchedHistory(results,"faraday",batchShape)};
    }
    const [e,b]=gridHistoryPair(electricHistory,magneticHalfHistory,{allowZeroSteps:emptyMagnetic}),steps=e.frames.length-1,delta=validateMaxwellControls(this.c,spacing,dt,steps,e.cells,{allowZeroSteps:emptyMagnetic});
    const source=currentGrid(current,e.shape,this.c),ampere=[],faraday=[];
    for(let n=0;n<steps;n++){
      const curlB=curlDirected(b.frames[n],delta,"backward");
      const c2=this.c*this.c;
      const scaledCurl=Number.isFinite(c2)&&c2>=2**-1022?scaleGrid(curlB,c2):scaleGridSpeedSquared(curlB,this.c);
      const residual=subtractVectorGrids(differenceGridQuotient(e.frames[n+1],e.frames[n],dt),scaledCurl);
      ampere.push(ungrid(addVectorGrids(residual,scaleGridRatio(source.current,1,this.epsilon0))));
    }
    for(let n=0;n<steps-1;n++){
      const curlE=curlDirected(e.frames[n+1],delta,"forward");
      faraday.push(ungrid(addVectorGrids(differenceGridQuotient(b.frames[n+1],b.frames[n],dt),curlE)));
    }
    return {ampere,faraday};
  }
  constraintResiduals(electricHistory,magneticHalfHistory,{spacing=[1,1,1],dt,current=null}={}){
    const eShape=shapeOf(electricHistory),bShape=shapeOf(magneticHalfHistory);
    if(eShape.length>5){
      const batchShape=eShape.slice(1,-4),gridShape=eShape.slice(-4,-1),steps=eShape[0]-1,cells=validateCellShape(gridShape,3),batchCount=shapeProduct(batchShape);
      if(eShape.length!==batchShape.length+5||eShape.at(-1)!==3||eShape[0]<2||bShape.length!==eShape.length||bShape[0]!==steps||bShape.slice(1).some((size,i)=>size!==eShape[i+1]))throw new RangeError("batched histories must have matching (frames,...,Nx,Ny,Nz,3) shapes with one more electric frame");
      if(gridShape.some(size=>size<2))throw new RangeError("each spatial grid dimension must have at least 2 points");
      if((2*steps+1)*cells*batchCount*3>MAX_FIELD_HISTORY_COMPONENTS)throw new RangeError("Maxwell history exceeds the bounded output-component limit");
      validateMaxwellControls(this.c,spacing,dt,steps,cells);
      validateBatchedCurrentShape(current,batchShape,gridShape,this.c);
      const indices=batchIndices(batchShape),results=indices.map(index=>this.constraintResiduals(electricHistory.map(frame=>prefixSlice(frame,index)),magneticHalfHistory.map(frame=>prefixSlice(frame,index)),{spacing,dt,current:currentBatchSlice(current,index)}));
      return {gauss_e:packBatchedHistory(results,"gauss_e",batchShape),gauss_b:packBatchedHistory(results,"gauss_b",batchShape)};
    }
    const [e,b]=gridHistoryPair(electricHistory,magneticHalfHistory),steps=e.frames.length-1,delta=validateMaxwellControls(this.c,spacing,dt,steps,e.cells);
    const source=currentGrid(current,e.shape,this.c),divJ=divergenceDirected(source.current,delta,"backward"),gaussE=[],gaussB=[];
    for(let n=0;n<=steps;n++){
      const divE=divergenceDirected(e.frames[n],delta,"backward");
      const ordinaryPermittivity=this.epsilon0>=2**-1022&&Number.isFinite(1/this.epsilon0);
      gaussE.push(ungridScalar(divE.map((value,i)=>ordinaryPermittivity?value-(source.rho[i]-n*dt*divJ[i])/this.epsilon0:value-(scaleRatioValue(source.rho[i],1,this.epsilon0)-scaleRatioValue(divJ[i],n*dt,this.epsilon0))),e.shape));
    }
    for(const frame of b.frames)gaussB.push(ungridScalar(divergenceDirected(frame,delta,"forward"),e.shape));
    return {gauss_e:gaussE,gauss_b:gaussB};
  }
  energyMomentumTensor(fieldTensor,metric=null){if(!(fieldTensor instanceof EMFieldTensor))throw new TypeError("fieldTensor must be an EMFieldTensor");if(fieldTensor.c!==this.c||fieldTensor.epsilon0!==this.epsilon0)throw new RangeError("field tensor and Maxwell equations must share physical constants");return fieldTensor.energyMomentumTensor(metric??identity(4).map((r,i)=>r.map((x,j)=>x*(i===0?-1:1))));}
  /** Directional derivative with respect to initial fields and static current at fixed solver controls. */
  evolveDirectional(electric,magnetic,electricDirection,magneticDirection,{current=null,currentDirection=null,spacing=[1,1,1],dt,steps=1}={}){
    const shape=shapeOf(electric),sameShape=value=>JSON.stringify(shapeOf(value))===JSON.stringify(shape);
    if(!sameShape(magnetic)||!sameShape(electricDirection)||!sameShape(magneticDirection))throw new RangeError("fields and directions must have matching grid and batch shapes");
    const values=this.evolve(electric,magnetic,{current,spacing,dt,steps});
    // The fixed-control Yee update is linear in initial E, B, and static J.
    const direction=this.evolve(electricDirection,magneticDirection,{current:currentDirection,spacing,dt,steps});
    return {values,direction};
  }
  evolve(electric,magnetic,{current=null,spacing=[1,1,1],dt,steps=1}={}){
    if(!Number.isSafeInteger(steps)||steps<1)throw new RangeError("steps must be a positive safe integer");
    if(Number.isFinite(dt)&&dt>0&&(!Number.isFinite(steps*dt)||dt/2===0))throw new RangeError("Maxwell output time range is not representable");
    const eShape=shapeOf(electric),bShape=shapeOf(magnetic);
    if(eShape.length>4){
      const batchShape=eShape.slice(0,-4),gridShape=eShape.slice(-4,-1),cells=validateCellShape(gridShape,3),batchCount=shapeProduct(batchShape);
      if(batchShape.length>7)throw new RangeError("Maxwell batch rank must not exceed 7");
      if(eShape.length!==batchShape.length+4||eShape.at(-1)!==3||bShape.length!==eShape.length||bShape.some((size,i)=>size!==eShape[i]))throw new RangeError("batched fields must have matching (...,Nx,Ny,Nz,3) shapes");
      if(gridShape.some(size=>size<2))throw new RangeError("each spatial grid dimension must have at least 2 points");
      if((2*steps+1)*cells*batchCount*3>MAX_FIELD_HISTORY_COMPONENTS)throw new RangeError("Maxwell history exceeds the bounded output-component limit");
      validateMaxwellControls(this.c,spacing,dt,steps,cells);
      validateBatchedCurrentShape(current,batchShape,gridShape,this.c);
      const results=batchIndices(batchShape).map(index=>this.evolve(prefixSlice(electric,index),prefixSlice(magnetic,index),{current:currentBatchSlice(current,index),spacing,dt,steps}));
      return [results[0][0],packBatchedHistory(results,1,batchShape),results[0][2],packBatchedHistory(results,3,batchShape)];
    }
    const E0=grid(electric),B0=grid(magnetic);sameGrid(E0,B0);
    if(E0.shape.some(size=>size<2))throw new RangeError("each spatial grid dimension must have at least 2 points");
    const delta=validateMaxwellControls(this.c,spacing,dt,steps,E0.values.length),source=currentGrid(current,E0.shape,this.c);
    let E=E0,B=subtractVectorGrids(B0,scaleGrid(curlDirected(E0,delta,"forward"),.5*dt));
    const electricHistory=[ungrid(E)],magneticHistory=[ungrid(B)];
    for(let n=0;n<steps;n++){
      const curlB=curlDirected(B,delta,"backward");
      const c2=this.c*this.c;
      let next;
      if(Number.isFinite(c2)&&c2>=2**-1022&&Number.isFinite(1/this.epsilon0))next=addVectorGrids(E,scaleGrid(subtractVectorGrids(scaleGrid(curlB,c2),source.current,1/this.epsilon0),dt));
      if(!next||next.values.some(row=>row.some(value=>!Number.isFinite(value))))next=addVectorGrids(E,subtractVectorGrids(scaleGridSpeedSquared(curlB,this.c,dt),scaleGridRatio(source.current,dt,this.epsilon0)));
      if(next.values.some(row=>row.some(value=>!Number.isFinite(value))))throw new RangeError("Maxwell electric field update is non-finite");
      E=next;
      electricHistory.push(ungrid(E));
      if(n+1<steps){B=subtractVectorGrids(B,scaleGrid(curlDirected(E,delta,"forward"),dt));magneticHistory.push(ungrid(B));}
    }
    const electricTimes=Array.from({length:steps+1},(_,n)=>n*dt),magneticTimes=Array.from({length:steps},(_,n)=>(n+.5)*dt);
    return [electricTimes,electricHistory,magneticTimes,magneticHistory];
  }
}
function shapeOf(x){const s=[],seen=new Set();while(Array.isArray(x)){if(seen.has(x))throw new RangeError("grid/history arrays cannot be cyclic");if(s.length>=12)throw new RangeError("grid/history array nesting exceeds the supported rank");seen.add(x);s.push(x.length);x=x[0];}return s;}
function validateCellShape(shape,rank){if(shape.length!==rank||shape.some(x=>!Number.isSafeInteger(x)||x<1))throw new RangeError(`grid must have ${rank} nonempty dimensions`);const cells=shape.reduce((a,b)=>a*b,1);if(!Number.isSafeInteger(cells)||cells>MAX_GRID_CELLS)throw new RangeError(`grid exceeds the ${MAX_GRID_CELLS}-cell limit`);return cells;}
function validateNestedShape(value,shape,depth=0){if(depth===shape.length){finiteNumber(value,"grid value");return;}if(!Array.isArray(value)||value.length!==shape[depth])throw new RangeError("grid must be rectangular and match its declared dimensions");for(const child of value)validateNestedShape(child,shape,depth+1);}
function flattenGrid(x,shape=shapeOf(x)){if(shape.length!==4||shape[3]!==3)throw new RangeError("field grid must have shape [Nx,Ny,Nz,3]");validateCellShape(shape.slice(0,3),3);validateNestedShape(x,shape);const out=[];for(const a of x)for(const b of a)for(const c of b)out.push(vector(c,3));return {values:out,shape:shape.slice(0,3)};}
function ungrid(g){const [nx,ny,nz]=g.shape,out=[];let n=0;for(let i=0;i<nx;i++){const a=[];for(let j=0;j<ny;j++){const b=[];for(let k=0;k<nz;k++){const cell=g.values[n++];if(cell.some(value=>!Number.isFinite(value)))throw new RangeError("Maxwell grid result is non-finite");b.push(cell.slice());}a.push(b);}out.push(a);}return out;}
function grid(x){return flattenGrid(x);}
function sameGrid(a,b){if(a.shape.some((x,i)=>x!==b.shape[i]))throw new RangeError("field grid dimensions must match");}

function gridHistory(history,name,components,minFrames=2){
  const shape=shapeOf(history);
  if(shape.length!==5||shape[4]!==components||shape[0]<minFrames)throw new RangeError(`${name} history must have shape (frames,Nx,Ny,Nz,${components})`);
  const cells=validateCellShape(shape.slice(1,4),3);
  if(shape.slice(1,4).some(size=>size<2))throw new RangeError("each spatial grid dimension must have at least 2 points");
  validateNestedShape(history,shape);
  return {frames:history.map(frame=>flattenGrid(frame)),shape:shape.slice(1,4),cells};
}
function gridHistoryPair(electricHistory,magneticHistory,{allowZeroSteps=false}={}){
  const eShape=shapeOf(electricHistory),bShape=shapeOf(magneticHistory);
  const emptyMagnetic=allowZeroSteps&&eShape[0]===1&&bShape.length===1&&bShape[0]===0;
  if(eShape.length!==5||eShape[4]!==3||(eShape[0]<2&&!emptyMagnetic)||(!emptyMagnetic&&(bShape.length!==5||bShape[4]!==3||bShape[0]!==eShape[0]-1||eShape.slice(1,4).some((size,axis)=>size!==bShape[axis+1]))))throw new RangeError("electric history must have one more frame than magnetic half-step history on the same grid");
  const cells=validateCellShape(eShape.slice(1,4),3);
  if(eShape.slice(1,4).some(size=>size<2))throw new RangeError("each spatial grid dimension must have at least 2 points");
  if((2*(eShape[0]-1)+1)*cells*3>MAX_FIELD_HISTORY_COMPONENTS)throw new RangeError("Maxwell history exceeds the bounded output-component limit");
  const electric=gridHistory(electricHistory,"electric",3,emptyMagnetic?1:2);
  return [electric,emptyMagnetic?{frames:[],shape:electric.shape.slice(),cells}:gridHistory(magneticHistory,"magnetic",3,1)];
}
function shapeProduct(shape){if(shape.some(value=>!Number.isSafeInteger(value)||value<1))throw new RangeError("batch dimensions must be positive safe integers");const size=shape.reduce((product,value)=>product*value,1);if(!Number.isSafeInteger(size))throw new RangeError("batch shape exceeds the supported size");return size;}
function batchIndices(shape){const indices=[];function visit(prefix){if(prefix.length===shape.length){indices.push(prefix);return;}for(let i=0;i<shape[prefix.length];i++)visit([...prefix,i]);}visit([]);return indices;}
function prefixSlice(value,index){let result=value;for(const axis of index)result=result[axis];return result;}
function validateBatchedCurrentShape(current,batchShape,gridShape,c){
  if(current==null)return;
  if(current.c!==undefined&&current.c!==c)throw new RangeError("FourCurrent and MaxwellEquations must use the same c");
  const value=current.components??current,shape=shapeOf(value);
  if(shape.length!==batchShape.length+4||batchShape.some((size,axis)=>shape[axis]!==size)||gridShape.some((size,axis)=>shape[batchShape.length+axis]!==size)||![3,4].includes(shape.at(-1)))throw new RangeError("batched current grid must match the fields' leading batch and spatial grid dimensions and end in 3 or 4 components");
}
function currentBatchSlice(current,index){if(current==null)return null;const value=current.components??current,sliced=prefixSlice(value,index);return current.components?{components:sliced,c:current.c}:sliced;}
function packBatchedHistory(results,key,batchShape){
  const frameCount=results[0]?.[key]?.length??0;
  return Array.from({length:frameCount},(_,frame)=>nestedFrom(batchShape,index=>results[linearBatchIndex(index,batchShape)][key][frame]));
}
function linearBatchIndex(index,shape){return index.reduce((linear,value,axis)=>linear*shape[axis]+value,0);}
function validateMaxwellControls(c,spacing,dt,steps,cells,{allowZeroSteps=false}={}){
  if(!Array.isArray(spacing)||spacing.length!==3||Array.from(spacing).some(value=>!Number.isFinite(value)||value<=0))throw new RangeError("spacing must contain three positive finite spatial steps");
  if(!Number.isFinite(dt)||dt<=0)throw new RangeError("dt must be positive and finite");
  if(!Number.isSafeInteger(steps)||steps<(allowZeroSteps?0:1))throw new RangeError(allowZeroSteps?"steps must be a nonnegative safe integer":"steps must be a positive safe integer");
  const cfl=c*dt*Math.hypot(...spacing.map(h=>1/h));
  let unstable=cfl>1;
  if(!Number.isFinite(cfl)||cfl===0){
    const logs=spacing.map(h=>Math.log(c)+Math.log(dt)-Math.log(h)),largest=Math.max(...logs);
    unstable=largest>0||largest+.5*Math.log(logs.reduce((sum,value)=>sum+Math.exp(2*(value-largest)),0))>0;
  }
  if(unstable)throw new RangeError("unstable timestep: violates the 3D Yee CFL limit");
  if((2*steps+1)*cells*3>MAX_FIELD_HISTORY_COMPONENTS)throw new RangeError("Maxwell history exceeds the bounded output-component limit");
  return spacing;
}
function currentGrid(current,shape,c){
  const cells=shape.reduce((product,size)=>product*size,1);
  if(current==null)return {rho:Array(cells).fill(0),current:{shape:shape.slice(),values:Array.from({length:cells},()=>[0,0,0])}};
  if(current.c!==undefined&&current.c!==c)throw new RangeError("FourCurrent and MaxwellEquations must use the same c");
  const value=current.components??current,dimensions=shapeOf(value);
  if(dimensions.length!==4||dimensions.slice(0,3).some((size,axis)=>size!==shape[axis])||![3,4].includes(dimensions[3]))throw new RangeError("current grid must match the spatial grid and have shape [Nx,Ny,Nz,3] or [Nx,Ny,Nz,4]");
  validateNestedShape(value,dimensions);
  const rho=[],values=[];
  for(const plane of value)for(const row of plane)for(const cell of row){rho.push(dimensions[3]===4?cell[0]/c:0);values.push(cell.slice(dimensions[3]===4?1:0));}
  return {rho,current:{shape:shape.slice(),values}};
}
function indexOf(shape,i,j,k){const [nx,ny,nz]=shape;return ((i+nx)%nx*ny+(j+ny)%ny)*nz+(k+nz)%nz;}
function differenceQuotient(a,b,step,kind){const difference=a-b,result=Number.isFinite(difference)?difference/step:a/step-b/step;if(!Number.isFinite(result))throw new RangeError(`Maxwell ${kind} derivative is non-finite`);return result;}
function differenceGridQuotient(a,b,step){sameGrid(a,b);return {shape:a.shape.slice(),values:a.values.map((row,i)=>row.map((value,axis)=>differenceQuotient(value,b.values[i][axis],step,"time")))};}
function derivativeDirected(grid,axis,spacing,direction){
  const [nx,ny,nz]=grid.shape,out=grid.values.map(()=>[0,0,0]),step=direction==="forward"?1:-1;
  for(let i=0;i<nx;i++)for(let j=0;j<ny;j++)for(let k=0;k<nz;k++){
    const point=[i,j,k],neighbor=point.slice();neighbor[axis]+=step;
    const here=grid.values[indexOf(grid.shape,i,j,k)],there=grid.values[indexOf(grid.shape,...neighbor)],at=indexOf(grid.shape,i,j,k);
    out[at]=here.map((value,component)=>{
      const a=direction==="forward"?there[component]:value,b=direction==="forward"?value:there[component];
      return differenceQuotient(a,b,spacing[axis],"spatial");
    });
  }
  return {shape:grid.shape.slice(),values:out};
}
function curlDirected(grid,spacing,direction){const d=spacing.map((_,axis)=>derivativeDirected(grid,axis,spacing,direction));return {shape:grid.shape.slice(),values:grid.values.map((_,i)=>[d[1].values[i][2]-d[2].values[i][1],d[2].values[i][0]-d[0].values[i][2],d[0].values[i][1]-d[1].values[i][0]])};}
function divergenceDirected(grid,spacing,direction){const d=spacing.map((_,axis)=>derivativeDirected(grid,axis,spacing,direction));return grid.values.map((_,i)=>d.reduce((sum,field,axis)=>sum+field.values[i][axis],0));}
function addVectorGrids(a,b){sameGrid(a,b);return {shape:a.shape.slice(),values:a.values.map((row,i)=>row.map((value,axis)=>value+b.values[i][axis]))};}
function subtractVectorGrids(a,b,factor=1){sameGrid(a,b);return {shape:a.shape.slice(),values:a.values.map((row,i)=>row.map((value,axis)=>value-factor*b.values[i][axis]))};}
function scaleGrid(grid,factor){return {shape:grid.shape.slice(),values:grid.values.map(row=>row.map(value=>value*factor))};}
function scaleGridSpeedSquared(grid,c,dt=1){return {shape:grid.shape.slice(),values:grid.values.map(row=>row.map(value=>{if(value===0)return 0;if(!Number.isFinite(value))throw new RangeError("Maxwell field update is non-finite");const result=Math.sign(value)*Math.exp(Math.log(Math.abs(value))+2*Math.log(c)+Math.log(dt));if(!Number.isFinite(result))throw new RangeError("Maxwell field update is non-finite");return result;}))};}
function scaleRatioValue(value,numerator,denominator){if(value===0||numerator===0)return 0;if(!Number.isFinite(value)||!Number.isFinite(numerator))throw new RangeError("Maxwell field update is non-finite");const direct=value/denominator*numerator,result=Number.isFinite(direct)&&direct!==0?direct:Math.sign(value)*Math.exp(Math.log(Math.abs(value))+Math.log(numerator)-Math.log(denominator));if(!Number.isFinite(result))throw new RangeError("Maxwell field update is non-finite");return result;}
function scaleGridRatio(grid,numerator,denominator){const ratio=numerator/denominator;if(Number.isFinite(ratio)&&ratio>=2**-1022)return scaleGrid(grid,ratio);return {shape:grid.shape.slice(),values:grid.values.map(row=>row.map(value=>scaleRatioValue(value,numerator,denominator)))};}
function ungridScalar(values,shape){const [nx,ny,nz]=shape,out=[];let index=0;for(let i=0;i<nx;i++){const plane=[];for(let j=0;j<ny;j++){const row=[];for(let k=0;k<nz;k++){const value=values[index++];if(!Number.isFinite(value))throw new RangeError("Maxwell grid result is non-finite");row.push(value);}plane.push(row);}out.push(plane);}return out;}

export class RetardedField {
  constructor(sourceWorldline,sourceVelocity,charge,maxSpeed,{c=C,epsilon0=8.8541878128e-12,maxIterations=96,tolerance=1e-12}={}){finiteNumber(charge,"charge");finiteNumber(maxSpeed,"maxSpeed");finiteNumber(c,"c");finiteNumber(epsilon0,"epsilon0");if(typeof sourceWorldline!=="function"||typeof sourceVelocity!=="function")throw new TypeError("sourceWorldline and sourceVelocity must be functions");if(maxSpeed<0||maxSpeed>=c||c<=0||epsilon0<=0)throw new RangeError("invalid retarded-field physical constants");if(!Number.isInteger(maxIterations)||maxIterations<1||maxIterations>4096)throw new RangeError("maxIterations must be an integer from 1 to 4096");if(!Number.isFinite(tolerance)||tolerance<=0)throw new RangeError("tolerance must be positive and finite");this.sourceWorldline=sourceWorldline;this.sourceVelocity=sourceVelocity;this.charge=charge;this.maxSpeed=maxSpeed;this.c=c;this.epsilon0=epsilon0;this.maxIterations=maxIterations;this.tolerance=tolerance;}
  retardedTime(event){
    const parsed=tensorShape(event,"event",{trailingSize:4}),samples=parsed.shape.slice(0,-1).reduce((count,size)=>count*size,1);if(samples*(this.maxIterations+3)>MAX_RETARDED_FIELD_BATCH_WORK)throw new RangeError("retarded-time batch exceeds the bounded root-work limit");return mapLastAxis(parsed.value,x=>this._retardedTimeScalar(x));
  }
  _retardedTimeScalar(x){
    const t=x[0]/this.c,position=x.slice(1);if(!Number.isFinite(t))throw new RangeError("observation time is non-finite");
    const now=vector(this.sourceWorldline(t),3,"source position"),distance=norm(sub(position,now));
    const coincident=distance<=64*Number.EPSILON*Math.max(1,norm(position),norm(now));
    const span=distance/(this.c-this.maxSpeed)+this.tolerance;
    let lo=t-span,hi=t;if(!Number.isFinite(distance)||!Number.isFinite(lo)||!Number.isFinite(span))throw new RangeError("retarded-time bracket is non-finite");
    const lowerSource=vector(this.sourceWorldline(lo),3,"source position"),lowerDistance=norm(sub(position,lowerSource)),lowerLightDistance=this.c*(t-lo),lowerResidual=lowerLightDistance-lowerDistance;
    if(!Number.isFinite(lowerResidual)||!Number.isFinite(lowerDistance)||!Number.isFinite(lowerLightDistance))throw new RangeError("retarded-time bracket residual is non-finite");
    if(lowerResidual < -64*Number.EPSILON*Math.max(lowerLightDistance,lowerDistance))throw new RangeError("retarded-time bracket does not contain a root; check source motion and maxSpeed");
    for(let i=0;i<this.maxIterations;i++){
      const mid=lo+(hi-lo)/2,source=vector(this.sourceWorldline(mid),3,"source position"),f=this.c*(t-mid)-norm(sub(position,source));
      if(!Number.isFinite(f))throw new RangeError("retarded-time residual is non-finite");
      if(f===0)return coincident?t:mid;
      if(mid===lo||mid===hi){
        if(hi-lo<=this.tolerance)return coincident?t:mid;
        const other=mid===lo?hi:lo,otherSource=vector(this.sourceWorldline(other),3,"source position"),otherResidual=this.c*(t-other)-norm(sub(position,otherSource));
        if(!Number.isFinite(otherResidual))throw new RangeError("retarded-time residual is non-finite");
        if(otherResidual===0)return coincident?t:other;
        throw new RangeError("retarded-time bracket cannot be refined at this floating-point scale");
      }
      if(f>0)lo=mid;else hi=mid;
      if(hi-lo<=this.tolerance)return coincident?t:lo+(hi-lo)/2;
    }
    throw new RangeError("retarded-time bisection did not converge within maxIterations");
  }
  retardedPotential(event){
    const parsed=tensorShape(event,"event",{trailingSize:4}),samples=parsed.shape.slice(0,-1).reduce((count,size)=>count*size,1);if(samples*(this.maxIterations+4)>MAX_RETARDED_FIELD_BATCH_WORK)throw new RangeError("retarded-potential batch exceeds the bounded work limit");return mapLastAxis(parsed.value,x=>this._retardedPotentialScalar(x));
  }
  _retardedPotentialScalar(x){
    const tr=this._retardedTimeScalar(x),position=x.slice(1),source=vector(this.sourceWorldline(tr),3,"source position"),velocity=vector(this.sourceVelocity(tr),3,"source velocity"),speed=norm(velocity),radius=norm(sub(position,source));
    const positionScale=Math.max(1,norm(position));
    if(speed>=this.c)throw new RangeError("source velocity must remain strictly subluminal");
    if(speed>this.maxSpeed+64*Number.EPSILON*this.c)throw new RangeError("source velocity exceeded the declared maxSpeed bound");
    if(radius===0||radius<=64*Number.EPSILON*positionScale)throw new RangeError("retarded potential is singular on or too close to the source worldline at this coordinate scale");
    const beta=velocity.map(component=>component/this.c),direction=sub(position,source).map(component=>component/radius),denominator=radius*(1-dot(direction,beta));
    if(!Number.isFinite(denominator)||denominator<=0)throw new RangeError("invalid source velocity produced a nonpositive or non-finite retarded denominator");
    const divisor=4*Math.PI*this.epsilon0*denominator;
    const phi=Number.isFinite(divisor)&&divisor>=2**-1022?this.charge/divisor:this.charge===0?0:Math.sign(this.charge)*Math.exp(Math.log(Math.abs(this.charge))-Math.log(4*Math.PI)-Math.log(this.epsilon0)-Math.log(denominator));
    let temporal=phi/this.c;
    if(this.charge!==0&&(!Number.isFinite(phi)||Math.abs(phi)<2**-1022||!Number.isFinite(temporal)||temporal===0))temporal=Math.sign(this.charge)*Math.exp(Math.log(Math.abs(this.charge))-Math.log(4*Math.PI)-Math.log(this.epsilon0)-Math.log(denominator)-Math.log(this.c));
    const result=[temporal,...beta.map(component=>temporal*component)];
    if(result.some(value=>!Number.isFinite(value)))throw new RangeError("retarded potential is non-finite");
    return result;
  }
  evaluate(event){return this.retardedPotential(event);}
  sample(events){return this.retardedPotential(events);}
}
