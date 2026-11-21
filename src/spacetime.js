import {weightedDot4} from "./weighted-dot-internal.js";
import {C, add, sub, scale, dot, norm, vector, matrix, mvec, mmul, transpose, inverse, minkowskiDot, identity, zeros, finiteNumber} from "./array.js";
import {Vector4, Metric4, LorentzTransform, PoincareTransform} from "./algebra.js";

const MAX_INTEGRATION_STEPS=10_000;
const MAX_INTEGRATION_COMPONENTS=1_000_000;
const MAX_CONE_SAMPLES=65_536;

function mapVectorLeaves(value,fn){if(!Array.isArray(value))throw new TypeError("batched four-vectors must be arrays");if(value.length===4&&value.every(component=>typeof component==="number"))return fn(value);return value.map(item=>mapVectorLeaves(item,fn));}
function mapScalarTree(value,fn){return Array.isArray(value)?value.map(item=>mapScalarTree(item,fn)):fn(value);}
function mapVectorsWithScalars(vectors,scalars,fn){if(vectors.length===4&&vectors.every(component=>typeof component==="number"))return fn(vectors,scalars);return vectors.map((item,index)=>mapVectorsWithScalars(item,scalars[index],fn));}
function mapScalarPairs(a,b,fn){if(Array.isArray(a)){if(!Array.isArray(b)||a.length!==b.length)throw new RangeError("scalar batch shapes do not match");return a.map((value,index)=>mapScalarPairs(value,b[index],fn));}if(Array.isArray(b))throw new RangeError("scalar batch shapes do not match");return fn(a,b);}
function everyScalar(value,predicate){return Array.isArray(value)?value.every(item=>everyScalar(item,predicate)):predicate(value);}
const MAX_SPACETIME_COMPONENTS=1_000_000;
function parseNumericTree(value,name,trailingShape=[]){let leaves=0;const active=new Set();function copy(node,depth){if(ArrayBuffer.isView(node)&&!(node instanceof DataView)){if(node.length>MAX_SPACETIME_COMPONENTS)throw new RangeError(`${name} exceeds the component limit`);node=Array.from(node);}if(Array.isArray(node)){if(node.length>MAX_SPACETIME_COMPONENTS)throw new RangeError(`${name} exceeds the component limit`);if(depth>=8)throw new RangeError(`${name} rank exceeds eight`);if(node.length===0)throw new RangeError(`${name} batch dimensions must be nonempty`);if(active.has(node))throw new TypeError(`${name} cannot contain cycles`);active.add(node);const out=Array.from(node,item=>copy(item,depth+1));active.delete(node);return out;}finiteNumber(node,name);if(++leaves>MAX_SPACETIME_COMPONENTS)throw new RangeError(`${name} exceeds the component limit`);return node;}const data=copy(value,0),shape=[];let cursor=data;while(Array.isArray(cursor)){shape.push(cursor.length);cursor=cursor[0];}function verify(node,axis){if(axis===shape.length)return;if(!Array.isArray(node)||node.length!==shape[axis])throw new RangeError(`${name} must be rectangular`);for(const item of node)verify(item,axis+1);}verify(data,0);if(shape.length<trailingShape.length||trailingShape.some((size,index)=>shape[shape.length-trailingShape.length+index]!==size))throw new RangeError(`${name} must end in shape ${trailingShape.join(" by ")}`);return {data,shape,batchShape:shape.slice(0,shape.length-trailingShape.length)};}
function parseSpacetimeParameters(value){try{return parseNumericTree(value,"parameter");}catch(error){if(error instanceof TypeError&&error.message==="parameter must be a finite number")throw new RangeError("parameter must be finite");throw error;}}
function spacetimeBatchShape(a,b){const rank=Math.max(a.length,b.length),out=Array(rank);for(let i=0;i<rank;i++){const x=a[a.length-rank+i]??1,y=b[b.length-rank+i]??1;if(x!==y&&x!==1&&y!==1)throw new RangeError("batch shapes are not broadcast-compatible");out[i]=x===1?y:y===1?x:x;}return out;}
function spacetimeBatchGet(data,shape,index,target){let node=data;const offset=target.length-shape.length;for(let axis=0;axis<shape.length;axis++)node=node[shape[axis]===1?0:index[offset+axis]];return node;}
function buildSpacetimeBatch(shape,fn,index=[]){if(index.length===shape.length)return fn(index);return Array.from({length:shape[index.length]},(_,i)=>buildSpacetimeBatch(shape,fn,[...index,i]));}
function product(shape){return shape.reduce((value,size)=>value*size,1);}
function boundSpacetimeBatch(shape,components){const size=product(shape)*components;if(!Number.isSafeInteger(size)||size>MAX_SPACETIME_COMPONENTS)throw new RangeError("spacetime batch exceeds the component limit");}
function batchMinkowskiDot(left,right){return minkowskiDot(left,right);}
function squaredScaleTolerance(v,atol){return Math.max(atol,weightedDot4(v,v,[atol,atol,atol,atol]));}
function timeScaleTolerance(v,atol){const scale=Math.max(...v.map(Math.abs));if(scale===0||atol===0)return atol;const length=Math.hypot(...v.map(value=>value/scale)),product=atol*scale;return Math.max(atol,(product<2**-1022?atol*length*scale:product*length));}

export class Event4 {
  constructor(position,frame=null,c=C){this.position=new Vector4(position).components;if(!Number.isFinite(c)||c<=0)throw new RangeError("c must be positive and finite");this.frame=frame;this.c=c;Object.freeze(this);}
  displacementTo(other){if(!(other instanceof Event4))throw new TypeError("other must be an Event4");return new Vector4(other.position).subtract(this.position).components;}
  intervalTo(other,{c=this.c,atol=1e-12}={}){return new SpacetimeInterval(this.displacementTo(other),{c,atol});}
  causalRelation(other,{atol=1e-12}={}){if(!Number.isFinite(atol)||atol<0)throw new RangeError("atol must be nonnegative and finite");const d=this.displacementTo(other),s=new Vector4(d).normSquared();return mapVectorsWithScalars(d,s,(v,squared)=>{if(!Number.isFinite(squared))throw new RangeError("squared interval is non-finite");const q=squared,tolerance=squaredScaleTolerance(v,atol),timeTolerance=timeScaleTolerance(v,atol);if(!Number.isFinite(tolerance)||!Number.isFinite(timeTolerance))throw new RangeError("causal classification scale is non-finite");if(q< -tolerance)return v[0]>=0?"future_timelike":"past_timelike";if(q>tolerance)return "spacelike";if(v[0]>timeTolerance)return "future_null";if(v[0]<-timeTolerance)return "past_null";return "coincident";});}
  transform(transform){let p;if(transform instanceof PoincareTransform)p=new Vector4(this.position).transform(transform.lorentz.matrix).add(transform.translation).components;else if(transform instanceof LorentzTransform)p=new Vector4(this.position).transform(transform.matrix).components;else throw new TypeError("transform must be a LorentzTransform or PoincareTransform");return new Event4(p,null,this.c);}
}

export class SpacetimeInterval {
  constructor(displacement,{c=C,atol=1e-12}={}){this.displacement=new Vector4(displacement).components;if(!Number.isFinite(c)||c<=0)throw new RangeError("c must be positive and finite");if(!Number.isFinite(atol)||atol<0)throw new RangeError("atol must be nonnegative and finite");this.c=c;this.atol=atol;this.squared;Object.freeze(this);}
  get squared(){const value=new Vector4(this.displacement).normSquared();if(!everyScalar(value,Number.isFinite))throw new RangeError("squared interval is non-finite");return value;}
  tolerance(){return mapVectorLeaves(this.displacement,vector=>{const tolerance=squaredScaleTolerance(vector,this.atol);if(!Number.isFinite(tolerance))throw new RangeError("interval classification scale is non-finite");return tolerance;});}
  intervalSquared(){return this.squared;}
  isTimelike(){return mapScalarPairs(this.squared,this.tolerance(),(squared,tolerance)=>squared< -tolerance);}
  isSpacelike(){const tolerance=this.tolerance();return mapScalarPairs(this.squared,tolerance,(squared,tol)=>squared>tol);}
  isNull(){const tolerance=this.tolerance();return mapScalarPairs(this.squared,tolerance,(squared,tol)=>Math.abs(squared)<=tol);}
  properTime(){const valid=this.isTimelike();if(!everyScalar(valid,value=>value))throw new RangeError("proper time requires timelike separation");return mapScalarTree(this.squared,value=>Math.sqrt(-value)/this.c);}
  properDistance(){const valid=this.isSpacelike();if(!everyScalar(valid,value=>value))throw new RangeError("proper distance requires spacelike separation");return mapScalarTree(this.squared,value=>Math.sqrt(value));}
}

function unitSpatialDirection(direction){const magnitude=Math.max(...direction.map(Math.abs));if(magnitude===0)throw new RangeError("directions must be finite and nonzero");const scaled=direction.map(value=>value/magnitude),length=Math.hypot(...scaled);return scaled.map(value=>value/length);}

export class LightCone {
  constructor(apex,{branch="both",c=C,atol=1e-10}={}){if(!(apex instanceof Event4)&&!Array.isArray(apex)&&!(ArrayBuffer.isView(apex)&&!(apex instanceof DataView)))throw new TypeError("apex must be an Event4 or four-component position");if(!["future","past","both"].includes(branch))throw new TypeError("branch must be future, past, or both");if(!Number.isFinite(c)||c<=0||!Number.isFinite(atol)||atol<0)throw new RangeError("c must be positive and atol must be nonnegative");this.apex=apex instanceof Event4?apex:new Event4(apex,null,c);this.apexBatchShape=parseNumericTree(this.apex.position,"cone apex",[4]).batchShape;this.branch=branch;this.c=c;this.atol=atol;Object.freeze(this);}
  _delta(event){return new Vector4(event instanceof Event4?event.position:new Event4(event).position).subtract(this.apex.position).components;}
  onSurface(event){const d=this._delta(event),s=new Vector4(d).normSquared();return mapVectorsWithScalars(d,s,(delta,squared)=>{const tolerance=squaredScaleTolerance(delta,this.atol);if(!Number.isFinite(squared)||!Number.isFinite(tolerance))throw new RangeError("cone surface classification is non-finite");if(Math.abs(squared)>tolerance)return false;if(this.branch==="future"&&delta[0]<-this.atol)return false;if(this.branch==="past"&&delta[0]>this.atol)return false;return true;});}
  contains(event){const d=this._delta(event),s=new Vector4(d).normSquared();return mapVectorsWithScalars(d,s,(delta,squared)=>{const tolerance=squaredScaleTolerance(delta,this.atol);if(!Number.isFinite(squared)||!Number.isFinite(tolerance))throw new RangeError("cone containment classification is non-finite");if(squared>tolerance)return false;if(this.branch==="future"&&delta[0]<-this.atol)return false;if(this.branch==="past"&&delta[0]>this.atol)return false;return true;});}
  sample(directions,spatialDistance){if(this.branch==="both")throw new RangeError("sample requires a future or past branch");if(Array.isArray(directions)&&directions.length>MAX_CONE_SAMPLES)throw new RangeError(`cone sampling exceeds the ${MAX_CONE_SAMPLES}-direction limit`);const parsedDirections=parseNumericTree(directions,"directions",[3]),parsedRadius=parseNumericTree(spatialDistance,"spatialDistance"),shape=spacetimeBatchShape(spacetimeBatchShape(parsedDirections.batchShape,parsedRadius.shape),this.apexBatchShape);if(product(parsedDirections.batchShape)>MAX_CONE_SAMPLES||product(shape)*4>MAX_SPACETIME_COMPONENTS)throw new RangeError("cone sampling exceeds its bounded direction or component limit");const sign=this.branch==="past"?-1:1;return buildSpacetimeBatch(shape,index=>{const direction=spacetimeBatchGet(parsedDirections.data,parsedDirections.batchShape,index,shape),radius=spacetimeBatchGet(parsedRadius.data,parsedRadius.shape,index,shape),unit=unitSpatialDirection(direction),apex=spacetimeBatchGet(this.apex.position,this.apexBatchShape,index,shape);if(!Number.isFinite(radius)||radius<0)throw new RangeError("spatialDistance must be finite and nonnegative");const event=[apex[0]+sign*radius,...apex.slice(1).map((value,axis)=>value+unit[axis]*radius)];if(event.some(value=>!Number.isFinite(value)))throw new RangeError("sampled cone event is non-finite");return event;});}
  intersectWorldline(origin,direction){
    if(this.apexBatchShape.length)throw new RangeError("intersectWorldline requires an unbatched cone apex");
    const p=vector(origin instanceof Event4?origin.position:origin,4,"origin"),v=vector(direction,4,"direction"),d=sub(p,this.apex.position);
    if(d.some(value=>!Number.isFinite(value)))throw new RangeError("worldline intersection offset is non-finite");
    const dMaximum=Math.max(...d.map(Math.abs)),vMaximum=Math.max(...v.map(Math.abs));
    const dScale=this.atol===0?(dMaximum||1):Math.max(1,dMaximum),vScale=this.atol===0?(vMaximum||1):Math.max(1,vMaximum),dn=d.map(value=>value/dScale),vn=v.map(value=>value/vScale);
    if(vn.some(value=>value!==0)&&dot(vn,vn)===0)throw new RangeError("worldline direction scale is non-finite or underflows");
    const a=minkowskiDot(vn,vn),b=2*minkowskiDot(dn,vn),c0=minkowskiDot(dn,dn),vUnitScale=this.atol===0?0:Math.max((1/vScale)**2,dot(vn,vn)),dUnitScale=this.atol===0?0:Math.max((1/dScale)**2,dot(dn,dn));
    const aTolerance=this.atol*vUnitScale,bTolerance=2*this.atol*Math.sqrt(vUnitScale)*Math.sqrt(dUnitScale),cTolerance=this.atol*dUnitScale;
    if(![a,b,c0,aTolerance,bTolerance,cTolerance].every(Number.isFinite))throw new RangeError("worldline intersection coefficients are non-finite");
    let roots=[];
    if(Math.abs(a)<=aTolerance){
      if(Math.abs(b)<=bTolerance){if(Math.abs(c0)<=cTolerance)throw new RangeError("worldline lies on the cone; intersection is not discrete");return [];}
      roots=[-c0/b];
    }else{
      const disc=b*b-4*a*c0,floor=this.atol===0?0:((1/dScale)/vScale)**2,discTol=this.atol*Math.max(floor,Math.abs(b*b)+Math.abs(4*a*c0));
      if(!Number.isFinite(disc)||!Number.isFinite(discTol))throw new RangeError("worldline intersection discriminant is non-finite");
      if(disc<-discTol)return [];
      const rootDisc=Math.sqrt(Math.max(0,disc));
      if(disc<=discTol)roots=[-b/(2*a)];
      else{const q=-(b/2+(b<0?-1:1)*rootDisc/2);if(q===0)throw new RangeError("worldline intersection roots are indeterminate");const stable=q/a,other=c0/q;roots=b<0?[other,stable]:[stable,other];}
    }
    return roots.map(root=>{
      if(!Number.isFinite(root))throw new RangeError("worldline intersection is non-finite");
      const event=p.map((value,i)=>weightedDot4([value,vn[i],0,0],[1,root,0,0],[1,dScale,0,0]));
      if(event.some(value=>!Number.isFinite(value)))throw new RangeError("worldline intersection is non-finite");
      return event;
    }).filter(event=>{if(this.branch==="both")return true;const deltaTime=event[0]-this.apex.position[0],timeTolerance=this.atol*Math.max(1,Math.abs(event[0]),Math.abs(this.apex.position[0]));return this.branch==="future"?deltaTime>=-timeTolerance:deltaTime<=timeTolerance;});
  }
  intersectHypersurface(normal,offset,spatialDirections){if(this.branch==="both")throw new RangeError("intersectHypersurface requires a future or past branch");if(Array.isArray(spatialDirections)&&spatialDirections.length>MAX_CONE_SAMPLES)throw new RangeError(`cone intersection exceeds the ${MAX_CONE_SAMPLES}-direction limit`);const n=vector(normal,4,"normal"),rOffset=finiteNumber(offset,"offset"),dirs=parseNumericTree(spatialDirections,"spatialDirections",[3]),shape=spacetimeBatchShape(dirs.batchShape,this.apexBatchShape);if(product(dirs.batchShape)>MAX_CONE_SAMPLES||product(shape)*4>MAX_SPACETIME_COMPONENTS)throw new RangeError("cone intersection exceeds its bounded direction or component limit");const sign=this.branch==="future"?1:-1;const samples=buildSpacetimeBatch(shape,index=>{const raw=spacetimeBatchGet(dirs.data,dirs.batchShape,index,shape),direction=unitSpatialDirection(raw),apex=spacetimeBatchGet(this.apex.position,this.apexBatchShape,index,shape);const denominator=sign*n[0]+dot(n.slice(1),direction),parallel=Math.abs(denominator)<=this.atol*Math.max(1,Math.abs(denominator)),radius=parallel?NaN:rOffset/denominator,hit=!parallel&&Number.isFinite(radius)&&radius>=-this.atol,clamped=hit?Math.max(0,radius):0,event=hit?[apex[0]+sign*clamped,...apex.slice(1).map((value,axis)=>value+direction[axis]*clamped)]:[NaN,NaN,NaN,NaN];if(hit&&event.some(value=>!Number.isFinite(value)))throw new RangeError("cone intersection event is non-finite");return {event,hit};});const split=node=>{if(node&&Array.isArray(node.event))return {events:node.event,hitMask:node.hit};if(Array.isArray(node)){if(node.length===0)return {events:[],hitMask:[]};if(node[0]&&Array.isArray(node[0].event))return {events:node.map(item=>item.event),hitMask:node.map(item=>item.hit)};const values=node.map(split);return {events:values.map(value=>value.events),hitMask:values.map(value=>value.hitMask)};}throw new TypeError("invalid cone intersection result");};return split(samples);}
}

export class SpatialSlice {
  constructor(origin,normal,{basis=null,c=C,atol=1e-9}={}){
    if(!Number.isFinite(c)||c<=0||!Number.isFinite(atol)||atol<0)throw new RangeError("c must be positive and atol must be nonnegative");
    this.c=c;this.atol=atol;this.originFrame=origin instanceof Event4?origin.frame:null;this.originC=origin instanceof Event4?origin.c:c;const rawOrigin=origin instanceof Event4?origin.position:origin,parsedOrigin=parseNumericTree(rawOrigin,"slice origin",[4]),parsedNormal=parseNumericTree(normal,"timelike normal",[4]);
    let shape=spacetimeBatchShape(parsedOrigin.batchShape,parsedNormal.batchShape);
    const parsedBasis=basis===null?null:parseNumericTree(basis,"spatial basis",[3,4]);if(parsedBasis)shape=spacetimeBatchShape(shape,parsedBasis.batchShape);boundSpacetimeBatch(shape,16);
    const normalizedShape=parsedNormal.batchShape,normalized=buildSpacetimeBatch(normalizedShape,index=>{const n=spacetimeBatchGet(parsedNormal.data,normalizedShape,index,normalizedShape),magnitude=Math.max(...n.map(Math.abs));if(magnitude===0)throw new RangeError("slice normal must be timelike");const scaled=n.map(x=>x/magnitude),q=minkowskiDot(scaled,scaled);if(q>=0||!Number.isFinite(q))throw new RangeError("slice normal must be timelike");const unit=scaled.map(x=>x/Math.sqrt(-q));if(unit[0]<=0)throw new RangeError("timelike normal must be future-directed");return unit;});
    this.originPosition=buildSpacetimeBatch(shape,index=>spacetimeBatchGet(parsedOrigin.data,parsedOrigin.batchShape,index,shape).slice());this.normal=buildSpacetimeBatch(shape,index=>spacetimeBatchGet(normalized,normalizedShape,index,shape).slice());this.basis=parsedBasis?buildSpacetimeBatch(shape,index=>spacetimeBatchGet(parsedBasis.data,parsedBasis.batchShape,index,shape).map(axis=>axis.slice())):buildSpacetimeBatch(shape,index=>makeSpatialBasis(spacetimeBatchGet(this.normal,shape,index,shape),atol));this.batchShape=shape;
    buildSpacetimeBatch(shape,index=>{const n=spacetimeBatchGet(this.normal,shape,index,shape),b=spacetimeBatchGet(this.basis,shape,index,shape);if(b.length!==3||b.some(x=>Math.abs(minkowskiDot(x,n))>atol)||b.some((x,i)=>b.some((y,j)=>Math.abs(minkowskiDot(x,y)-(i===j?1:0))>atol*(1+(i===j?1:0)))))throw new RangeError("spatial basis must be orthonormal and orthogonal to the normal");return true;});
    this.origin=shape.length?this.originPosition:new Event4(this.originPosition,this.originFrame,this.originC);Object.freeze(this);
  }
  _eventData(event,name="event"){return parseNumericTree(event instanceof Event4?event.position:event,name,[4]);}
  projectTo3D(event){const p=this._eventData(event),shape=spacetimeBatchShape(p.batchShape,this.batchShape);boundSpacetimeBatch(shape,3);return buildSpacetimeBatch(shape,index=>{const d=sub(spacetimeBatchGet(p.data,p.batchShape,index,shape),spacetimeBatchGet(this.originPosition,this.batchShape,index,shape)),n=spacetimeBatchGet(this.normal,this.batchShape,index,shape),b=spacetimeBatchGet(this.basis,this.batchShape,index,shape),spatial=add(d,scale(n,minkowskiDot(n,d)));return b.map(axis=>minkowskiDot(axis,spatial));});}
  embed3D(coords){const q=parseNumericTree(coords,"coordinates",[3]),shape=spacetimeBatchShape(q.batchShape,this.batchShape);boundSpacetimeBatch(shape,4);const positions=buildSpacetimeBatch(shape,index=>{const c=spacetimeBatchGet(q.data,q.batchShape,index,shape),o=spacetimeBatchGet(this.originPosition,this.batchShape,index,shape),b=spacetimeBatchGet(this.basis,this.batchShape,index,shape);return b.reduce((p,e,i)=>add(p,scale(e,c[i])),o.slice());});return new Event4(positions,this.originFrame,this.originC);}
  contains(event,{atol=this.atol}={}){if(!Number.isFinite(atol)||atol<0)throw new RangeError("atol must be nonnegative and finite");const p=this._eventData(event),shape=spacetimeBatchShape(p.batchShape,this.batchShape);boundSpacetimeBatch(shape,1);return buildSpacetimeBatch(shape,index=>{const d=sub(spacetimeBatchGet(p.data,p.batchShape,index,shape),spacetimeBatchGet(this.originPosition,this.batchShape,index,shape)),n=spacetimeBatchGet(this.normal,this.batchShape,index,shape),distance=Math.abs(minkowskiDot(d,n)),tolerance=timeScaleTolerance(d,atol);if(!Number.isFinite(distance)||!Number.isFinite(tolerance))throw new RangeError("slice containment classification is non-finite");return distance<=tolerance;});}
  _d(event){const p=this._eventData(event),shape=spacetimeBatchShape(p.batchShape,this.batchShape);boundSpacetimeBatch(shape,4);return buildSpacetimeBatch(shape,index=>sub(spacetimeBatchGet(p.data,p.batchShape,index,shape),spacetimeBatchGet(this.originPosition,this.batchShape,index,shape)));}
  intersectWorldline(origin,direction){const p=this._eventData(origin,"worldline origin"),v=parseNumericTree(direction,"direction",[4]),shape=spacetimeBatchShape(spacetimeBatchShape(p.batchShape,v.batchShape),this.batchShape);boundSpacetimeBatch(shape,5);const results=buildSpacetimeBatch(shape,index=>{const x=spacetimeBatchGet(p.data,p.batchShape,index,shape),u=spacetimeBatchGet(v.data,v.batchShape,index,shape),o=spacetimeBatchGet(this.originPosition,this.batchShape,index,shape),n=spacetimeBatchGet(this.normal,this.batchShape,index,shape),den=minkowskiDot(u,n);if(Math.abs(den)<=this.atol)throw new RangeError("worldline is parallel to or contained in the slice");const parameter=-minkowskiDot(sub(x,o),n)/den;return {parameter,event:add(x,scale(u,parameter))};});const peel=(node,key)=>Array.isArray(node)?node.map(item=>peel(item,key)):node[key];return {parameter:peel(results,"parameter"),event:new Event4(peel(results,"event"),origin instanceof Event4?origin.frame:null,origin instanceof Event4?origin.c:this.c)};}
  intersectLightcone(cone){if(!(cone instanceof LightCone))throw new TypeError("cone must be a LightCone");const apex=parseNumericTree(cone.apex.position,"cone apex",[4]),shape=spacetimeBatchShape(this.batchShape,apex.batchShape);boundSpacetimeBatch(shape,5);const results=buildSpacetimeBatch(shape,index=>{const o=spacetimeBatchGet(this.originPosition,this.batchShape,index,shape),n=spacetimeBatchGet(this.normal,this.batchShape,index,shape),b=spacetimeBatchGet(this.basis,this.batchShape,index,shape),a=spacetimeBatchGet(apex.data,apex.batchShape,index,shape),delta=sub(o,a),signed=minkowskiDot(n,delta),tolerance=timeScaleTolerance(delta,this.atol);if(!Number.isFinite(signed)||!Number.isFinite(tolerance))throw new RangeError("slice-cone intersection classification is non-finite");let hit=true;if(cone.branch==="future")hit=signed<=tolerance;else if(cone.branch==="past")hit=signed>=-tolerance;if(!hit)return {center:Array(3).fill(Number.NaN),radius:Number.NaN,hit:false};const da=sub(a,o),spatial=add(da,scale(n,minkowskiDot(n,da))),center=b.map(axis=>minkowskiDot(axis,spatial)),radius=Math.abs(signed);if(center.some(value=>!Number.isFinite(value)))throw new RangeError("slice-cone intersection center is non-finite");return {center,radius,hit:true};});const peel=(node,key)=>Array.isArray(node)?node.map(item=>peel(item,key)):node[key];return {center:peel(results,"center"),radius:peel(results,"radius"),hit:peel(results,"hit")};}
}
function makeSpatialBasis(n,tol){const candidates=[[0,1,0,0],[0,0,1,0],[0,0,0,1]],basis=[];for(const raw of candidates){let v=raw.slice();v=sub(v,scale(n,minkowskiDot(v,n)/minkowskiDot(n,n)));for(const e of basis)v=sub(v,scale(e,minkowskiDot(v,e)));const q=minkowskiDot(v,v);if(q>tol)basis.push(v.map(x=>x/Math.sqrt(q)));}if(basis.length!==3)throw new RangeError("could not construct an orthonormal spatial basis");return basis;}

function worldlineEndpoint(value,metadata,name){const event=value instanceof Event4?value:new Event4(value,metadata.frame,metadata.c);if(event.c!==metadata.c||event.frame!==metadata.frame)throw new RangeError("worldline segment endpoints must share c and frame");return new Event4(vector(event.position,4,name),metadata.frame,metadata.c);}

export class WorldlineSegment {
  constructor(start,end,u0,u1,parameterSpan=[0,1]){const metadata=start instanceof Event4?start:end instanceof Event4?end:{frame:null,c:C};this.start=worldlineEndpoint(start,metadata,"start event");this.end=worldlineEndpoint(end,metadata,"end event");this.u0=vector(u0);this.u1=vector(u1);this.span=vector(parameterSpan,2,"parameter span");if(this.span[1]<=this.span[0]||!Number.isFinite(this.span[1]-this.span[0]))throw new RangeError("parameter span must increase with a finite duration");}
  evaluate(t){const parsed=parseSpacetimeParameters(t);boundSpacetimeBatch(parsed.shape,4);return new Event4(mapScalarTree(parsed.data,value=>{const q=this._q(value),h=this.span[1]-this.span[0],p0=this.start.position,p1=this.end.position,h00=2*q**3-3*q*q+1,h10=q**3-2*q*q+q,h01=-2*q**3+3*q*q,h11=q**3-q*q;return p0.map((x,i)=>h00*x+h10*h*this.u0[i]+h01*p1[i]+h11*h*this.u1[i]);}),this.start.frame,this.start.c);}
  tangent(t){const parsed=parseSpacetimeParameters(t);boundSpacetimeBatch(parsed.shape,4);return new Vector4(mapScalarTree(parsed.data,value=>{const q=this._q(value),h=this.span[1]-this.span[0],p0=this.start.position,p1=this.end.position,a=(6*q*q-6*q)/h,b=3*q*q-4*q+1,c=(-6*q*q+6*q)/h,d=3*q*q-2*q;return p0.map((x,i)=>a*x+b*this.u0[i]+c*p1[i]+d*this.u1[i]);}));}
  _q(t){if(!Number.isFinite(t))throw new RangeError("parameter must be finite");return (t-this.span[0])/(this.span[1]-this.span[0]);}
}

export class Worldline {
  constructor(segments){if(!Array.isArray(segments)||!segments.length)throw new RangeError("worldline requires a nonempty sequence of WorldlineSegment objects");for(const segment of segments)if(!(segment instanceof WorldlineSegment))throw new RangeError("worldline requires a nonempty sequence of WorldlineSegment objects");for(let i=1;i<segments.length;i++){const left=segments[i-1],right=segments[i];if(left.span[1]!==right.span[0])throw new RangeError("adjacent segment parameter spans must meet");if(left.end.c!==right.start.c||left.end.frame!==right.start.frame)throw new RangeError("worldline segments must use the same c and frame");const close=(a,b)=>a.every((value,index)=>Math.abs(value-b[index])<=1e-8+1e-5*Math.abs(b[index]));if(!close(left.end.position,right.start.position)||!close(left.u1,right.u0))throw new RangeError("adjacent segments must share their event and four-velocity");}this.segments=segments.slice();}
  _segment(t){if(!Number.isFinite(t))throw new RangeError("parameter must be finite");let low=0,high=this.segments.length-1;while(low<high){const middle=Math.floor((low+high)/2);if(t<this.segments[middle].span[1])high=middle;else low=middle+1;}return this.segments[low];}
  _evaluate(parameter,method){const parsed=parseSpacetimeParameters(parameter);boundSpacetimeBatch(parsed.shape,4);return mapScalarTree(parsed.data,value=>this._segment(value)[method](value)[method==="evaluate"?"position":"components"]);}
  evaluate(t){const first=this.segments[0].start;return new Event4(this._evaluate(t,"evaluate"),first.frame,first.c);}
  tangent(t){return new Vector4(this._evaluate(t,"tangent"));}
  call(t){return this.evaluate(t);}
}

export function integrateRK4(rhs,initialState,start,end,steps){
  if(!Number.isInteger(steps)||steps<1||steps>MAX_INTEGRATION_STEPS)throw new RangeError(`steps must be an integer from 1 to ${MAX_INTEGRATION_STEPS}`);
  if(typeof rhs!=="function")throw new TypeError("rhs must be a function");
  if(!Number.isFinite(start)||!Number.isFinite(end))throw new RangeError("integration bounds must be finite");
  const parsed=parseNumericTree(initialState,"initialState");
  const components=product(parsed.shape),resultComponents=(steps+1)*components;if(!Number.isSafeInteger(resultComponents)||resultComponents>MAX_INTEGRATION_COMPONENTS)throw new RangeError("integration result exceeds the bounded component limit");
  const h=(end-start)/steps;if(!Number.isFinite(h))throw new RangeError("integration step size must be finite");
  const times=[start];
  for(let i=0;i<steps;i++){
    const previous=times[i],next=i===steps-1?end:previous+h,step=next-previous,midpoint=previous+step/2;
    if(!Number.isFinite(next)||!Number.isFinite(step)||(start!==end&&(h>0?(next<=previous||next>end):(next>=previous||next<end))))throw new RangeError("integration time grid cannot represent distinct steps");
    if(start!==end&&!(h>0?midpoint>previous&&midpoint<next:midpoint<previous&&midpoint>next))throw new RangeError("integration time grid cannot represent a distinct midpoint");
    if(start!==end&&Math.abs((midpoint-previous)-(next-midpoint))>Math.abs(step)*1e-8)throw new RangeError("integration time grid midpoint is too asymmetric for RK4; shift or rescale the parameter");
    times.push(next);
  }
  const copy=state=>Array.isArray(state)?state.map(copy):state;
  const map=(state,fn)=>Array.isArray(state)?state.map(value=>map(value,fn)):finiteNumber(fn(state),"integration state");
  const combine=(left,right,fn,name)=>{if(Array.isArray(left)){if(!Array.isArray(right)||left.length!==right.length)throw new RangeError(`${name} shape must match the integration state`);return left.map((value,index)=>combine(value,right[index],fn,name));}if(Array.isArray(right))throw new RangeError(`${name} shape must match the integration state`);return finiteNumber(fn(left,right),name);};
  const addTree=(left,right,name="integration state")=>combine(left,right,(a,b)=>a+b,name),scaleTree=(state,factor)=>map(state,value=>value*factor);
  const evaluate=(at,state)=>{if(!Number.isFinite(at))throw new RangeError("integration parameter became non-finite");const raw=rhs(at,copy(state)),parsedRhs=parseNumericTree(raw,"rhs result");if(parsedRhs.shape.length!==parsed.shape.length||parsedRhs.shape.some((size,index)=>size!==parsed.shape[index]))throw new RangeError("rhs result shape must match the integration state");return parsedRhs.data;};
  let t=start,y=parsed.data;const rows=[[start,copy(y)]];
  for(let i=0;i<steps;i++){
    const step=times[i+1]-t,midpoint=t+step/2;
    const k1=evaluate(t,y),k2=evaluate(midpoint,addTree(y,scaleTree(k1,step/2))),k3=evaluate(midpoint,addTree(y,scaleTree(k2,step/2))),k4=evaluate(times[i+1],addTree(y,scaleTree(k3,step)));
    const update=(state,a,b,c,d)=>{if(Array.isArray(state))return state.map((value,index)=>update(value,a[index],b[index],c[index],d[index]));const sum=a+2*b+2*c+d,product=step*sum,increment=Number.isFinite(product)?product/6:Number.isFinite(sum)?step*(sum/6):step*(a/6+b/3+c/3+d/6);return finiteNumber(state+increment,"integration state");},next=update(y,k1,k2,k3,k4);
    y=next;t=times[i+1];rows.push([t,copy(y)]);
  }
  return rows;
}
export class ODEWorldline {
  constructor(initialEvent,fourVelocity){this.initialEvent=initialEvent instanceof Event4?initialEvent:new Event4(initialEvent);if(typeof fourVelocity!=="function")throw new TypeError("fourVelocity must be a function of (parameter, event)");this.fourVelocity=fourVelocity;}
  integrate(parameterSpan,{steps=100}={}){const span=vector(parameterSpan,2,"parameter span"),rows=integrateRK4(this.fourVelocity,this.initialEvent.position,span[0],span[1],steps);return [rows.map(([parameter])=>parameter),rows.map(([,event])=>event)];}
}

export class InertialFrame {
  constructor({origin=[0,0,0,0],orientation=null,relativeVelocity=[0,0,0],c=C}={}){if(!Number.isFinite(c)||c<=0)throw new RangeError("c must be positive and finite");this.origin=origin instanceof Event4?origin:new Event4(origin,null,c);this.c=c;const parsedVelocity=parseNumericTree(relativeVelocity,"relative velocity",[3]),velocityShape=parsedVelocity.batchShape;if(product(velocityShape)*3>MAX_SPACETIME_COMPONENTS)throw new RangeError("relative-velocity batch exceeds the component limit");if(!everyScalar(buildSpacetimeBatch(velocityShape,index=>Math.hypot(...spacetimeBatchGet(parsedVelocity.data,velocityShape,index,velocityShape))),speed=>speed<c))throw new RangeError("relative velocity magnitude must be less than c");this.relativeVelocity=parsedVelocity.data;const parsedOrientation=orientation===null?{data:identity(3),batchShape:[]}:parseNumericTree(orientation,"orientation",[3,3]);if(product(parsedOrientation.batchShape)*16>MAX_SPACETIME_COMPONENTS)throw new RangeError("orientation batch exceeds the component limit");this.orientation=parsedOrientation.data;LorentzTransform.rotation(this.orientation);Object.freeze(this);}
  get toGlobalTransform(){const rotation=LorentzTransform.rotation(this.orientation),globalFromUnrotated=LorentzTransform.boost(mapScalarTree(this.relativeVelocity,value=>-value/this.c));return new PoincareTransform(globalFromUnrotated.compose(rotation),this.origin.position);}
  get fromGlobalTransform(){return this.toGlobalTransform.inverse();}
  toGlobal(position){const coordinates=position instanceof Event4?position.position:position?.position??position;return this.toGlobalTransform.applyEvent({position:coordinates});}
  fromGlobal(position){const coordinates=position instanceof Event4?position.position:position?.position??position;return this.fromGlobalTransform.applyEvent({position:coordinates});}
  transformEvent(event){return this.fromGlobalTransform.applyEvent(event);}
  transformVector(v){return this.fromGlobalTransform.applyVector(v);}
  relativeTransform(other){if(!(other instanceof InertialFrame))throw new TypeError("other must be an InertialFrame");return other.fromGlobalTransform.compose(this.toGlobalTransform);}
}
