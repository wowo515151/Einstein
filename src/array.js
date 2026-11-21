import {weightedDot4} from "./weighted-dot-internal.js";

export const C = 299_792_458;
export const SPEED_OF_LIGHT = C;
export const VACUUM_PERMITTIVITY = 8.8541878128e-12;
export const ETA = Object.freeze([-1, 1, 1, 1]);
const MAX_FLAT_COMPONENTS = 1_000_000;

export function finiteNumber(value, name = "value") {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new TypeError(`${name} must be a finite number`);
  return value;
}
export function vector(value, size = 4, name = "vector") {
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) throw new TypeError(`${name} must be an array-like value`);
  if (value.length !== size) throw new RangeError(`${name} must have ${size} components`);
  return Array.from(value, (x, i) => finiteNumber(x, `${name}[${i}]`));
}
export function matrix(value, n = 4, name = "matrix") {
  if (!Array.isArray(value) || value.length !== n) throw new RangeError(`${name} must be ${n} by ${n}`);
  return Array.from(value, (row, i) => vector(row, n, `${name}[${i}]`));
}
export const add = (a,b) => a.map((x,i)=>x+b[i]);
export const sub = (a,b) => a.map((x,i)=>x-b[i]);
export const scale = (a,k) => a.map(x=>x*k);
export const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
export const norm = a => Math.hypot(...a);
export const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const zeros = n => Array.from({length:n},()=>Array(n).fill(0));
export const identity = n => zeros(n).map((r,i)=>r.map((_,j)=>i===j?1:0));
export function transpose(a) { return a[0].map((_,j)=>a.map(r=>r[j])); }
export function mmul(a,b) { const bt=transpose(b); return a.map(r=>bt.map(c=>dot(r,c))); }
export function mvec(a,v) { return a.map(r=>dot(r,v)); }
export function determinant(a) {
  const scaledFallback=()=>{const scales=a.map(row=>Math.max(...row.map(Math.abs)));if(scales.some(value=>value===0))return 0;const normalized=a.map((row,i)=>row.map(value=>value/scales[i])),unit=determinant(normalized);if(unit===0)return 0;return Math.sign(unit)*Math.exp(Math.log(Math.abs(unit))+scales.reduce((sum,value)=>sum+Math.log(value),0));};
  const m=a.map(r=>r.slice()); let d=1,sign=1,logMagnitude=0,extreme=false;
  for(let i=0;i<m.length;i++){let p=i;for(let r=i+1;r<m.length;r++)if(Math.abs(m[r][i])>Math.abs(m[p][i]))p=r;
    if(m[p][i]===0)return 0;if(p!==i){[m[p],m[i]]=[m[i],m[p]];d=-d;sign=-sign;}const q=m[i][i];d*=q;sign*=Math.sign(q);logMagnitude+=Math.log(Math.abs(q));if(!Number.isFinite(d)||d===0)extreme=true;
    for(let r=i+1;r<m.length;r++){const f=m[r][i]/q;for(let c=i+1;c<m.length;c++){m[r][c]-=f*m[i][c];if(!Number.isFinite(m[r][c]))return scaledFallback();}}}
  return extreme?sign*Math.exp(logMagnitude):d;
}
export function determinantRoot(a) {
  const scaledFallback=()=>{const scales=a.map(row=>Math.max(...row.map(Math.abs)));if(scales.some(value=>value===0))throw new RangeError("metric is singular");const normalized=a.map((row,i)=>row.map(value=>value/scales[i])),unit=determinantRoot(normalized),root=Math.exp(Math.log(unit.root)+scales.reduce((sum,value)=>sum+Math.log(value),0)/2);if(!Number.isFinite(root)||root===0)throw new RangeError("metric volume scale is outside the finite range");return {sign:unit.sign,root};};
  const m=a.map(row=>row.slice());let sign=1,logMagnitude=0;
  for(let i=0;i<m.length;i++){
    let p=i;for(let r=i+1;r<m.length;r++)if(Math.abs(m[r][i])>Math.abs(m[p][i]))p=r;
    if(m[p][i]===0)throw new RangeError("metric is singular");
    if(p!==i){[m[p],m[i]]=[m[i],m[p]];sign=-sign;}
    const pivot=m[i][i];sign*=Math.sign(pivot);logMagnitude+=Math.log(Math.abs(pivot));
    for(let r=i+1;r<m.length;r++){const factor=m[r][i]/pivot;for(let c=i+1;c<m.length;c++){m[r][c]-=factor*m[i][c];if(!Number.isFinite(m[r][c]))return scaledFallback();}}
  }
  const root=Math.exp(logMagnitude/2);
  if(!Number.isFinite(root)||root===0)throw new RangeError("metric volume scale is outside the finite range");
  return {sign,root};
}
export function inverse(a, tolerance=0) { return inverseCore(a,tolerance,true); }
function inverseCore(a,tolerance,allowScaling) {
  const retry=()=>{if(!allowScaling||tolerance!==0)throw new RangeError("matrix inversion has a non-finite intermediate");const scales=a.map(row=>Math.max(...row.map(Math.abs)));if(scales.some(value=>value===0))throw new RangeError("matrix is singular");const scaled=a.map((row,i)=>row.map(value=>value/scales[i])),result=inverseCore(scaled,0,false).map(row=>row.map((value,j)=>value/scales[j]));if(result.some(row=>row.some(value=>!Number.isFinite(value))))throw new RangeError("matrix inverse is non-finite");return result;};
  const n=a.length, m=a.map((r,i)=>[...r,...identity(n)[i]]);
  for(let i=0;i<n;i++){let p=i;for(let r=i+1;r<n;r++)if(Math.abs(m[r][i])>Math.abs(m[p][i]))p=r;
    if(Math.abs(m[p][i])<=tolerance)throw new RangeError("matrix is singular");[m[p],m[i]]=[m[i],m[p]];
    const q=m[i][i];m[i]=m[i].map(x=>x/q);if(m[i].some(value=>!Number.isFinite(value)))return retry();for(let r=0;r<n;r++)if(r!==i){const f=m[r][i];m[r]=m[r].map((x,c)=>x-f*m[i][c]);if(m[r].some(value=>!Number.isFinite(value)))return retry();}}
  const result=m.map(r=>r.slice(n));
  if(result.some(row=>row.some(value=>!Number.isFinite(value))))throw new RangeError("matrix inverse is non-finite");
  return result;
}
function flatVectorTree(value,name){let leaves=0;const ancestors=new Set();function copy(node,depth){if(ArrayBuffer.isView(node)&&!(node instanceof DataView)){if(node.length>MAX_FLAT_COMPONENTS)throw new RangeError(`${name} exceeds the component limit`);node=Array.from(node);}if(Array.isArray(node)){if(node.length>MAX_FLAT_COMPONENTS)throw new RangeError(`${name} exceeds the component limit`);if(depth>=9)throw new RangeError(`${name} rank exceeds eight batch axes`);if(node.length===0)throw new RangeError(`${name} dimensions must be nonempty`);if(ancestors.has(node))throw new TypeError(`${name} cannot contain cycles`);ancestors.add(node);const result=Array.from(node,item=>copy(item,depth+1));ancestors.delete(node);return result;}if(++leaves>MAX_FLAT_COMPONENTS)throw new RangeError(`${name} exceeds the component limit`);return finiteNumber(node,name);}const data=copy(value,0),shape=[];let current=data;while(Array.isArray(current)){shape.push(current.length);current=current[0];}if(!shape.length||shape.at(-1)!==4)throw new RangeError(`${name} must end in four-vector shape`);function verify(node,axis){if(axis===shape.length)return;if(!Array.isArray(node)||node.length!==shape[axis])throw new RangeError(`${name} must be rectangular`);for(const item of node)verify(item,axis+1);}verify(data,0);return {data,batchShape:shape.slice(0,-1)};}
function flatBatchShape(a,b){const rank=Math.max(a.length,b.length),out=Array(rank);for(let i=0;i<rank;i++){const x=a[a.length-rank+i]??1,y=b[b.length-rank+i]??1;if(x!==y&&x!==1&&y!==1)throw new RangeError("Minkowski vector batches are not broadcast-compatible");out[i]=x===1?y:y===1?x:x;}return out;}
function flatBatchGet(data,shape,index,target){let value=data;const offset=target.length-shape.length;for(let axis=0;axis<shape.length;axis++)value=value[shape[axis]===1?0:index[offset+axis]];return value;}
function flatBatchBuild(shape,fn,index=[]){if(index.length===shape.length)return fn(index);return Array.from({length:shape[index.length]},(_,i)=>flatBatchBuild(shape,fn,[...index,i]));}
export function minkowskiDot(a,b,metric=ETA) {const left=flatVectorTree(a,"a"),right=flatVectorTree(b,"b"),batchShape=flatBatchShape(left.batchShape,right.batchShape),eta=vector(metric,4,"metric"),components=batchShape.reduce((count,size)=>count*size,1);if(components>MAX_FLAT_COMPONENTS)throw new RangeError("Minkowski dot exceeds the component limit");return flatBatchBuild(batchShape,index=>weightedDot4(flatBatchGet(left.data,left.batchShape,index,batchShape),flatBatchGet(right.data,right.batchShape,index,batchShape),eta));}
export function close(a,b,tol=1e-10) { return Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)); }
