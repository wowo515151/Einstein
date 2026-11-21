import {C,vector,matrix,zeros,identity,transpose,mmul,mvec,inverse,determinantRoot,dot,scale,add,sub,minkowskiDot} from "./array.js";
import {Metric4,LorentzTransform} from "./algebra.js";
import {Event4,integrateRK4} from "./spacetime.js";
import {forwardJacobian4,forwardHessian4,forwardSecondOrder4} from "./autodiff.js";

function geometryMetric(value){return matrix(value instanceof Metric4?value._checkedComponents().data:value);}
function differenceStencil(point,axis,step){const plus=point.slice(),minus=point.slice();plus[axis]+=step;minus[axis]-=step;const span=plus[axis]-minus[axis],forward=plus[axis]-point[axis],backward=point[axis]-minus[axis];if(!Number.isFinite(span)||!(forward>0)||!(backward>0))throw new RangeError("finite-difference step must produce distinct finite stencil points");return {plus,minus,span,forward,backward};}
function differenceQuotient(plus,minus,span){const delta=plus-minus,result=Number.isFinite(delta)?delta/span:plus/span-minus/span;if(!Number.isFinite(result))throw new RangeError("finite-difference derivative is non-finite");return result;}
function weightedStencilDerivative(plus,minus,center,stencil){
  if(stencil.forward===stencil.backward)return differenceQuotient(plus,minus,stencil.span);
  const left=differenceQuotient(center,minus,stencil.backward),right=differenceQuotient(plus,center,stencil.forward),value=(stencil.forward/stencil.span)*left+(stencil.backward/stencil.span)*right;
  if(!Number.isFinite(value))throw new RangeError("finite-difference derivative is non-finite");return value;
}
function partial(fn,x,axis,h=1e-5){const stencil=differenceStencil(x,axis,h),plus=fn(stencil.plus),minus=fn(stencil.minus),center=stencil.forward===stencil.backward?undefined:fn(x.slice());if(!Number.isFinite(plus)||!Number.isFinite(minus)||(stencil.forward!==stencil.backward&&!Number.isFinite(center)))throw new TypeError("scalar field samples must be finite");return weightedStencilDerivative(plus,minus,center,stencil);}
const MAX_COVARIANT_COMPONENTS=65536;
const MAX_TRANSPORT_INTEGRATION_STEPS=10_000;
const MAX_CURVATURE_BATCH_WORK=4_000_000;
const MAX_DIFFERENTIAL_BATCH_WORK=1_000_000;
function parseGeometryBatch(value,name,trailingShape){let leaves=0;const active=new Set();function copy(node,depth){if(ArrayBuffer.isView(node)&&!(node instanceof DataView)){if(node.length>MAX_COVARIANT_COMPONENTS)throw new RangeError(`${name} exceeds the bounded component limit`);node=Array.from(node);}if(Array.isArray(node)){if(node.length>MAX_COVARIANT_COMPONENTS)throw new RangeError(`${name} exceeds the bounded component limit`);if(depth>=8)throw new RangeError(`${name} rank exceeds eight`);if(node.length===0)throw new RangeError(`${name} dimensions must be nonempty`);if(active.has(node))throw new TypeError(`${name} cannot contain cycles`);active.add(node);const result=Array.from(node,item=>copy(item,depth+1));active.delete(node);return result;}if(typeof node!=="number"||!Number.isFinite(node))throw new TypeError(`${name} values must be finite numbers`);if(++leaves>MAX_COVARIANT_COMPONENTS)throw new RangeError(`${name} exceeds the bounded component limit`);return node;}const data=copy(value,0),shape=[];let current=data;while(Array.isArray(current)){shape.push(current.length);current=current[0];}if(shape.length<trailingShape.length||trailingShape.some((size,index)=>shape[shape.length-trailingShape.length+index]!==size))throw new RangeError(`${name} must end in shape ${trailingShape.join(" by ")}`);function verify(node,axis){if(axis===shape.length)return;if(!Array.isArray(node)||node.length!==shape[axis])throw new RangeError(`${name} must be rectangular`);for(const item of node)verify(item,axis+1);}verify(data,0);return {data,shape};}
function geometryBatchShape(a,b){const rank=Math.max(a.length,b.length),out=Array(rank);for(let i=0;i<rank;i++){const x=a[a.length-rank+i]??1,y=b[b.length-rank+i]??1;if(x!==y&&x!==1&&y!==1)throw new RangeError("batch shapes are not broadcast-compatible");out[i]=x===1?y:y===1?x:x;}return out;}
function geometryBatchGet(data,shape,index,target){let value=data;const offset=target.length-shape.length;for(let axis=0;axis<shape.length;axis++)value=value[shape[axis]===1?0:index[offset+axis]];return value;}
function geometryBuildBatch(shape,fn,index=[]){if(index.length===shape.length)return fn(index);return Array.from({length:shape[index.length]},(_,i)=>geometryBuildBatch(shape,fn,[...index,i]));}
function validateDifferenceStep(step){if(!Number.isFinite(step)||step<=0)throw new RangeError("step must be positive and finite");return step;}
function tensorShape(value,path=[],ancestors=new Set()){if(typeof value==="number"){if(!Number.isFinite(value))throw new TypeError("tensor field values must be finite");return [];}if(!Array.isArray(value)||value.length!==4)throw new RangeError(`tensor field axis ${path.length} must have length 4`);if(path.length>=8)throw new RangeError("tensor field rank exceeds the bounded rank-eight limit");if(ancestors.has(value))throw new RangeError("tensor field arrays cannot be cyclic");ancestors.add(value);try{const childShape=tensorShape(value[0],[...path,0],ancestors);for(let i=1;i<4;i++)if(JSON.stringify(tensorShape(value[i],[...path,i],ancestors))!==JSON.stringify(childShape))throw new RangeError("tensor field must have a rectangular shape");return [4,...childShape];}finally{ancestors.delete(value);}}
function tensorValue(value,indices){return indices.reduce((current,index)=>current[index],value);}
function setTensorValue(value,indices,next){let target=value;for(const index of indices.slice(0,-1))target=target[index];target[indices.at(-1)]=next;}
function zeroTensor(shape){if(!shape.length)return 0;return Array.from({length:shape[0]},()=>zeroTensor(shape.slice(1)));}
function numericTreeShape(value){if(!Array.isArray(value))return [];if(!value.length)throw new RangeError("batch dimensions must be nonempty");const childShape=numericTreeShape(value[0]);for(let index=1;index<value.length;index++)if(JSON.stringify(numericTreeShape(value[index]))!==JSON.stringify(childShape))throw new RangeError("field outputs must be rectangular across batch samples");return [value.length,...childShape];}
function fieldOutputShape(value){const active=new Set();let leaves=0;function visit(node,depth){if(typeof node==="number"){if(!Number.isFinite(node))throw new TypeError(depth===0?"scalar field samples must be finite":"field samples must be finite");if(++leaves>MAX_COVARIANT_COMPONENTS)throw new RangeError("field output exceeds the bounded component limit");return [];}if(!Array.isArray(node)||!node.length)throw new RangeError("field output must be a nonempty rectangular numeric array or scalar");if(depth>=8)throw new RangeError("field output rank exceeds eight");if(node.length>MAX_COVARIANT_COMPONENTS)throw new RangeError("field output exceeds the bounded component limit");if(active.has(node))throw new RangeError("field output cannot be cyclic");active.add(node);try{const first=visit(node[0],depth+1);for(let i=1;i<node.length;i++)if(JSON.stringify(visit(node[i],depth+1))!==JSON.stringify(first))throw new RangeError("field output must be rectangular");return [node.length,...first];}finally{active.delete(node);}}return visit(value,0);}
function eachIndex(rank,visit,prefix=[]){if(prefix.length===rank){visit(prefix);return;}for(let i=0;i<4;i++)eachIndex(rank,visit,[...prefix,i]);}
function eachShapeIndex(shape,visit,prefix=[]){if(prefix.length===shape.length){visit(prefix);return;}for(let i=0;i<shape[prefix.length];i++)eachShapeIndex(shape,visit,[...prefix,i]);}

export class DifferentialOperators4 {
  constructor(metric=null,{step=1e-5,metricDerivative="finite"}={}){this.metric=metric;this.step=validateDifferenceStep(step);this.connection=new Connection4(x=>this._rawMetric(x),{step:this.step,metricDerivative});}
  _rawMetric(x){const g=typeof this.metric==="function"?this.metric(x):(this.metric??new Metric4().components);return g instanceof Metric4?g._checkedComponents().data:g;}
  _g(x){const mode=this.connection.metricDerivative;if(!["finite","forward"].includes(mode))throw new TypeError("metricDerivative must be finite or forward");if(mode==="forward"&&typeof this.metric==="function")return geometryMetric(forwardJacobian4(p=>this._rawMetric(p),x).value);return geometryMetric(this._rawMetric(x));}
  _mapPoints(x,operation,workPerPoint=1){const parsed=parseGeometryBatch(x,"position",[4]),batchShape=parsed.shape.slice(0,-1),batchCount=batchShape.reduce((count,size)=>count*size,1),firstIndex=Array(batchShape.length).fill(0);if(typeof workPerPoint==="number"&&batchCount*workPerPoint>MAX_DIFFERENTIAL_BATCH_WORK)throw new RangeError("differential-operator batch exceeds the bounded work limit");const firstPoint=geometryBatchGet(parsed.data,batchShape,firstIndex,batchShape),firstValue=operation(firstPoint),shape=numericTreeShape(firstValue),components=shape.reduce((count,size)=>count*size,1);if(batchCount*components>MAX_COVARIANT_COMPONENTS)throw new RangeError("differential-operator batch exceeds the bounded component limit");if(typeof workPerPoint==="function"&&batchCount*workPerPoint(firstValue,components)>MAX_DIFFERENTIAL_BATCH_WORK)throw new RangeError("differential-operator batch exceeds the bounded work limit");if(!batchShape.length)return firstValue;return geometryBuildBatch(batchShape,index=>{if(index.every((value,axis)=>value===firstIndex[axis]))return firstValue;const result=operation(geometryBatchGet(parsed.data,batchShape,index,batchShape));if(JSON.stringify(numericTreeShape(result))!==JSON.stringify(shape))throw new RangeError("field result shape must remain constant across batch samples");return result;});}
  partial(fn,x){return this._mapPoints(x,p=>this._jacobianScalar(fn,p),(_value,components)=>Math.ceil(13*components/4));}
  _partialScalar(fn,p){return Array.from({length:4},(_,i)=>partial(fn,p,i,this.step));}
  jacobian(fn,x){return this._mapPoints(x,p=>this._jacobianScalar(fn,p),(_value,components)=>Math.ceil(13*components/4));}
  _jacobianScalar(fn,p){
    let shape=null,components=0,center,hasCenter=false;
    const derivatives=Array.from({length:4},(_,axis)=>{
      const stencil=differenceStencil(p,axis,this.step);
      const plus=fn(stencil.plus),minus=fn(stencil.minus);
      const plusShape=fieldOutputShape(plus),minusShape=fieldOutputShape(minus);
      if(JSON.stringify(plusShape)!==JSON.stringify(minusShape)||(shape!==null&&JSON.stringify(plusShape)!==JSON.stringify(shape)))throw new RangeError("field shape must remain constant in a neighborhood");
      if(shape===null){shape=plusShape;components=shape.reduce((product,size)=>product*size,1);if(13*components>MAX_COVARIANT_COMPONENTS)throw new RangeError("Jacobian exceeds the bounded finite-difference work limit");}
      if(stencil.forward!==stencil.backward&&!hasCenter){center=fn(p.slice());if(JSON.stringify(fieldOutputShape(center))!==JSON.stringify(shape))throw new RangeError("field shape must remain constant in a neighborhood");hasCenter=true;}
      const difference=(a,b,c)=>Array.isArray(a)?a.map((child,i)=>difference(child,b[i],c?.[i])):weightedStencilDerivative(a,b,c,stencil);
      return difference(plus,minus,center);
    });
    const out=zeroTensor([...shape,4]);
    eachShapeIndex(shape,indices=>{for(let axis=0;axis<4;axis++)setTensorValue(out,[...indices,axis],tensorValue(derivatives[axis],indices));});
    return out;
  }
  partialForward(fn,x){return this.jacobianForward(fn,x);}
  jacobianForward(fn,x){return this._mapPoints(x,p=>forwardJacobian4(fn,p).jacobian,(_value,components)=>2*components);}
  gradientForward(fn,x){return this._mapPoints(x,p=>{const field=forwardJacobian4(fn,p);if(typeof field.value!=="number")throw new TypeError("gradient requires a scalar field");const result=mvec(inverse(this._g(p)),field.jacobian);if(result.some(value=>!Number.isFinite(value)))throw new RangeError("gradient is non-finite");return result;},16);}
  gradient(fn,x){return this._mapPoints(x,p=>{const result=mvec(inverse(this._g(p)),this._partialScalar(fn,p));if(result.some(value=>!Number.isFinite(value)))throw new RangeError("gradient is non-finite");return result;},16);}
  divergenceForward(fn,x){return this._mapPoints(x,p=>{
    const differentiated=forwardJacobian4(fn,p),v=vector(differentiated.value),d=differentiated.jacobian,gamma=this.connection.christoffel(p);let out=0;
    for(let a=0;a<4;a++){out+=d[a][a];for(let b=0;b<4;b++)out+=gamma[a][a][b]*v[b];}
    if(!Number.isFinite(out))throw new RangeError("divergence is non-finite");return out;
  },128);}
  divergence(fn,x){return this._mapPoints(x,p=>{const v=vector(fn(p)),d=this._jacobianScalar(fn,p),gamma=this.connection.christoffel(p);let out=0;for(let a=0;a<4;a++){out+=d[a][a];for(let b=0;b<4;b++)out+=gamma[a][a][b]*v[b];}if(!Number.isFinite(out))throw new RangeError("divergence is non-finite");return out;},128);}
  exteriorDerivativeForward(form,x){return this._mapPoints(x,p=>{
    const differentiated=forwardJacobian4(form,p),shape=tensorShape(differentiated.value),rank=shape.length,inputComponents=shape.reduce((product,size)=>product*size,1),outputComponents=inputComponents*4;
    if(5*inputComponents+outputComponents>MAX_COVARIANT_COMPONENTS)throw new RangeError("exterior derivative exceeds the bounded forward-mode work limit");
    const out=zeroTensor([4,...shape]);
    eachIndex(rank+1,indices=>{
      let total=0;
      for(let position=0;position<=rank;position++){const derivativeIndex=indices[position],formIndices=indices.filter((_,axis)=>axis!==position),term=tensorValue(differentiated.jacobian,[...formIndices,derivativeIndex]);total+=(position%2===0?1:-1)*term;}
      if(!Number.isFinite(total))throw new RangeError("exterior derivative is non-finite");setTensorValue(out,indices,total);
    });
    return out;
  },(_value,components)=>3*components);}
  exteriorDerivative(form,x){return this._mapPoints(x,p=>this._exteriorDerivativeScalar(form,p),(_value,components)=>3*components);}
  _exteriorDerivativeScalar(form,p){
    const value=form(p),shape=tensorShape(value),rank=shape.length,inputComponents=shape.reduce((product,size)=>product*size,1),outputComponents=inputComponents*4;
    if(8*inputComponents+outputComponents>MAX_COVARIANT_COMPONENTS)throw new RangeError("exterior derivative exceeds the bounded finite-difference work limit");
    const derivatives=Array.from({length:4},(_,axis)=>{
      const stencil=differenceStencil(p,axis,this.step);
      const plus=form(stencil.plus),minus=form(stencil.minus);
      if(JSON.stringify(tensorShape(plus))!==JSON.stringify(shape)||JSON.stringify(tensorShape(minus))!==JSON.stringify(shape))throw new RangeError("form field shape must remain constant in a neighborhood");
      const difference=(a,b,c)=>Array.isArray(a)?a.map((child,i)=>difference(child,b[i],c[i])):weightedStencilDerivative(a,b,c,stencil);
      return difference(plus,minus,value);
    });
    const out=zeroTensor([4,...shape]);
    eachIndex(rank+1,indices=>{
      let total=0;
      for(let position=0;position<=rank;position++){
        const derivativeIndex=indices[position],formIndices=indices.filter((_,axis)=>axis!==position),term=tensorValue(derivatives[derivativeIndex],formIndices);
        total+=(position%2===0?1:-1)*term;
      }
      if(!Number.isFinite(total))throw new RangeError("exterior derivative is non-finite");
      setTensorValue(out,indices,total);
    });
    return out;
  }
  hodgeDual(twoForm,x=undefined){
    if(x===undefined&&typeof this.metric==="function")throw new TypeError("x is required when the metric depends on position");
    const forms=parseGeometryBatch(twoForm,"two-form",[4,4]),positions=parseGeometryBatch(x===undefined?[0,0,0,0]:x,"position",[4]),batchShape=geometryBatchShape(forms.shape.slice(0,-2),positions.shape.slice(0,-1)),components=batchShape.reduce((count,size)=>count*size,1)*16;if(components>MAX_COVARIANT_COMPONENTS)throw new RangeError("Hodge dual batch exceeds the bounded component limit");
    return geometryBuildBatch(batchShape,index=>this._hodgeDualScalar(geometryBatchGet(forms.data,forms.shape.slice(0,-2),index,batchShape),geometryBatchGet(positions.data,positions.shape.slice(0,-1),index,batchShape)));
  }
  _hodgeDualScalar(twoForm,x){
    const g=this._g(x);
    const epsilon=LeviCivitaTensor(g),f=matrix(twoForm),gi=inverse(g),fUp=mmul(mmul(gi,f),transpose(gi)),out=zeros(4);
    if(fUp.some(row=>row.some(value=>!Number.isFinite(value))))throw new RangeError("Hodge dual raised components are non-finite");
    for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++)out[a][b]+=.5*epsilon[a][b][c][d]*fUp[c][d];
    if(out.some(row=>row.some(value=>!Number.isFinite(value))))throw new RangeError("Hodge dual components are non-finite");
    return out;
  }
  tensorHessianForward(fn,x){return this._mapPoints(x,p=>forwardSecondOrder4(fn,p).hessian,(_value,components)=>3*components);}
  hessianForward(fn,x){return this._mapPoints(x,p=>forwardHessian4(fn,p).hessian,64);}
  dAlembertianForward(fn,x){return this._mapPoints(x,p=>{
    const field=forwardHessian4(fn,p),g=inverse(this._g(p)),gamma=this.connection.christoffel(p);let result=0;
    for(let mu=0;mu<4;mu++)for(let nu=0;nu<4;nu++){let trace=0;for(let a=0;a<4;a++)trace+=gamma[a][mu][nu]*field.gradient[a];result+=g[mu][nu]*(field.hessian[mu][nu]-trace);}
    if(!Number.isFinite(result))throw new RangeError("dAlembertian is non-finite");return result;
  },300);}
  dAlembertian(fn,x){return this._mapPoints(x,p=>{const g=inverse(this._g(p)),grad=this._partialScalar(fn,p),gamma=this.connection.christoffel(p);let result=0;for(let mu=0;mu<4;mu++)for(let nu=0;nu<4;nu++){const second=partial(q=>this._partialScalar(fn,q)[mu],p,nu,this.step);let trace=0;for(let a=0;a<4;a++)trace+=gamma[a][mu][nu]*grad[a];result+=g[mu][nu]*(second-trace);}if(!Number.isFinite(result))throw new RangeError("dAlembertian is non-finite");return result;},300);}
}
function LeviCivitaTensor(g){const root=-determinantRoot(g).root,symbol=(()=>{const e=zeros(4).map(()=>zeros(4).map(()=>zeros(4).map(()=>Array(4).fill(0))));for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++){const v=[a,b,c,d];if(new Set(v).size===4){let p=0;for(let i=0;i<4;i++)for(let j=i+1;j<4;j++)if(v[i]>v[j])p++;e[a][b][c][d]=p%2?-1:1;}}return e;})();return symbol.map(a=>a.map(b=>b.map(c=>c.map(v=>v*root))));}

function forwardConnectionDerivatives(metric,p){
    const field=forwardSecondOrder4(metric,p),g=geometryMetric(field.value),gi=inverse(g),dg=field.jacobian,ddg=field.hessian;
    const gamma=Array.from({length:4},()=>Array.from({length:4},()=>Array(4).fill(0)));
    const derivative=Array.from({length:4},()=>Array.from({length:4},()=>Array.from({length:4},()=>Array(4).fill(0))));
    const inverseDerivative=Array.from({length:4},()=>zeros(4));
    for(let e=0;e<4;e++)for(let a=0;a<4;a++)for(let d=0;d<4;d++)for(let i=0;i<4;i++)for(let j=0;j<4;j++)inverseDerivative[e][a][d]-=gi[a][i]*dg[i][j][e]*gi[j][d];
    for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++){
      const q=dg[d][c][b]+dg[d][b][c]-dg[b][c][d];gamma[a][b][c]+=.5*gi[a][d]*q;
      for(let e=0;e<4;e++)derivative[e][a][b][c]+=.5*(inverseDerivative[e][a][d]*q+gi[a][d]*(ddg[d][c][b][e]+ddg[d][b][c][e]-ddg[b][c][d][e]));
    }
    if(gamma.flat(2).some(value=>!Number.isFinite(value))||derivative.flat(3).some(value=>!Number.isFinite(value)))throw new RangeError("forward connection derivative is non-finite");
    return {gamma,derivative};
}

export class Connection4 {
  constructor(metric,{step=1e-5,metricDerivative="finite"}={}){if(!["finite","forward"].includes(metricDerivative))throw new TypeError("metricDerivative must be finite or forward");this.metric=metric;this.step=validateDifferenceStep(step);this.metricDerivative=metricDerivative;}
  _g(x){const g=this.metric(x);return geometryMetric(g);}
  christoffel(x){if(this.metricDerivative==="forward")return this.christoffelForward(x);if(this.metricDerivative!=="finite")throw new TypeError("metricDerivative must be finite or forward");const parsed=parseGeometryBatch(x,"position",[4]),batchShape=parsed.shape.slice(0,-1),components=batchShape.reduce((count,size)=>count*size,1)*64;if(components>MAX_COVARIANT_COMPONENTS)throw new RangeError("Christoffel batch exceeds the component limit");return geometryBuildBatch(batchShape,index=>this._christoffelScalar(geometryBatchGet(parsed.data,batchShape,index,batchShape)));}
  christoffelForward(x){
    const parsed=parseGeometryBatch(x,"position",[4]),batchShape=parsed.shape.slice(0,-1),components=batchShape.reduce((count,size)=>count*size,1)*64;
    if(components>MAX_COVARIANT_COMPONENTS)throw new RangeError("Christoffel batch exceeds the component limit");
    return geometryBuildBatch(batchShape,index=>this._christoffelForwardScalar(geometryBatchGet(parsed.data,batchShape,index,batchShape)));
  }
  christoffelDerivativeForward(x){
    const parsed=parseGeometryBatch(x,"position",[4]),batchShape=parsed.shape.slice(0,-1),count=batchShape.reduce((product,size)=>product*size,1);
    if(count*256>MAX_COVARIANT_COMPONENTS||count*4096>MAX_CURVATURE_BATCH_WORK)throw new RangeError("connection derivative batch exceeds the bounded component or work limit");
    return geometryBuildBatch(batchShape,index=>forwardConnectionDerivatives(this.metric,geometryBatchGet(parsed.data,batchShape,index,batchShape)).derivative);
  }
  _christoffelForwardScalar(p){
    const differentiated=forwardJacobian4(this.metric,p),g=geometryMetric(differentiated.value),gi=inverse(g),dg=differentiated.jacobian,out=zeros(4).map(()=>zeros(4));
    for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++)out[a][b][c]+=.5*gi[a][d]*(dg[d][c][b]+dg[d][b][c]-dg[b][c][d]);
    if(out.some(plane=>plane.some(row=>row.some(value=>!Number.isFinite(value)))))throw new RangeError("Christoffel symbols are non-finite");
    return out;
  }
  _christoffelScalar(p){const g=this._g(p),gi=inverse(g),dg=Array.from({length:4},(_,k)=>{const stencil=differenceStencil(p,k,this.step);const a=this._g(stencil.plus),b=this._g(stencil.minus);return a.map((r,i)=>r.map((v,j)=>weightedStencilDerivative(v,b[i][j],g[i][j],stencil)));}),out=zeros(4).map(()=>zeros(4));for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++)out[a][b][c]+=.5*gi[a][d]*(dg[b][d][c]+dg[c][d][b]-dg[d][b][c]);if(out.some(plane=>plane.some(row=>row.some(value=>!Number.isFinite(value)))))throw new RangeError("Christoffel symbols are non-finite");return out;}
  covariantDerivativeVectorForward(field,x){return this.covariantDerivativeForward(field,x,["up"]);}
  covariantDerivativeForward(field,x,indexTypes=["up"]){
    if(!Array.isArray(indexTypes)||Array.from(indexTypes).some(type=>type!=="up"&&type!=="down"))throw new TypeError("indexTypes entries must be 'up' or 'down'");
    if(indexTypes.length>24)throw new RangeError("tensor rank exceeds the covariant derivative index limit");
    const parsed=parseGeometryBatch(x,"position",[4]),batchShape=parsed.shape.slice(0,-1),batchCount=batchShape.reduce((count,size)=>count*size,1),firstIndex=Array(batchShape.length).fill(0),firstPoint=geometryBatchGet(parsed.data,batchShape,firstIndex,batchShape),first=forwardJacobian4(field,firstPoint),shape=tensorShape(first.value);
    if(shape.length!==indexTypes.length)throw new RangeError("indexTypes length must match tensor field rank");
    const components=shape.reduce((product,length)=>product*length,1);
    if(batchCount*components*4>MAX_COVARIANT_COMPONENTS)throw new RangeError("covariant derivative exceeds the bounded component limit");
    const evaluate=(point,differentiated)=>{
      if(JSON.stringify(tensorShape(differentiated.value))!==JSON.stringify(shape))throw new RangeError("tensor field shape must remain constant across batch samples");
      const g=this.christoffel(point),out=zeroTensor([...shape,4]);
      eachIndex(indexTypes.length,slots=>{for(let d=0;d<4;d++){
        let derivative=tensorValue(differentiated.jacobian,[...slots,d]);
        for(let axis=0;axis<indexTypes.length;axis++)for(let e=0;e<4;e++){
          const replaced=slots.slice();replaced[axis]=e;
          const connection=indexTypes[axis]==="up"?g[slots[axis]][d][e]:g[e][d][slots[axis]];
          derivative+=(indexTypes[axis]==="up"?1:-1)*connection*tensorValue(differentiated.value,replaced);
        }
        if(!Number.isFinite(derivative))throw new RangeError("covariant derivative is non-finite");
        setTensorValue(out,[...slots,d],derivative);
      }});
      return out;
    };
    if(!batchShape.length)return evaluate(firstPoint,first);
    return geometryBuildBatch(batchShape,index=>{const point=geometryBatchGet(parsed.data,batchShape,index,batchShape),value=index.every((part,axis)=>part===firstIndex[axis])?first:forwardJacobian4(field,point);return evaluate(point,value);});
  }
  covariantDerivativeVector(field,x){return this.covariantDerivative(field,x,["up"]);}
  covariantDerivative(field,x,indexTypes=["up"]){
    if(!Array.isArray(indexTypes)||Array.from(indexTypes).some(type=>type!=="up"&&type!=="down"))throw new TypeError("indexTypes entries must be 'up' or 'down'");
    if(indexTypes.length>24)throw new RangeError("tensor rank exceeds the covariant derivative index limit");
    const parsed=parseGeometryBatch(x,"position",[4]),batchShape=parsed.shape.slice(0,-1),batchCount=batchShape.reduce((count,size)=>count*size,1),firstPoint=geometryBatchGet(parsed.data,batchShape,Array(batchShape.length).fill(0),batchShape),firstValue=field(firstPoint),shape=tensorShape(firstValue);
    if(shape.length!==indexTypes.length)throw new RangeError("indexTypes length must match tensor field rank");
    const components=shape.reduce((product,length)=>product*length,1);
    if(batchCount*components*4>MAX_COVARIANT_COMPONENTS)throw new RangeError("covariant derivative exceeds the bounded component limit");
    if(!batchShape.length)return this._covariantDerivativeScalar(field,firstPoint,indexTypes,shape,firstValue);
    return geometryBuildBatch(batchShape,index=>{const point=geometryBatchGet(parsed.data,batchShape,index,batchShape),value=field(point),valueShape=tensorShape(value);if(JSON.stringify(valueShape)!==JSON.stringify(shape))throw new RangeError("tensor field shape must remain constant across batch samples");return this._covariantDerivativeScalar(field,point,indexTypes,shape,value);});
  }
  _covariantDerivativeScalar(field,p,indexTypes,shape,value){
    const g=this.christoffel(p),out=zeroTensor([...shape,4]);
    for(let d=0;d<4;d++){
      const stencil=differenceStencil(p,d,this.step);
      const plus=field(stencil.plus),minus=field(stencil.minus);
      if(JSON.stringify(tensorShape(plus))!==JSON.stringify(shape)||JSON.stringify(tensorShape(minus))!==JSON.stringify(shape))throw new RangeError("tensor field shape must remain constant in a neighborhood");
      eachIndex(indexTypes.length,slots=>{
        let derivative=weightedStencilDerivative(tensorValue(plus,slots),tensorValue(minus,slots),tensorValue(value,slots),stencil);
        for(let axis=0;axis<indexTypes.length;axis++)for(let e=0;e<4;e++){
          const replaced=slots.slice();replaced[axis]=e;
          const connection=indexTypes[axis]==="up"?g[slots[axis]][d][e]:g[e][d][slots[axis]];
          derivative+=(indexTypes[axis]==="up"?1:-1)*connection*tensorValue(value,replaced);
        }
        if(!Number.isFinite(derivative))throw new RangeError("covariant derivative is non-finite");
        setTensorValue(out,[...slots,d],derivative);
      });
    }
    return out;
  }
  parallelTransport(value,path,{stepsPerSegment=8}={}){
    if(!Number.isInteger(stepsPerSegment)||stepsPerSegment<1)throw new RangeError("stepsPerSegment must be a positive integer");
    const parsedPath=parseGeometryBatch(path,"path",[4]),parsedValue=parseGeometryBatch(value,"vector",[4]);
    if(parsedPath.shape.length<2)throw new RangeError("path must contain at least two coordinate points");
    const pathLength=parsedPath.shape.at(-2),pathBatchShape=parsedPath.shape.slice(0,-2),valueBatchShape=parsedValue.shape.slice(0,-1);
    if(pathLength<2)throw new RangeError("path must contain at least two coordinate points");
    const shape=geometryBatchShape(pathBatchShape,valueBatchShape),components=shape.reduce((p,n)=>p*n,1)*pathLength*4,work=shape.reduce((p,n)=>p*n,1)*(pathLength-1)*stepsPerSegment;
    if(!Number.isSafeInteger(components)||components>MAX_COVARIANT_COMPONENTS)throw new RangeError("parallel transport history exceeds the bounded component limit");
    if(!Number.isSafeInteger(work)||work>MAX_TRANSPORT_INTEGRATION_STEPS)throw new RangeError("parallel transport exceeds the bounded integration-step limit");
    return geometryBuildBatch(shape,index=>this._parallelTransportScalar(geometryBatchGet(parsedValue.data,valueBatchShape,index,shape),geometryBatchGet(parsedPath.data,pathBatchShape,index,shape),stepsPerSegment));
  }
  _parallelTransportScalar(value,path,stepsPerSegment){
    let transported=vector(value),result=[transported.slice()];
    for(let i=1;i<path.length;i++){
      const x0=vector(path[i-1]),x1=vector(path[i]),delta=x1.map((component,k)=>component-x0[k]);
      const rows=integrateRK4((s,current)=>{
        const x=x0.map((component,k)=>component+s*delta[k]),gamma=this.christoffel(x),derivative=Array(4).fill(0);
        for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)derivative[a]-=gamma[a][b][c]*current[c]*delta[b];
        return derivative;
      },transported,0,1,stepsPerSegment);
      transported=rows.at(-1)[1];result.push(transported.slice());
    }
    return result;
  }
  geodesicAcceleration(x,tangent){const position=parseGeometryBatch(x,"position",[4]),velocity=parseGeometryBatch(tangent,"tangent",[4]),batchShape=geometryBatchShape(position.shape.slice(0,-1),velocity.shape.slice(0,-1)),components=batchShape.reduce((count,size)=>count*size,1)*4;if(components>MAX_COVARIANT_COMPONENTS)throw new RangeError("geodesic acceleration batch exceeds the component limit");return geometryBuildBatch(batchShape,index=>{const point=geometryBatchGet(position.data,position.shape.slice(0,-1),index,batchShape),v=geometryBatchGet(velocity.data,velocity.shape.slice(0,-1),index,batchShape),g=this.christoffel(point),a=Array(4).fill(0);for(let i=0;i<4;i++)for(let j=0;j<4;j++)for(let k=0;k<4;k++)a[i]-=g[i][j][k]*v[j]*v[k];if(a.some(value=>!Number.isFinite(value)))throw new RangeError("geodesic acceleration is non-finite");return a;});}
}

export class Curvature4 {
  constructor(metric,{step=1e-5,metricDerivative="finite"}={}){if(!["finite","forward"].includes(metricDerivative))throw new TypeError("metricDerivative must be finite or forward");this.metric=metric;this.step=validateDifferenceStep(step);this.metricDerivative=metricDerivative;this.connection=new Connection4(metric,{step:this.step,metricDerivative});}
  _metricAt(point){return this.metricDerivative==="forward"?forwardSecondOrder4(this.metric,point).value:this.metric(point);}
  _mapPoints(x,components,operation,workPerPoint=4096){const parsed=parseGeometryBatch(x,"position",[4]),batchShape=parsed.shape.slice(0,-1),batchCount=batchShape.reduce((count,size)=>count*size,1),outputComponents=batchCount*components;if(outputComponents>MAX_COVARIANT_COMPONENTS)throw new RangeError("curvature batch exceeds the bounded component limit");if(batchCount*workPerPoint>MAX_CURVATURE_BATCH_WORK)throw new RangeError("curvature batch exceeds the bounded work limit");return geometryBuildBatch(batchShape,index=>operation(geometryBatchGet(parsed.data,batchShape,index,batchShape)));}
  riemann(x){return this._mapPoints(x,256,point=>this._riemannScalar(point));}
  _riemannScalar(p){
    if(this.metricDerivative==="forward")return this._riemannForwardScalar(p);
    if(this.metricDerivative!=="finite")throw new TypeError("metricDerivative must be finite or forward");
    const christoffel=this.connection instanceof Connection4&&this.connection.metricDerivative==="forward"?point=>this.connection._christoffelScalar(point):point=>this.connection.christoffel(point);
    const g=christoffel(p);
    const dg=Array.from({length:4},(_,d)=>{
      const stencil=differenceStencil(p,d,this.step);
      const plus=christoffel(stencil.plus),minus=christoffel(stencil.minus);
      return plus.map((row,i)=>row.map((plane,j)=>plane.map((value,k)=>weightedStencilDerivative(value,minus[i][j][k],g[i][j][k],stencil))));
    });
    const R=zeros(4).map(()=>zeros(4).map(()=>zeros(4).map(()=>Array(4).fill(0))));
    for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++){
      R[a][b][c][d]=dg[c][a][d][b]-dg[d][a][c][b];
      for(let e=0;e<4;e++)R[a][b][c][d]+=g[a][c][e]*g[e][d][b]-g[a][d][e]*g[e][c][b];
      if(!Number.isFinite(R[a][b][c][d]))throw new RangeError("Riemann curvature is non-finite");
    }
    return R;
  }
  _riemannForwardScalar(p){
    const {gamma,derivative}=forwardConnectionDerivatives(this.metric,p);
    const R=Array.from({length:4},()=>Array.from({length:4},()=>Array.from({length:4},()=>Array(4).fill(0))));
    for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++){
      let value=derivative[c][a][d][b]-derivative[d][a][c][b];
      for(let e=0;e<4;e++)value+=gamma[a][c][e]*gamma[e][d][b]-gamma[a][d][e]*gamma[e][c][b];
      if(!Number.isFinite(value))throw new RangeError("Riemann curvature is non-finite");R[a][b][c][d]=value;
    }
    return R;
  }
  ricci(x){return this._mapPoints(x,16,point=>this._ricciTensorScalar(point));}
  _ricciTensorScalar(point){const R=this._riemannScalar(point),out=zeros(4);for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)out[a][b]+=R[c][a][c][b];if(out.some(row=>row.some(value=>!Number.isFinite(value))))throw new RangeError("Ricci tensor is non-finite");return out;}
  ricciScalar(x){return this._mapPoints(x,1,point=>{const raw=this._metricAt(point),g=inverse(geometryMetric(raw)),r=this._ricciTensorScalar(point);let scalar=0;for(let a=0;a<4;a++)for(let b=0;b<4;b++)scalar+=g[a][b]*r[a][b];if(!Number.isFinite(scalar))throw new RangeError("Ricci scalar is non-finite");return scalar;});}
  einsteinTensor(x){return this._mapPoints(x,16,point=>{const r=this._ricciTensorScalar(point),raw=this._metricAt(point),g=geometryMetric(raw),inverseMetric=inverse(g);let scalar=0;for(let a=0;a<4;a++)for(let b=0;b<4;b++)scalar+=inverseMetric[a][b]*r[a][b];if(!Number.isFinite(scalar))throw new RangeError("Einstein tensor curvature scalar is non-finite");const result=r.map((row,i)=>row.map((value,j)=>value-.5*g[i][j]*scalar));if(result.some(row=>row.some(value=>!Number.isFinite(value))))throw new RangeError("Einstein tensor is non-finite");return result;});}
  kretschmannScalar(x){return this._mapPoints(x,1,point=>{const R=this._riemannScalar(point),raw=this._metricAt(point),g=geometryMetric(raw),gi=inverse(g),lower=zeros(4).map(()=>zeros(4).map(()=>zeros(4).map(()=>Array(4).fill(0))));for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++)for(let e=0;e<4;e++)lower[a][b][c][d]+=g[a][e]*R[e][b][c][d];let sum=0;for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++)for(let A=0;A<4;A++)for(let B=0;B<4;B++)for(let Cc=0;Cc<4;Cc++)for(let D=0;D<4;D++)sum+=gi[a][A]*gi[b][B]*gi[c][Cc]*gi[d][D]*lower[a][b][c][d]*lower[A][B][Cc][D];if(!Number.isFinite(sum))throw new RangeError("Kretschmann scalar is non-finite");return sum;},70_000);}
}

export class Geodesic4 {
  constructor(connection,initialEvent,initialTangent){this.connection=connection;this.eventFrame=initialEvent instanceof Event4?initialEvent.frame:null;this.eventC=initialEvent instanceof Event4?initialEvent.c:undefined;this.initialEvent=initialEvent instanceof Event4?initialEvent.position:initialEvent;this.initialTangent=initialTangent;const event=parseGeometryBatch(this.initialEvent,"initial event",[4]),tangent=parseGeometryBatch(this.initialTangent,"initial tangent",[4]);if(JSON.stringify(event.shape)!==JSON.stringify(tangent.shape))throw new RangeError("initial_event and initial_tangent must have matching (...,4) shapes");this.initialEvent=event.data;this.initialTangent=tangent.data;}
  integrate(span,{steps=100,format="samples"}={}){if(!["samples","components"].includes(format))throw new TypeError("format must be 'samples' or 'components'");const parameterSpan=vector(span,2,"parameter span"),event=parseGeometryBatch(this.initialEvent,"initial event",[4]),tangent=parseGeometryBatch(this.initialTangent,"initial tangent",[4]),batchShape=event.shape.slice(0,-1),state=geometryBuildBatch(batchShape,index=>[...geometryBatchGet(event.data,batchShape,index,batchShape),...geometryBatchGet(tangent.data,batchShape,index,batchShape)]),rows=integrateRK4((parameter,current)=>geometryBuildBatch(batchShape,index=>{const sample=geometryBatchGet(current,batchShape,index,batchShape);return [...sample.slice(4),...this.connection.geodesicAcceleration(sample.slice(0,4),sample.slice(4))];}),state,parameterSpan[0],parameterSpan[1],steps),samples=rows.map(([parameter,current])=>({parameter,event:new Event4(geometryBuildBatch(batchShape,index=>geometryBatchGet(current,batchShape,index,batchShape).slice(0,4)),this.eventFrame,this.eventC),tangent:geometryBuildBatch(batchShape,index=>geometryBatchGet(current,batchShape,index,batchShape).slice(4))}));return format==="samples"?samples:[samples.map(sample=>sample.parameter),samples.map(sample=>sample.event.position),samples.map(sample=>sample.tangent)];}
  _validateForwardSensitivityConnection(){
    if(!(this.connection instanceof Connection4)||this.connection.metricDerivative!=="forward")throw new TypeError("forward endpoint sensitivity requires a Connection4 configured with metricDerivative forward");
    for(const name of ["christoffel","christoffelForward","_christoffelForwardScalar","geodesicAcceleration"])if(this.connection[name]!==Connection4.prototype[name])throw new TypeError("forward endpoint sensitivity requires the standard metric connection");
  }
  /** Endpoint and its derivative with respect to the initial tangent. */
  endpointSensitivityForward(span,{steps=100}={}){
    if(this.initialEvent.length!==4||this.initialTangent.length!==4||this.initialEvent.some(Array.isArray)||this.initialTangent.some(Array.isArray))throw new RangeError("endpoint sensitivity requires unbatched four-vectors");
    this._validateForwardSensitivityConnection();
    if(!Number.isInteger(steps)||steps<1||steps>2000)throw new RangeError("endpoint sensitivity steps must be an integer from 1 to 2000");
    const interval=vector(span,2,"parameter span"),initial=[...this.initialEvent,...this.initialTangent,...Array(16).fill(0),...identity(4).flat()];
    const rows=integrateRK4((_parameter,state)=>{
      const position=state.slice(0,4),velocity=state.slice(4,8),S=Array.from({length:4},(_,i)=>state.slice(8+4*i,12+4*i)),T=Array.from({length:4},(_,i)=>state.slice(24+4*i,28+4*i));
      const {gamma,derivative}=forwardConnectionDerivatives(this.connection.metric,position),acceleration=Array(4).fill(0),change=zeros(4);
      for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++){
        acceleration[a]-=gamma[a][b][c]*velocity[b]*velocity[c];
        for(let j=0;j<4;j++){
          change[a][j]-=gamma[a][b][c]*(T[b][j]*velocity[c]+velocity[b]*T[c][j]);
          for(let d=0;d<4;d++)change[a][j]-=derivative[d][a][b][c]*S[d][j]*velocity[b]*velocity[c];
        }
      }
      return [...velocity,...acceleration,...T.flat(),...change.flat()];
    },initial,interval[0],interval[1],steps);
    const last=rows.at(-1)[1];return {event:last.slice(0,4),tangent:last.slice(4,8),jacobian:Array.from({length:4},(_,i)=>last.slice(8+4*i,12+4*i))};
  }
  solveBoundary(targetEvent,span,{steps=100,iterations=12,maxIterations=iterations,tolerance=1e-9,damping=1e-10,format="samples",jacobianMethod="finite"}={}){
    if(!["samples","components"].includes(format))throw new TypeError("format must be 'samples' or 'components'");
    if(!["finite","forward"].includes(jacobianMethod))throw new TypeError("jacobianMethod must be finite or forward");
    if(jacobianMethod==="forward")this._validateForwardSensitivityConnection();
    if(this.initialEvent.length!==4||this.initialTangent.length!==4||this.initialEvent.some(value=>Array.isArray(value))||this.initialTangent.some(value=>Array.isArray(value)))throw new RangeError("boundary shooting requires unbatched four-vectors");
    if(!Number.isInteger(steps)||steps<1||steps>2000)throw new RangeError("boundary solve steps must be an integer from 1 to 2000");
    if(!Number.isInteger(maxIterations)||maxIterations<0||maxIterations>12)throw new RangeError("boundary solve maxIterations must be an integer from 0 to 12");
    if(steps*(9*maxIterations+2)>12_000)throw new RangeError("boundary solve exceeds the bounded integration-step limit");
    if(!Number.isFinite(tolerance)||tolerance<=0)throw new RangeError("tolerance must be positive and finite");
    if(!Number.isFinite(damping)||damping<=0)throw new RangeError("damping must be positive and finite");
    const target=vector(targetEvent instanceof Event4?targetEvent.position:targetEvent);
    const endpoint=tangent=>new Geodesic4(this.connection,this.initialEvent,tangent).integrate(span,{steps}).at(-1).event.position;
    let tangent=this.initialTangent.slice();
    for(let iteration=0;iteration<=maxIterations;iteration++){
      const currentEndpoint=endpoint(tangent),residual=sub(currentEndpoint,target);
      if(Math.hypot(...residual)<=tolerance){const result=new Geodesic4(this.connection,new Event4(this.initialEvent,this.eventFrame,this.eventC),tangent).integrate(span,{steps,format});return format==="components"?[...result,tangent.slice()]:result;}
      if(iteration===maxIterations)break;
      let jacobian;
      if(jacobianMethod==="forward")jacobian=new Geodesic4(this.connection,this.initialEvent,tangent).endpointSensitivityForward(span,{steps}).jacobian;
      else{
        const h=1e-5;
        const columns=Array.from({length:4},(_,column)=>{
          const stencil=differenceStencil(tangent,column,h);
          const plusEnd=endpoint(stencil.plus),minusEnd=endpoint(stencil.minus);
          return plusEnd.map((value,row)=>weightedStencilDerivative(value,minusEnd[row],currentEndpoint[row],stencil));
        });
        jacobian=transpose(columns);
      }
      const jacobianTranspose=transpose(jacobian);
      const normal=mmul(jacobianTranspose,jacobian);
      for(let axis=0;axis<4;axis++)normal[axis][axis]+=damping;
      const update=mvec(inverse(normal),mvec(jacobianTranspose,residual));
      tangent=sub(tangent,update);
    }
    throw new RangeError("geodesic boundary shooting did not converge within the iteration limit");
  }
}
