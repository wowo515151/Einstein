// Explicit first-order forward differentiation for functions written with Dual4 methods.
const MAX_OUTPUT_COMPONENTS=16384;
function finite(value,name){if(typeof value!=="number"||!Number.isFinite(value))throw new TypeError(`${name} must be a finite number`);return value;}
function asDual(value){return value instanceof Dual4?value:Dual4.constant(value);}

export class Dual4 {
  constructor(value,derivatives=[0,0,0,0]){
    this.value=finite(value,"dual value");
    if(!Array.isArray(derivatives)||derivatives.length!==4)throw new TypeError("dual derivatives must contain four numbers");
    this.derivatives=Object.freeze(Array.from(derivatives,(v)=>finite(v,"dual derivative")));Object.freeze(this);
  }
  static constant(value){return new Dual4(value);}
  static variable(value,axis){if(!Number.isInteger(axis)||axis<0||axis>3)throw new RangeError("dual variable axis must be from 0 to 3");return new Dual4(value,Array.from({length:4},(_,i)=>i===axis?1:0));}
  [Symbol.toPrimitive](){throw new TypeError("Dual4 cannot be implicitly converted; use its arithmetic methods");}
  add(other){const b=asDual(other);return new Dual4(this.value+b.value,this.derivatives.map((v,i)=>v+b.derivatives[i]));}
  subtract(other){const b=asDual(other);return new Dual4(this.value-b.value,this.derivatives.map((v,i)=>v-b.derivatives[i]));}
  multiply(other){const b=asDual(other);return new Dual4(this.value*b.value,this.derivatives.map((v,i)=>v*b.value+this.value*b.derivatives[i]));}
  divide(other){const b=asDual(other);if(b.value===0)throw new RangeError("dual division by zero");const q=this.value/b.value;return new Dual4(q,this.derivatives.map((v,i)=>(v-q*b.derivatives[i])/b.value));}
  negate(){return new Dual4(-this.value,this.derivatives.map(v=>-v));}
  sin(){return new Dual4(Math.sin(this.value),this.derivatives.map(v=>Math.cos(this.value)*v));}
  cos(){return new Dual4(Math.cos(this.value),this.derivatives.map(v=>-Math.sin(this.value)*v));}
  sinh(){return new Dual4(Math.sinh(this.value),this.derivatives.map(v=>Math.cosh(this.value)*v));}
  cosh(){return new Dual4(Math.cosh(this.value),this.derivatives.map(v=>Math.sinh(this.value)*v));}
  exp(){const value=Math.exp(this.value);return new Dual4(value,this.derivatives.map(v=>value*v));}
  log(){if(this.value<=0)throw new RangeError("dual logarithm requires a positive value");return new Dual4(Math.log(this.value),this.derivatives.map(v=>v/this.value));}
  sqrt(){if(this.value<=0)throw new RangeError("dual square root requires a positive value");const value=Math.sqrt(this.value);return new Dual4(value,this.derivatives.map(v=>v/(2*value)));}
  pow(exponent){
    finite(exponent,"exponent");if(exponent===0)return Dual4.constant(1);
    if(this.value<0&&!Number.isInteger(exponent))throw new RangeError("dual power requires a real result");
    if(this.value===0&&exponent<1)throw new RangeError("dual power has an undefined derivative at zero");
    const value=this.value**exponent,factor=this.value===0?(exponent===1?1:0):exponent*this.value**(exponent-1);
    return new Dual4(value,this.derivatives.map(v=>factor*v));
  }
}

/** Evaluate a scalar or bounded rectangular field and its first derivatives at one point. */
export function forwardJacobian4(field,point){
  if(typeof field!=="function")throw new TypeError("field must be a function");
  if((!Array.isArray(point)&&!ArrayBuffer.isView(point))||point.length!==4)throw new RangeError("point must have four coordinates");
  const input=Array.from(point,(value,axis)=>Dual4.variable(value,axis)),output=field(input),active=new Set();let count=0;
  function unpack(node,depth){
    if(node instanceof Dual4||typeof node==="number"){
      if(++count>MAX_OUTPUT_COMPONENTS)throw new RangeError("forward field output exceeds the component limit");
      const d=asDual(node);return {value:d.value,jacobian:Array.from(d.derivatives),shape:[]};
    }
    if(!Array.isArray(node)||node.length===0)throw new TypeError("forward field output must be a nonempty rectangular array or scalar");
    if(depth>=8)throw new RangeError("forward field output rank exceeds eight");
    if(node.length>MAX_OUTPUT_COMPONENTS)throw new RangeError("forward field output exceeds the component limit");
    if(active.has(node))throw new RangeError("forward field output cannot contain cycles");
    active.add(node);
    try{
      const children=Array.from(node,child=>unpack(child,depth+1)),shape=children[0].shape;
      if(children.some(child=>JSON.stringify(child.shape)!==JSON.stringify(shape)))throw new RangeError("forward field output must be rectangular");
      return {value:children.map(child=>child.value),jacobian:children.map(child=>child.jacobian),shape:[node.length,...shape]};
    }finally{active.delete(node);}
  }
  const result=unpack(output,0);return {value:result.value,jacobian:result.jacobian};
}


/** Differentiate a scalar or rectangular output with respect to up to 32 inputs. */
export function forwardJacobian(field,point){
  if(typeof field!=="function")throw new TypeError("field must be a function");
  if((!Array.isArray(point)&&!(ArrayBuffer.isView(point)&&!(point instanceof DataView)))||point.length<1||point.length>32)throw new RangeError("point must contain from 1 to 32 coordinates");
  const input=Array.from(point,value=>finite(value,"point coordinate")),n=input.length,limit=Math.floor(65536/(n+1));let shape,values,jacobian;
  for(let first=0;first<n;first+=4){
    const variables=input.map((value,i)=>i>=first&&i<first+4?Dual4.variable(value,i-first):Dual4.constant(value)),output=field(variables),active=new Set(),leaves=[];
    function inspect(node,depth){
      if(node instanceof Dual4||typeof node==="number"){if(leaves.length>=limit)throw new RangeError("forward field output exceeds the component limit");leaves.push(asDual(node));return [];}
      if(!Array.isArray(node)||node.length===0)throw new TypeError("forward field output must be a nonempty rectangular array or scalar");
      if(depth>=8)throw new RangeError("forward field output rank exceeds eight");
      if(node.length>limit)throw new RangeError("forward field output exceeds the component limit");
      if(active.has(node))throw new RangeError("forward field output cannot contain cycles");
      active.add(node);try{const childShapes=Array.from(node,child=>inspect(child,depth+1)),childShape=childShapes[0];if(childShapes.some(child=>JSON.stringify(child)!==JSON.stringify(childShape)))throw new RangeError("forward field output must be rectangular");return [node.length,...childShape];}finally{active.delete(node);}
    }
    const outputShape=inspect(output,0),sample=leaves.map(leaf=>leaf.value);
    if(!shape){shape=outputShape;values=sample;jacobian=leaves.map(()=>Array(n).fill(0));}
    else if(JSON.stringify(outputShape)!==JSON.stringify(shape)||sample.some((value,i)=>!Object.is(value,values[i])))throw new RangeError("forward field output must remain consistent across sensitivity evaluations");
    for(let row=0;row<leaves.length;row++)for(let lane=0;lane<4&&first+lane<n;lane++)jacobian[row][first+lane]=leaves[row].derivatives[lane];
  }
  const rebuild=leaves=>{let i=0;const visit=depth=>depth===shape.length?leaves[i++]:Array.from({length:shape[depth]},()=>visit(depth+1));return visit(0);};
  return {value:rebuild(values),jacobian:rebuild(jacobian)};
}


function asJet(value){return value instanceof Jet4?value:Jet4.constant(value);}

/** A scalar value with four first derivatives and a 4×4 Hessian. */
export class Jet4 {
  constructor(value,derivatives=[0,0,0,0],hessian=Array.from({length:4},()=>[0,0,0,0])){
    this.value=finite(value,"jet value");
    if(!Array.isArray(derivatives)||derivatives.length!==4)throw new TypeError("jet derivatives must contain four numbers");
    if(!Array.isArray(hessian)||hessian.length!==4||Array.from(hessian).some(row=>!Array.isArray(row)||row.length!==4))throw new TypeError("jet Hessian must be a 4 by 4 array");
    this.derivatives=Object.freeze(Array.from(derivatives,v=>finite(v,"jet derivative")));
    this.hessian=Object.freeze(Array.from(hessian,row=>Object.freeze(Array.from(row,v=>finite(v,"jet Hessian")))));Object.freeze(this);
  }
  static constant(value){return new Jet4(value);}
  static variable(value,axis){if(!Number.isInteger(axis)||axis<0||axis>3)throw new RangeError("jet variable axis must be from 0 to 3");return new Jet4(value,Array.from({length:4},(_,i)=>i===axis?1:0));}
  [Symbol.toPrimitive](){throw new TypeError("Jet4 cannot be implicitly converted; use its arithmetic methods");}
  add(other){const b=asJet(other);return new Jet4(this.value+b.value,this.derivatives.map((v,i)=>v+b.derivatives[i]),this.hessian.map((row,i)=>row.map((v,j)=>v+b.hessian[i][j])));}
  subtract(other){const b=asJet(other);return new Jet4(this.value-b.value,this.derivatives.map((v,i)=>v-b.derivatives[i]),this.hessian.map((row,i)=>row.map((v,j)=>v-b.hessian[i][j])));}
  multiply(other){const b=asJet(other);return new Jet4(this.value*b.value,this.derivatives.map((v,i)=>v*b.value+this.value*b.derivatives[i]),this.hessian.map((row,i)=>row.map((v,j)=>v*b.value+this.derivatives[i]*b.derivatives[j]+this.derivatives[j]*b.derivatives[i]+this.value*b.hessian[i][j])));}
  divide(other){const b=asJet(other);if(b.value===0)throw new RangeError("jet division by zero");const value=this.value/b.value,derivatives=this.derivatives.map((v,i)=>(v-value*b.derivatives[i])/b.value),hessian=this.hessian.map((row,i)=>row.map((v,j)=>(v-value*b.hessian[i][j]-derivatives[i]*b.derivatives[j]-derivatives[j]*b.derivatives[i])/b.value));return new Jet4(value,derivatives,hessian);}
  negate(){return this._unary(-this.value,-1,0);}
  _unary(value,first,second){return new Jet4(value,this.derivatives.map(v=>first*v),this.hessian.map((row,i)=>row.map((v,j)=>first*v+second*this.derivatives[i]*this.derivatives[j])));}
  sin(){return this._unary(Math.sin(this.value),Math.cos(this.value),-Math.sin(this.value));}
  cos(){return this._unary(Math.cos(this.value),-Math.sin(this.value),-Math.cos(this.value));}
  sinh(){return this._unary(Math.sinh(this.value),Math.cosh(this.value),Math.sinh(this.value));}
  cosh(){return this._unary(Math.cosh(this.value),Math.sinh(this.value),Math.cosh(this.value));}
  exp(){const value=Math.exp(this.value);return this._unary(value,value,value);}
  log(){if(this.value<=0)throw new RangeError("jet logarithm requires a positive value");const inverse=1/this.value;return this._unary(Math.log(this.value),inverse,-inverse*inverse);}
  sqrt(){if(this.value<=0)throw new RangeError("jet square root requires a positive value");const value=Math.sqrt(this.value);return this._unary(value,1/(2*value),(-.25/this.value)/value);}
  pow(exponent){
    finite(exponent,"exponent");if(exponent===0)return Jet4.constant(1);if(exponent===1)return this;
    if(this.value<0&&!Number.isInteger(exponent))throw new RangeError("jet power requires a real result");
    if(this.value===0){if(!Number.isInteger(exponent)||exponent<2)throw new RangeError("jet power has an undefined second derivative at zero");return this._unary(0,0,exponent===2?2:0);}
    return this._unary(this.value**exponent,exponent*this.value**(exponent-1),exponent*(exponent-1)*this.value**(exponent-2));
  }
}

/** Evaluate a scalar field, gradient and Hessian with explicit second-order jets. */
export function forwardHessian4(field,point){
  if(typeof field!=="function")throw new TypeError("field must be a function");
  if((!Array.isArray(point)&&!ArrayBuffer.isView(point))||point.length!==4)throw new RangeError("point must have four coordinates");
  const output=field(Array.from(point,(value,axis)=>Jet4.variable(value,axis)));
  if(!(output instanceof Jet4)&&typeof output!=="number")throw new TypeError("forward Hessian requires a scalar field");
  const result=asJet(output);return {value:result.value,gradient:Array.from(result.derivatives),hessian:result.hessian.map(row=>Array.from(row))};
}

/** Evaluate a bounded rectangular field with explicit second-order jets. */
export function forwardSecondOrder4(field,point){
  if(typeof field!=="function")throw new TypeError("field must be a function");
  if((!Array.isArray(point)&&!ArrayBuffer.isView(point))||point.length!==4)throw new RangeError("point must have four coordinates");
  const limit=Math.floor(65536/21),output=field(Array.from(point,(value,axis)=>Jet4.variable(value,axis))),active=new Set();let count=0;
  function unpack(node,depth){
    if(node instanceof Jet4||typeof node==="number"){
      if(++count>limit)throw new RangeError("forward second-order output exceeds the component limit");
      const j=asJet(node);return {value:j.value,jacobian:Array.from(j.derivatives),hessian:j.hessian.map(row=>Array.from(row)),shape:[]};
    }
    if(!Array.isArray(node)||node.length===0)throw new TypeError("forward second-order output must be a nonempty rectangular array or scalar");
    if(depth>=8)throw new RangeError("forward second-order output rank exceeds eight");
    if(node.length>limit)throw new RangeError("forward second-order output exceeds the component limit");
    if(active.has(node))throw new RangeError("forward second-order output cannot contain cycles");
    active.add(node);
    try{
      const children=Array.from(node,child=>unpack(child,depth+1)),shape=children[0].shape;
      if(children.some(child=>JSON.stringify(child.shape)!==JSON.stringify(shape)))throw new RangeError("forward second-order output must be rectangular");
      return {value:children.map(child=>child.value),jacobian:children.map(child=>child.jacobian),hessian:children.map(child=>child.hessian),shape:[node.length,...shape]};
    }finally{active.delete(node);}
  }
  const result=unpack(output,0);return {value:result.value,jacobian:result.jacobian,hessian:result.hessian};
}


/** Differentiate bounded scalar or rectangular outputs through second order. */
export function forwardSecondOrder(field,point){
  if(typeof field!=="function")throw new TypeError("field must be a function");
  if((!Array.isArray(point)&&!(ArrayBuffer.isView(point)&&!(point instanceof DataView)))||point.length<1||point.length>32)throw new RangeError("point must contain from 1 to 32 coordinates");
  const input=Array.from(point,value=>finite(value,"point coordinate")),n=input.length,limit=Math.floor(65536/(1+n+n*n)),groups=Math.ceil(n/2);
  let shape,values,jacobian,hessian;
  for(let left=0;left<groups;left++)for(let right=left;right<groups;right++){
    const variables=input.map((value,i)=>{const local=i-2*left,other=i-2*right;if(local>=0&&local<2)return Jet4.variable(value,local);if(right!==left&&other>=0&&other<2)return Jet4.variable(value,other+2);return Jet4.constant(value);});
    const output=field(variables),active=new Set(),leaves=[];
    function inspect(node,depth){
      if(node instanceof Jet4||typeof node==="number"){if(leaves.length>=limit)throw new RangeError("forward second-order output exceeds the component limit");leaves.push(asJet(node));return [];}
      if(!Array.isArray(node)||node.length===0)throw new TypeError("forward second-order output must be a nonempty rectangular array or scalar");
      if(depth>=8)throw new RangeError("forward second-order output rank exceeds eight");
      if(node.length>limit)throw new RangeError("forward second-order output exceeds the component limit");
      if(active.has(node))throw new RangeError("forward second-order output cannot contain cycles");
      active.add(node);try{const children=Array.from(node,child=>inspect(child,depth+1)),childShape=children[0];if(children.some(child=>JSON.stringify(child)!==JSON.stringify(childShape)))throw new RangeError("forward second-order output must be rectangular");return [node.length,...childShape];}finally{active.delete(node);}
    }
    const outputShape=inspect(output,0),sample=leaves.map(leaf=>leaf.value);
    if(!shape){shape=outputShape;values=sample;jacobian=leaves.map(()=>Array(n).fill(0));hessian=leaves.map(()=>Array.from({length:n},()=>Array(n).fill(0)));}
    else if(JSON.stringify(outputShape)!==JSON.stringify(shape)||sample.some((value,i)=>!Object.is(value,values[i])))throw new RangeError("forward second-order output must remain consistent across sensitivity evaluations");
    const indices=[];for(let lane=0;lane<2&&2*left+lane<n;lane++)indices.push([2*left+lane,lane]);
    if(right!==left)for(let lane=0;lane<2&&2*right+lane<n;lane++)indices.push([2*right+lane,lane+2]);
    for(let row=0;row<leaves.length;row++){
      for(const [axis,lane] of indices)jacobian[row][axis]=leaves[row].derivatives[lane];
      for(const [axis,lane] of indices)for(const [otherAxis,otherLane] of indices)hessian[row][axis][otherAxis]=leaves[row].hessian[lane][otherLane];
    }
  }
  const rebuild=leaves=>{let i=0;const visit=depth=>depth===shape.length?leaves[i++]:Array.from({length:shape[depth]},()=>visit(depth+1));return visit(0);};
  return {value:rebuild(values),jacobian:rebuild(jacobian),hessian:rebuild(hessian)};
}


function prepareForwardTree(point){
  const active=new Set(),values=[];
  function inspect(node,depth){
    if(typeof node==="number"){if(values.length>=32)throw new RangeError("input exceeds the 32-component limit");values.push(finite(node,"input value"));return [];}
    if(ArrayBuffer.isView(node)&&!(node instanceof DataView))node=Array.from(node);
    if(!Array.isArray(node)||node.length===0)throw new TypeError("input must be a nonempty rectangular array or scalar");
    if(depth>=8)throw new RangeError("input rank exceeds eight");
    if(node.length>32)throw new RangeError("input exceeds the 32-component limit");
    if(active.has(node))throw new RangeError("input cannot contain cycles");
    active.add(node);try{const children=Array.from(node,child=>inspect(child,depth+1)),shape=children[0];if(children.some(child=>JSON.stringify(child)!==JSON.stringify(shape)))throw new RangeError("input must be rectangular");return [node.length,...shape];}finally{active.delete(node);}
  }
  const shape=inspect(point,0);
  const rebuild=flat=>{let i=0;const visit=depth=>depth===shape.length?flat[i++]:Array.from({length:shape[depth]},()=>visit(depth+1));return visit(0);};
  return {values,rebuild};
}

/** Forward Jacobian for a scalar or rectangular input, flattened in row-major order. */
export function forwardJacobianTree(field,point){
  if(typeof field!=="function")throw new TypeError("field must be a function");
  const input=prepareForwardTree(point);return forwardJacobian(values=>field(input.rebuild(values)),input.values);
}

/** Forward second derivatives for a scalar or rectangular input, flattened in row-major order. */
export function forwardSecondOrderTree(field,point){
  if(typeof field!=="function")throw new TypeError("field must be a function");
  const input=prepareForwardTree(point);return forwardSecondOrder(values=>field(input.rebuild(values)),input.values);
}
