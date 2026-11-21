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
